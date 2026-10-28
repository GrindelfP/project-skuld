"""
best_learning_curves.py — learning curves for the three best architectures.

Trains SECHIREN, SIREN and WIRE, each at the best configuration found by its
fine sweep, once per seed, and records the full per-epoch history.

Winning rows (ranked by mean correct digits over all runs of a configuration,
seeds x partitions; ties broken by the lowest min digits):

  SECHIREN  (SECHIREN sweep m3) hidden=[80,80,80] omega_0=15.0 lr=2e-4
            epochs=8000 n_per_param=2048          mean_dig 4.525
  SIREN     (SIREN sweep m1)    hidden=[128,128,128] omega_0=40.0 lr=1e-4
            epochs=6000 n_per_param=2048          mean_dig 4.933
  WIRE      (WIRE sweep m1)     entry_width=96 omega_0=20.0 lr=1e-4
            epochs=10000 n_per_param=2048         mean_dig 5.400
            (n_blocks=3, sigma_0=10.0 fixed in that sweep)

Fixed as in the sweeps: output_scale = 1/omega_0^3, n_params=4, n_int_vars=3,
the meson integrand and PARAM_SETS from physics.py.
Seeds are those the sweeps used: 42, 137, 2024, 7, 999.

Parallel dispatch: one worker process per visible CUDA GPU (a single worker on
MPS/CPU). Workers pull runs from a shared queue, so a slow architecture does
not leave a GPU idle. Each worker appends to its own results_gpu<k>.csv.

Resumable: every results_gpu*.csv in the output directory is read at start-up
and finished (architecture, seed) pairs are skipped.

Outputs (under results/grid/<this dir>/[<run-tag>/]):
  results_gpu<k>.csv           one row per run: config, min/mean/max digits,
                               per-integral digits I_1..I_8, timing
  history/history_<arch>_s<seed>.csv
                               per epoch: epoch, loss, clean_time,
                               data_gen_time, total_time, cumulative_time
  --smoke writes results_smoke_gpu<k>.csv and history_smoke/ instead.

Timing note: SechirenIntegrator.train() returns real per-epoch timings. The
SIREN and WIRE integrators return only the loss list, so their history timing
columns are the run's average step latency (loss is exact), as in the sweeps.

Usage:
    python best_learning_curves.py                 # 15 runs on all visible GPUs
    python best_learning_curves.py --smoke         # 50 epochs, tiny batch
    python best_learning_curves.py --arch siren    # one architecture
    python best_learning_curves.py --seeds 42      # one seed
    python best_learning_curves.py --run-tag ampere
"""
import argparse
import csv
import math
import os
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[4] / "skuld-lib"))
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from utils.paths import get_mirror_path

import numpy as np
import torch
import torch.multiprocessing as mp

from physics import integrand_transformed, reference_scipy, PARAM_SETS

# ─────────────────────────────────────────────────────────────────────────
# 1. CONFIGURATIONS (winning rows of the sweeps)
# ─────────────────────────────────────────────────────────────────────────
CONFIGS = {
    "sechiren": dict(hidden_sizes=[80, 80, 80], omega_0=15.0, lr=2e-4,
                     n_epochs=8000, n_per_param=2048),
    "siren": dict(hidden_sizes=[128, 128, 128], omega_0=40.0, lr=1e-4,
                  n_epochs=6000, n_per_param=2048),
    "wire": dict(entry_width=96, n_blocks=3, sigma_0=10.0, omega_0=20.0,
                 lr=1e-4, n_epochs=10000, n_per_param=2048),
}
ARCH_ORDER = ["sechiren", "siren", "wire"]

SEEDS = [42, 137, 2024, 7, 999]

SMOKE_EPOCHS = 50
SMOKE_NPP = 64

CSV_FIELDS = [
    "arch", "hidden_sizes", "entry_width", "n_blocks", "sigma_0", "omega_0",
    "lr", "n_epochs", "n_per_param", "seed",
    "n_params_total", "final_loss", "min_loss", "mean_rel_err", "max_rel_err",
    "I_1_dig", "I_2_dig", "I_3_dig", "I_4_dig",
    "I_5_dig", "I_6_dig", "I_7_dig", "I_8_dig",
    "min_dig", "mean_dig", "max_dig",
    "wall_clock_s", "epochs_per_sec", "step_latency_s", "mean_gpu_util",
    "elapsed_s",
]

HISTORY_FIELDS = [
    "epoch", "loss", "clean_time", "data_gen_time", "total_time", "cumulative_time",
]


# ─────────────────────────────────────────────────────────────────────────
# 2. GPU UTILIZATION MONITOR
# ─────────────────────────────────────────────────────────────────────────
class GPUMonitor:
    """Background thread sampling GPU utilization (pynvml, else torch)."""

    def __init__(self, device_id: int = 0, interval: float = 0.1):
        self.interval = interval
        self.utilizations = []
        self.running = False
        self.thread = None
        self._pynvml = None
        self._handle = None
        try:
            import pynvml
            pynvml.nvmlInit()
            self._handle = pynvml.nvmlDeviceGetHandleByIndex(device_id)
            self._pynvml = pynvml
        except Exception:
            self._pynvml = None

    def start(self):
        self.running = True
        self.thread = threading.Thread(target=self._monitor, daemon=True)
        self.thread.start()

    def _monitor(self):
        while self.running:
            try:
                if self._pynvml is not None:
                    self.utilizations.append(
                        self._pynvml.nvmlDeviceGetUtilizationRates(self._handle).gpu)
                elif torch.cuda.is_available():
                    self.utilizations.append(torch.cuda.utilization())
            except Exception:
                pass
            time.sleep(self.interval)

    def stop(self):
        self.running = False
        if self.thread is not None:
            self.thread.join(timeout=2.0)

    def mean_utilization(self) -> float:
        return sum(self.utilizations) / len(self.utilizations) if self.utilizations else 0.0


# ─────────────────────────────────────────────────────────────────────────
# 3. ONE TRAIN + EVAL RUN
# ─────────────────────────────────────────────────────────────────────────
def build_integrator(arch: str, cfg: dict):
    output_scale = 1.0 / (cfg["omega_0"] ** 3)
    if arch == "sechiren":
        from skuld.sechiren import SechirenIntegrator
        return SechirenIntegrator(n_params=4, n_int_vars=3,
                                  hidden_sizes=cfg["hidden_sizes"],
                                  omega_0=cfg["omega_0"], output_scale=output_scale)
    if arch == "siren":
        from skuld.siren import SirenIntegrator
        return SirenIntegrator(n_params=4, n_int_vars=3,
                               hidden_sizes=cfg["hidden_sizes"],
                               omega_0=cfg["omega_0"], output_scale=output_scale)
    if arch == "wire":
        from skuld.wire import WireIntegrator
        return WireIntegrator(n_params=4, n_int_vars=3,
                              entry_width=cfg["entry_width"], n_blocks=cfg["n_blocks"],
                              omega_0=cfg["omega_0"], sigma_0=cfg["sigma_0"],
                              output_scale=output_scale)
    raise ValueError(arch)


def run_one(arch: str, cfg: dict, seed: int, device: torch.device, refs: dict) -> tuple:
    torch.manual_seed(seed)
    np.random.seed(seed)

    integrator = build_integrator(arch, cfg)
    n_params_total = integrator.n_weights

    gpu_monitor = GPUMonitor(device_id=device.index if device.index is not None else 0)
    gpu_monitor.start()

    t0 = time.time()
    out = integrator.train(
        integrand_fn=integrand_transformed,
        param_sets=PARAM_SETS,
        n_epochs=cfg["n_epochs"],
        n_per_param=cfg["n_per_param"],
        lr=cfg["lr"],
        device=device,
        verbose_every=cfg["n_epochs"] + 1,
    )
    elapsed = time.time() - t0
    gpu_monitor.stop()
    mean_gpu_util = gpu_monitor.mean_utilization()

    # SECHIREN returns (history, norm_cache, epoch_data); the rest (history, norm_cache).
    if len(out) == 3:
        history, norm_cache, epoch_data = out
    else:
        history, norm_cache = out
        epoch_data = None

    n_epochs = cfg["n_epochs"]
    if epoch_data is not None:
        total_clean = sum(e["clean_time"] for e in epoch_data)
        total_step = sum(e["total_time"] for e in epoch_data)
        epochs_per_sec = n_epochs / total_clean if total_clean > 0 else 0.0
        step_latency = total_step / n_epochs
    else:
        step_latency = elapsed / n_epochs
        epochs_per_sec = n_epochs / elapsed if elapsed > 0 else 0.0
        epoch_data = [{
            "epoch": i, "loss": loss, "clean_time": step_latency,
            "data_gen_time": 0.0, "total_time": step_latency,
            "cumulative_time": i * step_latency,
        } for i, loss in enumerate(history, 1)]

    abs_errs, rel_errs, digits = [], [], []
    for (a, b, m, n) in PARAM_SETS:
        nni_val = integrator.integrate((a, b, m, n), norm_cache=norm_cache, device=device)
        ref_val, _ = refs[(a, b, m, n)]
        abs_err = abs(nni_val - ref_val)
        rel_errs.append(abs_err / (abs(ref_val) + 1e-30))
        abs_errs.append(abs_err)
        digits.append(max(0, -math.floor(math.log10(abs_err + 1e-30))))

    result = {
        "arch": arch,
        "hidden_sizes": str(cfg.get("hidden_sizes", "")),
        "entry_width": cfg.get("entry_width", ""),
        "n_blocks": cfg.get("n_blocks", ""),
        "sigma_0": cfg.get("sigma_0", ""),
        "omega_0": cfg["omega_0"],
        "lr": cfg["lr"],
        "n_epochs": n_epochs,
        "n_per_param": cfg["n_per_param"],
        "seed": seed,
        "n_params_total": n_params_total,
        "final_loss": history[-1],
        "min_loss": min(history),
        "mean_rel_err": sum(rel_errs) / len(rel_errs),
        "max_rel_err": max(rel_errs),
        "min_dig": min(digits),
        "mean_dig": sum(digits) / len(digits),
        "max_dig": max(digits),
        "wall_clock_s": elapsed,
        "epochs_per_sec": epochs_per_sec,
        "step_latency_s": step_latency,
        "mean_gpu_util": mean_gpu_util,
        "elapsed_s": elapsed,
    }
    for k, d in enumerate(digits, 1):
        result[f"I_{k}_dig"] = d
    return result, epoch_data


# ─────────────────────────────────────────────────────────────────────────
# 4. CSV HELPERS
# ─────────────────────────────────────────────────────────────────────────
def load_completed(results_dir: Path, prefix: str) -> set:
    """(arch, seed) pairs found in every <prefix>_gpu*.csv in results_dir."""
    done = set()
    for p in results_dir.glob(f"{prefix}_gpu*.csv"):
        with open(p, newline="", encoding="utf-8") as fh:
            for row in csv.DictReader(fh):
                done.add((row["arch"], int(row["seed"])))
    return done


def append_result(csv_path: Path, result: dict):
    exists = csv_path.exists()
    with open(csv_path, "a", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=CSV_FIELDS)
        if not exists:
            w.writeheader()
        w.writerow(result)
        fh.flush()


def save_history_csv(history_dir: Path, arch: str, seed: int, epoch_data: list):
    history_dir.mkdir(parents=True, exist_ok=True)
    with open(history_dir / f"history_{arch}_s{seed}.csv", "w", newline="",
              encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=HISTORY_FIELDS)
        w.writeheader()
        for e in epoch_data:
            w.writerow(e)


# ─────────────────────────────────────────────────────────────────────────
# 5. WORKER / MAIN
# ─────────────────────────────────────────────────────────────────────────
def worker(worker_id: int, device_str: str, queue, results_dir: str,
           prefix: str, history_name: str, smoke: bool):
    device = torch.device(device_str)
    if device.type == "cuda":
        torch.cuda.set_device(device)
    results_dir = Path(results_dir)
    csv_path = results_dir / f"{prefix}_gpu{worker_id}.csv"

    # Reference integrals are cheap relative to training; each worker computes its own.
    refs = {}
    for (a, b, m, n) in PARAM_SETS:
        refs[(a, b, m, n)] = reference_scipy(a, b, m, n)

    while True:
        job = queue.get()
        if job is None:
            break
        arch, seed = job
        cfg = dict(CONFIGS[arch])
        if smoke:
            cfg["n_epochs"], cfg["n_per_param"] = SMOKE_EPOCHS, SMOKE_NPP
        print(f"[worker {worker_id} {device_str}] {arch} seed={seed} "
              f"epochs={cfg['n_epochs']} npp={cfg['n_per_param']} ...", flush=True)
        try:
            res, epoch_data = run_one(arch, cfg, seed, device, refs)
            append_result(csv_path, res)
            save_history_csv(results_dir / history_name, arch, seed, epoch_data)
            print(f"    [worker {worker_id}] {arch} s{seed} -> "
                  + " ".join(f"I{k}={res[f'I_{k}_dig']}" for k in range(1, 9))
                  + f" min={res['min_dig']} mean={res['mean_dig']:.1f} max={res['max_dig']} "
                  f"({res['elapsed_s']:.1f}s, {res['epochs_per_sec']:.1f} ep/s, "
                  f"gpu={res['mean_gpu_util']:.0f}%)", flush=True)
        except Exception as exc:
            print(f"    [worker {worker_id}] {arch} s{seed} FAILED: {exc!r}", flush=True)


def main():
    parser = argparse.ArgumentParser(description="Learning curves for the best SECHIREN/SIREN/WIRE configs")
    parser.add_argument("--smoke", action="store_true",
                        help=f"{SMOKE_EPOCHS} epochs, batch {SMOKE_NPP}; writes *_smoke files")
    parser.add_argument("--arch", nargs="+", choices=ARCH_ORDER, default=ARCH_ORDER)
    parser.add_argument("--seeds", type=int, nargs="+", default=SEEDS)
    parser.add_argument("--run-tag", default=None,
                        help="subdirectory under the results dir (default: $SKULD_RUN_TAG, "
                             "then $SLURM_JOB_PARTITION)")
    args = parser.parse_args()

    run_tag = (args.run_tag or os.environ.get("SKULD_RUN_TAG")
               or os.environ.get("SLURM_JOB_PARTITION"))
    if run_tag:
        run_tag = "".join(c if (c.isalnum() or c in "-_.") else "_" for c in run_tag)

    if torch.cuda.is_available():
        devices = [f"cuda:{i}" for i in range(torch.cuda.device_count())]
    elif torch.backends.mps.is_available():
        devices = ["mps"]
    else:
        devices = ["cpu"]

    prefix = "results_smoke" if args.smoke else "results"
    history_name = "history_smoke" if args.smoke else "history"

    results_dir = get_mirror_path(__file__, "results")
    if run_tag:
        results_dir = results_dir / run_tag
        results_dir.mkdir(parents=True, exist_ok=True)

    completed = load_completed(results_dir, prefix)
    jobs = [(a, s) for s in args.seeds for a in args.arch if (a, s) not in completed]

    print(f"\n{'=' * 72}")
    print(f"  BEST-CONFIG LEARNING CURVES{'  [SMOKE]' if args.smoke else ''}")
    print(f"  {len(args.arch)} architectures x {len(args.seeds)} seeds; "
          f"{len(completed)} done, {len(jobs)} to run; workers={devices}")
    for a in args.arch:
        print(f"  {a:<9} {CONFIGS[a]}")
    print(f"{'=' * 72}\n", flush=True)

    if not jobs:
        print("  Nothing to do (all runs already completed).")
        return

    ctx = mp.get_context("spawn")
    queue = ctx.Queue()
    for j in jobs:
        queue.put(j)
    n_workers = min(len(devices), len(jobs))
    for _ in range(n_workers):
        queue.put(None)

    procs = [ctx.Process(target=worker,
                         args=(i, devices[i], queue, str(results_dir), prefix,
                               history_name, args.smoke))
             for i in range(n_workers)]
    for p in procs:
        p.start()
    for p in procs:
        p.join()

    # ── summary: min/mean/max digits per run, then per architecture ───────
    rows = []
    for p in sorted(results_dir.glob(f"{prefix}_gpu*.csv")):
        with open(p, newline="", encoding="utf-8") as fh:
            rows.extend(csv.DictReader(fh))
    print(f"\n{'=' * 72}\n  SUMMARY ({len(rows)} runs on disk)\n{'=' * 72}")
    for a in ARCH_ORDER:
        rs = [r for r in rows if r["arch"] == a]
        if not rs:
            continue
        for r in sorted(rs, key=lambda r: int(r["seed"])):
            print(f"  {a:<9} seed={r['seed']:>5} min={r['min_dig']} "
                  f"mean={float(r['mean_dig']):.2f} max={r['max_dig']} "
                  f"final_loss={float(r['final_loss']):.3e}")
        print(f"  {a:<9} MEAN over {len(rs)} seeds: "
              f"mean_dig={sum(float(r['mean_dig']) for r in rs) / len(rs):.2f}\n")
    print(f"  Results -> {results_dir}\n")


if __name__ == "__main__":
    main()
