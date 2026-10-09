"""
sechiren_sweep_m3.py — multi-GPU hyperparameter sweep for SechirenIntegrator.

Fine grid sweep around the best-known SECHIREN config overall (from m1):
  [64,64,64], omega_0=15, lr=1e-4, epochs=8000, n_per_param=1024
  (mean_dig=4.9, best individual run across all SECHIREN tests)

Explores 5 hyperparameters in a full factorial grid:
  hidden_sizes, omega_0, n_epochs, lr, n_per_param

Fixed:
  output_scale=1/omega_0^3, weight_decay=0, corner=off

Multi-GPU via SLURM job array:
  Each job reads SLURM_ARRAY_TASK_ID and processes a contiguous block
  of the (config x seed) list. Results are written to per-GPU CSVs.

Resumable:
  Reads existing per-GPU CSV at startup, skips completed runs.

Timing metrics per run:
  - epochs/sec (clean): forward-backward pass only, no data-gen overhead
  - step latency: total per-epoch time (data-gen + forward-backward)
  - GPU utilization: mean GPU utilization during training (pynvml or torch fallback)
  - wall-clock: total training time

Per-run history CSV:
  One CSV per run with per-epoch rows: epoch, loss, clean_time, data_gen_time,
  total_time, cumulative_time. Saved to results/history/.

Usage:
    python sechiren_sweep_m3.py                    # full grid, 5 seeds, 1 GPU
    python sechiren_sweep_m3.py --num-gpus 4       # 4 GPUs (SLURM array)
    python sechiren_sweep_m3.py --quick            # tiny grid for testing
    python sechiren_sweep_m3.py --seeds 42         # single seed
"""
import argparse
import csv
import itertools
import math
import os
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[4] / "skuld-lib"))

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from utils.paths import get_mirror_path

import numpy as np
import torch

from skuld.sechiren import SechirenIntegrator
from physics import integrand_transformed, reference_scipy, PARAM_SETS

# ─────────────────────────────────────────────────────────────────────────
# 1. HYPERPARAMETER GRID
# ─────────────────────────────────────────────────────────────────────────
GRID = {
    "hidden_sizes": [
        [48, 48, 48],
        [64, 64, 64],
        [80, 80, 80],
    ],
    "omega_0": [12.5, 15.0, 17.5],
    "n_epochs": [6000, 8000, 10000],
    "lr": [5e-5, 1e-4, 2e-4],
    "n_per_param": [512, 1024, 2048],
}

QUICK_GRID = {
    "hidden_sizes": [[64, 64, 64]],
    "omega_0": [15.0],
    "n_epochs": [8000],
    "lr": [1e-4],
    "n_per_param": [1024],
}

SEEDS = [42, 137, 2024, 7, 999]

CSV_FIELDS = [
    "hidden_sizes", "omega_0", "lr", "n_epochs", "n_per_param", "seed",
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
    """Background thread that samples GPU utilization during training.

    Uses pynvml if available, falls back to torch.cuda.utilization().
    """

    def __init__(self, device_id: int = 0, interval: float = 0.1):
        self.device_id = device_id
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
        except (ImportError, Exception):
            self._pynvml = None

    def start(self):
        self.running = True
        self.thread = threading.Thread(target=self._monitor, daemon=True)
        self.thread.start()

    def _monitor(self):
        while self.running:
            try:
                if self._pynvml is not None:
                    util = self._pynvml.nvmlDeviceGetUtilizationRates(self._handle)
                    self.utilizations.append(util.gpu)
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
        if not self.utilizations:
            return 0.0
        return sum(self.utilizations) / len(self.utilizations)


# ─────────────────────────────────────────────────────────────────────────
# 3. ONE TRAIN + EVAL RUN
# ─────────────────────────────────────────────────────────────────────────
def run_one(cfg: dict, seed: int, device: torch.device,
            refs: dict, verbose_every: int) -> dict:
    torch.manual_seed(seed)
    np.random.seed(seed)

    omega_0 = cfg["omega_0"]
    output_scale = 1.0 / (omega_0 ** 3)

    integrator = SechirenIntegrator(
        n_params=4,
        n_int_vars=3,
        hidden_sizes=cfg["hidden_sizes"],
        omega_0=omega_0,
        output_scale=output_scale,
    )
    n_params_total = integrator.n_weights

    # ── GPU monitor ──────────────────────────────────────────────────────
    gpu_monitor = GPUMonitor(device_id=device.index if device.index is not None else 0)
    gpu_monitor.start()

    t0 = time.time()
    history, norm_cache, epoch_data = integrator.train(
        integrand_fn=integrand_transformed,
        param_sets=PARAM_SETS,
        n_epochs=cfg["n_epochs"],
        n_per_param=cfg["n_per_param"],
        lr=cfg["lr"],
        device=device,
        verbose_every=verbose_every if verbose_every > 0 else cfg["n_epochs"] + 1,
    )
    elapsed = time.time() - t0

    gpu_monitor.stop()
    mean_gpu_util = gpu_monitor.mean_utilization()

    # ── timing metrics ───────────────────────────────────────────────────
    total_clean_time = sum(e["clean_time"] for e in epoch_data)
    total_data_gen_time = sum(e["data_gen_time"] for e in epoch_data)
    total_step_time = sum(e["total_time"] for e in epoch_data)
    n_epochs = cfg["n_epochs"]

    epochs_per_sec = n_epochs / total_clean_time if total_clean_time > 0 else 0.0
    step_latency = total_step_time / n_epochs if n_epochs > 0 else 0.0
    wall_clock = elapsed

    # ── accuracy vs. precomputed scipy reference ──────────────────────────
    abs_errs, rel_errs, digits = [], [], []
    for (a, b, m, n) in PARAM_SETS:
        nni_val = integrator.integrate((a, b, m, n),
                                        norm_cache=norm_cache, device=device)
        ref_val, _ = refs[(a, b, m, n)]
        abs_err = abs(nni_val - ref_val)
        rel_err = abs_err / (abs(ref_val) + 1e-30)
        d = max(0, -math.floor(math.log10(abs_err + 1e-30)))
        abs_errs.append(abs_err)
        rel_errs.append(rel_err)
        digits.append(d)

    result = {
        "hidden_sizes": str(cfg["hidden_sizes"]),
        "omega_0": omega_0,
        "lr": cfg["lr"],
        "n_epochs": cfg["n_epochs"],
        "n_per_param": cfg["n_per_param"],
        "seed": seed,
        "n_params_total": n_params_total,
        "final_loss": history[-1],
        "min_loss": min(history),
        "mean_rel_err": sum(rel_errs) / len(rel_errs),
        "max_rel_err": max(rel_errs),
        "I_1_dig": digits[0],
        "I_2_dig": digits[1],
        "I_3_dig": digits[2],
        "I_4_dig": digits[3],
        "I_5_dig": digits[4],
        "I_6_dig": digits[5],
        "I_7_dig": digits[6],
        "I_8_dig": digits[7],
        "min_dig": min(digits),
        "mean_dig": sum(digits) / len(digits),
        "max_dig": max(digits),
        "wall_clock_s": wall_clock,
        "epochs_per_sec": epochs_per_sec,
        "step_latency_s": step_latency,
        "mean_gpu_util": mean_gpu_util,
        "elapsed_s": elapsed,
    }

    return result, epoch_data


# ─────────────────────────────────────────────────────────────────────────
# 4. CSV HELPERS
# ─────────────────────────────────────────────────────────────────────────
def make_configs(grid: dict) -> list:
    keys = list(grid.keys())
    combos = list(itertools.product(*(grid[k] for k in keys)))
    return [dict(zip(keys, c)) for c in combos]


def make_key(cfg: dict, seed: int) -> tuple:
    """Hashable key for deduplication / resumability."""
    return (
        str(cfg["hidden_sizes"]),
        str(cfg["omega_0"]),
        str(cfg["lr"]),
        str(cfg["n_epochs"]),
        str(cfg["n_per_param"]),
        str(seed),
    )


def load_completed_runs(csv_path: str) -> set:
    """Return set of (hidden_sizes, omega_0, lr, n_epochs, n_per_param, seed) tuples."""
    completed = set()
    if not os.path.exists(csv_path):
        return completed
    with open(csv_path, "r", newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            key = (
                row["hidden_sizes"],
                row["omega_0"],
                row["lr"],
                row["n_epochs"],
                row["n_per_param"],
                row["seed"],
            )
            completed.add(key)
    return completed


def append_result(csv_path: str, result: dict):
    """Append one result row to CSV, writing header if file is new."""
    file_exists = os.path.exists(csv_path)
    with open(csv_path, "a", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=CSV_FIELDS)
        if not file_exists:
            writer.writeheader()
        writer.writerow(result)
        fh.flush()


def save_history_csv(history_dir: str, cfg: dict, seed: int, epoch_data: list):
    """Save per-epoch training history (loss curve + timing) to CSV."""
    os.makedirs(history_dir, exist_ok=True)
    filename = (
        f"history_{cfg['hidden_sizes']}_o{cfg['omega_0']}_"
        f"lr{cfg['lr']}_e{cfg['n_epochs']}_npp{cfg['n_per_param']}_s{seed}.csv"
    )
    # Sanitize filename
    filename = filename.replace("[", "").replace("]", "").replace(" ", "")
    filepath = os.path.join(history_dir, filename)

    with open(filepath, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=HISTORY_FIELDS)
        writer.writeheader()
        for e in epoch_data:
            writer.writerow(e)
        fh.flush()


# ─────────────────────────────────────────────────────────────────────────
# 5. MAIN SWEEP
# ─────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Multi-GPU hyperparameter sweep for SechirenIntegrator")
    parser.add_argument("--quick", action="store_true",
                        help="use a tiny hyperparameter grid for testing")
    parser.add_argument("--num-gpus", type=int, default=1,
                        help="total number of GPUs in the job array (default: 1)")
    parser.add_argument("--gpu-id", type=int, default=None,
                        help="GPU ID to process (default: SLURM_ARRAY_TASK_ID env var, or 0)")
    parser.add_argument("--seeds", type=int, nargs="+", default=SEEDS,
                        help=f"seeds to run (default: {SEEDS})")
    parser.add_argument("--verbose-every", type=int, default=0,
                        help="print training progress every N epochs (0 = silent)")
    args = parser.parse_args()

    # ── determine GPU ID ──────────────────────────────────────────────────
    gpu_id = args.gpu_id if args.gpu_id is not None else int(os.environ.get("SLURM_ARRAY_TASK_ID", "0"))
    num_gpus = args.num_gpus

    device = torch.device(
        "mps" if torch.backends.mps.is_available() else
        "cuda" if torch.cuda.is_available() else
        "cpu"
    )

    grid = QUICK_GRID if args.quick else GRID
    configs = make_configs(grid)
    all_runs = list(itertools.product(configs, args.seeds))

    # ── split into contiguous blocks for each GPU ─────────────────────────
    block_size = (len(all_runs) + num_gpus - 1) // num_gpus
    start_idx = gpu_id * block_size
    end_idx = min(start_idx + block_size, len(all_runs))
    my_runs = all_runs[start_idx:end_idx]

    print(f"\n{'=' * 72}")
    print(f"  SECHIREN SWEEP m3  —  {len(configs)} configs × {len(args.seeds)} seeds "
          f"= {len(all_runs)} total runs")
    print(f"  GPU {gpu_id}/{num_gpus}  —  processing runs [{start_idx}:{end_idx}] "
          f"= {len(my_runs)} runs, device={device}")
    print(f"{'=' * 72}\n")

    if not my_runs:
        print(f"  GPU {gpu_id}: no runs to process (block empty).")
        return

    # ── reference values (shared across all configs) ──────────────────────
    print("Computing scipy reference integrals (shared across all configs)...")
    refs = {}
    for (a, b, m, n) in PARAM_SETS:
        r, e = reference_scipy(a, b, m, n)
        refs[(a, b, m, n)] = (r, e)
    print("Done.\n")

    # ── resumability: load completed runs ─────────────────────────────────
    results_dir = get_mirror_path(__file__, "results")
    csv_path = os.path.join(results_dir, f"results_gpu{gpu_id}.csv")
    completed = load_completed_runs(csv_path)
    if completed:
        print(f"  Resuming: {len(completed)} runs already completed, skipping them.\n")

    # ── main loop ─────────────────────────────────────────────────────────
    results = []
    for i, (cfg, seed) in enumerate(my_runs, 1):
        key = make_key(cfg, seed)
        if key in completed:
            continue

        tag = (f"[{i}/{len(my_runs)}] hidden={cfg['hidden_sizes']} omega0={cfg['omega_0']} "
               f"lr={cfg['lr']} epochs={cfg['n_epochs']} npp={cfg['n_per_param']} seed={seed}")
        print(f"{tag} ...", flush=True)
        try:
            res, epoch_data = run_one(cfg, seed, device, refs, args.verbose_every)
            results.append(res)
            append_result(csv_path, res)

            # Save per-run history CSV
            history_dir = os.path.join(results_dir, "history")
            save_history_csv(history_dir, cfg, seed, epoch_data)

            print(f"    -> I1={res['I_1_dig']} I2={res['I_2_dig']} I3={res['I_3_dig']} "
                  f"I4={res['I_4_dig']} I5={res['I_5_dig']} I6={res['I_6_dig']} "
                  f"I7={res['I_7_dig']} I8={res['I_8_dig']} "
                  f"min={res['min_dig']} mean={res['mean_dig']:.1f} max={res['max_dig']} "
                  f"({res['elapsed_s']:.1f}s, {res['epochs_per_sec']:.1f} ep/s, "
                  f"step={res['step_latency_s'] * 1000:.1f}ms, gpu={res['mean_gpu_util']:.0f}%)")
        except Exception as exc:
            print(f"    -> FAILED: {exc}")

    if not results:
        print(f"\n  GPU {gpu_id}: no new runs completed (all were already done).")
        return

    # ── rank: best = highest min_dig, tiebreak by mean_rel_err, tiebreak by fewer epochs ──
    results.sort(key=lambda r: (-r["min_dig"], r["mean_rel_err"], r["n_epochs"]))

    SEP = "=" * 140
    print(f"\n{SEP}")
    print(f"  RANKED RESULTS (best first) — GPU {gpu_id}")
    print(SEP)
    header = (f"  {'#':>3} {'hidden':<16} {'omega0':>6} {'lr':>8} {'epochs':>6} "
              f"{'npp':>5} {'seed':>5} {'min_loss':>10} {'mean_relerr':>11} "
              f"{'I1':>3} {'I2':>3} {'I3':>3} {'I4':>3} {'I5':>3} {'I6':>3} {'I7':>3} {'I8':>3} "
              f"{'min':>4} {'mean':>5} {'max':>4} {'wall':>6} {'ep/s':>6} {'step':>6} {'gpu%':>5}")
    print(header)
    print("-" * 140)
    for i, r in enumerate(results, 1):
        print(f"  {i:>3} {r['hidden_sizes']:<16} {r['omega_0']:>6.1f} {r['lr']:>8.1e} "
              f"{r['n_epochs']:>6} {r['n_per_param']:>5} {r['seed']:>5} "
              f"{r['min_loss']:>10.3e} {r['mean_rel_err']:>11.3e} "
              f"{r['I_1_dig']:>3} {r['I_2_dig']:>3} {r['I_3_dig']:>3} {r['I_4_dig']:>3} "
              f"{r['I_5_dig']:>3} {r['I_6_dig']:>3} {r['I_7_dig']:>3} {r['I_8_dig']:>3} "
              f"{r['min_dig']:>4} {r['mean_dig']:>5.1f} {r['max_dig']:>4} "
              f"{r['wall_clock_s']:>6.1f} {r['epochs_per_sec']:>6.1f} "
              f"{r['step_latency_s'] * 1000:>6.1f} {r['mean_gpu_util']:>5.0f}")
    print(SEP)
    best = results[0]
    print(f"\n  BEST CONFIG (GPU {gpu_id}):")
    print(f"    hidden_sizes = {best['hidden_sizes']}")
    print(f"    omega_0      = {best['omega_0']}")
    print(f"    lr           = {best['lr']}")
    print(f"    n_epochs     = {best['n_epochs']}")
    print(f"    n_per_param  = {best['n_per_param']}")
    print(f"    seed         = {best['seed']}")
    print(f"    -> min {best['min_dig']} digits, "
          f"mean {best['mean_dig']:.1f} digits, "
          f"max {best['max_dig']} digits, "
          f"mean rel. err {best['mean_rel_err']:.3e}")
    print(f"    -> {best['epochs_per_sec']:.1f} ep/s, "
          f"step {best['step_latency_s'] * 1000:.1f}ms, "
          f"gpu {best['mean_gpu_util']:.0f}%, "
          f"wall {best['wall_clock_s']:.1f}s\n")

    # ── per-config seed-averaged summary ─────────────────────────────────
    print(f"{SEP}")
    print(f"  PER-CONFIG SUMMARY (seed-averaged) — GPU {gpu_id}")
    print(SEP)
    by_cfg = {}
    for r in results:
        key = (r["hidden_sizes"], r["omega_0"], r["lr"], r["n_epochs"], r["n_per_param"])
        by_cfg.setdefault(key, []).append(r)
    for key, rs in sorted(by_cfg.items(),
                          key=lambda kv: (-min(r["min_dig"] for r in kv[1]),
                                          sum(r["mean_rel_err"] for r in kv[1]) / len(kv[1]))):
        hs, om, lr, ep, npp = key
        min_d = min(r["min_dig"] for r in rs)
        mean_d = sum(r["mean_dig"] for r in rs) / len(rs)
        max_d = max(r["max_dig"] for r in rs)
        mean_rel = sum(r["mean_rel_err"] for r in rs) / len(rs)
        mean_ep_s = sum(r["epochs_per_sec"] for r in rs) / len(rs)
        mean_step = sum(r["step_latency_s"] for r in rs) / len(rs)
        mean_gpu = sum(r["mean_gpu_util"] for r in rs) / len(rs)
        print(f"  hidden={hs} omega0={om} lr={lr} epochs={ep} npp={npp}")
        print(f"      min_dig={min_d}  mean_dig={mean_d:.1f}  max_dig={max_d}  "
              f"mean_relerr={mean_rel:.3e}  n_seeds={len(rs)}")
        print(f"      ep/s={mean_ep_s:.1f}  step={mean_step * 1000:.1f}ms  gpu={mean_gpu:.0f}%")
    print(SEP)
    print(f"  Results saved -> {csv_path}\n")


if __name__ == "__main__":
    main()
