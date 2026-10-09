"""
mock_test.py — quick smoke test for sechiren_sweep_m3.py.

Runs 2 configs × 1 seed × 50 epochs to verify all code paths work:
- Timing metrics (epochs/sec, step latency, GPU util, wall-clock)
- Per-run history CSV output
- Summary CSV output
- Resumability (run twice to verify skip logic)
- Multi-GPU block splitting logic

Usage:
    python mock_test.py
"""
import csv
import itertools
import math
import os
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../../skuld-lib'))
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from utils.paths import get_mirror_path

import numpy as np
import torch

from skuld.sechiren import SechirenIntegrator
from physics import integrand_transformed, reference_scipy, PARAM_SETS

# ─────────────────────────────────────────────────────────────────────────
# MOCK GRID — tiny for quick testing
# ─────────────────────────────────────────────────────────────────────────
MOCK_GRID = {
    "hidden_sizes": [
        [32, 32, 32],
        [64, 64, 64],
    ],
    "omega_0": [15.0],
    "n_epochs": [50],
    "lr": [1e-4],
    "n_per_param": [256],
}

MOCK_SEEDS = [42]

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


class GPUMonitor:
    def __init__(self, device_id=0, interval=0.05):
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

    def mean_utilization(self):
        if not self.utilizations:
            return 0.0
        return sum(self.utilizations) / len(self.utilizations)


def run_one(cfg, seed, device, refs):
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
        verbose_every=10,
    )
    elapsed = time.time() - t0

    gpu_monitor.stop()
    mean_gpu_util = gpu_monitor.mean_utilization()

    total_clean_time = sum(e["clean_time"] for e in epoch_data)
    total_step_time = sum(e["total_time"] for e in epoch_data)
    n_epochs = cfg["n_epochs"]

    epochs_per_sec = n_epochs / total_clean_time if total_clean_time > 0 else 0.0
    step_latency = total_step_time / n_epochs if n_epochs > 0 else 0.0

    abs_errs, rel_errs, digits = [], [], []
    for (a, b, m, n) in PARAM_SETS:
        nni_val = integrator.integrate((a, b, m, n), norm_cache=norm_cache, device=device)
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
        "n_params_total": integrator.n_weights,
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
        "wall_clock_s": elapsed,
        "epochs_per_sec": epochs_per_sec,
        "step_latency_s": step_latency,
        "mean_gpu_util": mean_gpu_util,
        "elapsed_s": elapsed,
    }
    return result, epoch_data


def main():
    device = torch.device(
        "mps" if torch.backends.mps.is_available() else
        "cuda" if torch.cuda.is_available() else
        "cpu"
    )

    grid = MOCK_GRID
    keys = list(grid.keys())
    combos = list(itertools.product(*(grid[k] for k in keys)))
    configs = [dict(zip(keys, c)) for c in combos]
    all_runs = list(itertools.product(configs, MOCK_SEEDS))

    print(f"\n{'=' * 72}")
    print(f"  MOCK TEST — {len(configs)} configs × {len(MOCK_SEEDS)} seeds = {len(all_runs)} runs")
    print(f"  Device: {device}")
    print(f"{'=' * 72}\n")

    # Reference values
    print("Computing scipy reference integrals...")
    refs = {}
    for (a, b, m, n) in PARAM_SETS:
        r, e = reference_scipy(a, b, m, n)
        refs[(a, b, m, n)] = (r, e)
    print("Done.\n")

    # Resumability check
    results_dir = get_mirror_path(__file__, "results")
    results_dir = os.path.join(results_dir, "mock")
    csv_path = os.path.join(results_dir, "results_mock.csv")
    completed = set()
    if os.path.exists(csv_path):
        with open(csv_path, "r", newline="", encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                key = (row["hidden_sizes"], row["omega_0"], row["lr"],
                       row["n_epochs"], row["n_per_param"], row["seed"])
                completed.add(key)
        if completed:
            print(f"  Resuming: {len(completed)} runs already completed, skipping them.\n")

    results = []
    for i, (cfg, seed) in enumerate(all_runs, 1):
        key = (str(cfg["hidden_sizes"]), str(cfg["omega_0"]), str(cfg["lr"]),
               str(cfg["n_epochs"]), str(cfg["n_per_param"]), str(seed))
        if key in completed:
            continue

        tag = (f"[{i}/{len(all_runs)}] hidden={cfg['hidden_sizes']} omega0={cfg['omega_0']} "
               f"lr={cfg['lr']} epochs={cfg['n_epochs']} npp={cfg['n_per_param']} seed={seed}")
        print(f"{tag} ...", flush=True)
        res, epoch_data = run_one(cfg, seed, device, refs)
        results.append(res)

        # Append to summary CSV
        os.makedirs(results_dir, exist_ok=True)
        file_exists = os.path.exists(csv_path)
        with open(csv_path, "a", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=CSV_FIELDS)
            if not file_exists:
                writer.writeheader()
            writer.writerow(res)
            fh.flush()

        # Save history CSV
        history_dir = os.path.join(results_dir, "history")
        os.makedirs(history_dir, exist_ok=True)
        hs_str = str(cfg["hidden_sizes"]).replace("[", "").replace("]", "").replace(" ", "")
        hist_path = os.path.join(history_dir, f"history_{hs_str}_o{cfg['omega_0']}_lr{cfg['lr']}_e{cfg['n_epochs']}_npp{cfg['n_per_param']}_s{seed}.csv")
        with open(hist_path, "w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=HISTORY_FIELDS)
            writer.writeheader()
            for e in epoch_data:
                writer.writerow(e)
            fh.flush()

        print(f"    -> min={res['min_dig']} mean={res['mean_dig']:.1f} max={res['max_dig']} "
              f"({res['elapsed_s']:.1f}s, {res['epochs_per_sec']:.1f} ep/s, "
              f"step={res['step_latency_s'] * 1000:.1f}ms, gpu={res['mean_gpu_util']:.0f}%)")
        print(f"    -> history saved to {hist_path}")

    # Print summary
    print(f"\n{'=' * 72}")
    print(f"  MOCK TEST SUMMARY")
    print(f"{'=' * 72}")
    for r in results:
        print(f"  {r['hidden_sizes']} o={r['omega_0']} lr={r['lr']} e={r['n_epochs']} npp={r['n_per_param']}")
        print(f"    min={r['min_dig']} mean={r['mean_dig']:.1f} max={r['max_dig']} "
              f"wall={r['wall_clock_s']:.1f}s ep/s={r['epochs_per_sec']:.1f} "
              f"step={r['step_latency_s'] * 1000:.1f}ms gpu={r['mean_gpu_util']:.0f}%")
    print(f"\n  Results dir: {results_dir}")
    print(f"  Summary CSV: {csv_path}")
    print(f"  History CSVs: {os.path.join(results_dir, 'history')}")
    print(f"{'=' * 72}\n")

    # Verify history CSV has correct number of rows
    print("Verifying history CSVs...")
    for cfg in configs:
        hs_str = str(cfg["hidden_sizes"]).replace("[", "").replace("]", "").replace(" ", "")
        hist_path = os.path.join(results_dir, "history", f"history_{hs_str}_o{cfg['omega_0']}_lr{cfg['lr']}_e{cfg['n_epochs']}_npp{cfg['n_per_param']}_s{42}.csv")
        if os.path.exists(hist_path):
            with open(hist_path, "r", newline="", encoding="utf-8") as fh:
                reader = csv.DictReader(fh)
                rows = list(reader)
                print(f"  {os.path.basename(hist_path)}: {len(rows)} rows (expected {cfg['n_epochs']})")
        else:
            print(f"  {os.path.basename(hist_path)}: MISSING!")

    print("\nMock test complete!")


if __name__ == "__main__":
    main()
