"""Resumable multi-GPU canonical 6-D Genz sweep.

The production sweep covers every architecture registered in
``benchmarks.genz_runner.BUILDERS``, all five canonical Genz functions, and
five seeds. Each architecture keeps its n=6 runner configuration except for a
matched 3-by-3 grid over learning rate and epoch count.

Results are one row per training run and report only absolute error.  The
configuration summary averages that error across the five functions and five
seeds to select one configuration per architecture; the table export retains
the per-function error distribution for that selected configuration.

Examples:
    python genz_6d_sweep.py
    python genz_6d_sweep.py --num-gpus 4
    python genz_6d_sweep.py --run-tag hopper
    python genz_6d_sweep.py --smoke --architectures siren --functions gaussian

The integrators are imported exclusively from this experiment's ``lib6d``
package. Install/copy that package before running the experiment; the published
root ``skuld-lib`` is intentionally not used.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import math
import os
import re
import sys
from collections import defaultdict
from pathlib import Path

LIB6D = Path(__file__).resolve().parent / "lib6d"
if not (LIB6D / "skuld").is_dir():
    raise RuntimeError(f"isolated 6-D package is missing: {LIB6D / 'skuld'}")
sys.path.insert(0, str(LIB6D))
sys.path.insert(1, str(Path(__file__).resolve().parents[3]))

import numpy as np
import torch

from benchmarks.genz import GENZ_FUNCTIONS
from benchmarks.genz_runner import BUILDERS, train_one
from utils.paths import get_mirror_path


N_INT_VARS = 6
SEEDS = [42, 137, 2024, 7, 999]
LR_MULTIPLIERS = (0.5, 1.0, 2.0)
EPOCH_MULTIPLIERS = (0.75, 1.0, 1.25)
HISTORY_INTERVAL = 50

CSV_FIELDS = [
    "architecture_key", "architecture", "function", "n_int_vars", "difficulty",
    "params", "seed", "lr", "n_epochs", "n_per_param", "n_weights",
    "exact", "estimate", "abs_err", "final_loss", "min_loss",
    "wall_clock_s", "epochs_per_sec", "step_latency_s", "device",
]
HISTORY_FIELDS = ["epoch", "loss", "lr"]
TABLE_ARCHITECTURES = {
    "siren": "SIN 6D",
    "cosiren_ps": "COS 6D",
    "tanhiren": "TANH 6D",
    "sechiren": "SECH 6D",
    "gaussian": "GAUSS 6D",
    "wire": "G-M Wavelet 6D",
    "mixiren": "SIN+SECH 6D",
}


def matched_configs(arch_key: str) -> list[dict]:
    """Return the nine lr/epoch configurations around an architecture's runner setup."""
    base = BUILDERS[arch_key]().kwargs_for(N_INT_VARS)
    return [
        {
            **base,
            "lr": base["lr"] * lr_multiplier,
            "n_epochs": round(base["n_epochs"] * epoch_multiplier),
        }
        for lr_multiplier, epoch_multiplier in itertools.product(
            LR_MULTIPLIERS, EPOCH_MULTIPLIERS
        )
    ]


def make_runs(architectures: list[str], functions: list[str], seeds: list[int]) -> list[dict]:
    """Build the deterministic production work list."""
    function_by_name = {fn.name: fn for fn in GENZ_FUNCTIONS}
    return [
        {
            "architecture_key": arch_key,
            "function": function_by_name[function_name],
            "config": config,
            "seed": seed,
        }
        for arch_key in architectures
        for function_name in functions
        for config in matched_configs(arch_key)
        for seed in seeds
    ]


def smoke_runs(runs: list[dict]) -> list[dict]:
    """Shorten runs and deduplicate configs whose epoch axis collapses to 50."""
    shortened = [
        {**run, "config": {**run["config"], "n_epochs": 50}}
        for run in runs
    ]
    unique_runs = {}
    for run in shortened:
        unique_runs.setdefault(run_key(run), run)
    return list(unique_runs.values())


def run_key(run: dict) -> tuple[str, ...]:
    """Stable identity for one architecture/function/configuration/seed training run."""
    cfg = run["config"]
    return (
        run["architecture_key"], run["function"].name, str(cfg["lr"]),
        str(cfg["n_epochs"]), str(cfg["n_per_param"]), str(run["seed"]),
    )


def row_key(row: dict[str, str]) -> tuple[str, ...]:
    return (
        row["architecture_key"], row["function"], row["lr"], row["n_epochs"],
        row["n_per_param"], row["seed"],
    )


def load_completed(csv_path: Path) -> set[tuple[str, ...]]:
    if not csv_path.exists():
        return set()
    with csv_path.open(newline="", encoding="utf-8") as handle:
        return {row_key(row) for row in csv.DictReader(handle)}


def append_row(csv_path: Path, row: dict) -> None:
    exists = csv_path.exists()
    with csv_path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        if not exists:
            writer.writeheader()
        writer.writerow(row)
        handle.flush()


def safe_component(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", value)


def history_path(history_dir: Path, run: dict) -> Path:
    cfg = run["config"]
    return history_dir / (
        f"{run['architecture_key']}_{run['function'].name}_lr{cfg['lr']:.8g}_"
        f"epochs{cfg['n_epochs']}_npp{cfg['n_per_param']}_seed{run['seed']}.csv"
    )


def learning_rate_at(run: dict, epoch: int) -> float:
    """Learning rate after the scheduler step at a one-based epoch."""
    initial = run["config"]["lr"]
    total_epochs = run["config"]["n_epochs"]
    if run["architecture_key"] == "wire":
        cycle = max(1, total_epochs // 4)
        position = epoch % cycle
        minimum = initial / 50.0
        return minimum + (initial - minimum) * (1.0 + math.cos(math.pi * position / cycle)) / 2.0
    minimum = initial / 10.0
    return minimum + (initial - minimum) * (1.0 + math.cos(math.pi * epoch / total_epochs)) / 2.0


def save_history(history_dir: Path, run: dict, history: list[float]) -> None:
    """Save loss and lr at most once per 50 epochs, always retaining the final epoch."""
    history_dir.mkdir(parents=True, exist_ok=True)
    path = history_path(history_dir, run)
    indices = list(range(HISTORY_INTERVAL - 1, len(history), HISTORY_INTERVAL))
    if not indices or indices[-1] != len(history) - 1:
        indices.append(len(history) - 1)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=HISTORY_FIELDS)
        writer.writeheader()
        for index in indices:
            epoch = index + 1
            writer.writerow({"epoch": epoch, "loss": history[index], "lr": learning_rate_at(run, epoch)})


def run_one(run: dict, device: torch.device, verbose_every: int) -> tuple[dict, list[float]]:
    """Train against an integrand only, then evaluate its known closed-form integral."""
    arch_key = run["architecture_key"]
    gfn = run["function"]
    cfg = run["config"]
    seed = run["seed"]
    torch.manual_seed(seed)
    np.random.seed(seed)

    spec = BUILDERS[arch_key]()
    params = gfn.params_for("canonical")
    integrator, norm_cache, history, elapsed = train_one(
        spec, gfn, device, N_INT_VARS, verbose_every=verbose_every,
        params=params, train_kwargs=cfg,
    )
    estimate = integrator.integrate(tuple(params), norm_cache=norm_cache, device=device)
    exact = gfn.exact(N_INT_VARS, params)
    abs_err = abs(estimate - exact)
    epochs = cfg["n_epochs"]

    return {
        "architecture_key": arch_key,
        "architecture": spec.name,
        "function": gfn.name,
        "n_int_vars": N_INT_VARS,
        "difficulty": "canonical",
        "params": str(params),
        "seed": seed,
        "lr": cfg["lr"],
        "n_epochs": epochs,
        "n_per_param": cfg["n_per_param"],
        "n_weights": getattr(integrator, "n_weights", ""),
        "exact": f"{exact:.17g}",
        "estimate": f"{estimate:.17g}",
        "abs_err": f"{abs_err:.6e}",
        "final_loss": f"{history[-1]:.6e}",
        "min_loss": f"{min(history):.6e}",
        "wall_clock_s": f"{elapsed:.2f}",
        "epochs_per_sec": f"{epochs / elapsed if elapsed else 0.0:.6f}",
        "step_latency_s": f"{elapsed / epochs if epochs else 0.0:.6f}",
        "device": str(device),
    }, history


def read_all_rows(results_dir: Path, smoke: bool) -> list[dict[str, str]]:
    """Collect every array-task shard in one namespace for explicit summarisation."""
    suffix = "_smoke" if smoke else ""
    rows = []
    for path in sorted(results_dir.glob(f"results_gpu*{suffix}.csv")):
        with path.open(newline="", encoding="utf-8") as handle:
            rows.extend(csv.DictReader(handle))
    return rows


def write_summaries(results_dir: Path, smoke: bool) -> None:
    """Write config and Table-1-shaped absolute-error summaries from all shards.

    A configuration is selected per architecture by its mean absolute error over
    the five functions and five seeds.  The table export then retains each
    function's exact integral, mean estimate, and min/mean/max absolute error.
    """
    rows = read_all_rows(results_dir, smoke)
    if not rows:
        print("No result rows available for summary.")
        return

    by_config: dict[tuple[str, str, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        by_config[(row["architecture_key"], row["lr"], row["n_epochs"])].append(row)
    config_rows = []
    selected = {}
    for key, group in sorted(by_config.items()):
        errors = [float(row["abs_err"]) for row in group]
        architecture_key, lr, epochs = key
        config_rows.append({
            "architecture_key": architecture_key,
            "architecture": group[0]["architecture"],
            "lr": lr,
            "n_epochs": epochs,
            "n_runs": len(group),
            "n_functions": len({row["function"] for row in group}),
            "n_seeds": len({row["seed"] for row in group}),
            "abs_err_min": f"{min(errors):.6e}",
            "abs_err_mean": f"{sum(errors) / len(errors):.6e}",
            "abs_err_max": f"{max(errors):.6e}",
        })
        previous = selected.get(architecture_key)
        if previous is None or float(config_rows[-1]["abs_err_mean"]) < float(previous["abs_err_mean"]):
            selected[architecture_key] = config_rows[-1]

    suffix = "_smoke" if smoke else ""
    config_path = results_dir / f"config_abs_err_summary{suffix}.csv"
    with config_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(config_rows[0]))
        writer.writeheader()
        writer.writerows(config_rows)

    table_rows = []
    for architecture_key, chosen in sorted(selected.items()):
        chosen_rows = by_config[(architecture_key, chosen["lr"], chosen["n_epochs"])]
        by_function: dict[str, list[dict[str, str]]] = defaultdict(list)
        for row in chosen_rows:
            by_function[row["function"]].append(row)
        for function, group in sorted(by_function.items()):
            errors = [float(row["abs_err"]) for row in group]
            estimates = [float(row["estimate"]) for row in group]
            table_rows.append({
                "architecture_key": architecture_key,
                "architecture": group[0]["architecture"],
                "table_architecture": TABLE_ARCHITECTURES[architecture_key],
                "selected_lr": chosen["lr"],
                "selected_n_epochs": chosen["n_epochs"],
                "function": function,
                "I_star": group[0]["exact"],
                "I_mean": f"{sum(estimates) / len(estimates):.17g}",
                "abs_err_min": f"{min(errors):.6e}",
                "abs_err_mean": f"{sum(errors) / len(errors):.6e}",
                "abs_err_max": f"{max(errors):.6e}",
                "n_seeds": len(group),
            })
    table_path = results_dir / f"table1_absolute_error{suffix}.csv"
    with table_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(table_rows[0]))
        writer.writeheader()
        writer.writerows(table_rows)
    print(f"Absolute-error config summary: {config_path}")
    print(f"Table-1-shaped absolute-error export: {table_path}")


def resolve_results_dir(run_tag: str | None) -> Path:
    results_dir = get_mirror_path(__file__, "results")
    if run_tag:
        results_dir = results_dir / safe_component(run_tag)
        results_dir.mkdir(parents=True, exist_ok=True)
    return results_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Canonical 6-D Genz absolute-error sweep")
    parser.add_argument("--num-gpus", type=int, default=1)
    parser.add_argument("--gpu-id", type=int, default=None)
    parser.add_argument("--run-tag", default=None)
    parser.add_argument("--seeds", type=int, nargs="+", default=SEEDS)
    parser.add_argument("--architectures", nargs="+", choices=sorted(BUILDERS), default=sorted(BUILDERS))
    parser.add_argument("--functions", nargs="+", choices=[fn.name for fn in GENZ_FUNCTIONS],
                        default=[fn.name for fn in GENZ_FUNCTIONS])
    parser.add_argument("--smoke", action="store_true",
                        help="use 50 epochs and write smoke-named result files only")
    parser.add_argument("--summarize", action="store_true",
                        help="write summaries from every completed array-task shard, without training")
    parser.add_argument("--verbose-every", type=int, default=0)
    args = parser.parse_args()

    run_tag = args.run_tag or os.environ.get("SKULD_RUN_TAG") or os.environ.get("SLURM_JOB_PARTITION")
    gpu_id = args.gpu_id if args.gpu_id is not None else int(os.environ.get("SLURM_ARRAY_TASK_ID", "0"))
    # SLURM exposes a single allocated GPU to each array task; use that visible
    # device rather than treating the array-task index as a CUDA ordinal.
    device = torch.device("cuda:0" if torch.cuda.is_available() else
                          "mps" if torch.backends.mps.is_available() else "cpu")

    if args.summarize:
        write_summaries(resolve_results_dir(run_tag), args.smoke)
        return

    runs = make_runs(args.architectures, args.functions, args.seeds)
    if args.smoke:
        runs = smoke_runs(runs)
        # The three epoch axis values all become 50 for a smoke run, so retain
        # only the distinct runnable configurations rather than training them
        # three times under the same resume key.
        unique_runs = {}
        for run in runs:
            unique_runs.setdefault(run_key(run), run)
        runs = list(unique_runs.values())
    block_size = (len(runs) + args.num_gpus - 1) // args.num_gpus
    my_runs = runs[gpu_id * block_size:min((gpu_id + 1) * block_size, len(runs))]
    suffix = "_smoke" if args.smoke else ""

    print(
        f"Canonical 6-D Genz sweep: {len(runs)} training runs; GPU {gpu_id}/{args.num_gpus}; "
        f"local block={len(my_runs)}; device={device}"
    )
    if run_tag:
        print(f"Run tag: {safe_component(run_tag)}")
    if not my_runs:
        return

    results_dir = resolve_results_dir(run_tag)
    csv_path = results_dir / f"results_gpu{gpu_id}{suffix}.csv"
    completed = load_completed(csv_path)
    if completed:
        print(f"Resuming: skipping {len(completed)} completed runs.")

    rows = []
    for number, run in enumerate(my_runs, start=1):
        if run_key(run) in completed:
            continue
        cfg = run["config"]
        print(
            f"[{number}/{len(my_runs)}] {run['architecture_key']} {run['function'].name} "
            f"lr={cfg['lr']:.1e} epochs={cfg['n_epochs']} seed={run['seed']}", flush=True
        )
        row, history = run_one(run, device, args.verbose_every or cfg["n_epochs"] + 1)
        append_row(csv_path, row)
        save_history(results_dir / f"history{suffix}", run, history)
        rows.append(row)
        print(f"  abs_err={float(row['abs_err']):.3e} loss={float(row['final_loss']):.3e} "
              f"wall={float(row['wall_clock_s']):.1f}s")

    if not rows:
        print("No new runs completed.")
        return
    print(f"Results: {csv_path}")


if __name__ == "__main__":
    main()
