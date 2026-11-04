"""Seeded sigmoid Maître sweep; see README for grid, SLURM and provenance."""
import argparse
import csv
import itertools
import math
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[4] / "skuld-lib"))
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import torch
from physics import PARAM_SETS, integrand_transformed, reference_scipy
from skuld.sigmoid_maitre import SigmoidMaitreIntegrator
from utils.paths import get_mirror_path

GRID = {
    "width": (64, 128, 160),
    "lr": (1e-4, 5e-4, 1e-3),
    "epochs": (6000, 8000, 10000),
    "n_per_param": (512, 1024, 2048),
    "output_scale": (1e3, 1e4, 1e5),
}
SEEDS = (42, 137, 2024, 7, 999)
FIELDS = (["run_id", *GRID, "seed", "n_weights", "final_loss", "min_loss", "seconds"]
          + [f"I_{i}_{suffix}" for i in range(1, 9)
             for suffix in ("estimate", "reference", "dig")]
          + ["min_dig", "mean_dig", "max_dig"])


def configurations(smoke=False):
    if smoke:
        return [dict(width=8, lr=1e-3, epochs=2, n_per_param=8, output_scale=1e4)]
    return [dict(zip(GRID, values)) for values in itertools.product(*GRID.values())]


def run_one(cfg, seed, device, refs):
    torch.manual_seed(seed)
    model = SigmoidMaitreIntegrator(hidden_sizes=(cfg["width"],) * 3,
                                    output_scale=cfg["output_scale"])
    started = time.monotonic()
    history, norm = model.train(integrand_transformed, PARAM_SETS,
                                n_epochs=cfg["epochs"], n_per_param=cfg["n_per_param"],
                                lr=cfg["lr"], device=device, verbose_every=0)
    seconds = time.monotonic() - started
    if not all(math.isfinite(x) for x in history):
        raise ValueError("non-finite training loss")
    row = {**cfg, "seed": seed, "n_weights": model.n_weights,
           "final_loss": history[-1], "min_loss": min(history), "seconds": seconds}
    digits = []
    for i, params in enumerate(PARAM_SETS, 1):
        estimate = model.integrate(params, norm_cache=norm, device=device)
        reference = refs[params]
        if not math.isfinite(estimate) or not math.isfinite(reference) or reference == 0:
            raise ValueError(f"invalid estimate or reference for I_{i}")
        error = abs(estimate - reference) / abs(reference)
        digit = -math.log10(error) if error else math.inf
        row.update({f"I_{i}_estimate": estimate, f"I_{i}_reference": reference,
                    f"I_{i}_dig": digit})
        digits.append(digit)
    row.update(min_dig=min(digits), mean_dig=sum(digits) / len(digits), max_dig=max(digits))
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num-gpus", type=int, default=1)
    parser.add_argument("--gpu-id", type=int, default=None)
    parser.add_argument("--run-tag", default=None)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    shard = args.gpu_id if args.gpu_id is not None else int(os.getenv("SLURM_ARRAY_TASK_ID", "0"))
    if args.num_gpus < 1 or not 0 <= shard < args.num_gpus:
        parser.error("gpu-id must be in [0, num-gpus)")
    tag = args.run_tag or os.getenv("SKULD_RUN_TAG") or os.getenv("SLURM_JOB_PARTITION") or "default"
    if tag in (".", "..") or not all(c.isascii() and (c.isalnum() or c in "-_.") for c in tag):
        parser.error("run-tag must contain only ASCII letters, digits, '-', '_' or '.'")
    runs = list(itertools.product(configurations(args.smoke), (42,) if args.smoke else SEEDS))
    assigned = [(i, cfg, seed) for i, (cfg, seed) in enumerate(runs) if i % args.num_gpus == shard]
    if not assigned:
        print("No runs assigned to this shard.")
        return
    directory = get_mirror_path(__file__, "results") / tag / ("smoke" if args.smoke else "full")
    directory.mkdir(parents=True, exist_ok=True)
    # One atomically published CSV per run makes interruption and shard-count changes safe.
    pending = [(i, cfg, seed) for i, cfg, seed in assigned if not (directory / f"run_{i:04d}.csv").exists()]
    print(f"{len(runs)} runs total; shard {shard}/{args.num_gpus}: {len(pending)} pending", flush=True)
    if not pending:
        return
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_default_dtype(torch.float32)
    refs = {p: reference_scipy(*p)[0] for p in PARAM_SETS}
    for i, cfg, seed in pending:
        print(f"run {i}: {cfg}, seed={seed}, device={device}", flush=True)
        row = run_one(cfg, seed, device, refs)
        row["run_id"] = i
        target = directory / f"run_{i:04d}.csv"
        temporary = directory / f".run_{i:04d}.{os.getpid()}.tmp"
        with temporary.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=FIELDS)
            writer.writeheader()
            writer.writerow(row)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, target)
        print(f"  digits min/mean/max: {row['min_dig']:.3f}/{row['mean_dig']:.3f}/{row['max_dig']:.3f}; {row['seconds']:.1f}s", flush=True)


if __name__ == "__main__":
    main()
