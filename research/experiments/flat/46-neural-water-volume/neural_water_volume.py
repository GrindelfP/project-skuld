"""Learn a meson water surface end to end in PyTorch and integrate its volume."""

import csv
import math
import sys
from pathlib import Path
from statistics import mean
from time import perf_counter

import mpmath as mp
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
sys.path.insert(0, str(Path(__file__).resolve().parents[4] / "skuld-lib"))

from benchmarks.meson_reference import physical_references
from benchmarks.meson_surfaces import known_weights, meson_samples, training_values
from physics import L_LOG, PARAM_SETS, T_MAX
from skuld._neural_water_volume import fit_neural_surface
from utils.paths import get_mirror_path

STARTS = ((42, (3.7, -3.3)), (123, (5.1, -3.8)), (2026, (3.2, -5.2)))
TRAINING_SITES = 384
HOLDOUT_SITES = 4096
TARGET_DIGITS = 12
WEIGHTS = tuple((a, b, m) for a, b, m, _ in PARAM_SETS)


def evaluate(seed: int, start: tuple[float, float]) -> tuple:
    points = np.random.default_rng(seed).random((TRAINING_SITES, 3))
    fitted = fit_neural_surface(points, training_values(points), seed=seed, start_rates=start)
    if min(fitted.layer_updates.values()) <= 1e-6:
        raise RuntimeError("A layer did not learn")
    network = fitted.network
    moments = network.moment_representation()
    log_bound = moments.log_upper_bound()
    scale = L_LOG * math.exp(log_bound)

    holdout = np.random.default_rng(94).random((HOLDOUT_SITES, 3))
    with torch.inference_mode():
        log_output = network(torch.from_numpy(holdout), log=True).numpy()
        predicted = np.exp(log_output - log_bound) * known_weights(holdout, 0, 0, 1)
    conversion_error = float(np.max(np.abs(log_output - moments.log_surface(holdout))))
    if conversion_error > 1e-12:
        raise RuntimeError("Analytic expansion disagrees with the trained network")
    depth = 1 - predicted
    if not np.all((depth >= 0) & (depth <= 1)):
        raise RuntimeError("Water depth left the unit cube")
    expected = meson_samples(holdout, 0, 0, 1) / scale
    holdout_error = float(np.linalg.norm(predicted - expected) / np.linalg.norm(expected))

    coarse = L_LOG * np.array(moments.integrate_weighted(WEIGHTS, L_LOG))
    estimates = L_LOG * np.array(moments.integrate_weighted(WEIGHTS, L_LOG, series_tolerance=1e-24))
    occupied = estimates / scale
    change = np.abs((coarse - estimates) / estimates)
    if np.max(change) > 1e-13 or not np.all((occupied >= 0) & (occupied <= 1)):
        raise RuntimeError("Analytic water-volume calculation did not converge inside the cube")
    # I = scale * (1 - A) with the subtraction simplified before rounding A.
    return fitted, moments.rates, scale, estimates, occupied, holdout_error, conversion_error, change


def main() -> int:
    torch.set_num_threads(1)
    if any(n != m + 1 for _, _, m, n in PARAM_SETS):
        raise ValueError("This envelope requires n = m + 1")
    predictions = []
    for seed, start in STARTS:
        started = perf_counter()
        result = evaluate(seed, start)
        fit, rates, _, _, _, holdout, _, _ = result
        print(f"seed={seed} rates={rates} training RMS {fit.initial_rmse:.3e} -> {fit.final_rmse:.3e}, "
              f"holdout relative L2={holdout:.3e}, {perf_counter() - started:.2f}s", flush=True)
        predictions.append((seed, start, result))

    rows = []
    summaries = []
    with mp.workdps(45):
        coarse = physical_references(26)
        exact = physical_references(36)
        deltas = [abs(a - b) / abs(b) for a, b in zip(coarse, exact, strict=True)]
        if max(deltas) > mp.mpf("1e-15"):
            raise RuntimeError("Independent references did not converge")
        for seed, start, result in predictions:
            fit, rates, scale, estimates, occupied, holdout, expansion, changes = result
            scores = []
            for index, (params, estimate, reference) in enumerate(zip(PARAM_SETS, estimates, exact, strict=True)):
                a, b, m, n = params
                relative = abs(mp.mpf(float(estimate)) - reference) / abs(reference)
                score = float(-mp.log10(relative))
                rows.append({
                    "integral": f"I_{index+1}", "a": a, "b": b, "m": m, "n": n,
                    "seed": seed, "t_max": T_MAX, "estimate": estimate, "reference": mp.nstr(reference, 38),
                    "relative_error": float(relative), "correct_digits": score, "scale": scale,
                    "occupied_volume": occupied[index], "water_volume": 1 - occupied[index],
                    "series_relative_change": changes[index], "reference_relative_change": float(deltas[index]),
                })
                scores.append(score)
            summaries.append({
                "seed": seed, "sites": TRAINING_SITES,
                "parameters": sum(p.numel() for p in fit.network.parameters()),
                "initial_positive_rate": start[0], "initial_negative_rate": start[1],
                "positive_rate": rates[0], "negative_rate": rates[1], "jacobian_evaluations": fit.jacobian_evaluations,
                "initial_training_rmse": fit.initial_rmse, "final_training_rmse": fit.final_rmse,
                "holdout_relative_l2": holdout, "log_expansion_max_abs_error": expansion,
                **{f"{layer}_max_update": update for layer, update in fit.layer_updates.items()},
                "min_digits": min(scores), "mean_digits": mean(scores), "max_digits": max(scores),
                "reached_12": sum(score >= TARGET_DIGITS for score in scores),
            })
            print(f"seed={seed} correct digits min/mean/max {min(scores):.3f}/{mean(scores):.3f}/{max(scores):.3f}", flush=True)

    folder = get_mirror_path(__file__, "results")
    for name, data in (("integrals.csv", rows), ("training.csv", summaries)):
        with (folder / name).open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(data[0]), lineterminator="\n")
            writer.writeheader()
            writer.writerows(data)
    return 0 if all(row["reached_12"] == len(PARAM_SETS) for row in summaries) else 1


if __name__ == "__main__":
    raise SystemExit(main())
