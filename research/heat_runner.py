"""Shared fixed-configuration heat pilots; reference values are scoring-only."""
import argparse
import csv
import json
import math
import time
import statistics

import torch

from skuld.diffusion_integrator import DiffusionIntegrator
from utils.paths import get_mirror_path


def synthetic(x):
    d = x.shape[1]
    z = x[:, -1]
    radial = 2 * z * torch.exp(-z)
    if d == 1:
        return radial
    a = x[:, 0] * (1 - x[:, 0])
    if d == 2:
        return 6 * a * radial * (1 + a * z)
    y = x[:, 1]
    return 36 * a * y * (1 - y) * radial * (1 + a * y * z)


def cosine_solution(t, x):
    return (0.7 + torch.exp(-math.pi**2 * t[:, 0]) * torch.cos(math.pi * x[:, 0])
            + 0.3 * torch.exp(-4 * x.shape[1] * math.pi**2 * t[:, 0])
            * torch.cos(2 * math.pi * x).prod(dim=1))


def digits(estimate, reference):
    error = abs((estimate - reference) / reference)
    return -math.log10(max(error, torch.finfo(torch.float32).eps))


def run(dimension, script_path):
    parser = argparse.ArgumentParser(description=f"{dimension}D insulated heat PINN pilot")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--steps", type=int, default=512)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--benchmark", action="store_true", help="Measure warmed training throughput only")
    args = parser.parse_args()
    if args.steps <= 0:
        parser.error("--steps must be positive")
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available()
                          else "cpu" if args.device == "auto" else args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA unavailable in this PyTorch installation")
    steps = min(args.steps, 8) if args.smoke else args.steps
    check_count, batch = (128, 32) if args.smoke else (1024, 128)
    out = get_mirror_path(script_path, "results")
    if args.benchmark:
        torch.manual_seed(args.seed)
        model = DiffusionIntegrator(dimension).to(device)
        model.fit_initial(synthetic, steps=5, batch_size=128)
        timings = [model.fit_initial(synthetic, steps=16, batch_size=128)[1] / 16
                   for _ in range(3)]
        median = statistics.median(timings)
        report = dict(dimension=dimension, device=str(device), torch_version=torch.__version__,
                      gpu=torch.cuda.get_device_name(device) if device.type == "cuda" else None,
                      seconds_per_step=timings, median_seconds_per_step=median,
                      estimated_two_fits_seconds=1024*median,
                      caveat="Training-only extrapolation; startup, diagnostics, file I/O excluded. Adam restarts per block.")
        (out / "heat_benchmark.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(json.dumps(report, indent=2))
        return
    prefix = f"heat_seed{args.seed}" + ("_smoke" if args.smoke else "")
    rows, histories = [], {}
    start = time.perf_counter()
    for case_index, name in enumerate(("cosine", "synthetic")):
        torch.manual_seed(args.seed + case_index)
        model = DiffusionIntegrator(dimension).to(device)
        point_fn = (lambda x: cosine_solution(torch.zeros(len(x), 1, device=device), x)) if name == "cosine" else synthetic
        history, elapsed, observed_mean = model.fit_initial(point_fn, steps=steps, batch_size=batch)
        model.eval()
        # Independent diagnostics, generated only after fitting has ended.
        torch.manual_seed(args.seed + 10000 + case_index)
        x = torch.rand(check_count, dimension, device=device)
        t = torch.rand(check_count, 1, device=device)
        pde_rms = model.residual(t, x).square().mean().sqrt().item()
        # Measure the enforced boundary condition independently of the PDE loss.
        face = torch.rand(2 * dimension, dimension, device=device)
        for axis in range(dimension):
            face[2 * axis, axis] = 0.
            face[2 * axis + 1, axis] = 1.
        face.requires_grad_(True)
        face_gradient = torch.autograd.grad(
            model(torch.full((2 * dimension, 1), .5, device=device), face).sum(), face)[0]
        boundary_max_flux = max(abs(face_gradient[2 * axis + side, axis].item())
                                for axis in range(dimension) for side in (0, 1))
        with torch.no_grad():
            initial = model(torch.zeros(check_count, 1, device=device), x)
            truth = point_fn(x)
            terminal = model(torch.ones(check_count, 1, device=device), x)
            initial_rmse = (initial - truth).square().mean().sqrt().item()
            # Same diagnostic points at each time: drift, not training constraint.
            masses = [model(torch.full((check_count, 1), tval, device=device), x).mean().item()
                      for tval in (0., .25, .5, .75, 1.)]
            spacetime_rmse = ((model(t, x) - cosine_solution(t, x)).square().mean().sqrt().item()
                              if name == "cosine" else None)
        estimate = model.integrate()
        # Analytic references first enter here, after the optimizer is finished.
        reference = (0.7 if name == "cosine" else
                     (2 - 4 / math.e, 2 - 4 / math.e + (4 - 10 / math.e) / 5,
                      12 / 5 - 5 / math.e)[dimension - 1])
        row = dict(case=name, dimension=dimension, estimate=estimate, reference=reference,
                   signed_error=estimate-reference, absolute_error=abs(estimate-reference),
                   I_1_digits=digits(estimate, reference), digits_censored=abs(estimate-reference) <= abs(reference)*torch.finfo(torch.float32).eps,
                   initial_rmse=initial_rmse, pde_rms=pde_rms,
                   boundary_max_flux=boundary_max_flux,
                   terminal_spread=(terminal.max()-terminal.min()).item(),
                   diagnostic_mass_drift=max(masses)-min(masses),
                   cosine_spacetime_rmse=spacetime_rmse, train_seconds=elapsed,
                   training_oracle_calls=steps*batch, diagnostic_oracle_calls=check_count,
                   observed_mean=observed_mean, observed_mean_digits=digits(observed_mean, reference))
        rows.append(row)
        histories[name] = dict(loss=history, sampled_masses=masses)
        print(f"{name}: estimate={estimate:.9g}, reference={reference:.9g}, "
              f"digits={row['I_1_digits']:.3f}, train={elapsed:.2f}s", flush=True)
    values = [r["I_1_digits"] for r in rows]
    summary = dict(min_digits=min(values), mean_digits=sum(values)/len(values), max_digits=max(values))
    print("Relative correct digits: " + ", ".join(f"{k}={v:.3f}" for k,v in summary.items()))
    with (out / f"{prefix}.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    manifest = dict(dimension=dimension, steps=steps, batch_size=batch, initial_count=steps*batch,
                    seed=args.seed, device=str(device), torch_version=torch.__version__,
                    gpu=torch.cuda.get_device_name(device) if device.type == "cuda" else None,
                    dtype="float32", width=32, hidden_layers=2, horizon=1., lr=.001,
                    elapsed_seconds=time.perf_counter()-start, smoke=args.smoke,
                    summary=summary, histories=histories,
                    caveat="Finite-time biased readout; sampled residuals and drift are diagnostics, not certificates.")
    (out / f"{prefix}.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Results: {out}")
