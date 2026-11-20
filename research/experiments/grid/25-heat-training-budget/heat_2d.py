"""Experimental insulated-heat PINN. Integral labels are never used in fitting."""
import argparse
import csv
import json
import math
import os
from pathlib import Path
import re
import statistics
import subprocess
import time

import torch
from torch import nn


class DiffusionIntegrator(nn.Module):
    """Fit U_t = Laplacian(U), U(0,x)=f(x), with hard zero normal flux.

    The terminal point value is a biased experimental integral estimate. A small
    collocation loss or flat terminal field does not certify conservation.
    """

    def __init__(self, dimension, width=32, horizon=1.0):
        super().__init__()
        if dimension < 1 or horizon <= 0:
            raise ValueError("Positive dimension and horizon required")
        self.dimension, self.horizon = dimension, horizon
        self.net = nn.Sequential(nn.Linear(dimension + 1, width), nn.Tanh(),
                                 nn.Linear(width, width), nn.Tanh(), nn.Linear(width, 1))

    def forward(self, t, x):
        features = torch.cat((t / self.horizon, torch.cos(math.pi * x)), dim=1)
        return self.net(features).squeeze(-1)

    def residual(self, t, x):
        t = t.detach().requires_grad_(True)
        x = x.detach().requires_grad_(True)
        value = self(t, x)
        dt, dx = torch.autograd.grad(value.sum(), (t, x), create_graph=True)
        lap = torch.zeros_like(value)
        for axis in range(self.dimension):
            second = torch.autograd.grad(dx[:, axis].sum(), x, create_graph=True)[0]
            lap = lap + second[:, axis]
        return dt[:, 0] - lap

    def fit_initial(self, point_fn, steps=512, batch_size=128, lr=1e-3):
        if steps <= 0 or batch_size <= 0 or lr <= 0:
            raise ValueError("Positive steps, batch size and learning rate required")
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        parameter = next(self.parameters())
        device, dtype = parameter.device, parameter.dtype
        if device.type == "cuda":
            torch.cuda.synchronize()
        start = time.perf_counter()
        history = []
        observed_sum = 0.
        self.train()
        for step in range(steps):
            x0 = torch.rand(batch_size, self.dimension, device=device, dtype=dtype)
            with torch.no_grad():
                y0 = point_fn(x0)
            if y0.shape != (batch_size,):
                raise ValueError("Initial labels must have shape [N]")
            observed_sum += y0.sum().item()
            initial_loss = (self(torch.zeros(batch_size, 1, device=device, dtype=dtype), x0)
                            - y0).square().mean()
            x = torch.rand(batch_size, self.dimension, device=device, dtype=dtype)
            t = self.horizon * torch.rand(batch_size, 1, device=device, dtype=dtype)
            # Half of collocations resolve the fast initial transient.
            t[:batch_size // 2] = self.horizon * torch.rand(batch_size // 2, 1, device=device, dtype=dtype).square()
            pde_loss = self.residual(t, x).square().mean()
            loss = initial_loss + pde_loss
            if not torch.isfinite(loss):
                raise RuntimeError("Nonfinite training loss")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            if step == 0 or (step + 1) % 64 == 0 or step + 1 == steps:
                print(f"step={step + 1}/{steps} initial_mse={initial_loss.item():.4g} pde_mse={pde_loss.item():.4g}", flush=True)
                history.append(dict(step=step + 1, initial_mse=initial_loss.item(),
                                    pde_mse=pde_loss.item()))
        if device.type == "cuda":
            torch.cuda.synchronize()
        return history, time.perf_counter() - start, observed_sum / (steps * batch_size)

    def integrate(self):
        parameter = next(self.parameters())
        device, dtype = parameter.device, parameter.dtype
        with torch.no_grad():
            return self(torch.full((1, 1), self.horizon, device=device, dtype=dtype),
                        torch.full((1, self.dimension), 1 / 3, device=device, dtype=dtype)).item()


def get_mirror_path(current_file, target_module):
    """Mirror an in-repository experiment; copied scripts use sibling results."""
    script = Path(current_file).resolve()
    for ancestor in script.parents:
        if ancestor.name == "experiments" and ancestor.parent.name == "research":
            return ancestor.parent / target_module / script.parent.relative_to(ancestor)
    return script.parent / target_module


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


def source_revision(script_path):
    """Optional provenance; copied files work without Git."""
    try:
        revision = subprocess.run(["git", "rev-parse", "HEAD"],
                              cwd=Path(script_path).resolve().parent,
                              check=True, capture_output=True, text=True,
                              timeout=5).stdout.strip()
        status = subprocess.run(["git", "status", "--porcelain", "--untracked-files=normal"],
                                cwd=Path(script_path).resolve().parent,
                                check=True, capture_output=True, text=True, timeout=5)
        return revision, bool(status.stdout.strip())
    except (OSError, subprocess.SubprocessError):
        return None, None

def run(dimension, script_path):
    parser = argparse.ArgumentParser(description=f"{dimension}D insulated heat PINN pilot")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--benchmark", action="store_true", help="Measure warmed training throughput only")
    parser.add_argument("--run-tag", default=os.getenv("SKULD_RUN_TAG") or os.getenv("SLURM_JOB_PARTITION") or "default")
    args = parser.parse_args()
    if not re.fullmatch(r"[A-Za-z0-9_-]+", args.run_tag):
        parser.error("--run-tag accepts only letters, digits, underscores and hyphens")
    if args.steps <= 0:
        parser.error("--steps must be positive")
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available()
                          else "cpu" if args.device == "auto" else args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA unavailable in this PyTorch installation")
    steps = min(args.steps, 8) if args.smoke else args.steps
    check_count, batch = (128, 32) if args.smoke else (1024, 128)
    out = get_mirror_path(script_path, "results") / args.run_tag
    out.mkdir(parents=True, exist_ok=True)
    revision, dirty = source_revision(script_path)
    print(f"Heat pilot: dimension={dimension}, device={device}, run_tag={args.run_tag}", flush=True)
    if args.benchmark:
        torch.manual_seed(args.seed)
        model = DiffusionIntegrator(dimension).to(device)
        model.fit_initial(synthetic, steps=5, batch_size=128)
        timings = [model.fit_initial(synthetic, steps=16, batch_size=128)[1] / 16
                   for _ in range(3)]
        median = statistics.median(timings)
        report = dict(source_revision=revision, source_dirty=dirty, requested_steps=args.steps, run_tag=args.run_tag, slurm_job_id=os.getenv("SLURM_JOB_ID"), cuda_runtime=torch.version.cuda, dimension=dimension, device=str(device), torch_version=torch.__version__,
                      gpu=torch.cuda.get_device_name(device) if device.type == "cuda" else None,
                      seconds_per_step=timings, median_seconds_per_step=median,
                      estimated_two_fits_seconds=2*steps*median,
                      caveat="Training-only extrapolation; startup, diagnostics, file I/O excluded. Adam restarts per block.")
        (out / f"heat_{dimension}d_steps{steps}_seed{args.seed}_benchmark.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(json.dumps(report, indent=2))
        return
    prefix = f"heat_{dimension}d_steps{steps}_seed{args.seed}" + ("_smoke" if args.smoke else "")
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
                   relative_error=abs((estimate-reference)/reference),
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
    manifest = dict(source_revision=revision, source_dirty=dirty, requested_steps=args.steps, run_tag=args.run_tag, slurm_job_id=os.getenv("SLURM_JOB_ID"), cuda_runtime=torch.version.cuda, dimension=dimension, steps=steps, batch_size=batch, initial_count=steps*batch,
                    seed=args.seed, device=str(device), torch_version=torch.__version__,
                    gpu=torch.cuda.get_device_name(device) if device.type == "cuda" else None,
                    dtype="float32", width=32, hidden_layers=2, horizon=1., lr=.001,
                    elapsed_seconds=time.perf_counter()-start, smoke=args.smoke,
                    summary=summary, histories=histories,
                    caveat="Finite-time biased readout; sampled residuals and drift are diagnostics, not certificates.")
    (out / f"{prefix}.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Results: {out}")


if __name__ == '__main__':
    run(2, __file__)
