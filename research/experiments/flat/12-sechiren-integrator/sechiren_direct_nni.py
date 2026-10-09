"""
sechiren_direct_nni.py — SECHIREN trained on N directly (not mixed partial).

Instead of training the network to match the mixed partial derivative
(integrand), we precompute the antiderivative N at random points using
scipy and train the network to match N directly. This eliminates the
error accumulation from integrating the mixed partial.

Usage:
    python sechiren_direct_nni.py [seed]
"""
import math
import os
import sys
from pathlib import Path
import time
from datetime import datetime

import numpy as np
import torch
from scipy import integrate

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))
sys.path.insert(0, str(Path(__file__).resolve().parents[4] / "skuld-lib"))

from physics import integrand_transformed, reference_scipy, PARAM_SETS, FLOATING_POINT_PRECISION, D_R2, LA, M1, M2, M3, P1P1, P2P2, PP, P1P2, T_MAX, L_LOG, korobov, korobov_weight

##############################################################################
#  PRECOMPUTE N AT RANDOM POINTS
##############################################################################

def compute_N_at_points(param_sets, n_points=10000, seed=42, cache_file='n_data_cache.npz'):
    """Precompute the antiderivative N at random points using scipy.

    N(s, u) = integral from 0 to u of the integrand.

    Returns a dict: params_key -> (u_points, N_values)
    """
    # Try to load from cache
    if os.path.exists(cache_file):
        print(f"  Loading precomputed N from {cache_file}...")
        data = np.load(cache_file)
        results = {}
        for params in param_sets:
            params_key = (params[0], params[1], params[2], params[3])
            key_str = f"{params_key[0]}_{params_key[1]}_{params_key[2]}_{params_key[3]}"
            u_points = data[f"{key_str}_u"]
            N_values = data[f"{key_str}_N"]
            results[params_key] = (u_points, N_values)
        return results

    rng = np.random.RandomState(seed)
    results = {}

    for params in param_sets:
        a, b, m, n = params
        params_key = (a, b, m, n)

        # Generate random points in [0,1]^3
        u_points = rng.rand(n_points, 3)

        # Compute N at each point
        N_values = np.zeros(n_points)

        for i in range(n_points):
            u1, u2, u3 = u_points[i]

            # Transform to u-space
            def integrand_u(u3_inner, u2_inner, u1_inner):
                x1 = u1_inner ** 2 * (3.0 - 2.0 * u1_inner)
                x2 = u2_inner ** 2 * (3.0 - 2.0 * u2_inner)
                w1 = 6.0 * u1_inner * (1.0 - u1_inner)
                w2 = 6.0 * u2_inner * (1.0 - u2_inner)

                alpha1 = x1
                alpha2 = (1.0 - x1) * x2
                J_alpha = (1.0 - x1) * w1 * w2

                exp_Lu3 = np.exp(L_LOG * u3_inner)
                t = exp_Lu3 - 1.0
                Jt = L_LOG * exp_Lu3

                alpha3 = 1.0 - alpha1 - alpha2
                D = (alpha1 * alpha2 * PP
                     + P1P1 * alpha2 * alpha3
                     + P2P2 * alpha1 * alpha3
                     + alpha1 * M1 ** 2
                     + alpha2 * M2 ** 2
                     + alpha3 * M3 ** 2)
                R2 = (alpha1 ** 2 * P2P2
                      + alpha2 ** 2 * P1P1
                      - alpha1 * alpha2 * (PP - P1P1 - P2P2))
                exp_arg = -(t * D + t / (1.0 + t) * R2)
                if exp_arg > 700.0:
                    return 0.0
                a1 = alpha1 ** a if a > 0 else 1.0
                a2 = alpha2 ** b if b > 0 else 1.0
                return a1 * a2 * t ** m / (1.0 + t) ** n * np.exp(exp_arg) * J_alpha * Jt

            # Compute the integral
            val, err = integrate.nquad(
                integrand_u,
                [[0, u1], [0, u2], [0, u3]],
                opts={'epsabs': 1e-10, 'epsrel': 1e-10, 'limit': 100}
            )
            N_values[i] = val

        results[params_key] = (u_points, N_values)
        print(f"  Precomputed N for {params_key}: mean={np.mean(N_values):.6e}, std={np.std(N_values):.6e}")

    # Save to cache
    print(f"  Saving precomputed N to {cache_file}...")
    save_dict = {}
    for params_key, (u_points, N_values) in results.items():
        key_str = f"{params_key[0]}_{params_key[1]}_{params_key[2]}_{params_key[3]}"
        save_dict[f"{key_str}_u"] = u_points
        save_dict[f"{key_str}_N"] = N_values
    np.savez(cache_file, **save_dict)

    return results


##############################################################################
#  TRAIN ON N DIRECTLY
##############################################################################

def train_direct(n_data, param_sets, n_epochs=8000, lr=5e-4, device=None, verbose_every=500, seed=42):
    """Train a network to match N directly."""
    if device is None:
        device = torch.device("cpu")

    torch.manual_seed(seed)
    np.random.seed(seed)

    HIDDEN = [128, 128, 128]
    OMEGA_0 = 30.0

    # Build the network
    from skuld.sechiren import SechirenPrimitiveNet
    net = SechirenPrimitiveNet(
        n_params=4,
        n_int_vars=3,
        hidden_sizes=HIDDEN,
        omega_0=OMEGA_0,
        output_scale=1.0,
    )
    net.to(device)
    net.train()

    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs, eta_min=lr / 10)
    loss_fn = torch.nn.MSELoss()

    # Prepare training data (float32 for MPS compatibility)
    all_x, all_N = [], []
    for params in param_sets:
        params_key = tuple(params)
        u_points, N_values = n_data[params_key]
        s = torch.tensor(params, dtype=torch.float32).expand(u_points.shape[0], -1)
        u = torch.tensor(u_points, dtype=torch.float32)
        x = torch.cat([s, u], dim=1)
        all_x.append(x)
        all_N.append(torch.tensor(N_values, dtype=torch.float32))

    all_x = torch.cat(all_x, dim=0)
    all_N = torch.cat(all_N, dim=0)

    # Normalize N
    N_mean = all_N.mean()
    N_std = all_N.std()
    all_N_normalized = (all_N - N_mean) / N_std

    history = []
    t0 = time.time()

    for epoch in range(1, n_epochs + 1):
        # Sample a batch
        idx = torch.randint(0, all_x.shape[0], (1024,))
        batch_x = all_x[idx].to(device)
        batch_N = all_N_normalized[idx].to(device)

        # Forward pass
        N_pred = net(batch_x).squeeze(-1)

        # Loss
        loss = loss_fn(N_pred, batch_N)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=5.0)
        optimizer.step()
        scheduler.step()

        lv = loss.item()
        history.append(lv)

        if verbose_every > 0 and (epoch % verbose_every == 0 or epoch == 1):
            lr_now = scheduler.get_last_lr()[0]
            elapsed = time.time() - t0
            print(f"  Epoch {epoch:5d}/{n_epochs}  loss={lv:.4e}  lr={lr_now:.2e}  ({elapsed:.1f}s)")

    return net, history, N_mean, N_std


##############################################################################
#  MAIN
##############################################################################

def main():
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 42
    torch.manual_seed(seed)
    np.random.seed(seed)

    device = torch.device(
        "mps" if torch.backends.mps.is_available() else
        "cuda" if torch.cuda.is_available() else
        "cpu"
    )
    torch.set_default_dtype(torch.float32)

    N_EPOCHS = 8000
    LR = 5e-4
    N_PRECOMPUTE = 10000

    run_start = datetime.now()
    timestamp = run_start.strftime("%Y-%m-%d_%H-%M")

    print(f"\n  Run started : {run_start.strftime('%Y-%m-%d %H:%M')}")
    print(f"  Seed        : {seed}")
    print(f"  Device      : {device}")
    print(f"  Precision   : float64")

    # Step 1: Precompute N
    print(f"\n  Precomputing N at {N_PRECOMPUTE} random points per parameter set...")
    t0 = time.time()
    n_data = compute_N_at_points(PARAM_SETS, n_points=N_PRECOMPUTE, seed=seed)
    print(f"  Precomputation took {time.time() - t0:.1f}s")

    # Step 2: Train on N directly
    print(f"\n  Training on N directly...")
    net, history, N_mean, N_std = train_direct(
        n_data, PARAM_SETS,
        n_epochs=N_EPOCHS, lr=LR, device=device, verbose_every=500, seed=seed,
    )

    # Step 3: Evaluate via corner sum
    print("\n  Evaluating via corner sum...")
    net.eval()

    # Reference values
    print("\nComputing reference values (scipy) ...\n")
    refs = {}
    for (a, b, m, n) in PARAM_SETS:
        r, e = reference_scipy(a, b, m, n)
        refs[(a, b, m, n)] = (r, e)
        print(f"  ({int(a)},{int(b)},{int(m)},{int(n)}):  {r:.6e}  +/-  {e:.1e}")

    # Corner sum
    import itertools
    SEP = '=' * 84
    print(f"\n{SEP}")
    print(f"  {'(a,b,m,n)':^12}  {'NNI':^14}  {'scipy':^14}  "
          f"{'|delta|':^11}  {'|delta|/ref':^9}  {'Digits':^6}")
    print(SEP)

    for (a, b, m, n) in PARAM_SETS:
        params = (a, b, m, n)
        s_row = torch.tensor([params], dtype=torch.float32).to(device)
        I_tilde = 0.0

        with torch.no_grad():
            for corner in itertools.product([0.0, 1.0], repeat=3):
                sign = (-1) ** (3 - sum(corner))
                u_t = torch.tensor([corner], dtype=torch.float32).to(device)
                inp = torch.cat([s_row, u_t], dim=1)
                N_pred = net(inp).item()
                N_val = N_pred * N_std.item() + N_mean.item()
                I_tilde += sign * N_val

        ref_val, ref_err = refs[params]
        abs_err = abs(I_tilde - ref_val)
        rel_err = abs_err / (abs(ref_val) + 1e-30)
        correct_digits = max(0, -math.floor(math.log10(abs_err + 1e-30)))

        print(f"   ({int(a)},{int(b)},{int(m)},{int(n)})     "
              f"{I_tilde:>14.6e}  {ref_val:>14.6e}  "
              f"{abs_err:>11.3e}  {rel_err:>9.3e}  "
              f"{int(correct_digits):^6}")

    print(SEP)
    print(f"\n  Final loss   : {history[-1]:.4e}")
    print(f"  Minimum loss : {min(history):.4e}")
    print(f"\n  Run finished : {datetime.now().strftime('%Y-%m-%d %H:%M')}")

    # Save log
    results_dir = os.path.join(os.path.dirname(__file__), '../../../results/flat/12-sechiren-integrator')
    os.makedirs(results_dir, exist_ok=True)
    log_filename = os.path.join(results_dir, f"results_sechiren_direct_seed{seed}_{timestamp}.out")
    try:
        with open(log_filename, "w", encoding="utf-8") as fh:
            fh.write(f"Run: {run_start}\n")
            fh.write(f"Seed: {seed}\n")
            fh.write(f"Device: {device}\n")
            fh.write(f"Architecture: SECHIREN-DIRECT [128,128,128] omega_0=30\n")
            fh.write(f"Epochs: {N_EPOCHS}\n")
            fh.write(f"LR: {LR}\n")
            fh.write(f"N precompute: {N_PRECOMPUTE}\n")
            fh.write(f"Final loss: {history[-1]:.4e}\n")
            fh.write(f"Min loss: {min(history):.4e}\n")
            fh.write("\nResults:\n")
            for (a, b, m, n) in PARAM_SETS:
                params = (a, b, m, n)
                s_row = torch.tensor([params], dtype=torch.float32).to(device)
                I_tilde = 0.0
                with torch.no_grad():
                    for corner in itertools.product([0.0, 1.0], repeat=3):
                        sign = (-1) ** (3 - sum(corner))
                        u_t = torch.tensor([corner], dtype=torch.float32).to(device)
                        inp = torch.cat([s_row, u_t], dim=1)
                        N_pred = net(inp).item()
                        N_val = N_pred * N_std.item() + N_mean.item()
                        I_tilde += sign * N_val
                ref_val, _ = refs[params]
                abs_err = abs(I_tilde - ref_val)
                rel_err = abs_err / (abs(ref_val) + 1e-30)
                correct_digits = max(0, -math.floor(math.log10(abs_err + 1e-30)))
                fh.write(f"  ({int(a)},{int(b)},{int(m)},{int(n)}):  "
                         f"NNI={I_tilde:.6e}  ref={ref_val:.6e}  "
                         f"abs_err={abs_err:.3e}  rel_err={rel_err:.3e}  "
                         f"digits={int(correct_digits)}\n")
        print(f"\n  Log saved -> {log_filename}")
    except OSError as exc:
        print(f"\n  [WARNING] Could not save log: {exc}")


if __name__ == "__main__":
    main()
