"""
precision_diagnostic.py — is the m2 4-digit ceiling caused by float32, or by
hyperparameters/optimization? Cheap, targeted check before committing to a
big (possibly GPU-unfriendly) m3 sweep.

WHY THIS SCRIPT EXISTS:
  siren_nni.py hardcodes FLOATING_POINT_PRECISION = torch.float32 (line 35),
  and compute_integral() sums 8 signed corners of the unit cube (a classic
  catastrophic-cancellation pattern) using that same precision. float32 has
  ~7 significant decimal digits total, and cancellation in that sum can eat
  1-2 more before you even look at model error -- so the ~1-3e-4 mean_rel_err
  floor seen across ALL of m2's hyperparameters could easily be a precision
  wall rather than an optimization/architecture wall. If so, no amount of
  grid search in m3 would ever cross it.

WHAT THIS SCRIPT CHECKS (3 conditions, same config, same seed):
  A) TRAIN float32, EVAL float32  <- this is exactly what m2 did
  B) TRAIN float32, EVAL float64  <- cheap: only compute_integral() (8 forward
                                      passes, no grad) reruns in float64.
                                      If this alone gains a digit, the
                                      cancellation in the corner-sum was the
                                      bottleneck, not the network itself --
                                      and this fix is FREE on your GPU setup
                                      (training stays float32/GPU as-is).
  C) TRAIN float64, EVAL float64  <- the expensive case: does the network
                                      itself need higher precision internally
                                      (loss backprop, weights) to represent
                                      finer structure? Run this on CPU
                                      (--device cpu, the default) since
                                      float64 is typically 16-32x slower on
                                      consumer GPUs but only ~1.5-2x slower
                                      on CPU -- a fair, cheap comparison.

Run:
    python precision_diagnostic.py                  # 2 best m2 configs, CPU, 2000 epochs
    python precision_diagnostic.py --epochs 4000     # longer, more conclusive
    python precision_diagnostic.py --device cuda     # if you want condition C timed on GPU too

Reads its 2 default configs from the m2 ranked results (best two rows by
min_correct_digits / mean_rel_err from sweep_results_m2_2026-09-25_21-05.csv):
    #1: hidden=[128,128,128,128] omega0=30 lr=2e-4 n_epochs=2000 n_per_param=512
    #2: hidden=[128,128,128]     omega0=45 lr=2e-4 n_epochs=3000 n_per_param=512
"""

import argparse
import copy
import math
import time

import torch

import siren_nni as base


DEFAULT_CONFIGS = [
    dict(hidden_sizes=[128, 128, 128, 128], omega_0=30.0, lr=2e-4, n_per_param=512),
    dict(hidden_sizes=[128, 128, 128],      omega_0=45.0, lr=2e-4, n_per_param=512),
]


def build_net(cfg):
    omega_0 = cfg["omega_0"]
    return base.SirenPrimitiveNet(
        n_params=4, n_int_vars=3,
        hidden_sizes=cfg["hidden_sizes"],
        omega_0=omega_0,
        output_scale=1.0 / (omega_0 ** 3),
    )


def digits_for(net, refs, norm_cache, precision, device):
    """Recompute compute_integral() at the given precision, without retraining.

    net's own weights may be a different dtype than `precision` (e.g. net was
    trained in float32 and we want a float64 eval-only pass) -- cast a COPY of
    the net to match, so the input tensor built inside compute_integral()
    (which uses base.FLOATING_POINT_PRECISION) lines up with the weight dtype.
    """
    old_prec = base.FLOATING_POINT_PRECISION
    base.FLOATING_POINT_PRECISION = precision
    net_cast = copy.deepcopy(net)
    if precision == torch.float64:
        net_cast = net_cast.double()
    else:
        net_cast = net_cast.float()
    net_cast = net_cast.to(device)
    try:
        rel_errs, digits = [], []
        for (a, b, m, n) in base.PARAM_SETS:
            nni_val = base.compute_integral(net_cast, a, b, m, n, norm_cache=norm_cache, device=device)
            ref_val, _ = refs[(a, b, m, n)]
            abs_err = abs(nni_val - ref_val)
            rel_err = abs_err / (abs(ref_val) + 1e-30)
            d = max(0, -math.floor(math.log10(abs_err + 1e-30)))
            rel_errs.append(rel_err)
            digits.append(d)
        return min(digits), sum(rel_errs) / len(rel_errs), max(rel_errs)
    finally:
        base.FLOATING_POINT_PRECISION = old_prec


def run_condition(cfg, n_epochs, seed, precision, device, refs, label):
    torch.manual_seed(seed)
    old_prec = base.FLOATING_POINT_PRECISION
    old_default = torch.get_default_dtype()
    base.FLOATING_POINT_PRECISION = precision
    torch.set_default_dtype(precision)
    try:
        net = build_net(cfg)
        t0 = time.time()
        history, norm_cache = base.train(
            net, param_sets=base.PARAM_SETS, n_epochs=n_epochs,
            n_per_param=cfg["n_per_param"], lr=cfg["lr"], device=device,
            verbose_every=n_epochs + 1,
        )
        elapsed = time.time() - t0
        net_eval = net
        min_dig, mean_relerr, max_relerr = digits_for(net_eval, refs, norm_cache, precision, device)
        print(f"    [{label}] min_loss={min(history):.3e}  min_digits={min_dig}  "
              f"mean_relerr={mean_relerr:.3e}  ({elapsed:.1f}s)")
        return net, norm_cache, elapsed, min_dig, mean_relerr
    finally:
        base.FLOATING_POINT_PRECISION = old_prec
        torch.set_default_dtype(old_default)


def main():
    parser = argparse.ArgumentParser(description="float32 vs float64 precision diagnostic")
    parser.add_argument("--epochs", type=int, default=2000,
                         help="epochs per condition (keep small -- this is a diagnostic, not a sweep)")
    parser.add_argument("--device", type=str, default="cpu",
                         help="device for condition C (float64 train). Default cpu: fair vs float32, "
                              "since float64 is much slower on consumer GPUs. Conditions A/B always "
                              "use this same device too, for an apples-to-apples time comparison.")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    device = torch.device(args.device)

    print(f"\n{'═'*72}\n  PRECISION DIAGNOSTIC -- device={device}, epochs={args.epochs}, seed={args.seed}\n{'═'*72}")

    print("\nComputing scipy reference integrals (float64, done once)...")
    refs = {}
    for (a, b, m, n) in base.PARAM_SETS:
        r, e = base.reference_scipy(a, b, m, n)
        refs[(a, b, m, n)] = (r, e)
    print("Done.\n")

    for i, cfg in enumerate(DEFAULT_CONFIGS, 1):
        print(f"--- Config #{i}: hidden={cfg['hidden_sizes']} omega0={cfg['omega_0']} "
              f"lr={cfg['lr']} n_per_param={cfg['n_per_param']} epochs={args.epochs} ---")

        # A) train float32, eval float32 -- reproduces m2 exactly
        run_condition(cfg, args.epochs, args.seed, torch.float32, device, refs,
                      label="A: train f32 / eval f32 (=m2 baseline)")

        # B) train float32, eval float64 -- re-run compute_integral only, in float64
        net_b, cache_b, _, _, _ = run_condition(cfg, args.epochs, args.seed, torch.float32, device, refs,
                                                 label="B-train: (same as A, kept for eval-only reuse)")
        min_dig_b, mean_relerr_b, max_relerr_b = digits_for(net_b, refs, cache_b, torch.float64, device)
        print(f"    [B: train f32 / eval f64]  min_digits={min_dig_b}  "
              f"mean_relerr={mean_relerr_b:.3e}  (eval-only re-run, ~free)")

        # C) train float64, eval float64 -- the expensive, fully-double condition
        run_condition(cfg, args.epochs, args.seed, torch.float64, device, refs,
                      label="C: train f64 / eval f64")
        print()

    print(f"{'═'*72}")
    print("  HOW TO READ THIS:")
    print("  - If B ~= A (no digit gain from float64 eval alone): the corner-sum")
    print("    cancellation in compute_integral() is NOT the bottleneck.")
    print("  - If B > A (digit gain, still cheap): switch compute_integral() to")
    print("    float64 permanently -- free win, keep training on GPU/float32 as-is.")
    print("  - If C > B (float64 training needed to go further): the ceiling is in")
    print("    the optimization/representation itself, not just the final sum --")
    print("    you'd need float64 training, which is impractical on your GPU; the")
    print("    fallback is training on CPU in float64 for final high-precision runs")
    print("    only (small nets like [128,128,128] make this tolerable), or look at")
    print("    architecture changes (e.g. per-param-set heads) instead.")
    print(f"{'═'*72}\n")


if __name__ == "__main__":
    main()
