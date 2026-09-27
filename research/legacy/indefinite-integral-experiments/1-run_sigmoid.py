from common.physics import PARAM_SETS, FLOATING_POINT_PRECISION, L_LOG, T_MAX
from common.integrand import center_value, integrand_transformed
from common.training import make_batch, mixed_partial_3, compute_integral
from common.reference import reference_scipy
from common.logging_utils import tee_print
from nets.sigmoid_net import PrimitiveNet


import math
import time

import numpy as np
import torch
import torch.nn as nn
from zmq.sugar import device


##############################################################################
# 7.  TRAINING
##############################################################################

def train(net: PrimitiveNet,
          param_sets: list,
          n_epochs: int = 5000,
          n_per_param: int = 512,
          lr: float = 1e-3,
          device: torch.device = torch.device("cpu"),
          verbose_every: int = 500) -> tuple:
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=n_epochs, eta_min=lr / 10
    )
    loss_fn = nn.MSELoss()
    net.to(device)
    net.train()

    history = []
    norm_cache = {}
    t0 = time.time()

    print(f"\n{'═' * 64}")
    print(f"  Training: {n_epochs} epochs | {n_per_param} points per set | "
          f"{len(param_sets)} sets")
    print(f"  Per epoch: {len(param_sets) * n_per_param} points | "
          f"device: {device}")
    print(f"  lr: {lr} → {lr / 10} (CosineAnnealing)")
    print(f"{'═' * 64}")

    for epoch in range(1, n_epochs + 1):
        batch_xu, f_tilde, norm_cache = make_batch(
            param_sets, n_per_param, device, norm_cache
        )

        dN = mixed_partial_3(net, batch_xu)
        loss = loss_fn(dN, f_tilde)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=5.0)
        optimizer.step()
        scheduler.step()

        lv = loss.item()
        history.append(lv)

        if epoch % verbose_every == 0 or epoch == 1:
            lr_now = scheduler.get_last_lr()[0]
            print(f"  Epoch {epoch:5d}/{n_epochs}  "
                  f"loss={lv:.4e}  lr={lr_now:.2e}  "
                  f"({time.time() - t0:.1f}с)")

    print(f"\n  Done in {time.time() - t0:.1f}с | "
          f"min loss = {min(history):.4e}\n")
    return history, norm_cache


##############################################################################
# 10.  MAIN
##############################################################################
def main():
    torch.manual_seed(42)
    np.random.seed(42)

    device = torch.device(
        "mps" if torch.backends.mps.is_available() else
        "cuda" if torch.cuda.is_available() else
        "cpu"
    )

    torch.set_default_dtype(FLOATING_POINT_PRECISION)

    HIDDEN       = [128, 128, 128] # [64, 64, 64]
    N_EPOCHS     = 8000 # 5000
    N_PER_PARAM  = 1024  # 512
    LR           = 1e-3
    OUTPUT_SCALE = 1e4

    net = PrimitiveNet(
        n_params=4,
        n_int_vars=3,
        hidden_sizes=HIDDEN,
        output_scale=OUTPUT_SCALE,
    )
    n_p = sum(p.numel() for p in net.parameters())
    print(f"\nStructure: {HIDDEN} (Sigmoid) | {n_p:,} params")
    print(f"Subs for t: logarithmic  t = exp({L_LOG:.3f}·u₃) - 1")
    print(f"                 domain: t ∈ [0, {T_MAX:.0f}]")

    history, norm_cache = train(
        net,
        param_sets=PARAM_SETS,
        n_epochs=N_EPOCHS,
        n_per_param=N_PER_PARAM,
        lr=LR,
        device=device,
        verbose_every=500,
    )

    print("Reference values...\n")
    refs = {}
    for (a, b, m, n) in PARAM_SETS:
        r, e = reference_scipy(a, b, m, n)
        refs[(a, b, m, n)] = (r, e)
        print(f"  ({int(a)},{int(b)},{int(m)},{int(n)}):  {r:.6e}  ±  {e:.1e}")

    print(f"\n{'═' * 84}")
    print(f"  {'(a,b,m,n)':^12}  {'NNI':^14}  {'scipy':^14}  "
          f"{'|Δ|':^11}  {'|Δ|/ref':^9}  {'Digits':^6}")
    print(f"{'═' * 84}")

    for (a, b, m, n) in PARAM_SETS:
        nni_val = compute_integral(net, a, b, m, n,
                                   norm_cache=norm_cache,
                                   device=device)
        ref_val, ref_err = refs[(a, b, m, n)]

        abs_err = abs(nni_val - ref_val)
        rel_err = abs_err / (abs(ref_val) + 1e-30)

        correct_digits = max(0, -math.floor(math.log10(abs_err)))

        print(f"   ({int(a)},{int(b)},{int(m)},{int(n)})     "
              f"{nni_val:>14.6e}  {ref_val:>14.6e}  "
              f"{abs_err:>11.3e}  {rel_err:>9.3e}  "
              f"{int(correct_digits):^6}")

    print(f"{'═' * 84}")
    print(f"\n  Final loss: {history[-1]:.4e}")
    print(f"  Minimal loss: {min(history):.4e}")
    print(f"loss @ epoch 1: {history[0]:.4e}")


if __name__ == "__main__":
    main()
