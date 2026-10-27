# skuld

A neural network based numerical integration library.

Skuld computes definite integrals of multi-dimensional functions using neural
networks. Instead of sampling a quadrature mesh, it either learns the integrand
and integrates that network analytically, or learns the antiderivative and reads
the answer off the corners of a hypercube.

```bash
pip install skuld-lib
```

Requires Python 3.11 or newer.

## Two methods

These are independent approaches, not stages of one pipeline. Either can be used
on its own; which one suits a problem depends on the integrand.

**Approximate the integrand** — fit a shallow MLP to the integrand, then integrate
the network analytically using polylogarithms. Evaluation becomes a closed-form
expression, which pays off when one integrand is evaluated over many domains or
parameter values, since training is paid once.

**Approximate the antiderivative** — train a network so its mixed partial
derivatives match the integrand, then evaluate the integral as an
alternating-sign sum over the corners of the unit hypercube. There is no mesh
and no quadrature rule, only `2**n` forward passes over a trained network.

## Quick start

```python
from skuld.siren import SirenIntegrator

# An integrand on the unit cube: f(u, *params) -> Tensor of shape [N]
def integrand(u, a, b, m, n):
    ...

integrator = SirenIntegrator(
    n_params=4,
    n_int_vars=3,
    hidden_sizes=[64, 64, 64],
    omega_0=30.0,
)

history, norm_cache = integrator.train(
    integrand_fn=integrand,
    param_sets=[(0, 0, 1, 2), (1, 1, 2, 3)],
    n_epochs=8000,
)

result = integrator.integrate((0, 0, 1, 2), norm_cache=norm_cache)
```

The integrand must return shape `[N]`, not `[N, 1]`. An extra trailing dimension
broadcasts the mean-squared error to `[N, N]`, silently corrupting the loss with
no error raised.

## Architectures

Every integrator below answers to the same `train()` / `integrate()` interface, so
swapping one for another does not change the surrounding code.

| Class | Import | Network |
| --- | --- | --- |
| `SirenIntegrator` | `skuld.siren` | Sinusoidal representation network |
| `CosirenIntegrator` | `skuld.cosiren` | Cosine-activated SIREN variant |
| `CosirenPSIntegrator` | `skuld.cosiren_ps` | Cosine with a learnable per-layer phase shift |
| `WireIntegrator` | `skuld.wire` | Wavelet implicit representation (Morlet wavelet) |
| `KANIntegrator` | `skuld.kan` | Kolmogorov-Arnold network, learnable B-spline edges |
| `SechirenIntegrator` | `skuld.sechiren` | Hyperbolic-secant activation |
| `SechirenISIntegrator` | `skuld.sechiren_is` | SECHIREN with importance-sampled training data |
| `GudermannianIntegrator` | `skuld.gudermannian` | Gudermannian activation |
| `GaussianIntegrator` | `skuld.gaussian` | Gaussian activation |
| `SinhrenIntegrator` | `skuld.sinhren` | Hyperbolic-sine activation with frequency scaling |
| `MixirenIntegrator` | `skuld.mixiren` | Mixed activation stack, sech-sin-sech |
| `TanhirenIntegrator` | `skuld.tanhiren` | Tanh activation with frequency scaling |
| `ConvIntegrator` | `skuld.conv` | Convolutional antiderivative, no fully-connected layers |
| `BrokNetIntegrator` | `skuld.broknet` | Mixture of experts, one SIREN per parameter set |

Not every architecture is equally good at this task. Some are early
experiments kept for comparison, and one or two are known to underperform.

The integrand-approximation method lives at the package root:

```python
from skuld import (
    generate_data, scale_data, split_data,
    init_model, NeuralNumericalIntegration, descale_result,
)
```

## Utility API

```python
from skuld import generate_data, scale_data, split_data, set_global_device

# Sample a function on a box. With one strategy string, n_samples is the
# total row count (grid strategies round it up to a perfect n_dim-th power).
# With a per-dimension list, stochastic dims share n_samples rows and every
# grid dim adds n_samples points, crossed: here 200 x 200 = 40,000 rows.
X, y = generate_data(
    dis_type=["std", "std", "exp", "grd"],
    func=integrand,
    lower=[0.0, 0.0, 0.0, 0.1],
    upper=[1.0, 1.0, 50.0, 2.0],
    n_samples=200,
    n_dim=4,
)

X_scaled, y_scaled = scale_data(X, y, frange=(0, 1))
X_train, X_test, y_train, y_test = split_data(X_scaled, y_scaled, test_size=0.1)

# CUDA when available, else MPS on Apple Silicon, else CPU.
set_global_device("cpu")   # or "cuda" or "mps"
```

## Caveats

Two behaviours of the current implementation are easy to trip over, and neither
of them raises.

- **`SechirenIntegrator.train()` returns three values** —
  `(history, norm_cache, epoch_data)`, where `epoch_data` holds per-epoch timing.
  Every other integrator returns two.
- **`integrate()` does not guarantee the trained accuracy.** The training loss
  measures a mixed partial at random points; the corner sum reads values at
  vertices. Low training loss does not imply a correct integral.

## Versioning

Versions are `MAJOR.MINOR.PATCH`, with one addition: a `.postN` suffix marks a
**fix-only revision** of the release before it, rather than a new release. So
`0.4.0.post1` is the same `0.4.0` repackaged — the API is unchanged and nothing
new is added. It sorts after the plain release, so `pip install -U` picks it up.

Use `.postN` only when a release needs republishing without changing behaviour.
Any change to the API or to results gets a real version bump.

Install a specific revision with:

```bash
pip install skuld-lib==0.4.0.post1
```

Note that PEP 440 local versions (`0.4.0+build1`) would express the same idea
more neatly, but **PyPI rejects local versions outright**, so `.postN` is the
only option available here.

## License

Non-commercial use with attribution. Commercial use requires explicit written
permission. Use in the development of weapons or for military purposes is
prohibited. See [LICENSE](LICENSE).