# Sigmoid Maître meson sweep

One conditioned sigmoid primitive is trained jointly on all eight parameter sets,
using the mixed third partial against sampled integrand values, not integral
references. The three hidden layers, Xavier initialization, final-layer scale,
Adam, cosine schedule to lr/10, gradient clipping at 5, center-value
normalization, float32 and corner sum follow the original flat sigmoid
experiment. Shared `research/physics.py` supplies the same transformed
integrand. References are computed separately with its `reference_scipy` and
used only after training for evaluation. They integrate the finite domain
`t ∈ [0,80]`; they are numerical references, not exact infinite-domain values.
The returned outer quadrature error is not a complete error bound for nested
quadrature. Investigate reference precision before interpreting high digits.

## Grid and budget

Full factorial: widths {64, 128, 160}, learning rates {0.0001, 0.0005,
0.001}, epochs {6000, 8000, 10000}, points per parameter per epoch {512,
1024, 2048}, output scales {1000, 10000, 100000}. This is 243 configurations
× 5 seeds {42, 137, 2024, 7, 999} = **1,215 training runs**. Each epoch
evaluates eight times the points-per-parameter; sampling is refreshed each
epoch. This is a search budget, not a statistically or computationally
matched budget to any other architecture. Selection on these references is
in-sample; report all seeds and avoid presenting the selected score as an
independent test.

## Run and resume

From the remote submission directory containing the project's SLURM wrapper,
after syncing the repository and setting up the `jasher` environment:

```sh
sbatch --array=0-3 skuld_test_run.sh grid/24-sigmoid-maitre-sweep/sigmoid_sweep.py --num-gpus 4 --run-tag hopper
```

The wrapper must request one GPU per array task and place SLURM logs in the
experiment's mirrored results directory. The wrapper is site-provided, not
in this repository. Select the partition/resources as appropriate locally.
For a short plumbing check use the same command with `--smoke` and preferably
`--array=0-0` / `--num-gpus 1`; it trains 2 epochs with 8 samples per set and
writes only under the `smoke` namespace. Locally, from the repository root,
run `python research/experiments/grid/24-sigmoid-maitre-sweep/sigmoid_sweep.py --smoke --run-tag local` after activating jasher.

Each run writes one atomically published CSV under the mirror, partitioned
by run tag and `full` or `smoke`. The run ID is stable and independently
sharded by modulo, so re-submitting the same tag skips completed runs even
if the shard count changes. Do not launch overlapping arrays with the same
tag; distinct simultaneous hardware runs need distinct tags. Failures stop
the task with a nonzero exit; rerun to resume missing runs. The CSV includes
configuration, seed, elapsed training time, losses, all eight estimates,
references and relative correct digits, and their min/mean/max across the
eight integrals. Digits are `-log10(abs(estimate-reference)/abs(reference))`
without clipping or flooring; exact floating-point agreement yields `inf`.
Each CSV has a header and one row; concatenate rows while retaining one header
for aggregate analysis. There is no cross-seed ranking in the runner.
