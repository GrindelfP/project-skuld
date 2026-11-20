# Heat training-budget experiment

Three self-contained PyTorch scripts repeat the flat heat pilots with more
optimizer updates. The sole training change is the default `--steps 2000`.
Each script fits its cosine fixture and synthetic initial condition using the
same seed 42, two width-32 tanh layers, float32, batch 128, Adam learning rate
0.001, equal initial/PDE losses, horizon one, and fresh uniform initial points.
The PDE-time mixture and hard cosine-coordinate Neumann condition are unchanged.

The grid variable is training duration. The prior flat runs at 512 updates are
the first baseline. Compare 2,000 next; decide whether to run 8,000 only after
reviewing the 2,000-update results. No 8,000-update run is automatically launched.
“Steps” are optimizer updates, not epochs over a fixed dataset: every update
uses 128 new initial-function evaluations, giving 256,000 per fit at 2,000
updates and 1,024,000 at 8,000. More updates therefore also mean more target
observations. The cosine and synthetic fits retain separate seeded streams.
On the same software/device, their training trajectories have the same seeded
prefix as the flat runs; cross-version/device bitwise reproducibility is not
promised. Diagnostic RNG is reset after fitting, so held-out points stay fixed.

From this directory after activating jasher:

```powershell
python heat_1d.py --device cuda --run-tag local-gtx1050
python heat_2d.py --device cuda --run-tag local-gtx1050
python heat_3d.py --device cuda --run-tag local-gtx1050
```

`--steps 8000` is supported for a later explicitly selected run. `--smoke`
caps training at eight steps; `--benchmark` measures short warmed blocks and
extrapolates to the requested step budget. Both retain the flat protocol.
Each script needs only PyTorch and the standard library. Copy it alone to a
remote machine if desired. One GPU per process suffices; there is no multi-GPU
implementation. Standard project launcher convention, if configured remotely:

```text
sbatch skuld_test_run.sh grid/<this-directory>/heat_1d.py --device cuda --run-tag pilot
```

That launcher is absent from this checkout and the example is unvalidated.
Replace `<this-directory>` with this directory's actual name.

Outputs use the mirrored results directory and run-tag subfolder. Names contain
dimension, effective steps and seed, with separate smoke and benchmark names,
so the three scripts and training budgets coexist. Repeating the same
dimension/budget/seed/tag overwrites that run's files. CSV includes explicit
relative error, per-case relative correct digits and all prior diagnostics;
JSON includes min/mean/max digits, timing, configuration and optional Git source
revision and source_dirty flag. The revision identifies the checkout; dirty state flags local changes.
Outside the repository, files go to a sibling results/run-tag folder.

Preparation verification compares the copied training class and target definitions against the original flat pilots, then checks smoke outputs.
Analytic derivative and integral fixtures remain in `research/heat_selfcheck.py`.
Initial/transient errors must improve before interpreting a flat final field as
successful heat evolution. Compare against the observed mean of the same labels;
the prior 512-update neural estimates did not outperform it.
