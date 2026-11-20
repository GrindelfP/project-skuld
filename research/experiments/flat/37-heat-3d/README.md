# 3D insulated heat pilot

heat_nni.py is self-contained: copy this single file to another computer and
run it with Python 3.11+ and PyTorch. It uses PyTorch and the Python standard
library only; it does not import skuld or any repository helper. Duplication
between the three pilots is intentional so each experiment travels alone.

The MLP has two width-32 tanh layers and receives time and cos(pi*x) spatial
coordinates, enforcing zero normal flux. It fits U_t = Laplacian(U), with equal
initial-data and PDE squared-loss weights, then reads U(1,(1/3,...,1/3)).
Each invocation independently fits a cosine fixture (exact space-time solution)
and the synthetic function. Each fit uses 512 Adam updates, batch 128, learning
rate 0.001, float32 and horizon one. Every update samples fresh random initial
points and PDE collocations; half the PDE times concentrate near zero. There
are 65,536 initial-function calls per fit. Integral values never enter fitting.
The lower-dimensional cases retain radial z and progressively add angular
factors and coupling.

From this directory:

    python heat_nni.py --device cuda --run-tag local
    python heat_nni.py --device cuda --benchmark --run-tag local
    python heat_nni.py --device cuda --smoke --run-tag smoke

Other options: --steps N, --seed N, --device auto/cpu/cuda. The benchmark uses
five warmup steps and three blocks of sixteen full-batch steps, reporting median
step time and an extrapolation to two 512-update fits. Startup, diagnostics and
file I/O are excluded. Adam restarts per block.

In the repository, outputs follow the experiment's mirror results directory,
with a run-tag subfolder. A copied script outside the repository uses a sibling
results/run-tag folder. --run-tag defaults to SKULD_RUN_TAG, then
SLURM_JOB_PARTITION, then default, and accepts letters, digits, underscores and
hyphens. Repeated seeds overwrite their own files; smoke names are separate.
CSV/JSON record per-case relative digits, min/mean/max, initial RMSE, PDE RMS,
normal flux, terminal spread, sampled mass drift, observed-sample-mean control,
oracle counts, training time, device and SLURM job ID. References are scoring
only. Digits are capped at float32 epsilon and flagged when censored.

Run research/heat_selfcheck.py in the full repository to check the analytic PDE,
face derivatives, shapes and independently integrated synthetic references.
Initial boundary incompatibility in the synthetic function is expected; a flat
terminal field and small sampled residuals do not certify the correct mean.

For a remote SLURM installation configured with the project's launcher, submit
one GPU job for this dimension (activate jasher in that launcher):

    sbatch skuld_test_run.sh flat/37-heat-3d/heat_nni.py --device cuda --run-tag pilot

The launcher is not present in this checkout, so that command documents the
project convention and has not been validated here. These scripts use one GPU;
there are no multi-GPU flags. A direct scheduler wrapper may instead execute
python with this script path after requesting one GPU.
