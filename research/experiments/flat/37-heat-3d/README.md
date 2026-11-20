# 3D insulated heat pilot

Fixed configuration: a two-layer width-32 tanh network receives time and cos(pi*x)
coordinates. The cosine inputs enforce homogeneous Neumann boundaries. Fit
U_t = Laplacian(U) with equal initial-data and PDE squared-loss weights; read
U(1, (1/3,...,1/3)). Training never receives an integral value.

Each invocation fits two independent fields: an analytic cosine-mode fixture
(with exact space-time solution for checking the PDE solver), followed by the
synthetic integral. Both use fresh random initial points on every update, 512 Adam updates,
batch 128, learning rate 0.001, float32, and independent held-out diagnostics.
The last coordinate is radial z; lower-dimensional cases retain the radial
factor and progressively add angular factors and coupling.

Run heat_nni.py with --device cuda after activating jasher and installing the
project dependencies. Options: --smoke (eight updates), --steps N, --seed N,
--device auto/cpu/cuda. CUDA auto-selects when available. No plotting dependency.

CSV/JSON files are created only when running, in the experiment's mirror results
directory. A repeated seed replaces that seed's files. Smoke filenames are
separate. Outputs include per-case relative digits and min/mean/max, initial
RMSE, PDE residual RMS, terminal spread, sampled mass drift, observed-sample-mean control,
training time, oracle counts and configuration. The cosine fixture additionally
checks space-time RMSE. Zero-flux initial incompatibility is allowed: the
synthetic initial values have a nonzero face-normal derivative, so an early
boundary layer is expected. Half the PDE samples concentrate near time zero.

These are exploratory solver checks, not an accuracy guarantee. Terminal
flatness can coexist with an incorrect constant; finite-time error, float32
roundoff and training error remain. The reference is scoring-only. The observed-sample
mean is a cost-matched basic baseline. Numerical digits are censored at float32
epsilon and the CSV marks censored cases.

The default uses 65,536 initial oracle calls per fit (512 x 128), with no cached training grid. The observed mean is accumulated for reporting only and never enters optimization. --benchmark measures five warmup steps followed by three blocks of sixteen full-batch steps, then extrapolates to two 512-step fits; startup and diagnostics are excluded. Run research/heat_selfcheck.py first for analytic derivative, face-flux, shape and independent midpoint-integral checks.

