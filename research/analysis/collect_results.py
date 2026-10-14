"""Aggregate every recorded result in ``research/results`` into one snapshot table.

Reads both machine-written CSVs (grid sweeps, modern flat runs) and the
hand-printed ``.out`` / ``.log`` tables from the early flat experiments, then
reports, per architecture, the best configuration found (seed-averaged where
seeds exist) and the best individual run.

Usage::

    python research/analysis/collect_results.py [--json out.json]

Nothing here writes into ``research/results`` -- it is a read-only collector, so
it does not need ``get_mirror_path`` and cannot create phantom results dirs.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from statistics import mean, pstdev

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULTS = REPO_ROOT / "research" / "results"
EXPERIMENTS = REPO_ROOT / "research" / "experiments"

# --------------------------------------------------------------------------
# canonical names: directory name -> architecture label
# --------------------------------------------------------------------------

ARCH_BY_DIR = {
    "2-mlp-single": "MLP (Lloyd, integral approx.)",
    "4-mlp-maitre": "MLP (Maître, antiderivative)",
    "5-siren-integrator": "SIREN",
    "6-wire-integrator": "WIRE",
    "7-kan-integrator": "KAN",
    "9-brok-siren": "BrokNet-SIREN",
    "10-brok-wire-ensemble": "BrokNet-WIRE",
    "11-cosiren-ps-integrator": "COSIREN-PS",
    "12-sechiren-integrator": "SECHIREN",
    "13-guderiren-integrator": "GUDERMANIAN",
    "14-gaussian-integrator": "GAUSSIAN",
    "15-sinhren-integrator": "SINHREN",
    "16-mixiren-integrator": "MIXIREN",
    "17-tanhiren-integrator": "TANHIREN",
    "18-sechiren-is-integrator": "SECHIREN-IS",
    "19-conv-integrator": "CONV",
    "20-pcn-integrator": "PCN",
    "21-tropical-integrator": "TROP v1",
    "22-tropical-integrator-v2": "TROP v2",
    "23-tropical-integrator-v3": "TROP v3",
    "24-tropical-integrator-v4": "TROP v4",
    "25-tropical-integrator-v5": "TROP v5",
}

# grid directory -> {csv basename: architecture label}
ARCH_BY_GRID_DIR = {
    "1-mlp-grid-search": {"results_grid.csv": "MLP (Lloyd, integral approx.)"},
    "2-mlp-grid-search-fine": {"results_grid_finer.csv": "MLP (Lloyd, integral approx.)"},
    "3-mlp-distrib-sweep": {"results_t_dist_2026-04-20-2.csv": "MLP (Lloyd, integral approx.)"},
    "4-siren-precision-diagnostic": {"*": "SIREN"},
    "5-siren-wire-kan-sweep-m1": {
        "sweep_siren_*": "SIREN",
        "sweep_wire_*": "WIRE",
        "sweep_kan_*": "KAN",
    },
    "6-siren-wire-kan-sweep-m2": {
        "sweep_siren_*": "SIREN",
        "sweep_wire_*": "WIRE",
        "sweep_kan_*": "KAN",
    },
    # grid/7 stores SIREN / WIRE / KAN in identically named CSVs; the
    # architecture is recovered from the column signature instead.
    "7-siren-wire-kan-sweep-m3": {},
    "8-sinhren-omega-sweep": {"*": "SINHREN"},
    "9-tanhiren-omega-sweep": {"*": "TANHIREN"},
    "10-tanhiren-omega-sweep-low": {"*": "TANHIREN"},
    "12-sechiren-sweep-m1": {"*": "SECHIREN"},
    "13-sechiren-sweep-m2": {"sweep_results_m2_combined.csv": "SECHIREN", "results_gpu*.csv": "SECHIREN"},
    "15-conv-integrator-sweep": {"conv_sweep_combined.csv": "CONV"},
    "16-conv-integrator-sweep-m1.2": {"conv_sweep_m12_combined.csv": "CONV"},
    "17-tropical-integrator-sweep": {"results_gpu*.csv": "TROP v2"},
    # grid/18 writes only SLURM logs (CSVs were too large to commit), parsed below
    "18-sechiren-sweep-m3": {},
}

# grid/7 stores SIREN / WIRE / KAN in identically named CSVs; tell them apart
# from the column signature.
ARCH_BY_SIGNATURE = (
    (("sigma_0", "entry_width"), "WIRE"),
    (("width", "depth", "G"), "KAN"),
    (("hidden_sizes", "omega_0"), "SIREN"),
)

# CSVs that are flat-run outputs rather than sweep outputs
FLAT_CSV_PREFIXES = ("conv_nni.csv", "pcn_nni.csv", "trop_")

CONFIG_KEYS = (
    "hidden_sizes",
    "n_pieces",
    "pieces",
    "omega_0",
    "sigma_0",
    "entry_width",
    "n_blocks",
    "width",
    "depth",
    "G",
    "k",
    "lr",
    "n_epochs",
    "epochs",
    "n_per_param",
    "tau_start",
    "tau_end",
)


@dataclass
class Run:
    arch: str
    source: str          # human-readable provenance
    kind: str            # "flat" | "grid"
    config: dict = field(default_factory=dict)
    seed: int | None = None
    n_params: int | None = None
    min_dig: float | None = None
    mean_dig: float | None = None
    max_dig: float | None = None
    per_integral: list = field(default_factory=list)
    mean_rel_err: float | None = None
    final_loss: float | None = None
    elapsed_s: float | None = None
    device: str | None = None

    @property
    def label(self) -> str:
        cfg = self.config
        bits = []
        for k in CONFIG_KEYS:
            if k in cfg:
                bits.append(f"{k}={cfg[k]}")
        return ", ".join(bits) or "-"


# --------------------------------------------------------------------------
# flat CSV parsing
# --------------------------------------------------------------------------


def _f(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _i(value):
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None


def _digits_from_relerr(rel: float) -> float:
    """Correct digits from a relative error, capped the way the reports do."""
    if rel <= 0:
        return 15.0
    return -math.log10(rel)


DIGIT_COLS = {
    "min_correct_digits": "min_dig",
    "mean_correct_digits": "mean_dig",
    "max_correct_digits": "max_dig",
    "min_digits": "min_dig",
    "mean_digits": "mean_dig",
    "max_digits": "max_dig",
    "min_dig": "min_dig",
    "mean_dig": "mean_dig",
    "max_dig": "max_dig",
}


def parse_flat_csv(path: Path, arch: str, source: str, kind: str = "flat") -> list[Run]:
    rows = list(csv.DictReader(path.open()))
    if not rows:
        return []
    out = []
    for i, row in enumerate(rows):
        run = Run(arch=arch, source=source, kind=kind)
        for key, target in DIGIT_COLS.items():
            if key in row and getattr(run, target) is None:
                setattr(run, target, _f(row[key]))
        keys = [k for k in row if k]  # some CSVs have duplicated headers
        per = [
            _f(row[k])
            for k in sorted(keys)
            if re.fullmatch(r"(?:I_?\d+_?dig|I\d+)", k, flags=re.IGNORECASE)
        ]
        per = [v for v in per if v is not None]
        if not per:
            # early MLP sweeps stored per-integral *relative errors* instead
            errs = [
                _f(row[k])
                for k in sorted(keys)
                if re.fullmatch(r"(?:err|e)_?\d{4}", k, flags=re.IGNORECASE)
            ]
            errs = [e for e in errs if e is not None]
            if errs:
                per = [_digits_from_relerr(e) for e in errs]
        if per:
            run.per_integral = per
            run.min_dig = min(per)
            run.mean_dig = mean(per)
            run.max_dig = max(per)
        run.config = {k: row[k] for k in CONFIG_KEYS if row.get(k)}
        run.seed = _i(row.get("seed"))
        run.n_params = _i(row.get("n_weights") or row.get("n_params_total") or row.get("n_params"))
        run.mean_rel_err = _f(row.get("mean_rel_err"))
        run.final_loss = _f(row.get("final_loss"))
        run.elapsed_s = _f(row.get("elapsed_s"))
        run.device = row.get("device")
        if run.mean_dig is None and per:
            run.mean_dig = mean(per)
        out.append(run)
    return out


# --------------------------------------------------------------------------
# legacy .out table parsing
# --------------------------------------------------------------------------

ROW_RE = re.compile(
    r"\((?P<cfg>\d+\s*,\s*\d+\s*,\s*\d+\s*,\s*\d+)\)\s*[:\s]+"
    r"(?P<body>.*)"
)


def parse_out_table(path: Path, arch: str, source: str) -> Run | None:
    text = path.read_text(errors="replace")
    header = {}
    for key in ("Architecture", "Parameters", "Epochs", "N per param", "LR",
                "Final loss", "Min loss", "Device", "Seed"):
        m = re.search(rf"^{key}\s*:\s*(.+)$", text, flags=re.MULTILINE)
        if m:
            header[key] = m.group(1).strip()
    for key in ("Final loss", "Minimal loss", "Minimum loss", "Min loss"):
        m = re.search(rf"^\s*{key}\s*:?\s*([0-9.eE+-]+)\s*$", text, flags=re.MULTILINE)
        if m:
            header.setdefault("Final loss", m.group(1))

    per = []
    for m in ROW_RE.finditer(text):
        line = m.group(0)
        if "±" in line:  # scipy reference block, not a result row
            continue
        body = m.group("body")
        # digits always terminate the result row
        tail = re.search(r"(\d+)\s*$", body)
        if not tail:
            continue
        # guard: the row must carry the reference/NNI numbers too
        if len(re.findall(r"[-+]?\d*\.\d+(?:[eE][-+]?\d+)?", body)) < 2:
            continue
        per.append(float(tail.group(1)))
    if not per:
        return None
    # some logs reprint the table; the last 8 rows are the final evaluation
    if len(per) > 8:
        per = per[-8:]

    run = Run(arch=arch, source=source, kind="flat")
    run.per_integral = per
    run.min_dig, run.mean_dig, run.max_dig = min(per), mean(per), max(per)
    run.device = header.get("Device")
    run.final_loss = _f(header.get("Final loss"))
    params = header.get("Parameters", "")
    if params:
        cleaned = params.replace(",", "").replace("_", "")
        if cleaned.isdigit():
            run.n_params = int(cleaned)
    cfg = {"epochs": header.get("Epochs"), "lr": header.get("LR")}
    cfg = {k: v for k, v in cfg.items() if v and v != "None"}
    arch_line = header.get("Architecture", "")
    for token, key in (("omega_0", "omega_0"),):
        mm = re.search(rf"{token}=([0-9.]+)", arch_line)
        if mm:
            cfg[key] = mm.group(1)
    mm = re.search(r"\[([0-9,\s]+)\]", arch_line)
    if mm:
        cfg["hidden_sizes"] = "[" + mm.group(1).replace(" ", "") + "]"
    run.config = cfg
    return run


def parse_json_results(path: Path, arch: str, source: str) -> Run | None:
    data = json.loads(path.read_text())
    per = []
    for row in data:
        nni, ref = row.get("nni"), row.get("num")
        if nni is None or not ref:
            continue
        rel = abs(nni - ref) / abs(ref)
        per.append(_digits_from_relerr(rel))
    if not per:
        return None
    run = Run(arch=arch, source=source, kind="flat")
    run.per_integral = per
    run.min_dig, run.mean_dig, run.max_dig = min(per), mean(per), max(per)
    return run


def math_log10(x: float) -> float:
    return math.log10(x)


# --------------------------------------------------------------------------
# collection
# --------------------------------------------------------------------------


def collect_flat() -> list[Run]:
    runs: list[Run] = []
    for d in sorted(RESULTS.glob("flat/*")):
        if not d.is_dir():
            continue
        arch = ARCH_BY_DIR.get(d.name, d.name)
        for path in sorted(d.iterdir()):
            if not path.is_file():
                continue
            src = f"flat/{d.name}/{path.name}"
            if path.suffix == ".csv":
                header = path.open().readline()
                # long format: one row per integral, not per run
                if "digits" in header and re.search(r"(^|,)I(,|$)", header):
                    continue
                runs.extend(parse_flat_csv(path, arch, src))
            elif path.suffix in (".out", ".log"):
                run = parse_out_table(path, arch, src)
                if run:
                    runs.append(run)
            elif path.suffix == ".json":
                run = parse_json_results(path, arch, src)
                if run:
                    runs.append(run)
    return runs


def collect_grid() -> list[Run]:
    runs: list[Run] = []
    for d in sorted(RESULTS.glob("grid/*")):
        if not d.is_dir():
            continue
        spec = ARCH_BY_GRID_DIR.get(d.name)
        if spec is None:
            continue
        if d.name == "18-sechiren-sweep-m3":
            for log in sorted(d.glob("*.log")):
                runs.extend(parse_sweep_log(log, "SECHIREN", f"grid/{d.name}/{log.name}"))
            continue
        for path in sorted(d.glob("*.csv")):
            arch = None
            for pattern, label in spec.items():
                if fnmatch(path.name, pattern):
                    arch = label
                    break
            if arch is None and not spec:
                # grid/7 stores SIREN / WIRE / KAN in identically named CSVs
                arch = arch_from_signature(path)
            if arch is None:
                continue
            src = f"grid/{d.name}/{path.name}"
            runs.extend(parse_flat_csv(path, arch, src, kind="grid"))
    return runs


def arch_from_signature(path: Path) -> str | None:
    header = path.open().readline()
    for keys, arch in ARCH_BY_SIGNATURE:
        if all(k in header for k in keys):
            return arch
    return None


RUN_LINE = re.compile(
    r"hidden=(?P<hidden>\[[^\]]*\])\s+omega0=(?P<omega>[\d.]+)\s+lr=(?P<lr>[\d.e-]+)\s+"
    r"epochs=(?P<epochs>\d+)\s+npp=(?P<npp>\d+)\s+seed=(?P<seed>\d+)"
)
RESULT_LINE = re.compile(
    r"->\s+I1=(?P<per>.*?)\s+min=(?P<min>[\d.]+)\s+mean=(?P<mean>[\d.]+)\s+max=(?P<max>[\d.]+)"
)
SUMMARY_LINE = re.compile(
    r"min_dig=(?P<min>[\d.]+)\s+mean_dig=(?P<mean>[\d.]+)\s+max_dig=(?P<max>[\d.]+)"
    r".*?n_seeds=(?P<n>\d+)"
)


def parse_sweep_log(path: Path, arch: str, source: str) -> list[Run]:
    """Parse the SLURM logs of grid/18, where the CSVs were too large to commit."""
    text = path.read_text(errors="replace")
    lines = text.splitlines()
    runs: list[Run] = []
    pending: Run | None = None
    summary_cfg = re.compile(
        r"hidden=(?P<hidden>\[[^\]]*\])\s+omega0=(?P<omega>[\d.]+)\s+lr=(?P<lr>[\d.e-]+)\s+"
        r"epochs=(?P<epochs>\d+)\s+npp=(?P<npp>\d+)"
    )
    for line in lines:
        m = RUN_LINE.search(line)
        if m:
            pending = Run(arch=arch, source=source, kind="grid")
            pending.config = {
                "hidden_sizes": m.group("hidden").replace(" ", ""),
                "omega_0": m.group("omega"),
                "lr": m.group("lr"),
                "n_epochs": m.group("epochs"),
                "n_per_param": m.group("npp"),
            }
            pending.seed = _i(m.group("seed"))
            continue
        r = RESULT_LINE.search(line)
        if r and pending is not None:
            pending.per_integral = [float(v) for v in re.findall(r"I\d=(\d+)", r.group("per"))]
            pending.min_dig = float(r.group("min"))
            pending.mean_dig = float(r.group("mean"))
            pending.max_dig = float(r.group("max"))
            runs.append(pending)
            pending = None
            continue
        s = SUMMARY_LINE.search(line)
        c = summary_cfg.search(line)
        if s and c:
            run = Run(arch=arch, source=source, kind="grid_summary")
            run.config = {
                "hidden_sizes": c.group("hidden").replace(" ", ""),
                "omega_0": c.group("omega"),
                "lr": c.group("lr"),
                "n_epochs": c.group("epochs"),
                "n_per_param": c.group("npp"),
            }
            run.min_dig = float(s.group("min"))
            run.mean_dig = float(s.group("mean"))
            run.max_dig = float(s.group("max"))
            run.per_integral = []
            runs.append(run)
    return runs


def fnmatch(name: str, pattern: str) -> bool:
    from fnmatch import fnmatch as _f

    return _f(name, pattern)


# --------------------------------------------------------------------------
# aggregation
# --------------------------------------------------------------------------


FAMILY = {
    "TROP v1": "TROP (tropical max-plus)",
    "TROP v2": "TROP (tropical max-plus)",
    "TROP v3": "TROP (tropical max-plus)",
    "TROP v4": "TROP (tropical max-plus)",
    "TROP v5": "TROP (tropical max-plus)",
}


def _config_groups(valid: list[Run]) -> list[tuple]:
    """Score configs: prefer entries that are already seed-averaged."""
    valid = [r for r in valid if r.mean_dig is not None]
    summaries = [r for r in valid if r.kind == "grid_summary"]
    individuals = [r for r in valid if r.kind != "grid_summary"]

    scored = []
    if summaries:
        for r in summaries:
            scored.append((r.mean_dig, 5, r.max_dig or 0, r.min_dig or 0, r, True))
    groups: dict[str, list[Run]] = {}
    for r in individuals:
        groups.setdefault(r.label, []).append(r)
    for label, rs_ in groups.items():
        seeds = {r.seed for r in rs_ if r.seed is not None}
        scored.append(
            (
                mean(r.mean_dig for r in rs_),
                len(seeds) if len(seeds) > 1 else 1,
                max((r.max_dig or r.mean_dig or 0) for r in rs_),
                mean((r.min_dig or r.mean_dig or 0) for r in rs_),
                rs_[0],
                len(seeds) > 1,
            )
        )
    scored.sort(key=lambda s: (s[0], s[1], s[2], s[3]), reverse=True)
    return scored


def _row(arch: str, valid: list[Run]) -> dict | None:
    individuals = [r for r in valid if r.kind != "grid_summary" and r.mean_dig is not None]
    if not individuals:
        return None
    best_run = max(individuals, key=lambda r: (r.mean_dig, r.min_dig or 0))
    if best_run.n_params is None:
        hs = best_run.config.get("hidden_sizes")
        same = [r.n_params for r in individuals if hs and r.config.get("hidden_sizes") == hs]
        if same:
            best_run.n_params = max(set(same), key=same.count)
    scored = _config_groups(valid)
    robust = [s for s in scored if s[1] >= 3]
    cfg_mean, n_seeds, cfg_max, cfg_min, cfg_run, cfg_averaged = (robust or scored)[0]
    return {
        "architecture": arch,
        "n_runs": len(individuals),
        "best_config": cfg_run.label,
        "cfg_n_seeds": n_seeds,
        "cfg_seed_averaged": bool(cfg_averaged),
        "cfg_seed_avg_mean": round(cfg_mean, 3),
        "cfg_seed_avg_min": round(cfg_min, 2),
        "cfg_seed_avg_max": round(cfg_max, 2),
        "cfg_source": cfg_run.source,
        "best_run_mean": round(best_run.mean_dig, 3),
        "best_run_min": round(best_run.min_dig if best_run.min_dig is not None else best_run.mean_dig),
        "best_run_max": round(best_run.max_dig if best_run.max_dig is not None else best_run.mean_dig),
        "best_run_config": best_run.label,
        "best_run_source": best_run.source,
        "n_params": best_run.n_params,
        "overall_mean": round(mean(r.mean_dig for r in individuals), 3),
        "sd_mean_dig": round(pstdev([r.mean_dig for r in individuals]), 3)
        if len(individuals) > 1
        else 0.0,
    }


def aggregate(runs: list[Run]) -> list[dict]:
    by_arch: dict[str, list[Run]] = {}
    for r in runs:
        by_arch.setdefault(r.arch, []).append(r)
    table = [_row(arch, rs) for arch, rs in by_arch.items()]
    table = [r for r in table if r]
    table.sort(key=lambda d: -d["best_run_mean"])
    return table


def aggregate_families(runs: list[Run]) -> list[dict]:
    by_fam: dict[str, list[Run]] = {}
    for r in runs:
        by_fam.setdefault(FAMILY.get(r.arch, r.arch), []).append(r)
    table = [_row(fam, rs) for fam, rs in by_fam.items()]
    table = [r for r in table if r]
    table.sort(key=lambda d: -d["best_run_mean"])
    return table


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", type=Path)
    ap.add_argument("--markdown", type=Path)
    args = ap.parse_args()

    runs = collect_flat() + collect_grid()
    table = aggregate(runs)
    families = aggregate_families(runs)

    for title, rows in (("BY ARCHITECTURE FAMILY", families), ("BY VARIANT", table)):
        print(f"\n=== {title} ===")
        header = (
            f"{'architecture':<28}{'runs':>6}{'best':>7}{'min':>5}{'max':>5}"
            f"{'cfg':>7}{'sd':>6}  best_configuration"
        )
        print(header)
        print("-" * len(header))
        for row in rows:
            print(
                f"{row['architecture']:<28}{row['n_runs']:>6}"
                f"{row['best_run_mean']:>7.2f}{row['best_run_min']:>5.0f}{row['best_run_max']:>5.0f}"
                f"{row['cfg_seed_avg_mean']:>7.2f}{row['overall_mean']:>6.2f}  {row['best_run_config'][:70]}"
            )
    if args.json:
        args.json.write_text(json.dumps({"families": families, "variants": table}, indent=2))
        print(f"\nwrote {args.json}")
    if args.markdown:
        args.markdown.write_text(render_markdown(families, table))
        print(f"wrote {args.markdown}")


def render_markdown(families: list[dict], variants: list[dict]) -> str:
    def esc(s):
        return s.replace("_", "\\_") if s and s != "-" else "-"

    lines = [
        "| # | Architecture | Best run (mean / min / max) | Best-run configuration | Params | Seed-averaged best config | Seed-avg mean | Seeds | Runs |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for i, r in enumerate(families, 1):
        params = f"{r['n_params']:,}" if r.get("n_params") else "-"
        seeds = (
            f"{r['cfg_n_seeds']}"
            if r["cfg_seed_averaged"]
            else "**1** (no seed column)"
        )
        lines.append(
            f"| {i} | **{r['architecture']}** "
            f"| **{r['best_run_mean']:.2f}** / {r['best_run_min']:.0f} / {r['best_run_max']:.0f} "
            f"| {esc(r['best_run_config'])} | {params} "
            f"| {esc(r['best_config'])} | {r['cfg_seed_avg_mean']:.2f} | {seeds} | {r['n_runs']:,} |"
        )
    lines += [
        "",
        "| # | Variant | Best run | Seed-averaged best config | Seeds |",
        "|---|---|---|---|---|",
    ]
    for i, r in enumerate(variants, 1):
        lines.append(
            f"| {i} | {r['architecture']} "
            f"| {r['best_run_mean']:.2f} / {r['best_run_min']:.0f} / {r['best_run_max']:.0f} "
            f"| {r['cfg_seed_avg_mean']:.2f} ({esc(r['best_config'])}) | {r['cfg_n_seeds']} |"
        )
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    main()