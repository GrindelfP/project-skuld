#!/usr/bin/env python3
"""Reproducible PySecDec baseline for the eight meson reference integrals.

Run on Linux with the project's prescribed environment, for example:

    conda activate jasher
    python research/experiments/reference-integrals/i3ref-pysecdec.py

PySecDec creates and compiles a package for each integral.  Those generated
sources stay in a temporary directory outside the repository.  Results and
the accompanying log/metadata are written only to this experiment's mirror.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import subprocess
import sys
import tempfile
import time
from contextlib import redirect_stderr, redirect_stdout
from datetime import datetime, timezone
from pathlib import Path
from statistics import median
from typing import Any, TextIO

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from utils.paths import get_mirror_path


LA = 1.0
M1 = M2 = M3 = 0.3 / LA
P1P1 = P2P2 = -(0.14 / LA) ** 2
PP = -(0.70 / LA) ** 2
M1SQ, M2SQ, M3SQ = M1**2, M2**2, M3**2
R2MID = PP - P1P1 - P2P2
T_MAX = 80.0
L_LOG = math.log1p(T_MAX)

# This order is the established benchmark order and must not change.
PARAMETER_SETS = list(zip(
    [0, 0, 1, 1, 0, 0, 1, 1],
    [0, 1, 0, 1, 0, 1, 0, 1],
    [1, 1, 1, 1, 2, 2, 2, 2],
    [2, 2, 2, 2, 3, 3, 3, 3],
))

REAL_PARAMETERS = [PP, P1P1, P2P2, M1SQ, M2SQ, M3SQ, R2MID, L_LOG]
RESULT_FIELDS = [
    "run_id", "integral_index", "a", "b", "m", "n", "status", "message",
    "value", "pysecdec_estimated_uncertainty_abs",
    "pysecdec_estimated_uncertainty_rel", "prior_median_value",
    "prior_min_value", "prior_max_value", "abs_discrepancy_to_prior_median",
    "rel_discrepancy_to_prior_median", "max_abs_discrepancy_to_prior",
    "max_rel_discrepancy_to_prior",
    "build_seconds", "evaluation_seconds", "total_seconds", "maxeval",
    "epsrel", "epsabs", "qmc_transform", "pysecdec_version",
]


class Tee:
    """Send stdout/stderr to both the console and the run log."""

    def __init__(self, *streams: TextIO) -> None:
        self.streams = streams

    def write(self, text: str) -> int:
        for stream in self.streams:
            stream.write(text)
            stream.flush()
        return len(text)

    def flush(self) -> None:
        for stream in self.streams:
            stream.flush()


def build_integrand_string(a: int, b: int, m: int, n: int) -> str:
    alpha1 = "x1"
    alpha2 = "(x2*(1-x1))"
    alpha3 = "((1-x1)*(1-x2))"
    d_value = (
        f"({alpha1}*{alpha2}*PP + P1P1*{alpha2}*{alpha3}"
        f" + P2P2*{alpha1}*{alpha3} + {alpha1}*M1sq"
        f" + {alpha2}*M2sq + {alpha3}*M3sq)"
    )
    r_squared = (
        f"(({alpha1})^2*P2P2 + ({alpha2})^2*P1P1"
        f" - {alpha1}*{alpha2}*R2MID)"
    )
    z0 = f"((exp(LL*x3)-1)*{d_value} + (1-exp(-LL*x3))*{r_squared})"
    t_weight = f"(exp(LL*x3)-1)^{m} * exp(-{n}*LL*x3)"
    jacobian = f"(LL * exp(LL*x3) * (1-x1))"
    return (
        f"(({alpha1})^{a} * ({alpha2})^{b} * {t_weight}"
        f" * exp(-({z0})) * {jacobian})"
    )


_RESULT_RE = re.compile(
    r"\(\s*([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)\s*"
    r"\+/-\s*((?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)\s*\)"
)


def parse_secdec_result(result: object) -> tuple[float, float]:
    """Extract the central value and PySecDec's *estimated* QMC uncertainty."""
    match = _RESULT_RE.search(str(result))
    if match is None:
        raise ValueError(f"Cannot parse PySecDec result: {result!r}")
    value, uncertainty = float(match.group(1)), float(match.group(2))
    if not (math.isfinite(value) and math.isfinite(uncertainty) and uncertainty >= 0):
        raise ValueError(f"Non-finite PySecDec result: {result!r}")
    return value, uncertainty


def read_prior_values() -> list[list[float]]:
    """Read, but never modify, the three historical headerless reference CSVs."""
    prior_files = sorted(Path(__file__).parent.glob("i3ref_*.csv"))
    values = [[] for _ in PARAMETER_SETS]
    for prior_file in prior_files:
        with prior_file.open(newline="") as handle:
            rows = list(csv.reader(handle))
        if len(rows) != len(PARAMETER_SETS):
            raise ValueError(f"{prior_file} has {len(rows)} rows, expected 8")
        for index, row in enumerate(rows):
            values[index].append(float(row[0]))
    return values


def empty_row(run_id: str, index: int, params: tuple[int, int, int, int], settings: argparse.Namespace,
              pysecdec_version: str, prior_values: list[float]) -> dict[str, Any]:
    a, b, m, n = params
    return {
        "run_id": run_id, "integral_index": index, "a": a, "b": b, "m": m, "n": n,
        "status": "pending", "message": "", "value": "",
        "pysecdec_estimated_uncertainty_abs": "",
        "pysecdec_estimated_uncertainty_rel": "", "prior_median_value": median(prior_values),
        "prior_min_value": min(prior_values), "prior_max_value": max(prior_values),
        "abs_discrepancy_to_prior_median": "", "rel_discrepancy_to_prior_median": "",
        "max_abs_discrepancy_to_prior": "", "max_rel_discrepancy_to_prior": "",
        "build_seconds": "", "evaluation_seconds": "", "total_seconds": "",
        "maxeval": settings.maxeval, "epsrel": settings.epsrel, "epsabs": settings.epsabs,
        "qmc_transform": settings.transform, "pysecdec_version": pysecdec_version,
    }


def write_rows(output_path: Path, rows: list[dict[str, Any]]) -> None:
    """Atomically checkpoint the complete CSV after every completed case."""
    temporary_path = output_path.with_suffix(".tmp")
    with temporary_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=RESULT_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    temporary_path.replace(output_path)


def validate_rows(rows: list[dict[str, Any]]) -> None:
    if len(rows) != len(PARAMETER_SETS):
        raise ValueError(f"Expected 8 rows, found {len(rows)}")
    for index, row in enumerate(rows):
        if int(row["integral_index"]) != index:
            raise ValueError(f"Row {index} is out of benchmark order")
        if row["status"] == "ok":
            for field in ("value", "pysecdec_estimated_uncertainty_abs", "build_seconds",
                          "evaluation_seconds", "total_seconds"):
                value = float(row[field])
                if not math.isfinite(value):
                    raise ValueError(f"Row {index} has non-finite {field}")
            for field in ("build_seconds", "evaluation_seconds", "total_seconds"):
                if float(row[field]) <= 0:
                    raise ValueError(f"Row {index} has non-positive {field}")


def self_check() -> None:
    assert len(PARAMETER_SETS) == 8
    assert parse_secdec_result("( 1.25e+00 +/- 2.5e-06 )") == (1.25, 2.5e-06)
    assert all("exp" in build_integrand_string(*params) for params in PARAMETER_SETS)
    prior_values = read_prior_values()
    assert all(len(values) == 3 and all(math.isfinite(value) for value in values)
               for values in prior_values)
    print("Self-check passed: 8 parameter sets and 3 prior CSV values per integral.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--maxeval", type=int, default=100_000_000)
    parser.add_argument("--epsrel", type=float, default=1e-5)
    parser.add_argument("--epsabs", type=float, default=1e-12)
    parser.add_argument("--transform", default="korobov3")
    parser.add_argument("--output", type=Path, help="Existing/new CSV within the result mirror.")
    parser.add_argument("--keep-build-dir", action="store_true",
                        help="Retain the temporary PySecDec build directory on failure/success.")
    parser.add_argument("--self-check", action="store_true",
                        help="Validate parsing, parameter order, and historical CSVs without PySecDec.")
    return parser.parse_args()


def main() -> int:
    settings = parse_args()
    if settings.self_check:
        self_check()
        return 0

    try:
        import pySecDec
        from pySecDec.code_writer import make_package
        from pySecDec.integral_interface import IntegralLibrary
    except ModuleNotFoundError as error:
        print("PySecDec is unavailable in this Python environment; no packages were installed.", file=sys.stderr)
        print("Activate the configured environment (`conda activate jasher`) containing PySecDec and retry.",
              file=sys.stderr)
        raise SystemExit(2) from error

    results_dir = get_mirror_path(__file__, "results")
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%SZ")
    run_id = f"i3ref_pysecdec_{timestamp}"
    output_path = settings.output or results_dir / f"{run_id}.csv"
    output_path = output_path.resolve()
    if output_path.parent != results_dir.resolve():
        raise ValueError("--output must be directly inside this experiment's result mirror")
    log_path = results_dir / f"{run_id}.log"
    metadata_path = results_dir / f"{run_id}.json"
    pysecdec_version = getattr(pySecDec, "__version__", "unknown")
    prior_values = read_prior_values()
    prior_medians = [median(values) for values in prior_values]
    rows = [empty_row(run_id, index, params, settings, pysecdec_version, prior_values[index])
            for index, params in enumerate(PARAMETER_SETS)]

    metadata = {
        "run_id": run_id,
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "pysecdec_version": pysecdec_version,
        "settings": {"integrator": "Qmc", "maxeval": settings.maxeval, "epsrel": settings.epsrel,
                     "epsabs": settings.epsabs, "transform": settings.transform},
        "parameter_sets": PARAMETER_SETS,
        "prior_csvs": [path.name for path in sorted(Path(__file__).parent.glob("i3ref_*.csv"))],
        "uncertainty_note": "PySecDec-estimated QMC uncertainty; not a rigorous error bound.",
        "build_artifact_note": "Generated packages were built in a scoped temporary directory outside the repository.",
    }
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")

    with log_path.open("w") as log_handle, redirect_stdout(Tee(sys.stdout, log_handle)), \
            redirect_stderr(Tee(sys.stderr, log_handle)):
        print(json.dumps(metadata, indent=2))
        build_dir_context: Any
        if settings.keep_build_dir:
            build_dir_context = tempfile.mkdtemp(prefix="skuld-pysecdec-")
            build_dir = Path(build_dir_context)
            print(f"Keeping temporary build directory: {build_dir}")
        else:
            build_dir_context = tempfile.TemporaryDirectory(prefix="skuld-pysecdec-")
            build_dir = Path(build_dir_context.__enter__())
        original_cwd = Path.cwd()
        try:
            for index, params in enumerate(PARAMETER_SETS):
                a, b, m, n = params
                row = rows[index]
                case_started = time.perf_counter()
                package_name = f"integral_{index}"
                package_path = build_dir / package_name
                try:
                    os.chdir(build_dir)
                    build_started = time.perf_counter()
                    make_package(
                        name=package_name,
                        integration_variables=["x1", "x2", "x3"],
                        real_parameters=["PP", "P1P1", "P2P2", "M1sq", "M2sq", "M3sq", "R2MID", "LL"],
                        regulators=["eps"], requested_orders=[0], polynomials_to_decompose=["1"],
                        remainder_expression=build_integrand_string(a, b, m, n),
                    )
                    subprocess.run(["make", "-C", str(package_path)], check=True)
                    row["build_seconds"] = time.perf_counter() - build_started
                    integral = IntegralLibrary(str(package_path / f"{package_name}_pylink.so"))
                    integral.use_Qmc(transform=settings.transform, maxeval=settings.maxeval,
                                     epsrel=settings.epsrel, epsabs=settings.epsabs)
                    evaluation_started = time.perf_counter()
                    _, _, result = integral(real_parameters=REAL_PARAMETERS)
                    value, uncertainty = parse_secdec_result(result)
                    row["evaluation_seconds"] = time.perf_counter() - evaluation_started
                    row["value"] = value
                    row["pysecdec_estimated_uncertainty_abs"] = uncertainty
                    row["pysecdec_estimated_uncertainty_rel"] = uncertainty / abs(value) if value else math.inf
                    row["abs_discrepancy_to_prior_median"] = abs(value - prior_medians[index])
                    row["rel_discrepancy_to_prior_median"] = (
                        abs(value - prior_medians[index]) / abs(prior_medians[index]))
                    row["max_abs_discrepancy_to_prior"] = max(
                        abs(value - prior_value) for prior_value in prior_values[index])
                    row["max_rel_discrepancy_to_prior"] = max(
                        abs(value - prior_value) / abs(prior_value)
                        for prior_value in prior_values[index])
                    row["status"] = "ok"
                    print(f"I_{index + 1}: {value:.18e} +/- {uncertainty:.3e} (PySecDec estimate)")
                except Exception as error:  # Preserve prior successful rows and make the failure machine-readable.
                    row["status"] = "failed"
                    row["message"] = f"{type(error).__name__}: {error}"
                    print(f"I_{index + 1} FAILED: {row['message']}", file=sys.stderr)
                finally:
                    row["total_seconds"] = time.perf_counter() - case_started
                    write_rows(output_path, rows)
        finally:
            os.chdir(original_cwd)
            if not settings.keep_build_dir:
                build_dir_context.__exit__(None, None, None)

    validate_rows(rows)
    print(f"Validated {len(rows)} rows: {output_path}")
    return 0 if all(row["status"] == "ok" for row in rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
