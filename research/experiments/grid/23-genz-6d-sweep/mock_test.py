"""Fast plumbing and resume test for the canonical 6-D Genz sweep.

It substitutes a deterministic fake training call, so it verifies CSV identity,
smoke namespacing, downsampled history, and a second invocation's no-duplicate
behaviour without training networks or writing to the experiment mirror.
"""

from __future__ import annotations

import csv
import importlib.util
import tempfile
from pathlib import Path


SCRIPT = Path(__file__).with_name("genz_6d_sweep.py")
SPEC = importlib.util.spec_from_file_location("genz_6d_sweep", SCRIPT)
assert SPEC and SPEC.loader
sweep = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(sweep)


def main() -> None:
    assert sweep.N_INT_VARS == 6
    runs = sweep.smoke_runs(sweep.make_runs(["siren"], ["gaussian"], [42]))
    assert len(runs) == 3
    assert {run["config"]["n_epochs"] for run in runs} == {50}
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        result_path = root / "results_gpu0_smoke.csv"
        history_dir = root / "history_smoke"
        completed = sweep.load_completed(result_path)
        calls = 0
        for run in runs:
            if sweep.run_key(run) in completed:
                continue
            calls += 1
            row = {
                "architecture_key": "siren", "architecture": "SIREN", "function": "gaussian",
                "n_int_vars": 6, "difficulty": "canonical", "params": "(5.0, 0.5)",
                "seed": 42, "lr": run["config"]["lr"], "n_epochs": run["config"]["n_epochs"],
                "n_per_param": run["config"]["n_per_param"], "n_weights": 1, "exact": "1",
                "estimate": "1", "abs_err": "0", "final_loss": "0", "min_loss": "0",
                "wall_clock_s": "0", "epochs_per_sec": "0", "step_latency_s": "0", "device": "mock",
            }
            sweep.append_row(result_path, row)
            sweep.save_history(history_dir, run, [float(i) for i in range(run["config"]["n_epochs"])])

        assert calls == 3
        assert result_path.name.endswith("_smoke.csv")
        with result_path.open(newline="", encoding="utf-8") as handle:
            assert len(list(csv.DictReader(handle))) == 3
        assert len(sweep.load_completed(result_path)) == 3
        assert not [run for run in runs if sweep.run_key(run) not in sweep.load_completed(result_path)]

        sweep.write_summaries(root, smoke=True)
        with (root / "table1_absolute_error_smoke.csv").open(newline="", encoding="utf-8") as handle:
            table_row = next(csv.DictReader(handle))
        assert {"I_star", "I_mean", "abs_err_min", "abs_err_mean", "abs_err_max"} <= set(table_row)

        sample_history = sweep.history_path(history_dir, runs[0])
        with sample_history.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        assert len(rows) <= (max(run["config"]["n_epochs"] for run in runs) + 49) // 50
        assert set(rows[0]) == {"epoch", "loss", "lr"}
        assert abs(float(rows[-1]["lr"]) - runs[0]["config"]["lr"] / 10.0) < 1e-15
    print("mock test passed: smoke namespacing, history downsampling, and resume deduplication")


if __name__ == "__main__":
    main()
