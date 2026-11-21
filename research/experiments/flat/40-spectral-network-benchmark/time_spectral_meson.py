"""Time fresh Python processes through meson output and full scored completion."""

import csv
import subprocess
import sys
from pathlib import Path
from statistics import median
from time import perf_counter

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from utils.paths import get_mirror_path


def main() -> None:
    experiments = Path(__file__).resolve().parents[1]
    meson = next(experiments.glob("*-spectral-network/spectral_network_meson.py"))
    import_experiment = f"from runpy import run_path\nexperiment = run_path({str(meson)!r})\n"
    code = import_experiment + """
model = experiment["fit_meson_network"](42)
with experiment["torch"].inference_mode():
    print("INTEGRALS_READY", model.integrate().tolist(), flush=True)
"""
    rows = []
    for trial in range(1, 8):
        started = perf_counter()
        with subprocess.Popen([sys.executable, "-u", "-c", code], stdout=subprocess.PIPE, text=True) as process:
            for line in process.stdout:
                if line.startswith("INTEGRALS_READY "):
                    ready_seconds = perf_counter() - started
                    break
            else:
                raise RuntimeError("The child did not produce integrals")
            if process.wait() != 0:
                raise RuntimeError("The child did not exit successfully")
        rows.append({"trial": trial, "ready_seconds": ready_seconds, "exit_seconds": perf_counter() - started})
        print(f"Trial {trial}: {ready_seconds:.3f} s to integrals", flush=True)

    # Run the complete scoring path once and keep its CSV in a temporary directory.
    code = import_experiment + """
from pathlib import Path
from tempfile import TemporaryDirectory
main = experiment["main"]
with TemporaryDirectory() as directory:
    main.__globals__["get_mirror_path"] = lambda *_: Path(directory)
    raise SystemExit(main())
"""
    started = perf_counter()
    subprocess.run([sys.executable, "-u", "-c", code], stdout=subprocess.DEVNULL, check=True)
    rows.append({"trial": "scored", "ready_seconds": "", "exit_seconds": perf_counter() - started})
    path = get_mirror_path(__file__, "results") / "meson_latency.csv"
    with path.open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    print(f"Median time to integrals: {median(r['ready_seconds'] for r in rows[:-1]):.3f} s")
    print(f"Scored run including CSV: {rows[-1]['exit_seconds']:.3f} s")
    print(path)


if __name__ == "__main__":
    main()
