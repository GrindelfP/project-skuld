import json
import matplotlib.pyplot as plt
import numpy as np
from utils.paths import get_mirror_path


def main():
    data_dir = get_mirror_path(__file__, "results")
    data_file = data_dir / "integration_results.json"

    if not data_file.exists():
        print(f"Error: File not found {data_file}")
        print("Probably, experiment wasn't run!")
        return

    with open(data_file, "r", encoding="utf-8") as f:
        results = json.load(f)

    configs = [item["config"] for item in results]
    nni_vals = [item["nni"] for item in results]
    num_vals = [item["num"] for item in results]
    errors = [item["error"] for item in results]

    x = np.arange(len(configs))
    width = 0.35

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    ax1.bar(x - width / 2, nni_vals, width, label='NNI', color='skyblue')
    ax1.bar(x + width / 2, num_vals, width, label='NUM', color='salmon')
    ax1.set_ylabel('Integral value')
    ax1.set_title('NNI vs Num Integration')
    ax1.legend()
    ax1.grid(axis='y', linestyle='--', alpha=0.7)

    ax2.bar(x, errors, width=0.5, color='orange', label='Abs error')
    ax2.set_ylabel('Error (log scale)')
    ax2.set_yscale('log')
    ax2.set_xlabel('(a, b, m, n)')
    ax2.set_xticks(x)
    ax2.set_xticklabels(configs)
    ax2.legend()
    ax2.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plot_file = data_dir / "comparison_plot.png"
    plt.savefig(plot_file, dpi=300)
    print(f"[OK] Plot saved to: {plot_file}")
    plt.show()


if __name__ == "__main__":
    main()
