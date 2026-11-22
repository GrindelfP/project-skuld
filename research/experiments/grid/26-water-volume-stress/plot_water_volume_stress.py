"""Render the completed water-volume sweep as a Russian-language HTML report."""

import csv
import sys
from collections import Counter
from pathlib import Path
from statistics import mean, median

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from utils.paths import get_mirror_path

LABELS = {
    "two_ridges": "Две косые волны",
    "six_fast_ridges": "Шесть быстрых волн",
    "coupled_polynomial": "Связанный полином",
    "coupled_exponential": "Связанная экспонента",
    "genz_oscillatory": "Осцилляция Genz",
    "genz_product_peak": "Лоренцевы пики",
    "genz_corner_peak": "Угловой пик",
    "genz_gaussian": "Гауссов пик",
    "genz_c0": "Излом Genz",
    "oblique_jump": "Наклонный скачок",
    "tiny_integral": "Малый интеграл",
}
DIMENSIONS = (2, 3, 4, 5, 6)
COLORS = {"target": "#16786a", "below": "#e0a546", "uncertified": "#bfcbd9"}
TARGET = 12


def scores(rows: list[dict[str, str]]) -> list[float]:
    return [float(row["correct_digits"]) for row in rows if row["status"] == "ok"]


def outcomes(rows: list[dict[str, str]]) -> Counter:
    measured = scores(rows)
    return Counter({
        "target": sum(score >= TARGET for score in measured),
        "below": sum(score < TARGET for score in measured),
        "uncertified": len(rows) - len(measured),
    })


def number(value: float, places: int = 2) -> str:
    return f"{value:.{places}f}".replace(".", ",")


def precision_plot(cells: dict, path: Path) -> None:
    values = [
        [median(cell_scores) if (cell_scores := scores(cells[name, d])) else np.nan for d in DIMENSIONS]
        for name in LABELS
    ]
    fig, ax = plt.subplots(figsize=(8.0, 6.15), layout="constrained")
    palette = plt.colormaps["viridis"].with_extremes(bad="#e4eaf1")
    image = ax.imshow(np.ma.masked_invalid(values), cmap=palette, vmin=0, vmax=16, aspect="auto")
    ax.set_xticks(range(len(DIMENSIONS)), [f"{d}D" for d in DIMENSIONS])
    ax.set_yticks(range(len(LABELS)), LABELS.values())
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    for i, name in enumerate(LABELS):
        for j, d in enumerate(DIMENSIONS):
            value = values[i][j]
            measured = len(scores(cells[name, d]))
            label = "—" if not measured else number(value, 1)
            color = "white" if measured and value < 9 else "#14283e"
            ax.text(j, i, label, ha="center", va="center", color=color, fontweight="bold")
    ax.set_title("Медиана правильных цифр среди полученных оценок", loc="left", pad=18)
    fig.colorbar(image, ax=ax, label="Правильные цифры", shrink=.84, extend="both")
    fig.savefig(path, dpi=170)
    plt.close(fig)


def coverage_plot(cells: dict, path: Path) -> None:
    counts = [
        outcomes([row for d in DIMENSIONS for row in cells[name, d]])
        for name in LABELS
    ]
    fig, ax = plt.subplots(figsize=(8.0, 5.7), layout="constrained")
    left = np.zeros(len(LABELS))
    for key, label in (
        ("target", "≥ 12 цифр"), ("below", "< 12 цифр"), ("uncertified", "Допуск не подтверждён"),
    ):
        widths = [count[key] for count in counts]
        bars = ax.barh(list(LABELS.values()), widths, left=left, color=COLORS[key], label=label)
        ax.bar_label(bars, labels=[str(width) if width else "" for width in widths],
                     label_type="center", color="white" if key == "target" else "#14283e")
        left += widths
    ax.invert_yaxis()
    ax.set_xlim(0, max(left))
    ax.set_xlabel("Число запусков")
    ax.set_title("Исходы по функциям и всем размерностям", loc="left", pad=46)
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.015), frameon=False, ncols=3)
    ax.spines[["left", "top", "right"]].set_visible(False)
    ax.tick_params(axis="y", length=0)
    fig.savefig(path, dpi=170)
    plt.close(fig)


def comparison_plot(rows: list[dict[str, str]], path: Path) -> None:
    pairs = [
        (float(row["sample_mean_digits"]), float(row["correct_digits"]))
        for row in rows if row["status"] == "ok"
    ]
    fig, ax = plt.subplots(figsize=(7.4, 5.3), layout="constrained")
    lower = min(min(pair) for pair in pairs) - .3
    upper = max(max(pair) for pair in pairs) + .5
    ax.plot([lower, upper], [lower, upper], ":", color="#788da3", label="Равная точность")
    ax.axhline(TARGET, linestyle="--", color="#16786a", label="Цель 12 цифр")
    x, y = np.array(pairs).T
    ax.scatter(x, y, c=np.where(y >= TARGET, COLORS["target"], "#c27019"), alpha=.57, s=20,
               linewidths=0)
    ax.set(xlim=(lower, upper), ylim=(lower, upper),
           xlabel="Среднее по тем же точкам, правильные цифры",
           ylabel="Нейросеть, правильные цифры")
    ax.set_title("Сравнение на запусках, вернувших оценку", loc="left", pad=12)
    ax.grid(alpha=.17)
    ax.legend(loc="upper right", frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.savefig(path, dpi=170)
    plt.close(fig)


def table_cell(rows: list[dict[str, str]]) -> str:
    measured = scores(rows)
    passed = sum(x >= TARGET for x in measured)
    state = "empty" if not measured else "good" if passed else "low"
    value = number(median(measured)) if measured else "—"
    return (
        f'<td class="{state}"><strong>{value}</strong>'
        f'<small>допуск {len(measured)}/{len(rows)} · ≥12 {passed}</small></td>'
    )


def budget_row(name: str, rows: list[dict[str, str]]) -> str:
    config = rows[0]
    measured = scores(rows)
    counts = outcomes(rows)
    accuracy = " / ".join(number(x) for x in (min(measured), mean(measured), max(measured))) if measured else "—"
    return (
        f"<tr><th scope='row'>{name}</th><td>{config['width']}</td>"
        f"<td>{config['training_sites']}</td><td>{config['frequency_limit']}</td>"
        f"<td>{len(measured)}/{len(rows)}</td><td>{counts['target']}</td><td>{accuracy}</td></tr>"
    )


def html_report(rows: list[dict[str, str]], cells: dict) -> str:
    totals = outcomes(rows)
    measured = scores(rows)
    wins = sum(
        float(row["correct_digits"]) > float(row["sample_mean_digits"])
        for row in rows if row["status"] == "ok"
    )
    header = "".join(f'<th scope="col">{d}D</th>' for d in DIMENSIONS)
    table = "".join(
        f'<tr><th scope="row">{label}</th>'
        + "".join(table_cell(cells[name, d]) for d in DIMENSIONS) + "</tr>"
        for name, label in LABELS.items()
    )
    budgets = "".join(
        budget_row(name, [row for row in rows if row["name"] == name])
        for name in dict.fromkeys(row["name"] for row in rows)
    )
    digits = " / ".join(number(x) for x in (min(measured), mean(measured), max(measured)))
    return f"""<!doctype html>
<html lang="ru">
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Нейросетевое интегрирование методом объёма воды</title>
<style>
* {{ box-sizing: border-box }}
body {{ margin: 0; font: 16px/1.55 system-ui, sans-serif; color: #14283e; background: #f3f6fb }}
main {{ width: min(1100px, 100%); margin: auto; padding: 38px 24px 56px }}
h1 {{ font-size: clamp(30px, 5vw, 48px); line-height: 1.06; margin: 0 0 .4em }}
h2 {{ font-size: 24px; margin: 30px 0 8px }}
p {{ max-width: 78ch; margin: 0 0 13px }}
a {{ color: #135c91 }}
.metrics {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(155px, 1fr)); gap: 10px; margin: 20px 0 }}
.metric {{ background: #14283e; color: #f4f8ff; border-radius: 10px; padding: 12px 16px }}
.metric strong {{ display: block; font-size: 24px; color: #b7efde }}
.table-wrap {{ overflow-x: auto; border-radius: 10px; background: white }}
table {{ border-collapse: collapse; width: 100%; font-variant-numeric: tabular-nums }}
th, td {{ text-align: center; padding: 8px; border: 2px solid white }}
thead th {{ background: #213c59; color: white }}
tbody th {{ text-align: left; background: #eef3f9; white-space: nowrap }}
td small {{ display: block; min-width: 110px; white-space: nowrap }}
.empty {{ background: #e4eaf1 }}
.good {{ background: #c9eee4 }}
.low {{ background: #fce7bc }}
.plots {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(min(430px, 100%), 1fr)); gap: 12px }}
figure {{ margin: 0; background: white; padding: 10px; border-radius: 10px }}
img {{ width: 100%; height: auto; display: block }}
figcaption {{ padding: 6px 10px }}
</style>
<main>
<h1>Интегрирование через объём воды</h1>
<p>2D–6D · 11 поверхностей · 4 бюджета · 3 seed на каждый бюджет.
Правильные цифры вычислены как −log<sub>10</sub> относительной ошибки.</p>
<div class="metrics">
  <div class="metric"><strong>{len(rows)}</strong>запусков</div>
  <div class="metric"><strong>{totals['target']}</strong>с точностью ≥12 цифр</div>
  <div class="metric"><strong>{totals['uncertified']}</strong>отказов проверки диапазона</div>
  <div class="metric"><strong>{len(measured)}</strong>численных оценок</div>
</div>
<h2>Точность по функциям и размерностям</h2>
<p>В ячейке указана медиана правильных цифр среди полученных оценок,
число допусков из 12 запусков и число оценок с ≥12 цифрами.
Бирюзовым выделены ячейки, где хотя бы один запуск достиг цели.</p>
<div class="table-wrap"><table>
<thead><tr><th scope="col">Функция</th>{header}</tr></thead><tbody>{table}</tbody></table></div>
<h2>Сравнение бюджетов</h2>
<p>По всем полученным оценкам мин / среднее / макс составили {digits}.
Допуск означает, что амплитудная оценка гарантирует глубину воды в [0, 1] на кубе.</p>
<div class="table-wrap"><table>
<thead><tr><th>Бюджет</th><th>Нейроны</th><th>Точки</th><th>Предел частот</th>
<th>Допущено</th><th>≥12 цифр</th><th>Цифры, мин / сред. / макс</th></tr></thead>
<tbody>{budgets}</tbody></table></div>
<h2>Графики</h2>
<div class="plots">
<figure><img src="precision.png" alt="Тепловая карта медианной точности">
<figcaption>Серые ячейки соответствуют функциям и размерностям без допущенных оценок.</figcaption></figure>
<figure><img src="coverage.png" alt="Число точных, неточных и недопущенных запусков">
<figcaption>По каждой функции было 60 запусков.</figcaption></figure>
<figure><img src="comparison.png" alt="Правильные цифры сети и среднего по тем же точкам">
<figcaption>На допущенных запусках нейросеть превзошла среднее по тем же точкам в {wins} из {len(measured)} случаев.</figcaption></figure>
</div>
<h2>Данные</h2>
<p><a href="integrals.csv">Все запуски</a> · <a href="summary.csv">Сводка по бюджетам</a></p>
</main>
</html>
"""


def main() -> None:
    destination = get_mirror_path(__file__, "results")
    with (destination / "integrals.csv").open(encoding="utf-8", newline="") as source:
        rows = list(csv.DictReader(source))
    cells = {(name, d): [] for name in LABELS for d in DIMENSIONS}
    for row in rows:
        cells[row["function"], int(row["dimension"])].append(row)
    if any(len(cell) != 12 for cell in cells.values()):
        raise ValueError("Render the report after completing the full sweep")
    precision_plot(cells, destination / "precision.png")
    coverage_plot(cells, destination / "coverage.png")
    comparison_plot(rows, destination / "comparison.png")
    (destination / "report.html").write_text(html_report(rows, cells), encoding="utf-8")
    print(destination / "report.html")


if __name__ == "__main__":
    main()
