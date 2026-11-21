"""Build charts and a Russian-language PDF report from the recorded benchmark."""

import csv
import sys
from pathlib import Path
from statistics import median

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from reportlab.lib.colors import HexColor
from reportlab.lib.pagesizes import A4
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen.canvas import Canvas

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from utils.paths import get_mirror_path

LABELS = {
    "oscillatory": "Осцилляция",
    "product_peak": "Пик Лоренца",
    "corner_peak": "Угловой пик",
    "gaussian": "Гауссиан",
    "c0_continuous": "Излом C⁰",
    "coupled_polynomial": "Связанный полином",
    "coupled_exponential": "Связанная экспонента",
}
FORMULAS = (
    "cos(π + 5 ∑ xᵢ)",
    "∏ 25 / (25 + (xᵢ − ½)²)",
    "(1 + ∑ xᵢ)⁻⁽ᵈ⁺¹⁾",
    "exp(−25 ∑ (xᵢ − ½)²)",
    "exp(−5 ∑ |xᵢ − ½|)",
    "∏ (1 + xᵢ xᵢ₊₁), индексы циклические",
    "exp(2 ∏ xᵢ)",
)
DIMS = range(1, 7)
INK = "#14243a"
MUTED = "#465870"
ACCENT = "#14677a"
LINE = "#d7e1ea"


def read_rows(path: Path) -> list[dict]:
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def save_charts(dest: Path, digits: np.ndarray, timings: list[dict]) -> None:
    plt.rcParams.update({"font.family": "DejaVu Sans", "text.color": INK, "axes.labelcolor": INK})
    fig, ax = plt.subplots(figsize=(10.6, 4.7), layout="constrained")
    heatmap = ax.imshow(digits, cmap="YlGnBu", vmin=0, vmax=16, aspect="auto")
    ax.set_xticks(np.arange(len(DIMS)), labels=list(DIMS))
    ax.set_yticks(np.arange(len(LABELS)), labels=list(LABELS.values()))
    ax.set_xlabel("Число переменных")
    ax.set_xticks(np.arange(-0.5, len(DIMS), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(LABELS), 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=1.5)
    ax.tick_params(which="both", length=0, pad=8)
    for row, column in np.ndindex(digits.shape):
        value = digits[row, column]
        ax.text(column, row, f"{value:.2f}", ha="center", va="center",
                color="white" if value >= 10 else INK, fontweight="bold" if value >= 12 else "normal")
    fig.colorbar(heatmap, ax=ax, pad=0.02, label="Корректные цифры, худший запуск")
    ax.spines[:].set_visible(False)
    fig.savefig(dest / "accuracy.png", dpi=200, facecolor="white")
    plt.close(fig)

    fig, (ax_time, ax_sites) = plt.subplots(1, 2, figsize=(10.6, 4.15), layout="constrained")
    by_dim = [[r for r in timings if int(r["dimension"]) == d] for d in DIMS]
    for key, label, color in (("fit_seconds", "Обучение", ACCENT), ("readout_seconds", "Интеграл", "#a34b25")):
        values = [1000 * median(float(r[key]) for r in group) for group in by_dim]
        ax_time.plot(list(DIMS), values, marker="o", color=color, label=label)
    ax_time.legend(frameon=False)
    ax_time.set_ylabel("Медианное время, мс")
    ax_sites.plot(list(DIMS), [int(group[0]["training_sites"]) for group in by_dim], marker="s", color=ACCENT)
    ax_sites.set_ylabel("Точек обучения")
    for ax in (ax_time, ax_sites):
        ax.set_xlabel("Число переменных")
        ax.set_xticks(list(DIMS))
        ax.set_yscale("log")
        ax.grid(axis="y", color=LINE, alpha=0.9)
        ax.spines[["top", "right"]].set_visible(False)
    fig.savefig(dest / "cost.png", dpi=200, facecolor="white")
    plt.close(fig)


def make_pdf(
    dest: Path, data: list[dict], timings: list[dict], latency: list[dict],
    ablation: list[dict], digits: np.ndarray,
) -> None:
    fonts = Path(matplotlib.get_data_path()) / "fonts" / "ttf"
    pdfmetrics.registerFont(TTFont("Body", str(fonts / "DejaVuSans.ttf")))
    pdfmetrics.registerFont(TTFont("Bold", str(fonts / "DejaVuSans-Bold.ttf")))
    path = dest / "spectral_benchmark.pdf"
    pdf = Canvas(str(path), pagesize=A4, pageCompression=1, invariant=True)
    pdf.setTitle("Спектральная нейросеть для интегрирования")
    pdf.setAuthor("Skuld")
    width, height = A4
    margin = 42
    content = width - 2 * margin

    def write(x, y, message, size=10, *, bold=False, color=INK, right=False):
        pdf.setFont("Bold" if bold else "Body", size)
        pdf.setFillColor(HexColor(color))
        draw = pdf.drawRightString if right else pdf.drawString
        draw(x, y, message)

    def heading(page, title, subtitle):
        write(margin, height - 48, "SKULD  /  ЭКСПЕРИМЕНТЫ", 9, bold=True, color=ACCENT)
        write(margin, height - 84, title, 22, bold=True)
        write(margin, height - 106, subtitle, 10, color=MUTED)
        pdf.setStrokeColor(HexColor(LINE))
        pdf.line(margin, 45, width - margin, 45)
        write(margin, 29, "Единичный куб  •  CPU, float64", 8, color=MUTED)
        write(width - margin, 29, f"{page} / 4", 8, color=MUTED, right=True)

    by_dim = [[r for r in timings if int(r["dimension"]) == d] for d in DIMS]
    ready = [float(row["ready_seconds"]) for row in latency if row["trial"] != "scored"]
    full = float(next(row["exit_seconds"] for row in latency if row["trial"] == "scored"))
    heading(1, "Точность на семи функциях", "Три обучающие выборки для каждой размерности от 1 до 6")
    write(margin, 696, f"Мезонный пример: {median(ready):.2f} с от старта Python до восьми интегралов.", 11, bold=True)
    write(margin, 677, f"Диапазон {min(ready):.2f}–{max(ready):.2f} с. Полный запуск с проверкой и CSV: {full:.2f} с.", 9, color=MUTED)
    pdf.drawImage(str(dest / "accuracy.png"), margin, 423, width=content, height=227)
    write(margin, 402, "Более высокое число означает меньшую относительную ошибку.", 9, color=MUTED)
    headings = ("d", "Степень", "Точек", "Мин. / сред. / макс.", "Прошли 10⁻¹²", "Готово, мс")
    columns = [margin + 15, margin + 88, margin + 169, margin + 298, margin + 409, width - margin]
    for x, label in zip(columns, headings, strict=True):
        write(x, 362, label, 9, bold=True, color=ACCENT, right=True)
    for i, (dimension, group) in enumerate(zip(DIMS, by_dim, strict=True)):
        scores = [float(r["correct_digits"]) for r in data if int(r["dimension"]) == dimension]
        y = 335 - i * 28
        if i % 2 == 0:
            pdf.setFillColor(HexColor("#f3f7fa"))
            pdf.rect(margin, y - 8, content, 23, fill=1, stroke=0)
        cells = (
            str(dimension), group[0]["degree"], f"{int(group[0]['training_sites']):,}".replace(",", " "),
            f"{min(scores):.2f} / {np.mean(scores):.2f} / {max(scores):.2f}",
            f"{np.count_nonzero(digits[:, i] >= 12)} / {len(LABELS)}",
            f"{1000 * median(float(r['ready_seconds']) for r in group):.2f}",
        )
        for x, label in zip(columns, cells, strict=True):
            write(x, y, label, 9, right=True)
    write(margin, 144, "Порог засчитан, если функция прошла его на каждой из трёх выборок.", 9, color=MUTED)
    write(margin, 127, "Время в таблице включает обучение и расчёт семи интегралов после импортов.", 9, color=MUTED)
    pdf.showPage()

    heading(2, "Стоимость вычисления", "Медианы трёх запусков после импорта библиотек. Обе шкалы Y логарифмические.")
    pdf.drawImage(str(dest / "cost.png"), margin, 495, width=content, height=200)
    write(margin, 458, "Как устроена проверка", 14, bold=True)
    for y, line in zip((436, 419, 402, 385), (
        "Общая сеть учится на значениях семи функций в каждой размерности.",
        "Интегралы читаются аналитически из обученных весов сети.",
        "Эталоны вычислены по формулам в 50-значной арифметике после обучения.",
        "Степень снижается с ростом d, чтобы сдержать рост тензорной сетки.",
    ), strict=True):
        write(margin, y, line, 9.5)
    write(margin, 345, "Подынтегральные функции на [0, 1]ᵈ", 14, bold=True)
    for i, (label, formula) in enumerate(zip(LABELS.values(), FORMULAS, strict=True)):
        y = 317 - i * 29
        write(margin, y, label, 9.5, bold=True)
        write(margin + 166, y, formula, 10)
    write(margin, 94, "Первые пять функций взяты из канонического набора Genz в Skuld.", 9, color=MUTED)
    pdf.showPage()

    heading(3, "Интегралы и ошибки", "Оценки сети для seed 42. Худшая ошибка учитывает все три обучающие выборки.")
    positions = (margin, margin + 27, margin + 248, margin + 368, width - margin)
    for i, (x, label) in enumerate(zip(positions, ("d", "Функция", "Сеть, seed 42", "Эталон", "Худшая отн. ошибка"), strict=True)):
        write(x, 703, label, 8.5, bold=True, color=ACCENT, right=i >= 2)
    rows = [r for r in data if r["seed"] == "42"]
    for i, row in enumerate(rows):
        dimension = int(row["dimension"])
        name = row["function"]
        function_index = list(LABELS).index(name)
        y = 682 - 13.8 * i
        if dimension % 2 == 1:
            pdf.setFillColor(HexColor("#f3f7fa"))
            pdf.rect(margin - 2, y - 3, content + 2, 13.8, fill=1, stroke=0)
        cells = (
            str(dimension) if function_index == 0 else "",
            LABELS[name], f"{float(row['estimate']):.8e}", f"{float(row['exact']):.8e}",
            f"{10.0 ** (-digits[function_index, dimension - 1]):.3e}",
        )
        for j, (x, label) in enumerate(zip(positions, cells, strict=True)):
            write(x, y, label, 8.1, right=j >= 2)
    write(margin, 80, "Значения в таблице округлены. В CSV сохранены исходные результаты.", 9, color=MUTED)
    pdf.showPage()

    heading(4, "Что дала обучаемая часть", "Те же значения мезонной функции в тех же точках, степень 47")
    columns = (margin, margin + 263, width - margin)
    for i, (x, label) in enumerate(zip(columns, ("Seed", "Обученные признаки", "Фиксированные полиномы"), strict=True)):
        write(x, 698, label, 9.5, bold=True, color=ACCENT, right=i > 0)
    for index, seed in enumerate(dict.fromkeys(row["seed"] for row in ablation)):
        group = [row for row in ablation if row["seed"] == seed]
        worst_learned = max(float(row["learned_relative_error"]) for row in group)
        worst_fixed = max(float(row["fixed_relative_error"]) for row in group)
        y = 671 - index * 26
        write(columns[0], y, seed)
        write(columns[1], y, f"{worst_learned:.3e}", right=True)
        write(columns[2], y, f"{worst_fixed:.3e}", right=True)
    write(margin, 585, "В таблице указана максимальная относительная ошибка по восьми интегралам.", 9, color=MUTED)
    write(margin, 545, "Откуда берётся формула", 14, bold=True)
    lines = (
        "На каждой оси сеть учит линейные комбинации полиномов Чебышёва по срезам функции.",
        "Пусть A - значения этих признаков на оси, A⁺ - псевдообратная, m - вектор их интегралов.",
        "На этой оси подгонка использует A⁺, а интегрирование умножает на mᵀ.",
        "После композиции остаётся ряд весов mᵀA⁺. Тензорное произведение таких весов",
        "даёт кубатурную формулу для отсчётов на сетке. Веса зависят от обученных признаков.",
    )
    for index, line in enumerate(lines):
        write(margin, 521 - index * 18, line, 9)
    write(margin, 396, "Что показало сравнение", 14, bold=True)
    lines = (
        "Оба варианта дали около 14–15 цифр. Для этого мезонного теста высокая точность",
        "сохранилась при замене обученных признаков фиксированным базисом Чебышёва.",
        "Результат подтверждает точность спектрального приближения гладкой функции.",
        "Преимущество обучаемых признаков по точности в этом сравнении не установлено.",
    )
    for index, line in enumerate(lines):
        write(margin, 372 - index * 18, line, 9)
    pdf.save()
    print(path)


def main() -> None:
    dest = get_mirror_path(__file__, "results")
    data = read_rows(dest / "integrals.csv")
    timings = read_rows(dest / "timings.csv")
    latency = read_rows(dest / "meson_latency.csv")
    ablation = read_rows(dest / "meson_ablation.csv")
    digits = np.array([
        [min(float(r["correct_digits"]) for r in data if r["function"] == name and int(r["dimension"]) == d)
         for d in DIMS]
        for name in LABELS
    ])
    save_charts(dest, digits, timings)
    make_pdf(dest, data, timings, latency, ablation, digits)


if __name__ == "__main__":
    main()
