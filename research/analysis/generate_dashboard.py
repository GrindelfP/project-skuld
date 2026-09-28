"""
generate_dashboard.py — build an interactive HTML dashboard from sweep CSVs.

Reads the three architecture sweep results (SIREN, WIRE, KAN) and produces
a single self-contained dashboard.html with Plotly.js visualisations.

Usage:
    python generate_dashboard.py
"""
import csv
import json
import os
from pathlib import Path

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results" / "grid" / "6-siren-wire-kan-sweep-m2"
OUTPUT_PATH = Path(__file__).resolve().parent / "dashboard.html"

# ─────────────────────────────────────────────────────────────────────────
# 1. LOAD DATA
# ─────────────────────────────────────────────────────────────────────────

def load_csv(path: Path) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def parse_hidden_sizes(s: str) -> str:
    """Convert '[128, 128, 128]' -> '128x128x128' for display."""
    return s.replace("[", "").replace("]", "").replace(", ", "x").replace(",", "x")


def normalise_rows(rows: list[dict], arch: str) -> list[dict]:
    """Add architecture tag and normalise column names."""
    out = []
    for r in rows:
        d = dict(r)
        d["arch"] = arch
        # normalise epoch column
        if "epochs" in d and "n_epochs" not in d:
            d["n_epochs"] = d["epochs"]
        # parse hidden_sizes for SIREN
        if "hidden_sizes" in d:
            d["config_label"] = parse_hidden_sizes(d["hidden_sizes"])
        elif "entry_width" in d:
            d["config_label"] = f"W{d['entry_width']}xB{d['n_blocks']}"
        elif "width" in d:
            d["config_label"] = f"W{d['width']}xD{d['depth']}"
        else:
            d["config_label"] = "?"
        out.append(d)
    return out


def load_all() -> dict[str, list[dict]]:
    files = {
        "SIREN": "sweep_siren_results_m2_2026-09-25_21-05.csv",
        "WIRE": "sweep_wire_results_m2_2026-09-27_10-34.csv",
        "KAN": "sweep_kan_results_m2_2026-09-28_13-48.csv",
    }
    data = {}
    for arch, fname in files.items():
        path = RESULTS_DIR / fname
        if path.exists():
            data[arch] = normalise_rows(load_csv(path), arch)
    return data


# ─────────────────────────────────────────────────────────────────────────
# 2. BUILD HTML
# ─────────────────────────────────────────────────────────────────────────

COLORS = {
    "SIREN": "#636efa",
    "WIRE": "#ef553b",
    "KAN": "#00cc96",
}

CSS = """
* { margin: 0; padding: 0; box-sizing: border-box; }
body {
    font-family: 'Segoe UI', system-ui, -apple-system, sans-serif;
    background: #0f1117;
    color: #e1e4e8;
    line-height: 1.6;
}
.container { max-width: 1400px; margin: 0 auto; padding: 20px; }
h1 { font-size: 2em; margin-bottom: 8px; color: #fff; }
h2 { font-size: 1.4em; margin: 24px 0 12px; color: #fff; border-bottom: 1px solid #30363d; padding-bottom: 6px; }
h3 { font-size: 1.1em; margin: 16px 0 8px; color: #e1e4e8; }
.subtitle { color: #8b949e; margin-bottom: 24px; }

/* Summary cards */
.cards { display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 16px; margin-bottom: 24px; }
.card {
    background: #161b22;
    border: 1px solid #30363d;
    border-radius: 8px;
    padding: 16px 20px;
}
.card .label { font-size: 0.85em; color: #8b949e; text-transform: uppercase; letter-spacing: 0.5px; }
.card .value { font-size: 1.8em; font-weight: 700; margin-top: 4px; }
.card .sub { font-size: 0.85em; color: #8b949e; margin-top: 2px; }

/* Tabs */
.tabs { display: flex; gap: 4px; margin-bottom: 16px; flex-wrap: wrap; }
.tab {
    padding: 8px 20px;
    border-radius: 6px 6px 0 0;
    cursor: pointer;
    font-weight: 500;
    background: #161b22;
    border: 1px solid #30363d;
    border-bottom: none;
    color: #8b949e;
    transition: all 0.15s;
}
.tab:hover { color: #e1e4e8; background: #1c2128; }
.tab.active { background: #0f1117; color: #fff; border-color: #58a6ff; }

/* Tab panels */
.panel { display: none; }
.panel.active { display: block; }

/* Plots */
.plot-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 16px; }
.plot-full { grid-column: 1 / -1; }
.plot-box {
    background: #161b22;
    border: 1px solid #30363d;
    border-radius: 8px;
    padding: 12px;
    min-height: 380px;
}
.plot-box .js-plotly-plot {
    width: 100%;
    height: 100%;
    min-height: 350px;
}
.js-plotly-plot .plotly .modebar { right: 8px !important; top: 8px !important; }

/* Table */
.table-wrap { overflow-x: auto; }
table { width: 100%; border-collapse: collapse; font-size: 0.85em; }
th, td { padding: 8px 12px; text-align: left; border-bottom: 1px solid #21262d; }
th { background: #161b22; color: #8b949e; font-weight: 600; cursor: pointer; position: sticky; top: 0; z-index: 1; }
th:hover { color: #58a6ff; }
tr:hover td { background: #1c2128; }
td.num { text-align: right; font-variant-numeric: tabular-nums; }
.arch-badge {
    display: inline-block;
    padding: 2px 8px;
    border-radius: 12px;
    font-size: 0.8em;
    font-weight: 600;
}

/* Filter bar */
.filters { display: flex; gap: 12px; margin-bottom: 16px; flex-wrap: wrap; align-items: center; }
.filters label { font-size: 0.85em; color: #8b949e; }
.filters select, .filters input {
    background: #161b22;
    border: 1px solid #30363d;
    color: #e1e4e8;
    padding: 6px 10px;
    border-radius: 4px;
    font-size: 0.85em;
}
.filters input { width: 200px; }

@media (max-width: 900px) {
    .plot-grid { grid-template-columns: 1fr; }
}
"""


def build_summary_cards(data: dict[str, list[dict]]) -> str:
    total = sum(len(v) for v in data.values())
    cards = [f'<div class="card"><div class="label">Total Runs</div><div class="value">{total}</div><div class="sub">{len(data)} architectures</div></div>']
    for arch, rows in data.items():
        best = max(rows, key=lambda r: int(r["min_correct_digits"]))
        cards.append(
            f'<div class="card">'
            f'<div class="label">Best {arch}</div>'
            f'<div class="value">{best["min_correct_digits"]} dig</div>'
            f'<div class="sub">{best["config_label"]} · {int(float(best["n_epochs"]))} epochs</div>'
            f'</div>'
        )
    return '<div class="cards">' + "".join(cards) + "</div>"


def build_tab_bar(tabs: list[tuple[str, str]]) -> str:
    btns = [f'<div class="tab{" active" if i == 0 else ""}" data-tab="{tid}">{label}</div>' for i, (tid, label) in enumerate(tabs)]
    return '<div class="tabs">' + "".join(btns) + "</div>"


def build_plots_js(data: dict[str, list[dict]]) -> str:
    """Build all Plotly traces and layout configs, return JS."""
    # Prepare serialisable data
    serialisable = {}
    for arch, rows in data.items():
        serialisable[arch] = [
            {
                "arch": r["arch"],
                "config_label": r["config_label"],
                "n_epochs": int(float(r["n_epochs"])),
                "n_params_total": int(float(r["n_params_total"])),
                "min_loss": float(r["min_loss"]),
                "mean_rel_err": float(r["mean_rel_err"]),
                "max_rel_err": float(r["max_rel_err"]),
                "min_correct_digits": int(r["min_correct_digits"]),
                "mean_correct_digits": float(r["mean_correct_digits"]),
                "elapsed_s": float(r["elapsed_s"]),
                "lr": float(r["lr"]),
            }
            for r in rows
        ]

    return f"""
const DATA = {json.dumps(serialisable)};
const COLORS = {json.dumps(COLORS)};

function archRows(arch) {{ return DATA[arch] || []; }}
function allRows() {{ return Object.values(DATA).flat(); }}

// ── Helper: histogram trace ──────────────────────────────────────────────
function histTrace(rows, key, color, name) {{
    return {{
        x: rows.map(r => r[key]),
        type: 'histogram',
        name: name,
        marker: {{ color: color, opacity: 0.75, line: {{ color: color, width: 1 }} }},
        nbinsx: 30,
        hovertemplate: `%{{x}}<br>%{{y}} runs<extra>${{name}}</extra>`,
    }};
}}

// ── Helper: scatter trace ────────────────────────────────────────────────
function scatterTrace(rows, xKey, yKey, color, name, xTitle, yTitle) {{
    return {{
        x: rows.map(r => r[xKey]),
        y: rows.map(r => r[yKey]),
        mode: 'markers',
        type: 'scatter',
        name: name,
        marker: {{
            color: color,
            size: 7,
            opacity: 0.7,
            line: {{ color: '#0f1117', width: 1 }}
        }},
        text: rows.map(r => `${{r.arch}}<br>${{r.config_label}}<br>epochs=${{r.n_epochs}}<br>lr=${{r.lr}}`),
        hovertemplate: `%{{text}}<br>${{xTitle}}: %{{x}}<br>${{yTitle}}: %{{y:.4g}}<extra></extra>`,
    }};
}}

// ── Helper: box trace ────────────────────────────────────────────────────
function boxTrace(rows, key, color, name) {{
    return {{
        y: rows.map(r => r[key]),
        type: 'box',
        name: name,
        marker: {{ color: color }},
        boxpoints: 'outliers',
        hovertemplate: `%{{y:.4g}}<extra>${{name}}</extra>`,
    }};
}}

// ── Layout factory ───────────────────────────────────────────────────────
function baseLayout(title, xTitle, yTitle) {{
    return {{
        title: {{ text: title, font: {{ color: '#e1e4e8', size: 16 }} }},
        xaxis: {{ title: xTitle, gridcolor: '#21262d', zerolinecolor: '#30363d', tickfont: {{ color: '#8b949e' }}, titlefont: {{ color: '#8b949e' }} }},
        yaxis: {{ title: yTitle, gridcolor: '#21262d', zerolinecolor: '#30363d', tickfont: {{ color: '#8b949e' }}, titlefont: {{ color: '#8b949e' }} }},
        paper_bgcolor: '#161b22',
        plot_bgcolor: '#161b22',
        font: {{ color: '#8b949e' }},
        margin: {{ t: 50, b: 50, l: 60, r: 20 }},
        showlegend: true,
        legend: {{ font: {{ color: '#8b949e' }}, bgcolor: 'rgba(0,0,0,0)' }},
    }};
}}

// ── PLOT DEFINITIONS ─────────────────────────────────────────────────────
const PLOTS = {{
    // Tab: Distributions
    'hist-digits': {{
        traces: Object.keys(DATA).map(a => histTrace(archRows(a), 'min_correct_digits', COLORS[a], a)),
        layout: baseLayout('Distribution of Min Correct Digits', 'Min Correct Digits', 'Count'),
        barmode: 'overlay',
    }},
    'hist-relerr': {{
        traces: Object.keys(DATA).map(a => histTrace(archRows(a), 'mean_rel_err', COLORS[a], a)),
        layout: baseLayout('Distribution of Mean Relative Error', 'Mean Rel. Error', 'Count'),
        barmode: 'overlay',
    }},
    'hist-loss': {{
        traces: Object.keys(DATA).map(a => histTrace(archRows(a), 'min_loss', COLORS[a], a)),
        layout: baseLayout('Distribution of Min Loss', 'Min Loss', 'Count'),
        barmode: 'overlay',
    }},
    'hist-time': {{
        traces: Object.keys(DATA).map(a => histTrace(archRows(a), 'elapsed_s', COLORS[a], a)),
        layout: baseLayout('Distribution of Training Time', 'Elapsed (seconds)', 'Count'),
        barmode: 'overlay',
    }},
    'hist-params': {{
        traces: Object.keys(DATA).map(a => histTrace(archRows(a), 'n_params_total', COLORS[a], a)),
        layout: baseLayout('Distribution of Network Parameters', 'Total Parameters', 'Count'),
        barmode: 'overlay',
    }},

    // Tab: Scatter
    'scatter-epochs-digits': {{
        traces: Object.keys(DATA).map(a => scatterTrace(archRows(a), 'n_epochs', 'min_correct_digits', COLORS[a], a, 'Epochs', 'Min Correct Digits')),
        layout: baseLayout('Accuracy vs Epoch Budget', 'Epochs', 'Min Correct Digits'),
    }},
    'scatter-params-digits': {{
        traces: Object.keys(DATA).map(a => scatterTrace(archRows(a), 'n_params_total', 'min_correct_digits', COLORS[a], a, 'Parameters', 'Min Correct Digits')),
        layout: baseLayout('Accuracy vs Network Size', 'Total Parameters', 'Min Correct Digits'),
    }},
    'scatter-loss-digits': {{
        traces: Object.keys(DATA).map(a => scatterTrace(archRows(a), 'min_loss', 'min_correct_digits', COLORS[a], a, 'Min Loss', 'Min Correct Digits')),
        layout: baseLayout('Accuracy vs Training Loss', 'Min Loss', 'Min Correct Digits'),
    }},
    'scatter-time-digits': {{
        traces: Object.keys(DATA).map(a => scatterTrace(archRows(a), 'elapsed_s', 'min_correct_digits', COLORS[a], a, 'Elapsed (s)', 'Min Correct Digits')),
        layout: baseLayout('Accuracy vs Training Time', 'Elapsed (seconds)', 'Min Correct Digits'),
    }},
    'scatter-epochs-loss': {{
        traces: Object.keys(DATA).map(a => scatterTrace(archRows(a), 'n_epochs', 'min_loss', COLORS[a], a, 'Epochs', 'Min Loss')),
        layout: baseLayout('Loss vs Epoch Budget', 'Epochs', 'Min Loss'),
    }},
    'scatter-params-loss': {{
        traces: Object.keys(DATA).map(a => scatterTrace(archRows(a), 'n_params_total', 'min_loss', COLORS[a], a, 'Parameters', 'Min Loss')),
        layout: baseLayout('Loss vs Network Size', 'Total Parameters', 'Min Loss'),
    }},

    // Tab: Comparison
    'box-digits': {{
        traces: Object.keys(DATA).map(a => boxTrace(archRows(a), 'min_correct_digits', COLORS[a], a)),
        layout: baseLayout('Min Correct Digits by Architecture', '', 'Min Correct Digits'),
    }},
    'box-relerr': {{
        traces: Object.keys(DATA).map(a => boxTrace(archRows(a), 'mean_rel_err', COLORS[a], a)),
        layout: baseLayout('Mean Relative Error by Architecture', '', 'Mean Rel. Error'),
    }},
    'box-loss': {{
        traces: Object.keys(DATA).map(a => boxTrace(archRows(a), 'min_loss', COLORS[a], a)),
        layout: baseLayout('Min Loss by Architecture', '', 'Min Loss'),
    }},
    'box-time': {{
        traces: Object.keys(DATA).map(a => boxTrace(archRows(a), 'elapsed_s', COLORS[a], a)),
        layout: baseLayout('Training Time by Architecture', '', 'Elapsed (s)'),
    }},
    'box-params': {{
        traces: Object.keys(DATA).map(a => boxTrace(archRows(a), 'n_params_total', COLORS[a], a)),
        layout: baseLayout('Network Size by Architecture', '', 'Parameters'),
    }},
}};

// ── Render all plots ─────────────────────────────────────────────────────
function renderPlots() {{
    for (const [id, def] of Object.entries(PLOTS)) {{
        const el = document.getElementById(id);
        if (!el) continue;
        Plotly.newPlot(el, def.traces, def.layout, {{
            responsive: true,
            displayModeBar: true,
            modeBarButtonsToRemove: ['lasso2d', 'select2d'],
        }});
    }}
}}

// ── Tab switching ────────────────────────────────────────────────────────
function initTabs() {{
    document.querySelectorAll('.tab').forEach(tab => {{
        tab.addEventListener('click', () => {{
            document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
            document.querySelectorAll('.panel').forEach(p => p.classList.remove('active'));
            tab.classList.add('active');
            document.getElementById('panel-' + tab.dataset.tab).classList.add('active');
            // Resize plots in the newly visible panel
            setTimeout(() => {{
                document.querySelectorAll('#panel-' + tab.dataset.tab + ' .js-plotly-plot').forEach(el => {{
                    Plotly.Plots.resize(el);
                }});
            }}, 50);
        }});
    }});
}}

// ── Data table ───────────────────────────────────────────────────────────
function buildTable() {{
    const all = allRows();
    const cols = [
        {{ key: 'arch', label: 'Arch', type: 'str' }},
        {{ key: 'config_label', label: 'Config', type: 'str' }},
        {{ key: 'n_epochs', label: 'Epochs', type: 'num' }},
        {{ key: 'n_params_total', label: 'Params', type: 'num' }},
        {{ key: 'min_loss', label: 'Min Loss', type: 'num' }},
        {{ key: 'mean_rel_err', label: 'Mean Rel Err', type: 'num' }},
        {{ key: 'min_correct_digits', label: 'Min Dig', type: 'num' }},
        {{ key: 'mean_correct_digits', label: 'Mean Dig', type: 'num' }},
        {{ key: 'elapsed_s', label: 'Time (s)', type: 'num' }},
        {{ key: 'lr', label: 'LR', type: 'num' }},
    ];

    let sortKey = 'min_correct_digits';
    let sortDir = -1;
    let filterArch = 'all';
    let filterText = '';

    function render() {{
        let rows = all.slice();
        if (filterArch !== 'all') rows = rows.filter(r => r.arch === filterArch);
        if (filterText) {{
            const q = filterText.toLowerCase();
            rows = rows.filter(r => r.config_label.toLowerCase().includes(q) || r.arch.toLowerCase().includes(q));
        }}
        rows.sort((a, b) => {{
            const va = a[sortKey], vb = b[sortKey];
            if (typeof va === 'string') return sortDir * va.localeCompare(vb);
            return sortDir * (va - vb);
        }});

        const tbody = document.getElementById('table-body');
        tbody.innerHTML = rows.map(r => {{
            const color = COLORS[r.arch] || '#888';
            return `<tr>
                <td><span class="arch-badge" style="background:${{color}}22;color:${{color}};border:1px solid ${{color}}44">${{r.arch}}</span></td>
                <td>${{r.config_label}}</td>
                <td class="num">${{r.n_epochs}}</td>
                <td class="num">${{r.n_params_total.toLocaleString()}}</td>
                <td class="num">${{r.min_loss.toExponential(3)}}</td>
                <td class="num">${{r.mean_rel_err.toExponential(3)}}</td>
                <td class="num"><strong>${{r.min_correct_digits}}</strong></td>
                <td class="num">${{r.mean_correct_digits.toFixed(2)}}</td>
                <td class="num">${{r.elapsed_s.toFixed(1)}}</td>
                <td class="num">${{r.lr.toExponential(1)}}</td>
            </tr>`;
        }}).join('');
        document.getElementById('table-count').textContent = `${{rows.length}} / ${{all.length}} runs`;
    }}

    // Header sorting
    document.querySelectorAll('#results-table th').forEach((th, i) => {{
        th.addEventListener('click', () => {{
            const key = cols[i].key;
            if (sortKey === key) sortDir *= -1;
            else {{ sortKey = key; sortDir = cols[i].type === 'str' ? 1 : -1; }}
            render();
        }});
    }});

    // Filters
    document.getElementById('filter-arch').addEventListener('change', e => {{ filterArch = e.target.value; render(); }});
    document.getElementById('filter-text').addEventListener('input', e => {{ filterText = e.target.value; render(); }});

    render();
}}

// ── Init ─────────────────────────────────────────────────────────────────
document.addEventListener('DOMContentLoaded', () => {{
    renderPlots();
    initTabs();
    buildTable();
}});
"""


def build_table_section() -> str:
    return """
<div class="filters">
    <label>Architecture:</label>
    <select id="filter-arch">
        <option value="all">All</option>
        <option value="SIREN">SIREN</option>
        <option value="WIRE">WIRE</option>
        <option value="KAN">KAN</option>
    </select>
    <label>Search:</label>
    <input type="text" id="filter-text" placeholder="Filter by config...">
    <span id="table-count" style="margin-left:auto;color:#8b949e;font-size:0.85em;"></span>
</div>
<div class="table-wrap">
<table id="results-table">
    <thead><tr>
        <th>Arch</th><th>Config</th><th>Epochs</th><th>Params</th>
        <th>Min Loss</th><th>Mean Rel Err</th><th>Min Dig</th>
        <th>Mean Dig</th><th>Time (s)</th><th>LR</th>
    </tr></thead>
    <tbody id="table-body"></tbody>
</table>
</div>
"""


def build_html(data: dict[str, list[dict]]) -> str:
    tabs = [
        ("dist", "Distributions"),
        ("scatter", "Scatter Plots"),
        ("compare", "Comparison"),
        ("table", "Data Table"),
    ]

    panels = []
    # Distributions panel
    dist_plots = ["hist-digits", "hist-relerr", "hist-loss", "hist-time", "hist-params"]
    dist_html = '<div class="plot-grid">'
    for i, pid in enumerate(dist_plots):
        cls = "plot-box plot-full" if i == 0 else "plot-box"
        dist_html += f'<div class="{cls}" id="{pid}"></div>'
    dist_html += "</div>"
    panels.append(f'<div class="panel active" id="panel-dist">{dist_html}</div>')

    # Scatter panel
    scatter_plots = [
        "scatter-epochs-digits", "scatter-params-digits",
        "scatter-loss-digits", "scatter-time-digits",
        "scatter-epochs-loss", "scatter-params-loss",
    ]
    scatter_html = '<div class="plot-grid">'
    for pid in scatter_plots:
        scatter_html += f'<div class="plot-box" id="{pid}"></div>'
    scatter_html += "</div>"
    panels.append(f'<div class="panel" id="panel-scatter">{scatter_html}</div>')

    # Comparison panel
    compare_plots = ["box-digits", "box-relerr", "box-loss", "box-time", "box-params"]
    compare_html = '<div class="plot-grid">'
    for i, pid in enumerate(compare_plots):
        cls = "plot-box plot-full" if i == 0 else "plot-box"
        compare_html += f'<div class="{cls}" id="{pid}"></div>'
    compare_html += "</div>"
    panels.append(f'<div class="panel" id="panel-compare">{compare_html}</div>')

    # Table panel
    panels.append(f'<div class="panel" id="panel-table">{build_table_section()}</div>')

    summary = build_summary_cards(data)
    tab_bar = build_tab_bar(tabs)
    plots_js = build_plots_js(data)

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Skuld Sweep m2 — Interactive Results</title>
<script src="https://cdn.plot.ly/plotly-2.32.0.min.js"></script>
<style>{CSS}</style>
</head>
<body>
<div class="container">
    <h1>Skuld Sweep m2 — Grid Search Results</h1>
    <p class="subtitle">SIREN vs WIRE vs KAN · Maître et al. method · 8 parameter sets · physics integrand</p>

    {summary}
    {tab_bar}
    {"".join(panels)}
</div>
<script>{plots_js}</script>
</body>
</html>"""


def main():
    data = load_all()
    if not data:
        print("No CSV files found. Run the sweep experiments first.")
        return
    html = build_html(data)
    OUTPUT_PATH.write_text(html, encoding="utf-8")
    print(f"Dashboard written to {OUTPUT_PATH}")
    print(f"  Architectures: {list(data.keys())}")
    print(f"  Total runs: {sum(len(v) for v in data.values())}")


if __name__ == "__main__":
    main()
