"""
Live dashboard for the evaluator. It watches the JSONL stream and meta.json and
updates automatically (1s interval). Beautiful Bootstrap theme + Plotly charts.

Run:
    python dash_app.py

Note:
  - Add your favicon/logo where indicated in the layout (ICON PLACEHOLDER).
"""

import json
import os
from collections import defaultdict, Counter

import pandas as pd
import numpy as np
from dash import Dash, html, dcc, Input, Output, no_update, dash_table
import dash_bootstrap_components as dbc
import plotly.express as px
import plotly.graph_objects as go

OUTPUT_DIR = "./eval_out"
JSONL = os.path.join(OUTPUT_DIR, "runs.jsonl")
META = os.path.join(OUTPUT_DIR, "meta.json")

# ---- Data loading helpers ----

def load_meta():
    try:
        with open(META, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}

def read_jsonl() -> pd.DataFrame:
    if not os.path.exists(JSONL) or os.path.getsize(JSONL) == 0:
        return pd.DataFrame()
    rows = []
    with open(JSONL, "r", encoding="utf-8") as f:
        for line in f:
            try:
                rows.append(json.loads(line))
            except Exception:
                continue
    return pd.DataFrame(rows)

# ---- Layout ----

app = Dash(__name__, external_stylesheets=[dbc.themes.LUX])

app.title = "Agent Evaluation Dashboard"

controls = dbc.Card(
    [
        html.Div([
            # ICON PLACEHOLDER: add your <img src="path/to/logo.png" ...> below
            # e.g., html.Img(src="/assets/logo.png", style={"height":"40px"})
        ]),
        html.H5("Controls", className="mt-2"),
        dbc.Row([
            dbc.Col([
                html.Label("Field"),
                dcc.Dropdown(id="field-dd", placeholder="Select attribute/field")
            ], width=6),
            dbc.Col([
                html.Label("Metric"),
                dcc.Dropdown(
                    id="metric-dd",
                    options=[
                        {"label": "Accuracy (fixed/int)", "value": "accuracy"},
                        {"label": "F1 (fixed)", "value": "f1"},
                        {"label": "Jaccard (multi)", "value": "jaccard"},
                        {"label": "Cosine (free/open)", "value": "cosine"},
                        {"label": "ROUGE-L (free)", "value": "rougeL"},
                        {"label": "Consistency", "value": "consistency"},
                    ],
                    placeholder="Pick metric to plot"
                )
            ], width=6),
        ]),
        html.Br(),
        dbc.Alert(id="run-status", color="secondary", is_open=True),
    ],
    className="p-3 mb-3",
    style={"borderRadius": "1rem", "boxShadow": "0 10px 20px rgba(0,0,0,0.08)"},
)

summary_cards = dbc.Row([
    dbc.Col(dbc.Card(dbc.CardBody([html.H6("Samples"), html.H2(id="n-samples", className="text-primary")])), md=3),
    dbc.Col(dbc.Card(dbc.CardBody([html.H6("Runs/Sample"), html.H2(id="runs-per-sample", className="text-primary")])), md=3),
    dbc.Col(dbc.Card(dbc.CardBody([html.H6("Avg Correct Rate"), html.H2(id="avg-acc", className="text-success")])), md=3),
    dbc.Col(dbc.Card(dbc.CardBody([html.H6("Avg Consistency"), html.H2(id="avg-cons", className="text-success")])), md=3),
], className="g-3")

graphs = dbc.Row([
    dbc.Col(dcc.Graph(id="metric-graph"), md=8),
    dbc.Col(dcc.Graph(id="confusion-graph"), md=4),
])

table = dbc.Card(
    [
        html.H5("Per-sample details"),
        dash_table.DataTable(
            id="detail-table",
            columns=[
                {"name": "sample_id", "id": "sample_id"},
                {"name": "field", "id": "field"},
                {"name": "kind", "id": "kind"},
                {"name": "correct_rate", "id": "correct_rate", "type":"numeric", "format": {"specifier": ".3f"}},
                {"name": "consistency", "id": "consistency", "type":"numeric", "format": {"specifier": ".3f"}},
                {"name": "stable_and_correct", "id": "stable_and_correct"},
            ],
            page_size=10,
            style_table={"overflowX":"auto"},
            style_cell={"fontFamily":"Inter, system-ui, -apple-system, Segoe UI, Roboto", "fontSize":"14px"},
            style_header={"fontWeight":"700"},
            filter_action="native", sort_action="native"
        )
    ],
    className="p-3",
    style={"borderRadius": "1rem", "boxShadow": "0 10px 20px rgba(0,0,0,0.08)"}
)

app.layout = dbc.Container(
    [
        html.H1("Agent Evaluation Dashboard", className="mt-3 mb-2"),
        html.P("Live results stream in as evaluations run. Choose a field and metric to explore."),
        controls,
        summary_cards,
        graphs,
        html.Br(),
        table,
        dcc.Interval(id="ticker", interval=1000, n_intervals=0),  # 1s refresh
    ],
    fluid=True
)

# ---- Callbacks ----

@app.callback(
    Output("run-status", "children"),
    Output("run-status", "color"),
    Output("field-dd", "options"),
    Output("n-samples", "children"),
    Output("runs-per-sample", "children"),
    Output("avg-acc", "children"),
    Output("avg-cons", "children"),
    Input("ticker", "n_intervals")
)
def refresh_summary(_):
    meta = load_meta()
    df = read_jsonl()

    status = meta.get("status", "waiting")
    color = "secondary"
    if status == "running":
        color = "warning"
    elif status == "completed":
        color = "success"

    if df.empty:
        return f"Status: {status}", color, [], "0", "0", "0.00", "0.00"

    # Fields from summary rows: rely on the existence of the 'stable_and_correct' column,
    # not the nested 'details' dict (which may be NaN).
    if "stable_and_correct" in df.columns:
        summ_rows = df[df["stable_and_correct"].notnull()]
        fields_list = sorted(summ_rows["field"].dropna().unique().tolist())
    else:
        fields_list = []

    # Count samples using the summary rows only
    n_samples = int(summ_rows["sample_id"].nunique()) if fields_list else 0

    # Runs/sample: infer from per-run rows (those with a metrics dict)
    if "metrics" in df.columns:
        run_rows = df[df["metrics"].notnull()]
        if not run_rows.empty:
            g = run_rows.groupby(["sample_id", "field"]).size().reset_index(name="n")
            runs_per = int(g["n"].median())
        else:
            runs_per = 0
    else:
        run_rows = pd.DataFrame()
        runs_per = 0

    # Averages across summary rows
    if fields_list:
        avg_acc = float(summ_rows["correct_rate"].astype(float).mean())
        avg_cons = float(summ_rows["consistency"].astype(float).mean())
    else:
        avg_acc, avg_cons = 0.0, 0.0

    field_options = [{"label": f, "value": f} for f in fields_list]
    return f"Status: {status}", color, field_options, str(n_samples), str(runs_per), f"{avg_acc:.2f}", f"{avg_cons:.2f}"

@app.callback(
    Output("metric-graph", "figure"),
    Output("confusion-graph", "figure"),
    Output("detail-table", "data"),
    Input("ticker", "n_intervals"),
    Input("field-dd", "value"),
    Input("metric-dd", "value"),
)
def refresh_plots(_, field, metric):
    df = read_jsonl()
    if df.empty or field is None:
        return go.Figure(), go.Figure(), []

    # Detail table from summary rows
    if "stable_and_correct" in df.columns:
        summ = df[(df["field"] == field) & (df["stable_and_correct"].notnull())]
    else:
        summ = pd.DataFrame()
    detail_data = summ[["sample_id", "field", "kind", "correct_rate", "consistency", "stable_and_correct"]].to_dict("records") if not summ.empty else []

    # Metric plot
    metric_fig = go.Figure()
    if metric in {"accuracy", "consistency"} and not summ.empty:
        y = "correct_rate" if metric == "accuracy" else "consistency"
        metric_fig = px.bar(summ, x="sample_id", y=y, color="kind", title=("Correctness over runs" if metric=="accuracy" else "Consistency over runs"))
        metric_fig.update_layout(margin=dict(l=10, r=10, t=50, b=10))
    elif metric in {"jaccard", "f1", "cosine", "rougeL"}:
        # Per-run rows for the field
        runs = df[(df["field"] == field) & (df["metrics"].notnull())] if "metrics" in df.columns else pd.DataFrame()
        vals = []
        for _, r in runs.iterrows():
            m = r["metrics"]
            # Coerce metrics into a dict
            if not isinstance(m, dict):
                try:
                    m = json.loads(m) if isinstance(m, str) else {}
                except Exception:
                    m = {}
            if metric == "cosine":
                v = m.get("cosine")
            elif metric == "rougeL":
                v = m.get("rougeL")
            elif metric == "jaccard":
                v = m.get("jaccard")
            else:  # "f1"
                v = m.get("f1")
            if v is not None:
                vals.append({"sample_id": r["sample_id"], "val": float(v)})
        if vals:
            d = pd.DataFrame(vals).groupby("sample_id")["val"].mean().reset_index()
            metric_fig = px.bar(d, x="sample_id", y="val", title=f"{metric} (avg across runs)")

    # Confusion matrix (approx) for categorical/int single-label
    runs = df[(df["field"] == field) & (df["metrics"].notnull())] if "metrics" in df.columns else pd.DataFrame()
    labels = set()
    cm = {}
    for _, r in runs.iterrows():
        exp = r.get("expected")
        pred = r.get("predicted")
        # Skip rows without actual predictions (e.g., summary rows)
        if pd.isna(exp) or pd.isna(pred):
            continue
        g = str(exp)
        p = str(pred)
        labels.add(g); labels.add(p)
        cm[(g, p)] = cm.get((g, p), 0) + 1
    labels = sorted(labels)
    if labels:
        mat = np.zeros((len(labels), len(labels)), dtype=int)
        for (g, p), c in cm.items():
            gi = labels.index(g); pi = labels.index(p)
            mat[gi, pi] = c
        conf_fig = px.imshow(
            mat, x=labels, y=labels, text_auto=True, color_continuous_scale="Blues",
            labels=dict(x="Predicted", y="Gold", color="Count"),
            title="Confusion Matrix (approx.)"
        )
    else:
        conf_fig = go.Figure()

    return metric_fig, conf_fig, detail_data

if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    app.run(debug=True, port=8050)
 