"""Build the headline chart for the RAG eval: recall@10 by retriever,
original vs degraded queries, faceted by question type.

Numbers are the measured results (see METHODOLOGY.md). Hardcoded here so the
chart is reproducible without re-running the cloud eval; update if you re-run.

Outputs:
  report/recall_by_type.html  (interactive)
  report/recall_by_type.png   (for slides; needs kaleido)

Run:  .venv/bin/python scripts/make_report_chart.py
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

OUT = Path(__file__).resolve().parents[1] / "report"
OUT.mkdir(exist_ok=True)

# recall@10 by question type, per retriever, per query-set.
DATA = {
    "original": {
        "comparison": {"BM25": 0.913, "Dense": 0.828, "Hybrid": 0.898},
        "inference":  {"BM25": 0.783, "Dense": 0.744, "Hybrid": 0.808},
        "temporal":   {"BM25": 0.888, "Dense": 0.839, "Hybrid": 0.882},
    },
    "degraded": {
        "comparison": {"BM25": 0.832, "Dense": 0.778, "Hybrid": 0.847},
        "inference":  {"BM25": 0.658, "Dense": 0.716, "Hybrid": 0.751},
        "temporal":   {"BM25": 0.789, "Dense": 0.782, "Hybrid": 0.821},
    },
}
RETRIEVERS = ["BM25", "Dense", "Hybrid"]
COLORS = {"BM25": "#4e79a7", "Dense": "#59a14f", "Hybrid": "#003057"}  # Spectrum-ish
TYPES = ["comparison", "inference", "temporal"]


def main() -> None:
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=("Native (benchmark) queries", "Degraded (realistic) queries"),
        shared_yaxes=True,
    )
    for col, qset in enumerate(["original", "degraded"], start=1):
        for r in RETRIEVERS:
            fig.add_trace(
                go.Bar(
                    name=r, legendgroup=r, showlegend=(col == 1),
                    x=TYPES, y=[DATA[qset][t][r] for t in TYPES],
                    marker_color=COLORS[r],
                    text=[f"{DATA[qset][t][r]:.2f}" for t in TYPES],
                    textposition="outside",
                ),
                row=1, col=col,
            )
    fig.update_yaxes(title_text="recall@10", range=[0, 1.0], row=1, col=1)
    fig.update_layout(
        barmode="group",
        title_text="Retrieval recall@10 by query type — BM25 vs Dense vs Hybrid<br>"
                   "<sup>BM25 collapses on realistic queries; Hybrid is most robust (Mobile Insights – Advanced Analytics)</sup>",
        legend_title_text="Retriever",
        template="plotly_white",
        height=500, width=950,
        font=dict(size=13),
    )
    html = OUT / "recall_by_type.html"
    fig.write_html(html)
    print("wrote", html)
    try:
        png = OUT / "recall_by_type.png"
        fig.write_image(png, scale=2)
        print("wrote", png)
    except Exception as e:
        print("PNG export skipped (install kaleido for static image):", str(e)[:80])

    # Also emit the overall original-vs-degraded summary as a small table/csv.
    overall = pd.DataFrame({
        "BM25":   {"original": 0.859, "degraded": 0.758},
        "Dense":  {"original": 0.801, "degraded": 0.757},
        "Hybrid": {"original": 0.862, "degraded": 0.806},
    })
    overall.to_csv(OUT / "recall10_overall.csv")
    print("wrote", OUT / "recall10_overall.csv")
    print("\noverall recall@10:\n", overall.to_string())

    make_degradation_impact_chart()


def make_degradation_impact_chart() -> None:
    """Single chart: how much recall@10 each retriever LOST going from native to
    degraded (realistic) queries. Bigger bar = more fragile. BM25 towers."""
    # recall@10 drop (original - degraded): overall + per question type.
    drop = {
        "Overall":    {"BM25": 0.859 - 0.758, "Dense": 0.801 - 0.757, "Hybrid": 0.862 - 0.806},
        "comparison": {"BM25": 0.913 - 0.832, "Dense": 0.828 - 0.778, "Hybrid": 0.898 - 0.847},
        "inference":  {"BM25": 0.783 - 0.658, "Dense": 0.744 - 0.716, "Hybrid": 0.808 - 0.751},
        "temporal":   {"BM25": 0.888 - 0.789, "Dense": 0.839 - 0.782, "Hybrid": 0.882 - 0.821},
    }
    groups = ["Overall", "comparison", "inference", "temporal"]
    fig = go.Figure()
    for r in RETRIEVERS:
        fig.add_trace(go.Bar(
            name=r, x=groups, y=[round(drop[g][r], 3) for g in groups],
            marker_color=COLORS[r],
            text=[f"-{drop[g][r]:.3f}" for g in groups], textposition="outside",
        ))
    fig.update_yaxes(title_text="recall@10 lost (original − degraded)", rangemode="tozero")
    fig.update_layout(
        barmode="group",
        title_text="Degradation impact: recall@10 lost on realistic queries<br>"
                   "<sup>Taller = more fragile. BM25 loses the most everywhere; Dense the least; "
                   "inference queries hit BM25 hardest (Mobile Insights – Advanced Analytics)</sup>",
        legend_title_text="Retriever",
        template="plotly_white", height=500, width=900, font=dict(size=13),
    )
    html = OUT / "degradation_impact.html"; fig.write_html(html); print("wrote", html)
    try:
        png = OUT / "degradation_impact.png"; fig.write_image(png, scale=2); print("wrote", png)
    except Exception as e:
        print("PNG export skipped:", str(e)[:80])


if __name__ == "__main__":
    main()
