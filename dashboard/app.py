"""BMMM dashboard: an interactive view of the Bayesian Marketing Mix Model.

Reads the compact artifact written by ``bmmm export-dashboard`` (no PyMC, no
92MB model) and renders headline metrics, an interactive budget planner and the
pre-built figures.
"""

from __future__ import annotations

import plotly.graph_objects as go
import streamlit as st
from core import ASSETS, DashboardData, interp_profit, load_data

st.set_page_config(page_title="BMMM Dashboard", page_icon="📈", layout="wide")

DATA = load_data()
IMG = ASSETS / "img"
GREEN, RED, BLUE, GREY = "#2ca02c", "#d62728", "#1f77b4", "#9aa0a6"


def img(name: str) -> str:
    return str(IMG / f"{name}.png")


# --------------------------------------------------------------------------- #
# Header + headline metrics
# --------------------------------------------------------------------------- #
st.title("📈 Bayesian Marketing Mix Modeling")
st.caption(
    "How much did each channel drive sales, and how should the budget be split? "
    "Fitted on 3 years of weekly data, with full uncertainty."
)

m = DATA.metrics
c1, c2, c3, c4 = st.columns(4)
c1.metric("Model fit (R²)", f"{m['r2']:.2f}")
c2.metric("Error (MAPE)", f"{m['mape'] * 100:.1f}%")
c3.metric("Channels", m["n_channels"])
c4.metric(
    "Sampler health",
    "OK" if m["num_divergences"] == 0 and m["max_r_hat"] < 1.05 else "check",
    help=f"max R-hat {m['max_r_hat']}, {m['num_divergences']} divergences",
)

st.divider()


# --------------------------------------------------------------------------- #
# Interactive: scenario planner
# --------------------------------------------------------------------------- #
def scenario_planner(data: DashboardData) -> None:
    st.subheader("🎛 Scenario planner")
    st.write(
        "Set a total weekly budget and how it splits across channels. Each green bar "
        "is the sales that channel drives; the red line is what it costs. Bar above "
        "the line means the channel earns more than it spends."
    )

    cur_spend = sum(ch.current_spend for ch in data.channels)
    cur_total = sum(float(ch.response(ch.current_spend)) for ch in data.channels)

    left, right = st.columns([1, 2])
    with left:
        budget = st.slider("Total weekly budget", min_value=0.0,
                           max_value=round(2.5 * cur_spend, -1),
                           value=float(round(cur_spend, -1)), step=50.0)
        st.caption("Channel split (normalised to 100%)")
        weights = {}
        for ch in data.channels:
            default = round(100 * ch.current_spend / cur_spend)
            weights[ch.name] = st.slider(f"{ch.label}  %", 0, 100, default, key=f"w_{ch.name}")

    wsum = sum(weights.values()) or 1
    shares = {k: v / wsum for k, v in weights.items()}
    spend = [budget * shares[ch.name] for ch in data.channels]
    sales = [float(ch.response(s)) for ch, s in zip(data.channels, spend, strict=True)]
    new_total, new_spend = sum(sales), sum(spend)

    with right:
        labels = [f"{ch.label}<br>{shares[ch.name] * 100:.0f}%" for ch in data.channels]
        fig = go.Figure()
        fig.add_bar(name="sales driven", x=labels, y=sales, marker_color=GREEN)
        fig.add_scatter(
            name="spend", x=labels, y=spend, mode="markers",
            marker={"symbol": "line-ew", "size": 60, "line": {"color": RED, "width": 4}},
        )
        fig.update_layout(
            height=360, margin={"t": 10, "b": 10, "l": 10, "r": 10},
            yaxis_title="weekly sales vs spend", legend={"orientation": "h", "y": 1.12},
        )
        st.plotly_chart(fig, use_container_width=True)

    k1, k2, k3 = st.columns(3)
    k1.metric("Weekly sales from ads", f"{new_total:,.0f}",
              delta=f"{new_total - cur_total:,.0f} vs current")
    k2.metric("Weekly spend", f"{new_spend:,.0f}",
              delta=f"{new_spend - cur_spend:,.0f} vs current")
    k3.metric("Net (sales - spend)", f"{new_total - new_spend:,.0f}",
              delta=f"{(new_total - new_spend) - (cur_total - cur_spend):,.0f} vs current")


# --------------------------------------------------------------------------- #
# Interactive: optimal budget
# --------------------------------------------------------------------------- #
def budget_optimizer(data: DashboardData) -> None:
    st.subheader("💰 Optimal total budget")
    st.write(
        "Pick a total weekly budget. The bars show the profit-maximising split, "
        "and the chart shows where profit peaks (marginal ROAS = 1)."
    )

    pc = data.profit_curve
    b_cur, b_star = data.budget["current"], data.budget["profit_max"]
    hi = float(pc["budget"][-1])
    chosen = st.slider("Total weekly budget", min_value=0.0, max_value=round(hi, -1),
                       value=0.0, step=10.0)

    point = interp_profit(pc, chosen)
    left, right = st.columns(2)

    with left:
        labels = [ch.label for ch in data.channels]
        fig = go.Figure()
        fig.add_bar(name="current", x=labels, y=[ch.current_spend for ch in data.channels],
                    marker_color=GREY)
        fig.add_bar(name="optimal", x=labels,
                    y=[point["allocation"][ch.name] for ch in data.channels], marker_color=GREEN)
        fig.update_layout(barmode="group", height=320, margin={"t": 10, "b": 10, "l": 10, "r": 10},
                          yaxis_title="weekly spend", legend={"orientation": "h", "y": 1.1})
        st.plotly_chart(fig, use_container_width=True)
        st.metric("Profit at this budget", f"{point['profit']:,.0f}",
                  delta=f"marginal ROAS {point['marginal_roas']:.2f}")

    with right:
        fig = go.Figure()
        fig.add_scatter(x=pc["budget"], y=pc["profit"], name="profit", line_color=GREEN)
        fig.add_scatter(x=pc["budget"], y=pc["ad_sales"], name="ad-driven sales", line_color=BLUE)
        fig.add_vline(x=b_cur, line_dash="dot", line_color=GREY, annotation_text="current")
        fig.add_vline(x=b_star, line_dash="dash", line_color=RED, annotation_text="profit-max")
        fig.add_vline(x=chosen, line_color="#444", annotation_text="you")
        fig.update_layout(height=320, margin={"t": 10, "b": 10, "l": 10, "r": 10},
                          xaxis_title="total weekly budget", yaxis_title="weekly sales / profit",
                          legend={"orientation": "h", "y": 1.1})
        st.plotly_chart(fig, use_container_width=True)


# --------------------------------------------------------------------------- #
# Tabs
# --------------------------------------------------------------------------- #
tabs = st.tabs(
    ["🎛 Scenario planner", "💰 Optimal total budget", "Parameter recovery", "Model fit",
     "Contributions", "Trend & seasonality", "ROAS & budget"]
)

with tabs[0]:
    scenario_planner(DATA)

with tabs[1]:
    budget_optimizer(DATA)

with tabs[2]:
    st.subheader("Does the model recover the truth?")
    st.write(
        "The data is synthetic, so the true adstock of each channel is known (red). "
        "**All three true values land inside the model's 94% interval**, so the "
        "estimates can be trusted. `Web Search` is the least certain, which is honest: "
        "its short carry-over and overlap with `Social Networks` leave less to learn from."
    )
    st.image(img("recovery"), width=640)

with tabs[3]:
    st.subheader("Predicted vs actual sales")
    st.write(
        "**The model explains 93% of weekly sales (R² 0.93, MAPE 2.6%).** The posterior "
        "mean tracks the actuals across all three years, and the real series stays inside "
        "the credible band."
    )
    st.image(img("posterior_predictive"), width=820)

with tabs[4]:
    st.subheader("Where do sales come from?")
    st.write(
        "**About half of sales is baseline** (organic demand that runs without ads); "
        "advertising drives the rest. Among channels, **Social Networks contributes the "
        "most (~49%)** and Web Search the least (~19%)."
    )
    st.image(img("contributions"), width=820)
    st.image(img("contributions_share"), width=820)

with tabs[5]:
    st.subheader("Recovered trend and yearly seasonality")
    st.write(
        "Both baseline components come back cleanly: a **steady upward trend** and a "
        "**strong yearly cycle** that repeats with the same shape each year. The tight "
        "bands mean they are well identified, not guesswork."
    )
    st.image(img("trend_seasonality"), width=760)

with tabs[6]:
    st.subheader("Efficiency and budget")
    st.write(
        "**Social Networks is the most efficient channel, TV the least.** At current "
        "spend TV's marginal ROAS is already below 1 (the next dollar loses money), and "
        "**the total budget sits past its profit-maximising point (~1090 vs ~1310)** — "
        "the takeaway is to trim the total and shift spend out of TV."
    )
    col1, col2 = st.columns(2)
    col1.image(img("roas"), width=460)
    col2.image(img("marginal_roas"), width=460)
    st.image(img("budget_profit"), width=620)

st.divider()
st.caption("Built with PyMC-Marketing, ArviZ and Streamlit. The model is trained offline; "
           "this app reads a small precomputed artifact.")
