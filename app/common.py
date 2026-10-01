"""Shared helpers for the DriftGuard dashboard. Everything is read from the committed artifacts/ folder."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

ARTIFACTS = Path(__file__).resolve().parents[1] / "artifacts"
SHUTDOWN_STEP = 43

# Categorical slots validated for CVD separation and >=3:1 contrast on the navy surface.
COLORS = {
    "Static": "#9aa3b5",
    "DriftGuard (pre-registered)": "#3987e5",
    "DriftGuard v2 (post-hoc)": "#d95926",
    "Full retrain (oracle)": "#199e70",
    "extra": "#9085e9",
}
ALARM = "#e66767"
INK, MUTED, GRID = "#e6edf7", "#9aa3b5", "#22314f"

POLICY_LABELS = {
    "static": "Static (no labels)",
    "drift_triggered": "Drift-triggered (random audit)",
    "drift_triggered_v2": "Drift-triggered v2 (uncertainty audit, post-hoc)",
    "always": "Always query",
    "full_retrain": "Full retrain (oracle)",
}
WINDOW_LABELS = {"pre_shutdown": "Steps 35–42 (before)", "post_shutdown": "Steps 43–49 (after)",
                 "all_test": "Steps 35–49 (all)"}
MODEL_LABELS = {"trivial": "Trivial (prior)", "logreg": "Logistic Regression", "rf": "Random Forest",
                "xgb": "XGBoost"}

DISCLAIMER = (
    "**Research prototype.** Not a financial, compliance, or AML product, and not financial advice. "
    "No real funds or live systems are involved. All numbers are produced by the pipeline and read from "
    "`artifacts/results.json`."
)


def page_header() -> None:
    """Sidebar disclaimer shown on every page (page config and navigation live in app.py)."""
    with st.sidebar:
        st.caption("Drift-aware AML monitoring on the Elliptic Bitcoin dataset.")
        st.info(DISCLAIMER)


@st.cache_data
def results() -> dict:
    return json.loads((ARTIFACTS / "results.json").read_text())


@st.cache_data
def parquet(name: str) -> pd.DataFrame:
    return pd.read_parquet(ARTIFACTS / f"{name}.parquet")


def grid() -> pd.DataFrame:
    return pd.DataFrame(results()["adaptive_grid"])


def cfg_rows(df: pd.DataFrame, policy: str, strategy: str, k: int) -> pd.DataFrame:
    return df[(df["policy"] == policy) & (df["strategy"] == strategy) & (df["k"] == k)]


def headline_configs() -> dict[str, dict]:
    cfg = results()["config"]
    return {
        "Static": {"policy": "static", "strategy": "none", "k": 0},
        "DriftGuard (pre-registered)": cfg["headline_config"],
        "DriftGuard v2 (post-hoc)": cfg["headline_v2_config"],
        "Full retrain (oracle)": {"policy": "full_retrain", "strategy": "none", "k": 0},
    }


def pm(mean: float, std: float, fmt: str = ".3f") -> str:
    return f"{mean:{fmt}} ± {std:{fmt}}"


def style(fig: go.Figure, height: int = 380, **kwargs) -> go.Figure:
    layout = dict(
        height=height, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color=INK), margin=dict(l=10, r=10, t=40, b=10), hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0, bgcolor="rgba(0,0,0,0)"),
    )
    fig.update_layout(**{**layout, **kwargs})
    fig.update_xaxes(gridcolor=GRID, zeroline=False, color=MUTED)
    fig.update_yaxes(gridcolor=GRID, zeroline=False, color=MUTED)
    return fig


def shutdown_marker(fig: go.Figure, row: int | None = None) -> None:
    kw = {"row": row, "col": 1} if row else {}
    fig.add_vline(x=SHUTDOWN_STEP - 0.5, line_dash="dash", line_color=MUTED,
                  annotation_text="dark-market shutdown", annotation_font_color=MUTED,
                  annotation_position="top right", **kw)


def citation() -> None:
    st.caption(
        "Data: Elliptic Data Set, Weber et al. (2019), *Anti-Money Laundering in Bitcoin: Experimenting with Graph "
        "Convolutional Networks for Financial Forensics*. Licence CC BY-NC-ND 4.0. Not redistributed here; only "
        "derived metrics are shown."
    )
