"""
Streamlit demo app for distribution fitting and risk adjustment.

Run with: streamlit run app.py
"""

from collections import Counter
import math as _math

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from distributions import DISTRIBUTIONS, compare_fits, fit_all
from examples import ALL_EXAMPLES
from risk_analysis import (
    RiskParams, analyze, analyze_all,
    compute_dmreu, compute_wlu, compute_ambiguity_aversion,
    ev_eu_percentile_table, ev_eu_percentile_table_all,
    FormalModelRun, FORMAL_MODEL_TYPES,
    compute_formal_run, compute_formal_runs_all,
)

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Distribution Fitting & Risk Analysis",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("Cost-Effectiveness Distribution Fitting")
st.markdown(
    "Fit parametric distributions to expert-elicited percentile data, "
    "then explore how different risk preferences affect expected value."
)

# ---------------------------------------------------------------------------
# Sidebar: Input data (inside a form so changes require explicit submit)
# ---------------------------------------------------------------------------

has_duplicate_percentiles = False
duplicate_percentiles = []

with st.sidebar:
    fit_clicked = st.button("Fit distributions", type="primary", width="stretch")

    with st.expander("Input Data", expanded=True):
        st.caption("Does not automatically refresh — click **Fit distributions** to recalculate with changes")

        example_names = ["Custom"] + [ex["name"] for ex in ALL_EXAMPLES]
        example_choice = st.selectbox("Load example", example_names)

        if example_choice != "Custom":
            selected_example = next(ex for ex in ALL_EXAMPLES if ex["name"] == example_choice)
            st.caption(selected_example["description"])
            default_pcts = selected_example["percentiles"]
        else:
            default_pcts = {0.10: -10.0, 0.50: 5.0, 0.90: 30.0}

        if "pct_rows" not in st.session_state or example_choice != st.session_state.get("last_example"):
            st.session_state.pct_rows = [
                {"q": q, "v": v} for q, v in sorted(default_pcts.items())
            ]
            st.session_state.last_example = example_choice
            st.session_state.pct_version = st.session_state.get("pct_version", 0) + 1
            st.rerun()

        pv = st.session_state.get("pct_version", 0)

        with st.form("pct_form"):
            st.subheader("Percentile constraints")

            pct_rows_input = []
            for i, row in enumerate(st.session_state.pct_rows):
                cols = st.columns([2, 3])
                with cols[0]:
                    new_q = st.number_input(
                        "Percentile", value=row["q"], min_value=0.01, max_value=0.99,
                        step=0.05, format="%.2f", key=f"q_{pv}_{i}",
                    )
                with cols[1]:
                    new_v = st.number_input(
                        "Value", value=row["v"], step=1.0, format="%.2f", key=f"v_{pv}_{i}",
                    )
                pct_rows_input.append({"q": new_q, "v": new_v})

            pct_submitted = st.form_submit_button("Update percentiles", width="stretch")

        col_add, col_rm = st.columns(2)
        with col_add:
            if st.button("+ Add row"):
                st.session_state.pct_rows.append({"q": 0.50, "v": 0.0})
                st.rerun()
        with col_rm:
            if len(st.session_state.pct_rows) > 2 and st.button("- Remove last"):
                st.session_state.pct_rows.pop()
                st.rerun()

        if pct_submitted:
            st.session_state.pct_rows = pct_rows_input

    with st.expander("Distributions to fit", expanded=True):
        has_nonpositive_values = any(row["v"] <= 0 for row in st.session_state.pct_rows)
        positive_only_names = {
            name for name, cfg in DISTRIBUTIONS.items() if getattr(cfg, "positive_only", False)
        }
        if has_nonpositive_values:
            st.caption(
                "Note: Positive-only distributions are disabled because one or more "
                "percentile values are non-positive."
            )
        dist_tooltips = {
            "normal": "Symmetric, thin-tailed. Good baseline but rarely realistic for cost-effectiveness.",
            "lognormal": "Right-skewed, positive values only. Natural default for cost-effectiveness ratios.",
            "skew_normal": "Like normal but allows asymmetry. Good when outcomes could be slightly negative.",
            "students_t": "Symmetric with heavy tails. Use when you expect more 'surprises' than a normal.",
            "gev": "Models extreme tail behaviour. Best when worst/best case outcomes matter most.",
            "log_students_t": "Heavy-tailed lognormal. For positive quantities where 1000x upside is plausible.",
        }
        dist_choices = {}
        for name in DISTRIBUTIONS:
            is_positive_only = name in positive_only_names
            disabled = has_nonpositive_values and is_positive_only
            default_value = False if disabled else True
            dist_choices[name] = st.checkbox(
                name.replace("_", " ").title(),
                value=default_value,
                key=f"dist_{name}",
                disabled=disabled,
                help=dist_tooltips.get(name, ""),
            )


# ---------------------------------------------------------------------------
# Build inputs from session state
# ---------------------------------------------------------------------------

percentiles = {row["q"]: row["v"] for row in st.session_state.pct_rows}
quantiles = [row["q"] for row in st.session_state.pct_rows]
duplicate_percentiles = sorted({q for q, count in Counter(quantiles).items() if count > 1})
has_duplicate_percentiles = len(duplicate_percentiles) > 0

selected_dists = [name for name, checked in dist_choices.items() if checked]

# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

if len(percentiles) < 2:
    st.error("Please specify at least 2 percentile constraints.")
    st.stop()

if has_duplicate_percentiles:
    st.error(
        "Duplicate percentiles detected: "
        + ", ".join(f"{int(q * 100)}%" for q in duplicate_percentiles)
        + ". Please make each percentile unique."
    )
    st.stop()

# Check monotonicity: values should increase with quantile
sorted_pcts = sorted(percentiles.items())
non_monotonic = [
    (q1, v1, q2, v2)
    for (q1, v1), (q2, v2) in zip(sorted_pcts, sorted_pcts[1:])
    if v2 < v1
]
if non_monotonic:
    q1, v1, q2, v2 = non_monotonic[0]
    st.warning(
        f"Non-monotonic percentiles: p{int(q1*100)}={v1:.2f} > p{int(q2*100)}={v2:.2f}. "
        "Values should generally increase with percentile. Fits may be poor."
    )

if not selected_dists:
    st.error("Please select at least one distribution to fit.")
    st.stop()

# ---------------------------------------------------------------------------
# Fitting (only on button click, cached in session state)
# ---------------------------------------------------------------------------

if fit_clicked or "fits" not in st.session_state:
    with st.spinner("Fitting distributions..."):
        st.session_state.fits = fit_all(percentiles, distributions=selected_dists)
        st.session_state.fit_percentiles = percentiles
        st.session_state.fit_dists = selected_dists

fits = st.session_state.get("fits", [])

if not fits:
    st.info("Configure percentile inputs in the sidebar and click **Fit distributions**.")
    st.stop()

# ---------------------------------------------------------------------------
# Risk & formal model config (read from session state; widgets live in tabs)
# ---------------------------------------------------------------------------

trunc_pct = st.session_state.get("trunc_pct", 0.99)
loss_lambda = st.session_state.get("loss_lambda", 2.5)
ref_point = st.session_state.get("ref_point", 0.0)
use_median_ref = st.session_state.get("use_median_ref", False)

risk_params = RiskParams(
    truncation_percentile=trunc_pct,
    loss_aversion_lambda=loss_lambda,
    reference_point=ref_point,
)

_FORMAL_PRESETS = {
    "WLU sweep": [
        {"model": "wlu", "param": 0.01},
        {"model": "wlu", "param": 0.05},
        {"model": "wlu", "param": 0.10},
    ],
    "One of each": [
        {"model": "dmreu", "param": 0.05},
        {"model": "wlu", "param": 0.05},
        {"model": "ambiguity", "param": 4.0},
    ],
    "Custom": None,
}
_MODEL_PARAM_CFG = {
    "dmreu": {"min": 0.01, "max": 0.10, "step": 0.01, "fmt": "%.2f", "default": 0.05},
    "wlu": {"min": 0.0, "max": 0.25, "step": 0.01, "fmt": "%.2f", "default": 0.05},
    "ambiguity": {"min": 0.0, "max": 8.0, "step": 0.5, "fmt": "%.1f", "default": 4.0},
}

if "formal_runs" not in st.session_state:
    st.session_state.formal_runs = [dict(r) for r in _FORMAL_PRESETS["WLU sweep"]]

formal_runs = [
    FormalModelRun(model=r["model"], param=r["param"], epsilon=r.get("epsilon", 0.0))
    for r in st.session_state.formal_runs
]
formal_run_labels = [r.label for r in formal_runs]

# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------

tab1, tab2, tab3, tab4 = st.tabs([
    "Fitted Distributions", "Risk Adjustments", "Formal Risk Models", "Explorer",
])

# ---- Tab 1: Fitted Distributions ----
with tab1:
    st.subheader("PDF Comparison")

    # Determine x range
    all_ppf_lo = [f.ppf(0.005) for f in fits]
    all_ppf_hi = [f.ppf(0.995) for f in fits]
    x_min = min(v for v in all_ppf_lo if np.isfinite(v))
    x_max = max(v for v in all_ppf_hi if np.isfinite(v))
    padding = (x_max - x_min) * 0.05
    x = np.linspace(x_min - padding, x_max + padding, 500)

    fig_pdf = go.Figure()
    for fit in fits:
        y = fit.pdf(x)
        y = np.where(np.isfinite(y), y, 0.0)
        fig_pdf.add_trace(go.Scatter(
            x=x, y=y, mode="lines",
            name=fit.name.replace("_", " ").title(),
        ))

    # Percentile markers
    for q, v in sorted(percentiles.items()):
        fig_pdf.add_vline(
            x=v, line_dash="dash", line_color="gray", opacity=0.5,
            annotation_text=f"p{int(q * 100)}={v:.1f}",
            annotation_position="top",
        )

    fig_pdf.update_layout(
        xaxis_title="Value",
        yaxis_title="Probability Density",
        height=450,
        margin=dict(t=30),
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )
    st.plotly_chart(fig_pdf, width="stretch")

    st.subheader("Fit Comparison")
    df_compare = compare_fits(fits)
    st.dataframe(
        df_compare.style.format({
            "fit_error": "{:.6f}",
            "mean": "{:.2f}",
            "median": "{:.2f}",
            "std": "{:.2f}",
            "skewness": "{:.3f}",
            "kurtosis": "{:.3f}",
        }).highlight_min(subset=["fit_error"], color="#d4edda"),
        width="stretch",
    )

# ---------------------------------------------------------------------------
# Compute formal model runs (shared across tabs)
# ---------------------------------------------------------------------------

df_formal = compute_formal_runs_all(fits, formal_runs)

# ---- Tab 2: Risk Adjustments ----
with tab2:
    st.subheader("Risk-Adjusted Expected Values")
    st.markdown(
        "Informal risk adjustments that modify the expected value to reflect "
        "cautious decision-making: ignoring implausibly good outcomes (*upside "
        "skepticism*), penalising potential harm (*downside protection*), or both "
        "(*combined*)."
    )

    with st.expander("Adjust parameters", expanded=False):
        _ra_cols = st.columns([2, 2, 2, 1])
        with _ra_cols[0]:
            trunc_pct = st.slider(
                "Truncation percentile (upside skepticism)",
                min_value=0.90, max_value=0.999, value=trunc_pct, step=0.001,
                format="%.3f", key="trunc_pct",
                help="Outcomes above this percentile are ignored. Lower = more skeptical of upside.",
            )
        with _ra_cols[1]:
            loss_lambda = st.slider(
                "Loss aversion multiplier",
                min_value=1.0, max_value=5.0, value=loss_lambda, step=0.1,
                key="loss_lambda",
                help="How much worse losses feel than equivalent gains. 1.0 = neutral, 2.5 = moderate.",
            )
        with _ra_cols[2]:
            ref_point = st.number_input(
                "Reference point",
                value=ref_point, step=1.0, key="ref_point",
                help="Outcomes below this value count as losses.",
            )
        with _ra_cols[3]:
            use_median_ref = st.checkbox(
                "Use fitted median as reference", value=use_median_ref,
                key="use_median_ref",
            )
    st.caption(
        f"Current settings: truncate at **p{int(trunc_pct * 100)}** · "
        f"loss multiplier **{loss_lambda:.1f}×** · "
        f"reference point **{ref_point:.1f}**"
        + (" (fitted median)" if use_median_ref else "")
    )

    risk_params = RiskParams(
        truncation_percentile=trunc_pct,
        loss_aversion_lambda=loss_lambda,
        reference_point=ref_point,
    )

    # Informal adjustments (median-ref recomputes per distribution)
    if use_median_ref:
        risk_rows = []
        for fit in fits:
            params_i = RiskParams(
                truncation_percentile=trunc_pct,
                loss_aversion_lambda=loss_lambda,
                reference_point=fit.median(),
            )
            result = analyze(fit, params_i)
            row = result.to_dict()
            row["fit_error"] = float(fit.error)
            row["reference_point"] = float(fit.median())
            risk_rows.append(row)
        df_risk = pd.DataFrame(risk_rows)
    else:
        df_risk = analyze_all(fits, risk_params)

    # Merge formal run columns into the informal table
    informal_cols = ["distribution", "risk_neutral_ev", "upside_skepticism_ev",
                     "downside_protection_eu", "combined_eu", "fit_error"]
    if "reference_point" in df_risk.columns:
        informal_cols.insert(-1, "reference_point")
    df_merged = df_risk[informal_cols].merge(
        df_formal.drop(columns=["risk_neutral_ev", "fit_error"]),
        on="distribution",
    )

    ev_cols = ["risk_neutral_ev", "upside_skepticism_ev", "downside_protection_eu", "combined_eu"]
    display_df = df_merged.copy()
    display_df.columns = [c.replace("_", " ").title() if c not in formal_run_labels else c
                          for c in display_df.columns]

    fmt_dict = {col.replace("_", " ").title(): "{:.2f}" for col in ev_cols + ["fit_error"]}
    for lbl in formal_run_labels:
        fmt_dict[lbl] = "{:.2f}"

    st.dataframe(
        display_df.style.format(fmt_dict, na_rep="—"),
        width="stretch",
    )

    st.download_button(
        "Download Full Risk Analysis CSV",
        data=display_df.to_csv(index=False).encode("utf-8"),
        file_name="risk_analysis.csv",
        mime="text/csv",
        width="stretch",
    )

    # Bar chart
    st.subheader("Comparison Chart")
    fig_bar = go.Figure()
    labels = {
        "risk_neutral_ev": "Risk Neutral",
        "upside_skepticism_ev": "Upside Skepticism",
        "downside_protection_eu": "Downside Protection",
        "combined_eu": "Combined",
    }
    colors = ["#2196F3", "#FF9800", "#F44336", "#9C27B0"]
    for col, color in zip(ev_cols, colors):
        fig_bar.add_trace(go.Bar(
            x=df_merged["distribution"].str.replace("_", " ").str.title(),
            y=df_merged[col],
            name=labels[col],
            marker_color=color,
        ))

    fig_bar.update_layout(
        barmode="group",
        yaxis_title="Expected Value / Utility",
        height=400,
        margin=dict(t=30),
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )
    st.plotly_chart(fig_bar, width="stretch")

    # Explanation
    with st.expander("What do these risk adjustments mean?"):
        st.markdown(f"""
**Risk Neutral**: Standard expected value — the probability-weighted average
across the entire distribution. This is what a purely rational,
risk-indifferent decision-maker would use.

**Upside Skepticism** (truncated at p{int(trunc_pct * 100)}): Ignores outcomes
above the {trunc_pct:.1%} percentile. This reflects the belief that
extremely good outcomes are less likely than the distribution suggests —
perhaps due to model error, optimism bias, or diminishing returns.

**Downside Protection** (lambda={loss_lambda:.1f}): Applies a loss-averse
utility function where negative outcomes (below the reference point of
{ref_point:.1f}) are weighted {loss_lambda:.1f}x more heavily than
equivalent gains. This captures the intuition that doing harm is worse
than failing to do good.

**Combined**: Applies both upside skepticism and downside protection
simultaneously. This is the most conservative estimate.
        """)

# ---- Tab 3: Formal Risk Models ----
_MODEL_DISPLAY_NAMES = {"dmreu": "DMREU", "wlu": "WLU", "ambiguity": "Ambiguity Aversion"}
_MODEL_KEYS = list(FORMAL_MODEL_TYPES.keys())
_MODEL_DISPLAY_LIST = [_MODEL_DISPLAY_NAMES[k] for k in _MODEL_KEYS]

with tab3:
    st.subheader("Formal Risk Models")
    st.markdown(
        "Compare how different formal risk aversion models affect expected value. "
        "These models — from [Duffy (2023)](https://rethinkpriorities.org/research-area/"
        "how-can-risk-aversion-affect-your-cause-prioritization/) — apply mathematical "
        "adjustments that make the evaluation more conservative, reflecting different "
        "attitudes to risk."
    )

    # --- Configuration ---
    _preset_descriptions = {
        "WLU sweep": "WLU at three levels of risk aversion (c = 0.01, 0.05, 0.10)",
        "One of each": "One run of each model type (DMREU, WLU, Ambiguity Aversion)",
        "Custom": "Build your own set of model runs",
    }
    _current_preset = st.session_state.get("_last_formal_preset", "WLU sweep")
    _preset_names = list(_FORMAL_PRESETS.keys())

    formal_preset = st.selectbox(
        "Quick setup",
        _preset_names,
        index=_preset_names.index(_current_preset) if _current_preset in _preset_names else 0,
        format_func=lambda p: f"{p}  —  {_preset_descriptions[p]}",
    )

    if formal_preset != "Custom":
        preset_rows = _FORMAL_PRESETS[formal_preset]
        if (st.session_state.get("formal_runs") != preset_rows
                or st.session_state.get("_last_formal_preset") != formal_preset):
            st.session_state.formal_runs = [dict(r) for r in preset_rows]
            st.session_state._last_formal_preset = formal_preset
            st.rerun()

    with st.expander("Edit individual runs", expanded=(formal_preset == "Custom")):
        for i, run_row in enumerate(st.session_state.formal_runs):
            cols = st.columns([3, 3, 2])
            with cols[0]:
                display_idx = _MODEL_KEYS.index(run_row["model"])
                chosen_display = st.selectbox(
                    "Model type", _MODEL_DISPLAY_LIST,
                    index=display_idx,
                    key=f"fm_model_{i}",
                    label_visibility="collapsed",
                )
                run_row["model"] = _MODEL_KEYS[_MODEL_DISPLAY_LIST.index(chosen_display)]
            cfg = _MODEL_PARAM_CFG[run_row["model"]]
            clamped_param = min(max(float(run_row["param"]), cfg["min"]), cfg["max"])
            with cols[1]:
                run_row["param"] = st.number_input(
                    FORMAL_MODEL_TYPES[run_row["model"]]["param_name"],
                    min_value=cfg["min"], max_value=cfg["max"],
                    value=clamped_param, step=cfg["step"],
                    format=cfg["fmt"], key=f"fm_param_{i}",
                )
            with cols[2]:
                run_row["epsilon"] = st.number_input(
                    "ε", min_value=0.0, max_value=0.20, step=0.01,
                    value=float(run_row.get("epsilon", 0.0)),
                    format="%.2f", key=f"fm_eps_{i}",
                    help="Probability rounding: zero out positive outcomes with survival P(X≥x) < ε",
                )

        col_fa, col_fr = st.columns(2)
        with col_fa:
            if st.button("+ Add run", key="add_formal_run"):
                st.session_state.formal_runs.append({"model": "wlu", "param": 0.05, "epsilon": 0.0})
                st.session_state._last_formal_preset = "Custom"
                st.rerun()
        with col_fr:
            if len(st.session_state.formal_runs) > 1 and st.button("- Remove last", key="rm_formal_run"):
                st.session_state.formal_runs.pop()
                st.session_state._last_formal_preset = "Custom"
                st.rerun()

    # Recompute formal runs from (possibly updated) session state
    formal_runs = [
        FormalModelRun(model=r["model"], param=r["param"], epsilon=r.get("epsilon", 0.0))
        for r in st.session_state.formal_runs
    ]
    formal_run_labels = [r.label for r in formal_runs]
    df_formal = compute_formal_runs_all(fits, formal_runs)

    # Show a plain-English summary of what's being compared
    st.caption("Currently comparing: " + ", ".join(f"**{r.label}**" for r in formal_runs))

    # Summary table: Risk Neutral + each configured run
    formal_display_cols = ["distribution", "risk_neutral_ev"] + formal_run_labels + ["fit_error"]
    formal_display = df_formal[formal_display_cols].copy()

    fmt_formal = {"risk_neutral_ev": "{:.2f}", "fit_error": "{:.2f}"}
    for lbl in formal_run_labels:
        fmt_formal[lbl] = "{:.2f}"

    st.dataframe(
        formal_display.style.format(fmt_formal, na_rep="—"),
        width="stretch",
    )

    st.download_button(
        "Download Formal Risk Models CSV",
        data=formal_display.to_csv(index=False).encode("utf-8"),
        file_name="formal_risk_models.csv",
        mime="text/csv",
        width="stretch",
        key="formal_csv",
    )

    # Bar chart
    st.subheader("Comparison Chart")
    fig_formal = go.Figure()
    _bar_colors = ["#2196F3", "#4CAF50", "#00BCD4", "#795548", "#FF9800",
                   "#F44336", "#9C27B0", "#607D8B", "#E91E63", "#3F51B5"]
    fig_formal.add_trace(go.Bar(
        x=df_formal["distribution"].str.replace("_", " ").str.title(),
        y=df_formal["risk_neutral_ev"],
        name="Risk Neutral",
        marker_color=_bar_colors[0],
    ))
    for i, lbl in enumerate(formal_run_labels):
        fig_formal.add_trace(go.Bar(
            x=df_formal["distribution"].str.replace("_", " ").str.title(),
            y=df_formal[lbl],
            name=lbl,
            marker_color=_bar_colors[(i + 1) % len(_bar_colors)],
        ))

    fig_formal.update_layout(
        barmode="group",
        yaxis_title="Expected Value",
        height=400,
        margin=dict(t=30),
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )
    st.plotly_chart(fig_formal, width="stretch")

    # Sensitivity analysis for each unique model type in the configured runs
    active_model_types = sorted({r.model for r in formal_runs})
    if active_model_types:
        st.subheader("Sensitivity Analysis")
        best_fit = fits[0]
        st.caption(f"Showing sensitivity for **{best_fit.name.replace('_', ' ').title()}** (best fit)")

        _sens_configs = {
            "dmreu": {"range": np.linspace(0.01, 0.10, 30), "compute": compute_dmreu,
                      "param_key": "p", "title": "DMREU", "xlabel": "p (risk aversion)"},
            "wlu": {"range": np.linspace(0.0, 0.25, 30), "compute": compute_wlu,
                    "param_key": "c", "title": "WLU", "xlabel": "c (concavity)"},
            "ambiguity": {"range": np.linspace(0.0, 8.0, 30), "compute": compute_ambiguity_aversion,
                          "param_key": "k", "title": "Ambiguity Aversion", "xlabel": "k (strength)"},
        }
        sens_cols = st.columns(len(active_model_types))
        for col_s, mtype in zip(sens_cols, active_model_types):
            with col_s:
                cfg = _sens_configs[mtype]
                vals = [cfg["compute"](best_fit, **{cfg["param_key"]: v}, n_samples=2000) for v in cfg["range"]]
                fig_s = go.Figure()
                fig_s.add_trace(go.Scatter(x=cfg["range"], y=vals, mode="lines"))
                for r in formal_runs:
                    if r.model == mtype:
                        fig_s.add_vline(x=r.param, line_dash="dash", line_color="red", opacity=0.7,
                                        annotation_text=f"{r.param}", annotation_position="top")
                fig_s.update_layout(
                    title=cfg["title"], xaxis_title=cfg["xlabel"], yaxis_title="Expected Value",
                    height=300, margin=dict(t=40, b=30), showlegend=False,
                )
                st.plotly_chart(fig_s, width="stretch")

    # Explanation
    with st.expander("What do these formal models mean?"):
        st.markdown("""
**DMREU — Difference-Making Risk-Weighted Expected Utility**

Sorts all possible outcomes from worst to best and applies a probability-weighting
function m(P) = P^a that overweights the probability of achieving only the worst
outcomes. When a=1 (p=0.01), this recovers risk-neutral expected value. Higher values
of a (higher p) penalize interventions that have a high probability of making no
difference or causing harm, even if their upside is enormous.

---

**WLU — Weighted Linear Utility**

Reweights based on outcome magnitude. Worse outcomes (negative or small) receive
proportionally higher weight, while better outcomes (large positive) receive lower
weight. This captures "stakes-sensitive" risk aversion.

The weighting function w(x; c) = 1/(1+|x|^c) gives small outcomes (near zero) a
weight close to 1, while large positive outcomes get a weight approaching 0.

---

**Ambiguity Aversion — Expected Difference Made**

Sorts expected utilities from worst to best and applies a cubic weighting function
that overweights worse outcomes. At k=4 (mild), the worst outcome gets 1.5x weight
and the best gets 0.5x. At k=8 (strong), the worst gets 2x and the best gets 0x.

*Note:* In the paper, this is a second-order model that aggregates across multiple
expected-utility estimates under model uncertainty. Here we apply the same cubic
weighting as a single-distribution proxy.
        """)

# ---- Tab 4: Explorer ----
with tab4:

    @st.fragment
    def explorer_fragment():
        """Fragment so slider changes re-run only this section, not the whole page."""
        st.subheader("Distribution Explorer")
        st.caption("Sliders update live without refitting.")

        selected_name = st.selectbox(
            "Select distribution",
            [f.name.replace("_", " ").title() for f in fits],
        )
        selected_fit = next(f for f in fits if f.name.replace("_", " ").title() == selected_name)

        col_left, col_right = st.columns(2)

        with col_left:
            # CDF plot with shaded regions
            st.markdown("**CDF with risk regions**")
            x_exp = np.linspace(
                selected_fit.ppf(0.002),
                selected_fit.ppf(0.998),
                500,
            )
            cdf_vals = selected_fit.cdf(x_exp)

            fig_cdf = go.Figure()
            fig_cdf.add_trace(go.Scatter(
                x=x_exp, y=cdf_vals, mode="lines", name="CDF", line=dict(color="#1976D2"),
            ))

            # Shade truncation region
            exp_trunc = st.session_state.get("exp_trunc", trunc_pct)
            trunc_val = selected_fit.ppf(exp_trunc)
            fig_cdf.add_vrect(
                x0=trunc_val, x1=x_exp[-1],
                fillcolor="orange", opacity=0.15,
                annotation_text="Truncated",
                annotation_position="top right",
            )

            # Shade loss region
            exp_lambda = st.session_state.get("exp_lambda", loss_lambda)
            actual_ref = selected_fit.median() if use_median_ref else ref_point
            if x_exp[0] < actual_ref:
                fig_cdf.add_vrect(
                    x0=x_exp[0], x1=actual_ref,
                    fillcolor="red", opacity=0.1,
                    annotation_text=f"Loss region (x{exp_lambda:.1f})",
                    annotation_position="bottom left",
                )

            fig_cdf.add_hline(y=exp_trunc, line_dash="dot", line_color="orange", opacity=0.5)
            fig_cdf.add_vline(x=actual_ref, line_dash="dot", line_color="red", opacity=0.5)

            fig_cdf.update_layout(
                xaxis_title="Value", yaxis_title="Cumulative Probability",
                height=400, margin=dict(t=30),
            )
            st.plotly_chart(fig_cdf, width="stretch")

        with col_right:
            st.markdown("**Informal adjustments**")

            # Live parameter adjustment — these are cheap (no refitting)
            exp_trunc = st.slider(
                "Truncation percentile",
                0.90, 0.999, trunc_pct, 0.001, format="%.3f",
                key="exp_trunc",
            )
            exp_lambda = st.slider(
                "Loss aversion multiplier",
                1.0, 5.0, loss_lambda, 0.1,
                key="exp_lambda",
            )

            exp_params = RiskParams(
                truncation_percentile=exp_trunc,
                loss_aversion_lambda=exp_lambda,
                reference_point=selected_fit.median() if use_median_ref else ref_point,
            )
            exp_result = analyze(selected_fit, exp_params)

            informal_metrics = {
                "Risk Neutral EV": exp_result.risk_neutral_ev,
                "Upside Skepticism EV": exp_result.upside_skepticism_ev,
                "Downside Protection EU": exp_result.downside_protection_eu,
                "Combined EU": exp_result.combined_eu,
            }

            for label, value in informal_metrics.items():
                delta = value - exp_result.risk_neutral_ev if label != "Risk Neutral EV" else None
                st.metric(label, f"{value:.2f}", delta=f"{delta:.2f}" if delta is not None else None)

        # Formal models section — metric cards for each configured run
        st.divider()
        st.markdown("**Formal risk models**")

        n_runs = len(formal_runs)
        if n_runs > 0:
            run_cols = st.columns(min(n_runs, 4))
            for idx, run in enumerate(formal_runs):
                with run_cols[idx % len(run_cols)]:
                    val = compute_formal_run(selected_fit, run)
                    delta = val - exp_result.risk_neutral_ev
                    st.metric(run.label, f"{val:.2f}", delta=f"{delta:.2f}")

        # Fitted percentile check
        st.divider()
        st.subheader("Percentile fit check")
        check_rows = []
        for q, target in sorted(percentiles.items()):
            fitted_val = selected_fit.ppf(q)
            check_rows.append({
                "Percentile": f"p{int(q * 100)}",
                "Target": target,
                "Fitted": fitted_val,
                "Error": fitted_val - target,
                "Relative Error": f"{abs(fitted_val - target) / max(abs(target), 1e-6):.2%}",
            })
        st.dataframe(pd.DataFrame(check_rows), width="stretch")

    explorer_fragment()
