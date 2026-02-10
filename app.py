"""
Streamlit demo app for distribution fitting and risk adjustment.

Run with: streamlit run app.py
"""

from collections import Counter

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from distributions import DISTRIBUTIONS, compare_fits, fit_all
from examples import ALL_EXAMPLES
from risk import RiskParams, analyze, analyze_all

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
    fit_clicked = st.button("Fit distributions", type="primary", use_container_width=True)

    st.header("Input Data")
    st.caption("Does not automatically refresh — click **Fit distributions** to recalculate with changes")

    # Example selector lives outside the form so it can reset pct_rows
    example_names = ["Custom"] + [ex["name"] for ex in ALL_EXAMPLES]
    example_choice = st.selectbox("Load example", example_names)

    if example_choice != "Custom":
        selected_example = next(ex for ex in ALL_EXAMPLES if ex["name"] == example_choice)
        st.caption(selected_example["description"])
        default_pcts = selected_example["percentiles"]
    else:
        default_pcts = {0.10: -10.0, 0.50: 5.0, 0.90: 30.0}

    # Initialize session state for percentile rows
    if "pct_rows" not in st.session_state or example_choice != st.session_state.get("last_example"):
        st.session_state.pct_rows = [
            {"q": q, "v": v} for q, v in sorted(default_pcts.items())
        ]
        st.session_state.last_example = example_choice
        # Bump version so form widget keys are fresh (avoids stale key reuse)
        st.session_state.pct_version = st.session_state.get("pct_version", 0) + 1
        st.rerun()

    pv = st.session_state.get("pct_version", 0)

    # --- Percentile input form ---
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

        pct_submitted = st.form_submit_button("Update percentiles", use_container_width=True)

    # Add/remove buttons directly below the percentile list
    col_add, col_rm = st.columns(2)
    with col_add:
        if st.button("+ Add row"):
            st.session_state.pct_rows.append({"q": 0.50, "v": 0.0})
            st.rerun()
    with col_rm:
        if len(st.session_state.pct_rows) > 2 and st.button("- Remove last"):
            st.session_state.pct_rows.pop()
            st.rerun()

    # Sync percentile form inputs back to session state on submit
    if pct_submitted:
        st.session_state.pct_rows = pct_rows_input

    st.divider()

    # --- Distribution & risk settings (no form — these don't trigger refits) ---
    st.subheader("Distributions to fit")
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

    st.divider()

    st.subheader("Risk Parameters")
    trunc_pct = st.slider(
        "Truncation percentile (upside skepticism)",
        min_value=0.90, max_value=0.999, value=0.99, step=0.001, format="%.3f",
    )
    loss_lambda = st.slider(
        "Loss aversion multiplier",
        min_value=1.0, max_value=5.0, value=2.5, step=0.1,
    )
    ref_point = st.number_input("Reference point (gains vs losses threshold)", value=0.0, step=1.0)
    use_median_ref = st.checkbox("Use fitted median as reference point", value=False)

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
    st.info("Configure inputs in the sidebar, then click **Fit distributions**.")
    st.stop()

risk_params = RiskParams(
    truncation_percentile=trunc_pct,
    loss_aversion_lambda=loss_lambda,
    reference_point=ref_point,
)

# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------

tab1, tab2, tab3 = st.tabs(["Fitted Distributions", "Risk Adjustments", "Explorer"])

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
    st.plotly_chart(fig_pdf, use_container_width=True)

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
        use_container_width=True,
    )

# ---- Tab 2: Risk Adjustments ----
with tab2:
    st.subheader("Risk-Adjusted Expected Values")

    # If using median as reference, recompute per distribution
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
            row["fit_error"] = fit.error
            row["reference_point"] = fit.median()
            risk_rows.append(row)
        df_risk = pd.DataFrame(risk_rows)
    else:
        df_risk = analyze_all(fits, risk_params)

    # Format and display
    ev_cols = ["risk_neutral_ev", "upside_skepticism_ev", "downside_protection_eu", "combined_eu"]
    display_df = df_risk.copy()
    display_df.columns = [c.replace("_", " ").title() for c in display_df.columns]

    st.dataframe(
        display_df.style.format(
            {col.replace("_", " ").title(): "{:.2f}" for col in ev_cols + ["fit_error"]},
            na_rep="—",
        ),
        use_container_width=True,
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
            x=df_risk["distribution"].str.replace("_", " ").str.title(),
            y=df_risk[col],
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
    st.plotly_chart(fig_bar, use_container_width=True)

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

# ---- Tab 3: Explorer ----
with tab3:

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
            st.plotly_chart(fig_cdf, use_container_width=True)

        with col_right:
            st.markdown("**Risk-adjusted values**")

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

            metrics = {
                "Risk Neutral EV": exp_result.risk_neutral_ev,
                "Upside Skepticism EV": exp_result.upside_skepticism_ev,
                "Downside Protection EU": exp_result.downside_protection_eu,
                "Combined EU": exp_result.combined_eu,
            }

            for label, value in metrics.items():
                delta = value - exp_result.risk_neutral_ev if label != "Risk Neutral EV" else None
                st.metric(label, f"{value:.2f}", delta=f"{delta:.2f}" if delta is not None else None)

        # Fitted percentile check
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
        st.dataframe(pd.DataFrame(check_rows), use_container_width=True)

    explorer_fragment()
