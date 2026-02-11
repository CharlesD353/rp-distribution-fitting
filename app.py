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
from risk import (
    RiskParams, analyze, analyze_all,
    compute_dmreu, compute_wlu, compute_ambiguity_aversion,
    ev_eu_percentile_table, ev_eu_percentile_table_all,
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

    st.divider()

    st.subheader("Formal Risk Models")
    st.caption("[Duffy (2023)](https://rethinkpriorities.org/research-area/how-can-risk-aversion-affect-your-cause-prioritization/), Rethink Priorities")

    dmreu_p = st.slider(
        "DMREU risk aversion (p)",
        min_value=0.01, max_value=0.10, value=0.01, step=0.01, format="%.2f",
        help=(
            "Thought-experiment probability: what chance of saving 1000 lives "
            "makes you indifferent to saving 10 for certain? "
            "p=0.01 is risk-neutral, p=0.05 is moderate, p=0.10 is high risk aversion. "
            "Internally converted to power exponent a = −2/log₁₀(p)."
        ),
    )
    wlu_c = st.slider(
        "WLU concavity (c)",
        min_value=0.0, max_value=0.25, value=0.0, step=0.01, format="%.2f",
        help=(
            "Stakes-sensitive risk aversion: worse outcomes contribute more to the "
            "weighted expected value than their probability alone would suggest. "
            "c=0 is risk-neutral, c=0.25 is high risk aversion."
        ),
    )
    ambiguity_k = st.slider(
        "Ambiguity aversion (k)",
        min_value=0.0, max_value=8.0, value=0.0, step=0.5, format="%.1f",
        help=(
            "Overweights worse-ranked expected utilities and underweights better ones. "
            "k=0 is neutral, k=4 is mild (paper's f₂), k=8 is strong (paper's f₁)."
        ),
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
    st.info("Configure inputs in the sidebar, then click **Fit distributions**.")
    st.stop()

risk_params = RiskParams(
    truncation_percentile=trunc_pct,
    loss_aversion_lambda=loss_lambda,
    reference_point=ref_point,
    dmreu_p=dmreu_p,
    wlu_c=wlu_c,
    ambiguity_k=ambiguity_k,
)

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
                dmreu_p=dmreu_p,
                wlu_c=wlu_c,
                ambiguity_k=ambiguity_k,
            )
            result = analyze(fit, params_i)
            row = result.to_dict()
            row["fit_error"] = fit.error
            row["reference_point"] = fit.median()
            risk_rows.append(row)
        df_risk = pd.DataFrame(risk_rows)

        export_frames = []
        for fit in fits:
            params_i = RiskParams(
                truncation_percentile=trunc_pct,
                loss_aversion_lambda=loss_lambda,
                reference_point=fit.median(),
                dmreu_p=dmreu_p,
                wlu_c=wlu_c,
                ambiguity_k=ambiguity_k,
            )
            export_frames.append(ev_eu_percentile_table(fit, params_i))
        df_percentile_export = pd.concat(export_frames, ignore_index=True)
    else:
        df_risk = analyze_all(fits, risk_params)
        df_percentile_export = ev_eu_percentile_table_all(fits, risk_params)

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

    st.download_button(
        "Download 1-99 percentile EV/EU CSV",
        data=df_percentile_export.to_csv(index=False).encode("utf-8"),
        file_name="ev_eu_percentiles_p1_to_p99.csv",
        mime="text/csv",
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

# ---- Tab 3: Formal Risk Models ----
with tab3:
    st.subheader("Formal Risk Models (Duffy 2023)")
    st.caption("From *How Can Risk Aversion Affect Your Cause Prioritization?*, Rethink Priorities")

    # Summary table
    formal_cols = ["risk_neutral_ev", "dmreu_ev", "wlu_ev", "ambiguity_aversion_ev"]
    dmreu_a = -2.0 / _math.log10(dmreu_p) if dmreu_p > 0 and dmreu_p < 1 else 1.0
    formal_labels = {
        "risk_neutral_ev": "Risk Neutral",
        "dmreu_ev": f"DMREU (p={dmreu_p:.2f}, a={dmreu_a:.2f})",
        "wlu_ev": f"WLU (c={wlu_c:.2f})",
        "ambiguity_aversion_ev": f"Ambiguity (k={ambiguity_k:.1f})",
    }

    formal_df = df_risk[["distribution"] + formal_cols + ["fit_error"]].copy()
    formal_display = formal_df.copy()
    formal_display.columns = [
        formal_labels.get(c, c.replace("_", " ").title())
        for c in formal_display.columns
    ]

    st.dataframe(
        formal_display.style.format(
            {col: "{:.2f}" for col in formal_display.columns if col not in ("Distribution", formal_labels.get("distribution", ""))},
            na_rep="—",
        ),
        use_container_width=True,
    )

    # CSV download for formal models
    formal_export = df_risk[["distribution"] + formal_cols + ["fit_error"]].copy()
    st.download_button(
        "Download formal risk models CSV",
        data=formal_export.to_csv(index=False).encode("utf-8"),
        file_name="formal_risk_models.csv",
        mime="text/csv",
        use_container_width=True,
    )

    # Bar chart
    st.subheader("Comparison Chart")
    fig_formal = go.Figure()
    formal_colors = ["#2196F3", "#4CAF50", "#00BCD4", "#795548"]
    for col, color in zip(formal_cols, formal_colors):
        fig_formal.add_trace(go.Bar(
            x=df_risk["distribution"].str.replace("_", " ").str.title(),
            y=df_risk[col],
            name=formal_labels[col],
            marker_color=color,
        ))

    fig_formal.update_layout(
        barmode="group",
        yaxis_title="Expected Value",
        height=400,
        margin=dict(t=30),
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )
    st.plotly_chart(fig_formal, use_container_width=True)

    # Sensitivity analysis (for best-fit distribution)
    st.subheader("Sensitivity Analysis")
    best_fit = fits[0]
    st.caption(f"Showing sensitivity for **{best_fit.name.replace('_', ' ').title()}** (best fit)")

    col_s1, col_s2, col_s3 = st.columns(3)

    with col_s1:
        dmreu_p_range = np.linspace(0.01, 0.10, 30)
        dmreu_vals = [compute_dmreu(best_fit, p=pv, n_samples=2000) for pv in dmreu_p_range]
        fig_s1 = go.Figure()
        fig_s1.add_trace(go.Scatter(
            x=dmreu_p_range, y=dmreu_vals, mode="lines",
            line=dict(color="#4CAF50"),
        ))
        fig_s1.add_vline(x=dmreu_p, line_dash="dash", line_color="red", opacity=0.7)
        fig_s1.update_layout(
            title="DMREU",
            xaxis_title="p (risk aversion)",
            yaxis_title="Expected Value",
            height=300, margin=dict(t=40, b=30),
            showlegend=False,
        )
        st.plotly_chart(fig_s1, use_container_width=True)

    with col_s2:
        wlu_c_range = np.linspace(0.0, 0.25, 30)
        wlu_vals = [compute_wlu(best_fit, c=cv, n_samples=2000) for cv in wlu_c_range]
        fig_s2 = go.Figure()
        fig_s2.add_trace(go.Scatter(
            x=wlu_c_range, y=wlu_vals, mode="lines",
            line=dict(color="#00BCD4"),
        ))
        fig_s2.add_vline(x=wlu_c, line_dash="dash", line_color="red", opacity=0.7)
        fig_s2.update_layout(
            title="WLU",
            xaxis_title="c (concavity)",
            yaxis_title="Expected Value",
            height=300, margin=dict(t=40, b=30),
            showlegend=False,
        )
        st.plotly_chart(fig_s2, use_container_width=True)

    with col_s3:
        amb_k_range = np.linspace(0.0, 8.0, 30)
        amb_vals = [compute_ambiguity_aversion(best_fit, k=kv, n_samples=2000) for kv in amb_k_range]
        fig_s3 = go.Figure()
        fig_s3.add_trace(go.Scatter(
            x=amb_k_range, y=amb_vals, mode="lines",
            line=dict(color="#795548"),
        ))
        fig_s3.add_vline(x=ambiguity_k, line_dash="dash", line_color="red", opacity=0.7)
        fig_s3.update_layout(
            title="Ambiguity Aversion",
            xaxis_title="k (strength)",
            yaxis_title="Expected Value",
            height=300, margin=dict(t=40, b=30),
            showlegend=False,
        )
        st.plotly_chart(fig_s3, use_container_width=True)

    # Explanation
    with st.expander("What do these formal models mean?"):
        st.markdown(f"""
**DMREU — Difference-Making Risk-Weighted Expected Utility** (p={dmreu_p:.2f}, a={dmreu_a:.2f})

Sorts all possible outcomes from worst to best and applies a probability-weighting
function m(P) = P^a that overweights the probability of achieving only the worst
outcomes. When a=1 (p=0.01), this recovers risk-neutral expected value. Higher values
of a (higher p) penalize interventions that have a high probability of making no
difference or causing harm, even if their upside is enormous.

*Intuition*: "Would you take a {dmreu_p:.0%} chance of saving 1000 lives over saving
10 for certain?" If yes, your risk-aversion level corresponds to p={dmreu_p:.2f}.

---

**WLU — Weighted Linear Utility** (c={wlu_c:.2f})

Unlike DMREU which reweights probabilities based on outcome rank, WLU reweights
based on outcome magnitude. Worse outcomes (negative or small) receive
proportionally higher weight, while better outcomes (large positive) receive lower
weight. This captures "stakes-sensitive" risk aversion: structurally identical bets
might be valued differently depending on whether the stakes are tens of lives vs
billions of lives.

The weighting function w(x; c) = 1/(1+|x|^c) gives small outcomes (near zero) a
weight close to 1, while large positive outcomes get a weight approaching 0. After
normalization, this shifts expected value downward.

---

**Ambiguity Aversion — Expected Difference Made** (k={ambiguity_k:.1f})

When we have uncertainty about the model itself (not just the outcomes), we might
want to be ambiguity-averse: giving more weight to pessimistic model estimates
and less to optimistic ones. This model sorts expected utilities from worst to best
and applies a cubic weighting function that overweights worse outcomes.

At k=4 (mild), the worst outcome gets 1.5x weight and the best gets 0.5x.
At k=8 (strong), the worst gets 2x weight and the best gets 0x.

*Note:* In the paper, this is a second-order model that aggregates across multiple
expected-utility estimates under model uncertainty. Here we apply the same cubic
weighting as a single-distribution proxy — outcomes are rank-ordered and reweighted
to capture the directional intent (be more conservative when uncertain) without
requiring a set of competing models.
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
            st.plotly_chart(fig_cdf, use_container_width=True)

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
                dmreu_p=st.session_state.get("exp_dmreu_p", dmreu_p),
                wlu_c=st.session_state.get("exp_wlu_c", wlu_c),
                ambiguity_k=st.session_state.get("exp_amb_k", ambiguity_k),
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

        # Formal models section
        st.divider()
        st.markdown("**Formal risk models**")

        col_f1, col_f2, col_f3 = st.columns(3)

        with col_f1:
            exp_dmreu_p = st.slider(
                "DMREU p", 0.01, 0.10, dmreu_p, 0.01, format="%.2f",
                key="exp_dmreu_p",
            )
            delta_d = exp_result.dmreu_ev - exp_result.risk_neutral_ev
            st.metric("DMREU EV", f"{exp_result.dmreu_ev:.2f}", delta=f"{delta_d:.2f}")

        with col_f2:
            exp_wlu_c = st.slider(
                "WLU c", 0.0, 0.25, wlu_c, 0.01, format="%.2f",
                key="exp_wlu_c",
            )
            delta_w = exp_result.wlu_ev - exp_result.risk_neutral_ev
            st.metric("WLU EV", f"{exp_result.wlu_ev:.2f}", delta=f"{delta_w:.2f}")

        with col_f3:
            exp_amb_k = st.slider(
                "Ambiguity k", 0.0, 8.0, ambiguity_k, 0.5, format="%.1f",
                key="exp_amb_k",
            )
            delta_a = exp_result.ambiguity_aversion_ev - exp_result.risk_neutral_ev
            st.metric("Ambiguity EV", f"{exp_result.ambiguity_aversion_ev:.2f}", delta=f"{delta_a:.2f}")

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
        st.dataframe(pd.DataFrame(check_rows), use_container_width=True)

    explorer_fragment()
