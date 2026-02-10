"""
Example percentile data for charitable giving cost-effectiveness analysis.

These represent plausible expert-elicited beliefs about the marginal
cost-effectiveness of different charitable funds. Values are illustrative.
"""

GLOBAL_HEALTH_FUND = {
    "name": "Global Health & Development Fund",
    "description": "GiveWell-style global health interventions (DALYs averted per $1000)",
    "percentiles": {0.05: 0.5, 0.10: 1.0, 0.25: 2.5, 0.50: 5.0, 0.75: 10.0, 0.90: 20.0, 0.95: 35.0},
}

ANIMAL_WELFARE_FUND = {
    "name": "Animal Welfare Fund",
    "description": "Cage-free campaigns, fish/shrimp welfare (welfare-adjusted units per $1000)",
    "percentiles": {0.10: -2.0, 0.50: 8.0, 0.90: 50.0},
}

AI_SAFETY_FUND = {
    "name": "AI Safety & Governance Fund",
    "description": "Longtermist AI safety research and policy (expected value units per $1000)",
    "percentiles": {0.10: -50.0, 0.50: 5.0, 0.90: 500.0},
}

META_RESEARCH_FUND = {
    "name": "Meta-Research & Capacity Building",
    "description": "Improving EA research quality (counterfactual value per $1000)",
    "percentiles": {0.10: 0.1, 0.50: 3.0, 0.90: 15.0},
}

ALL_EXAMPLES = [GLOBAL_HEALTH_FUND, ANIMAL_WELFARE_FUND, AI_SAFETY_FUND, META_RESEARCH_FUND]
