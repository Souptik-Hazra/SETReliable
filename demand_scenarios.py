import numpy as np


GROUP = "MEDICINE_HOSPITAL_ID"


def create_stable_scenario(df):
    out = df.copy()
    out["SCENARIO_ID"] = "STABLE"
    out["DEMAND_REGIME"] = "STABLE"
    out["DEMAND_SCENARIO"] = out["DEMAND"]
    return out


def _eligible_indexes(group, min_history):
    observed = group["DEMAND"].notna()
    if int(observed.sum()) < min_history:
        return []
    return list(group.index)


def create_abrupt_surge_scenario(df, surge_factor=2.0, surge_months=3):
    out = df.sort_values([GROUP, "DATE"]).copy()
    out["DEMAND_SCENARIO"] = out["DEMAND"]
    out["SCENARIO_ID"] = f"ABRUPT_SURGE_{surge_factor}"
    out["DEMAND_REGIME"] = "STABLE"
    for _, group in out.groupby(GROUP, sort=False):
        indexes = _eligible_indexes(group, surge_months + 6)
        if not indexes:
            continue
        start = int(len(indexes) * 0.60)
        selected = indexes[start:start + surge_months]
        out.loc[selected, "DEMAND_SCENARIO"] = (
            out.loc[selected, "DEMAND"] * surge_factor
        )
        out.loc[selected, "DEMAND_REGIME"] = "ABRUPT_SURGE"
    return out


def create_gradual_drift_scenario(df, total_increase=0.50, drift_months=6):
    out = df.sort_values([GROUP, "DATE"]).copy()
    out["DEMAND_SCENARIO"] = out["DEMAND"]
    out["SCENARIO_ID"] = f"GRADUAL_DRIFT_{total_increase}"
    out["DEMAND_REGIME"] = "STABLE"
    for _, group in out.groupby(GROUP, sort=False):
        indexes = _eligible_indexes(group, drift_months + 6)
        if not indexes:
            continue
        start = int(len(indexes) * 0.60)
        selected = indexes[start:start + drift_months]
        factors = np.linspace(1.0, 1.0 + total_increase, len(selected))
        out.loc[selected, "DEMAND_SCENARIO"] = (
            out.loc[selected, "DEMAND"].to_numpy() * factors
        )
        out.loc[selected, "DEMAND_REGIME"] = "GRADUAL_DRIFT"
    return out


def create_volatility_scenario(df, volatility_multiplier=2.0):
    out = df.sort_values([GROUP, "DATE"]).copy()
    out["DEMAND_SCENARIO"] = out["DEMAND"]
    out["SCENARIO_ID"] = f"VOLATILITY_{volatility_multiplier}"
    out["DEMAND_REGIME"] = "STABLE"
    for _, group in out.groupby(GROUP, sort=False):
        observed = group["DEMAND"].dropna()
        if len(observed) < 12:
            continue
        start = int(len(group) * 0.60)
        selected = group.index[start:]
        baseline = group.loc[selected, "DEMAND"].mean()
        if not np.isfinite(baseline):
            continue
        values = group.loc[selected, "DEMAND"]
        modified = baseline + volatility_multiplier * (values - baseline)
        out.loc[selected, "DEMAND_SCENARIO"] = modified.clip(lower=0)
        out.loc[selected, "DEMAND_REGIME"] = "VOLATILITY_INCREASE"
    return out
