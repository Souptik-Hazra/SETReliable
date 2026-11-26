import numpy as np

GROUP = "MEDICINE_HOSPITAL_ID"


def _prepare(df, target, scenario_id):
    if target not in df:
        raise ValueError(f"Target column not found: {target}")
    out = df.sort_values([GROUP, "DATE"]).copy()
    out["SCENARIO_TARGET"] = target
    out["SCENARIO_QUANTITY"] = out[target]
    out["SCENARIO_ID"] = scenario_id
    out["SCENARIO_MODIFIED"] = 0
    return out


def create_stable_scenario(df, target="GROSS_POSITIVE_QUANTITY"):
    return _prepare(df, target, f"STABLE_{target}")


def create_abrupt_surge_scenario(df, target="GROSS_POSITIVE_QUANTITY",
                                 surge_factor=2.0, surge_months=3):
    if surge_factor < 0 or surge_months < 1:
        raise ValueError("Surge factor must be non-negative and months must be positive")
    out = _prepare(df, target, f"ABRUPT_SURGE_{target}_{surge_factor}")
    for _, group in out.groupby(GROUP, sort=False):
        observed = group.index[group["SCENARIO_QUANTITY"].notna()]
        if len(observed) < surge_months + 6:
            continue
        start = int(len(observed) * 0.60)
        selected = observed[start:start + surge_months]
        out.loc[selected, "SCENARIO_QUANTITY"] *= surge_factor
        out.loc[selected, "SCENARIO_MODIFIED"] = 1
    return out


def create_gradual_drift_scenario(df, target="GROSS_POSITIVE_QUANTITY",
                                  total_increase=0.50, drift_months=6):
    if total_increase < -1 or drift_months < 1:
        raise ValueError("Increase must be >= -1 and months must be positive")
    out = _prepare(df, target, f"GRADUAL_DRIFT_{target}_{total_increase}")
    for _, group in out.groupby(GROUP, sort=False):
        observed = group.index[group["SCENARIO_QUANTITY"].notna()]
        if len(observed) < drift_months + 6:
            continue
        selected = observed[int(len(observed) * 0.60):][:drift_months]
        factors = np.linspace(1.0, 1.0 + total_increase, len(selected))
        out.loc[selected, "SCENARIO_QUANTITY"] *= factors
        out.loc[selected, "SCENARIO_MODIFIED"] = 1
    return out


def create_volatility_scenario(df, target="GROSS_POSITIVE_QUANTITY",
                               volatility_multiplier=2.0):
    if volatility_multiplier < 0:
        raise ValueError("Volatility multiplier must be non-negative")
    out = _prepare(df, target, f"VOLATILITY_{target}_{volatility_multiplier}")
    for _, group in out.groupby(GROUP, sort=False):
        observed = group["SCENARIO_QUANTITY"].dropna()
        if len(observed) < 12:
            continue
        selected = group.index[int(len(group) * 0.60):]
        values = out.loc[selected, "SCENARIO_QUANTITY"]
        baseline = values.mean()
        if np.isfinite(baseline):
            out.loc[selected, "SCENARIO_QUANTITY"] = (
                baseline + volatility_multiplier * (values - baseline)
            ).clip(lower=0)
            out.loc[selected, "SCENARIO_MODIFIED"] = out.loc[selected, "SCENARIO_QUANTITY"].notna().astype("int8")
    return out
