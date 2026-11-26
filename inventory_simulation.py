import numpy as np
import pandas as pd

GROUP = "MEDICINE_HOSPITAL_ID"


def assign_lead_times(df, seed=42):
    """Assign clearly labelled assumed lead times; supplier lead times are unavailable."""
    out = df.copy()
    rng = np.random.default_rng(seed)
    ids = out[GROUP].drop_duplicates().to_numpy()
    mapping = {key: int(rng.choice([1, 2, 3], p=[0.60, 0.30, 0.10])) for key in ids}
    out["LEAD_TIME_MONTHS"] = out[GROUP].map(mapping).astype("int8")
    out["LEAD_TIME_ASSUMED"] = 1
    return out


def calculate_initial_inventory(df, initial_inventory=0.0):
    if initial_inventory < 0:
        raise ValueError("initial_inventory cannot be negative")
    out = df.copy()
    out["INITIAL_INVENTORY"] = float(initial_inventory)
    out["OPENING_STOCK_ASSUMED"] = 1
    return out


def simulate_inventory(df, target_stock_multiplier=1.5, reorder_quantile=0.90,
                       lead_time_delay_months=0, apply_negative_movements_as_returns=False):
    """Hypothetical movement stress test, not a reconstruction of actual Trust stock.

    SCMD does not supply verified opening balances, receipts or supplier lead
    times. Negative movements are excluded from simulated supply by default.
    """
    if not 0 < reorder_quantile < 1:
        raise ValueError("reorder_quantile must be between 0 and 1")
    if target_stock_multiplier < 0 or lead_time_delay_months < 0:
        raise ValueError("Stock multiplier and lead-time delay must be non-negative")
    required = {"GROSS_POSITIVE_QUANTITY", "NEGATIVE_QUANTITY_TOTAL",
                "LEAD_TIME_MONTHS", "INITIAL_INVENTORY"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Inventory input missing columns: {sorted(missing)}")

    frames = []
    for _, source in df.sort_values([GROUP, "DATE"]).groupby(GROUP, sort=False):
        group = source.copy().reset_index(drop=True)
        issue_col = "SCENARIO_QUANTITY" if "SCENARIO_QUANTITY" in group else "GROSS_POSITIVE_QUANTITY"
        gross = pd.to_numeric(group[issue_col], errors="coerce").to_numpy(float)
        negative = pd.to_numeric(group["NEGATIVE_QUANTITY_TOTAL"], errors="coerce").to_numpy(float)
        n = len(group)
        lead = max(1, int(group["LEAD_TIME_MONTHS"].iloc[0]) + lead_time_delay_months)
        current = float(group["INITIAL_INVENTORY"].iloc[0])
        pending, history = {}, []
        output = {name: np.zeros(n) for name in (
            "INCOMING_ORDER", "RETURN_ADJUSTMENT_INFLOW", "AVAILABLE_SUPPLY",
            "INVENTORY_ON_HAND", "ORDER_QUANTITY", "ORDER_PIPELINE",
            "SHORTAGE_QUANTITY", "STOCKOUT_EVENT")}
        output["FORECAST_GROSS_ISSUES"] = np.full(n, np.nan)
        output["REORDER_POINT"] = np.full(n, np.nan)

        for t in range(n):
            output["INCOMING_ORDER"][t] = pending.pop(t, 0.0)
            inflow = abs(negative[t]) if (
                apply_negative_movements_as_returns and np.isfinite(negative[t])
            ) else 0.0
            output["RETURN_ADJUSTMENT_INFLOW"][t] = inflow
            available = current + output["INCOMING_ORDER"][t] + inflow
            output["AVAILABLE_SUPPLY"][t] = available
            actual = gross[t]
            if np.isfinite(actual):
                forecast = float(np.mean(history[-6:])) if history else 0.0
                output["FORECAST_GROSS_ISSUES"][t] = forecast
                lead_totals = [
                    sum(history[max(0, i - lead + 1):i + 1])
                    for i in range(lead - 1, len(history))
                ]
                reorder = float(np.quantile(lead_totals, reorder_quantile)) if lead_totals else lead * forecast
                output["REORDER_POINT"][t] = reorder
                shortage = max(0.0, actual - available)
                output["SHORTAGE_QUANTITY"][t] = shortage
                output["STOCKOUT_EVENT"][t] = int(shortage > 0)
                current = max(0.0, available - actual)
                history.append(float(actual))
                position = current + sum(pending.values())
                target_stock = max(reorder, (lead + target_stock_multiplier) * forecast)
                if position <= reorder:
                    order = max(0.0, target_stock - position)
                    output["ORDER_QUANTITY"][t] = order
                    if order:
                        pending[t + lead] = pending.get(t + lead, 0.0) + order
            else:
                current = available
            output["INVENTORY_ON_HAND"][t] = current
            output["ORDER_PIPELINE"][t] = sum(pending.values())

        for col, values in output.items():
            group[col] = values
        group["NEGATIVE_MOVEMENTS_TREATED_AS_RETURNS"] = int(apply_negative_movements_as_returns)
        group["INVENTORY_SIMULATION_IS_HYPOTHETICAL"] = 1
        group["STOCKOUTS_ARE_OBSERVED"] = 0
        frames.append(group)
    return pd.concat(frames, ignore_index=True) if frames else df.copy()
