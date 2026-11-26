import numpy as np
import pandas as pd


GROUP = "MEDICINE_HOSPITAL_ID"


def assign_lead_times(df, seed=42):
    """Assign reproducible scenario lead times; these are assumptions, not observed supplier data."""
    out = df.copy()
    rng = np.random.default_rng(seed)
    ids = out[GROUP].drop_duplicates().to_numpy()
    mapping = {key: int(rng.choice([1, 2, 3], p=[0.60, 0.30, 0.10])) for key in ids}
    out["LEAD_TIME_MONTHS"] = out[GROUP].map(mapping).astype("int8")
    out["LEAD_TIME_ASSUMED"] = 1
    return out


def calculate_initial_inventory(df, multiplier=1.5):
    out = df.copy()
    rows = []
    for key, group in out.groupby(GROUP, sort=False):
        observed = group["DEMAND"].dropna()
        initial = float(observed.head(6).mean() * multiplier) if not observed.empty else 0.0
        rows.append({GROUP: key, "INITIAL_INVENTORY": initial})
    initial = pd.DataFrame(rows)
    return out.merge(initial, on=GROUP, how="left", validate="many_to_one")


def simulate_inventory(df, target_stock_multiplier=1.5, reorder_quantile=0.90,
                       lead_time_delay_months=0):
    """Periodic-review inventory simulation in each medicine's native unit.

    Missing observations are not treated as zero consumption. Unmet demand is
    recorded as shortage and is not carried as backorder in this baseline model.
    """
    if not 0 < reorder_quantile < 1:
        raise ValueError("reorder_quantile must be between 0 and 1")

    out = df.sort_values([GROUP, "DATE"]).copy()
    frames = []

    for _, group in out.groupby(GROUP, sort=False):
        group = group.copy().reset_index(drop=True)
        demand = pd.to_numeric(group["DEMAND_SCENARIO"], errors="coerce").to_numpy(float)
        n = len(group)
        lead = max(1, int(group["LEAD_TIME_MONTHS"].iloc[0]) + int(lead_time_delay_months))
        inventory = np.zeros(n)
        incoming = np.zeros(n)
        orders = np.zeros(n)
        reorder_points = np.zeros(n)
        available = np.zeros(n)
        shortages = np.zeros(n)
        stockouts = np.zeros(n, dtype="int8")
        forecasts = np.full(n, np.nan)
        pending = {}

        initial = group["INITIAL_INVENTORY"].iloc[0]
        current = float(initial) if pd.notna(initial) else 0.0
        observed_history = []

        for t in range(n):
            incoming[t] = pending.pop(t, 0.0)
            available[t] = current + incoming[t]
            observed = demand[t]

            if np.isfinite(observed):
                forecast = float(np.mean(observed_history[-6:])) if observed_history else max(observed, 0.0)
                forecasts[t] = forecast
                history_std = float(np.std(observed_history[-6:])) if len(observed_history) >= 2 else 0.0
                safety_stock = max(0.0, 1.645 * history_std)
                reorder_points[t] = lead * forecast + safety_stock

                shortages[t] = max(0.0, observed - available[t])
                stockouts[t] = int(shortages[t] > 0)
                current = max(0.0, available[t] - observed)
                observed_history.append(float(observed))

                outstanding = sum(pending.values())
                inventory_position = current + outstanding
                target = (lead + target_stock_multiplier) * max(forecast, 0.0)
                if inventory_position <= reorder_points[t]:
                    qty = max(0.0, target - inventory_position)
                    orders[t] = qty
                    pending[t + lead] = pending.get(t + lead, 0.0) + qty
            else:
                # No report for this month: preserve stock and do not infer zero demand.
                current = available[t]

            inventory[t] = current

        group["FORECAST_DEMAND"] = forecasts
        group["REORDER_POINT"] = reorder_points
        group["ORDER_QUANTITY"] = orders
        group["INCOMING_ORDER"] = incoming
        group["INVENTORY_ON_HAND"] = inventory
        group["AVAILABLE_SUPPLY"] = available
        group["STOCKOUT_EVENT"] = stockouts
        group["SHORTAGE_QUANTITY"] = shortages
        frames.append(group)

    return pd.concat(frames, ignore_index=True) if frames else out
