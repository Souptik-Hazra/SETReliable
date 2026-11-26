import numpy as np
import pandas as pd

GROUP = "MEDICINE_HOSPITAL_ID"


def assign_lead_times(df, seed=42):
    """Assign reproducible assumed lead times; no supplier lead-time data is present."""
    out = df.copy()
    rng = np.random.default_rng(seed)
    ids = out[GROUP].drop_duplicates().to_numpy()
    mapping = {key: int(rng.choice([1, 2, 3], p=[0.60, 0.30, 0.10])) for key in ids}
    out["LEAD_TIME_MONTHS"] = out[GROUP].map(mapping).astype("int8")
    out["LEAD_TIME_ASSUMED"] = 1
    return out


def calculate_initial_inventory(df, initial_inventory=0.0):
    """Set an explicit opening balance; never inspect future demand to initialize stock."""
    if initial_inventory < 0:
        raise ValueError("initial_inventory cannot be negative")
    out = df.copy()
    out["INITIAL_INVENTORY"] = float(initial_inventory)
    return out


def simulate_inventory(df, target_stock_multiplier=1.5, reorder_quantile=0.90,
                       lead_time_delay_months=0):
    """Monthly periodic-review simulation, operating in each series' native unit.

    Forecast and lead-time-demand quantile use past observed demand only.
    Missing reports do not imply zero consumption; stock is carried unchanged.
    """
    if not 0 < reorder_quantile < 1:
        raise ValueError("reorder_quantile must be between 0 and 1")
    if target_stock_multiplier < 0 or lead_time_delay_months < 0:
        raise ValueError("Stock multiplier and lead-time delay must be non-negative")

    frames = []
    for _, source in df.sort_values([GROUP, "DATE"]).groupby(GROUP, sort=False):
        group = source.copy().reset_index(drop=True)
        demand = pd.to_numeric(group["DEMAND_SCENARIO"], errors="coerce").to_numpy(float)
        n = len(group)
        lead = max(1, int(group["LEAD_TIME_MONTHS"].iloc[0]) + lead_time_delay_months)
        initial = group["INITIAL_INVENTORY"].iloc[0]
        current = float(initial) if pd.notna(initial) else 0.0
        pending = {}
        history = []

        available = np.zeros(n)
        on_hand = np.zeros(n)
        incoming = np.zeros(n)
        orders = np.zeros(n)
        reorder_points = np.full(n, np.nan)
        forecasts = np.full(n, np.nan)
        shortages = np.zeros(n)
        stockouts = np.zeros(n, dtype="int8")
        order_pipeline = np.zeros(n)

        for t, actual in enumerate(demand):
            incoming[t] = pending.pop(t, 0.0)
            available[t] = current + incoming[t]
            if np.isfinite(actual):
                forecast = float(np.mean(history[-6:])) if history else 0.0
                forecasts[t] = forecast
                lead_demands = [
                    sum(history[max(0, i - lead + 1):i + 1])
                    for i in range(lead - 1, len(history))
                ]
                if lead_demands:
                    reorder_point = float(np.quantile(lead_demands, reorder_quantile))
                else:
                    reorder_point = lead * forecast
                reorder_points[t] = reorder_point

                shortages[t] = max(0.0, actual - available[t])
                stockouts[t] = int(shortages[t] > 0)
                current = max(0.0, available[t] - actual)
                history.append(float(actual))

                outstanding = sum(pending.values())
                inventory_position = current + outstanding
                target = max(reorder_point, (lead + target_stock_multiplier) * forecast)
                if inventory_position <= reorder_point:
                    quantity = max(0.0, target - inventory_position)
                    orders[t] = quantity
                    if quantity:
                        pending[t + lead] = pending.get(t + lead, 0.0) + quantity
            else:
                current = available[t]
            on_hand[t] = current
            order_pipeline[t] = sum(pending.values())

        group["FORECAST_DEMAND"] = forecasts
        group["REORDER_POINT"] = reorder_points
        group["ORDER_QUANTITY"] = orders
        group["INCOMING_ORDER"] = incoming
        group["ORDER_PIPELINE"] = order_pipeline
        group["INVENTORY_ON_HAND"] = on_hand
        group["AVAILABLE_SUPPLY"] = available
        group["STOCKOUT_EVENT"] = stockouts
        group["SHORTAGE_QUANTITY"] = shortages
        frames.append(group)

    return pd.concat(frames, ignore_index=True) if frames else df.copy()
