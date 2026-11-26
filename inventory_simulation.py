import numpy as np
import pandas as pd


def assign_lead_times(df, seed=42):
    df = df.copy()

    rng = np.random.default_rng(seed)

    series_ids = (
        df["MEDICINE_HOSPITAL_ID"]
        .drop_duplicates()
        .to_numpy()
    )

    lead_time_map = {}

    for series_id in series_ids:
        lead_time_map[series_id] = int(
            rng.choice([1, 2, 3], p=[0.60, 0.30, 0.10])
        )

    df["LEAD_TIME_MONTHS"] = df["MEDICINE_HOSPITAL_ID"].map(
        lead_time_map
    )

    return df


def calculate_initial_inventory(df, multiplier=1.5):
    df = df.copy()

    initial_inventory = (
        df.groupby("MEDICINE_HOSPITAL_ID")["DEMAND"]
        .apply(
            lambda x: (
                multiplier * x.iloc[:6].mean()
                if len(x) >= 6
                else multiplier * x.mean()
            )
        )
        .reset_index(name="INITIAL_INVENTORY")
    )

    df = df.merge(
        initial_inventory,
        on="MEDICINE_HOSPITAL_ID",
        how="left",
    )

    return df


def simulate_inventory(
    df,
    target_stock_multiplier=1.5,
    reorder_quantile=0.90,
    lead_time_delay_months=0,
):
    """
    Simulates a monthly inventory system.

    Demand is taken from DEMAND_SCENARIO.
    Orders are placed when inventory falls below the reorder point.
    Orders arrive after the assigned lead time.
    """

    df = df.copy()
    df = df.sort_values(
        by=["MEDICINE_HOSPITAL_ID", "DATE"]
    )

    output_frames = []

    for series_id, group in df.groupby("MEDICINE_HOSPITAL_ID"):
        group = group.copy().reset_index(drop=True)

        demand = group["DEMAND_SCENARIO"].to_numpy()
        lead_time = int(
            group["LEAD_TIME_MONTHS"].iloc[0]
            + lead_time_delay_months
        )

        n = len(group)

        inventory = np.zeros(n)
        incoming_order = np.zeros(n)
        order_quantity = np.zeros(n)
        reorder_point = np.zeros(n)
        available_supply = np.zeros(n)
        lead_time_demand = np.zeros(n)
        stockout_event = np.zeros(n, dtype=int)
        shortage_quantity = np.zeros(n)

        current_inventory = group["INITIAL_INVENTORY"].iloc[0]

        average_demand = max(demand.mean(), 1.0)

        target_stock = (
            target_stock_multiplier + lead_time
        ) * average_demand

        pending_orders = {}

        for t in range(n):
            # Forecast proxy: rolling mean of previous demand
            if t == 0:
                forecast = demand[t]
            else:
                forecast = demand[:t].mean()

            # Forecast uncertainty proxy
            if t >= 3:
                residual_proxy = np.abs(
                    demand[:t] - np.repeat(
                        np.convolve(
                            demand[:t],
                            np.ones(3) / 3,
                            mode="valid",
                        ),
                        1,
                    )
                )
                uncertainty = max(
                    np.percentile(residual_proxy, reorder_quantile * 100),
                    0.1 * forecast,
                )
            else:
                uncertainty = 0.25 * forecast

            reorder_point[t] = (
                lead_time * forecast
                + uncertainty
            )

            # Place order if needed
            if current_inventory <= reorder_point[t]:
                order_quantity[t] = max(
                    0,
                    target_stock - current_inventory,
                )

                arrival_month = t + lead_time

                if arrival_month < n:
                    pending_orders[arrival_month] = (
                        pending_orders.get(arrival_month, 0)
                        + order_quantity[t]
                    )

            # Receive orders due this month
            incoming_order[t] = pending_orders.get(t, 0)

            # Available supply before current demand
            available_supply[t] = current_inventory + incoming_order[t]

            # Demand over lead-time horizon
            end_index = min(t + lead_time, n)
            lead_time_demand[t] = demand[t:end_index].sum()

            # Stockout determination
            if lead_time_demand[t] > available_supply[t]:
                stockout_event[t] = 1
                shortage_quantity[t] = (
                    lead_time_demand[t] - available_supply[t]
                )

            # Update inventory after current demand
            current_inventory = max(
                0,
                current_inventory
                + incoming_order[t]
                - demand[t],
            )

            inventory[t] = current_inventory

        group["FORECAST_DEMAND"] = forecast
        group["REORDER_POINT"] = reorder_point
        group["ORDER_QUANTITY"] = order_quantity
        group["INCOMING_ORDER"] = incoming_order
        group["INVENTORY_ON_HAND"] = inventory
        group["AVAILABLE_SUPPLY"] = available_supply
        group["LEAD_TIME_DEMAND"] = lead_time_demand
        group["STOCKOUT_EVENT"] = stockout_event
        group["SHORTAGE_QUANTITY"] = shortage_quantity

        output_frames.append(group)

    df = pd.concat(output_frames, ignore_index=True)

    return df