import numpy as np
import pandas as pd


def rolling_mae(residuals, window):
    return (
        pd.Series(np.abs(residuals))
        .rolling(window, min_periods=1)
        .mean()
        .to_numpy()
    )


def rolling_wape(actual, forecast, window):
    residual = actual - forecast

    numerator = (
        pd.Series(np.abs(residual))
        .rolling(window, min_periods=1)
        .sum()
    )

    denominator = (
        pd.Series(np.abs(actual))
        .rolling(window, min_periods=1)
        .sum()
    )

    wape = numerator / denominator.replace(0, np.nan)

    return wape.to_numpy()


def rolling_signed_bias(residuals, window):
    return (
        pd.Series(residuals)
        .rolling(window, min_periods=1)
        .mean()
        .to_numpy()
    )


def simple_cusum(residuals, k=0.5, h=5.0):
    """
    Simple one-sided CUSUM for increasing absolute forecast errors.
    """

    absolute_residual = np.abs(residuals)

    if len(absolute_residual) < 5:
        return np.zeros(len(absolute_residual), dtype=int)

    baseline = np.median(
        absolute_residual[: max(3, len(absolute_residual) // 3)]
    )

    scale = np.std(absolute_residual[: max(3, len(absolute_residual) // 3)])

    if scale == 0 or np.isnan(scale):
        scale = 1.0

    cusum = 0.0
    flags = []

    for value in absolute_residual:
        standardized = (value - baseline) / scale

        cusum = max(0, cusum + standardized - k)

        flags.append(int(cusum > h))

    return np.array(flags, dtype=int)


def simple_adwin_like_detector(values, window=6, threshold=2.0):
    """
    Lightweight adaptive-window detector.

    Compares the mean of the most recent window with the mean of the
    preceding window. This is a practical baseline implementation, not
    the full original ADWIN algorithm.
    """

    values = np.asarray(values)
    flags = np.zeros(len(values), dtype=int)

    for t in range(2 * window, len(values)):
        recent = values[t - window:t]
        previous = values[t - 2 * window:t - window]

        recent_mean = recent.mean()
        previous_mean = previous.mean()

        pooled_std = np.std(
            np.concatenate([recent, previous])
        )

        if pooled_std == 0 or np.isnan(pooled_std):
            continue

        difference = abs(recent_mean - previous_mean)

        standardized_difference = difference / pooled_std

        if standardized_difference > threshold:
            flags[t] = 1

    return flags


def rolling_calibration_error(
    actual,
    lower_bound,
    upper_bound,
    window,
    nominal_coverage=0.90,
):
    covered = (
        (actual >= lower_bound)
        & (actual <= upper_bound)
    ).astype(float)

    observed_coverage = (
        pd.Series(covered)
        .rolling(window, min_periods=1)
        .mean()
    )

    return (
        observed_coverage - nominal_coverage
    ).to_numpy()


def normalize(values):
    values = np.asarray(values, dtype=float)

    if len(values) == 0:
        return values

    minimum = np.nanmin(values)
    maximum = np.nanmax(values)

    if maximum == minimum:
        return np.zeros_like(values)

    return (values - minimum) / (maximum - minimum)


def calculate_reliability_score(
    rolling_mae_values,
    signed_bias_values,
    calibration_error_values,
    drift_flags,
    weight_mae=0.35,
    weight_bias=0.25,
    weight_calibration=0.25,
    weight_drift=0.15,
):
    mae_component = normalize(rolling_mae_values)
    bias_component = normalize(np.abs(signed_bias_values))

    undercoverage = np.maximum(
        0,
        -np.asarray(calibration_error_values),
    )

    calibration_component = normalize(undercoverage)

    reliability_score = (
        weight_mae * mae_component
        + weight_bias * bias_component
        + weight_calibration * calibration_component
        + weight_drift * drift_flags
    )

    return reliability_score


def add_reliability_columns(df, window=6, nominal_coverage=0.90):
    df = df.copy()
    df = df.sort_values(
        by=["MEDICINE_HOSPITAL_ID", "DATE"]
    )

    output_frames = []

    for series_id, group in df.groupby("MEDICINE_HOSPITAL_ID"):
        group = group.copy().reset_index(drop=True)

        actual = group["DEMAND_SCENARIO"].to_numpy()

        # Baseline forecast: previous-month demand
        forecast = np.roll(actual, 1)
        forecast[0] = actual[0]

        # Baseline prediction interval
        rolling_std = (
            pd.Series(actual)
            .rolling(window, min_periods=2)
            .std()
            .fillna(0)
            .to_numpy()
        )

        lower_bound = forecast - 1.645 * rolling_std
        upper_bound = forecast + 1.645 * rolling_std

        residual = actual - forecast

        group["FORECAST_DEMAND"] = forecast
        group["FORECAST_P10"] = lower_bound
        group["FORECAST_P50"] = forecast
        group["FORECAST_P90"] = upper_bound
        group["FORECAST_RESIDUAL"] = residual

        group["ROLLING_MAE"] = rolling_mae(
            residual,
            window,
        )

        group["ROLLING_WAPE"] = rolling_wape(
            actual,
            forecast,
            window,
        )

        group["SIGNED_BIAS"] = rolling_signed_bias(
            residual,
            window,
        )

        group["CUSUM_DRIFT_FLAG"] = simple_cusum(
            residual
        )

        group["ADWIN_LIKE_DRIFT_FLAG"] = (
            simple_adwin_like_detector(
                np.abs(residual),
                window=window,
            )
        )

        group["CALIBRATION_ERROR"] = (
            rolling_calibration_error(
                actual,
                lower_bound,
                upper_bound,
                window,
                nominal_coverage,
            )
        )

        group["RELIABILITY_SCORE"] = (
            calculate_reliability_score(
                group["ROLLING_MAE"].to_numpy(),
                group["SIGNED_BIAS"].to_numpy(),
                group["CALIBRATION_ERROR"].to_numpy(),
                group["CUSUM_DRIFT_FLAG"].to_numpy(),
            )
        )

        group["RISK_ALERT"] = (
            group["RELIABILITY_SCORE"]
            >= 0.50
        ).astype(int)

        output_frames.append(group)

    df = pd.concat(output_frames, ignore_index=True)

    return df