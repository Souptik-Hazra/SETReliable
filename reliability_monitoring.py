import numpy as np
import pandas as pd


def rolling_mae(residuals, window):
    return pd.Series(np.abs(residuals)).rolling(window, min_periods=1).mean().to_numpy()


def rolling_wape(actual, forecast, window):
    error = pd.Series(np.abs(np.asarray(actual) - np.asarray(forecast)))
    denominator = pd.Series(np.abs(actual)).rolling(window, min_periods=1).sum()
    numerator = error.rolling(window, min_periods=1).sum()
    return (numerator / denominator.replace(0, np.nan)).to_numpy()


def rolling_signed_bias(residuals, window):
    return pd.Series(residuals).rolling(window, min_periods=1).mean().to_numpy()


def simple_cusum(residuals, k=0.5, h=5.0):
    values = np.abs(np.asarray(residuals, dtype=float))
    flags = np.zeros(len(values), dtype=int)
    valid = values[np.isfinite(values)]
    if len(valid) < 5:
        return flags
    baseline = np.median(valid[:max(3, len(valid) // 3)])
    scale = np.std(valid[:max(3, len(valid) // 3)]) or 1.0
    score = 0.0
    for i, value in enumerate(values):
        if not np.isfinite(value):
            continue
        score = max(0.0, score + (value - baseline) / scale - k)
        flags[i] = int(score > h)
    return flags


def simple_adwin_like_detector(values, window=6, threshold=2.0):
    values = np.asarray(values, dtype=float)
    flags = np.zeros(len(values), dtype=int)
    for t in range(2 * window, len(values)):
        recent = values[t-window:t]
        previous = values[t-2*window:t-window]
        if not (np.isfinite(recent).all() and np.isfinite(previous).all()):
            continue
        scale = np.std(np.concatenate([recent, previous]))
        if scale > 0 and abs(recent.mean() - previous.mean()) / scale > threshold:
            flags[t] = 1
    return flags


def rolling_calibration_error(actual, lower_bound, upper_bound, window,
                              nominal_coverage=0.90):
    covered = ((actual >= lower_bound) & (actual <= upper_bound)).astype(float)
    return (pd.Series(covered).rolling(window, min_periods=1).mean()
            - nominal_coverage).to_numpy()


def normalize(values):
    values = np.asarray(values, dtype=float)
    valid = np.isfinite(values)
    result = np.zeros(len(values), dtype=float)
    if not valid.any():
        return result
    low, high = np.nanmin(values), np.nanmax(values)
    if high > low:
        result[valid] = (values[valid] - low) / (high - low)
    return result


def calculate_reliability_score(rolling_mae_values, signed_bias_values,
                                calibration_error_values, drift_flags,
                                weight_mae=0.35, weight_bias=0.25,
                                weight_calibration=0.25, weight_drift=0.15):
    mae = normalize(rolling_mae_values)
    bias = normalize(np.abs(signed_bias_values))
    undercoverage = np.maximum(0, -np.asarray(calibration_error_values))
    calibration = normalize(undercoverage)
    drift = np.nan_to_num(np.asarray(drift_flags, dtype=float))
    return (weight_mae * mae + weight_bias * bias
            + weight_calibration * calibration + weight_drift * drift)


def add_reliability_columns(df, window=6, nominal_coverage=0.90):
    df = df.sort_values(["MEDICINE_HOSPITAL_ID", "DATE"]).copy()
    frames = []
    for _, group in df.groupby("MEDICINE_HOSPITAL_ID", sort=False):
        group = group.copy().reset_index(drop=True)
        actual = group["DEMAND_SCENARIO"].to_numpy(dtype=float)
        # One-step naive forecast: previous observed month only.
        forecast = pd.Series(actual).shift(1).to_numpy()
        residual = actual - forecast
        historical_error = pd.Series(residual).shift(1)
        error_scale = historical_error.abs().rolling(
            window, min_periods=3
        ).quantile(nominal_coverage).to_numpy()
        error_scale = np.nan_to_num(error_scale, nan=0.0)
        lower = forecast - error_scale
        upper = forecast + error_scale

        group["FORECAST_DEMAND"] = forecast
        group["FORECAST_P10"] = lower
        group["FORECAST_P50"] = forecast
        group["FORECAST_P90"] = upper
        group["FORECAST_RESIDUAL"] = residual
        group["ROLLING_MAE"] = rolling_mae(residual, window)
        group["ROLLING_WAPE"] = rolling_wape(actual, forecast, window)
        group["SIGNED_BIAS"] = rolling_signed_bias(residual, window)
        group["CUSUM_DRIFT_FLAG"] = simple_cusum(residual)
        group["ADWIN_LIKE_DRIFT_FLAG"] = simple_adwin_like_detector(
            np.abs(residual), window=window
        )
        group["CALIBRATION_ERROR"] = rolling_calibration_error(
            actual, lower, upper, window, nominal_coverage
        )
        group["RELIABILITY_SCORE"] = calculate_reliability_score(
            group["ROLLING_MAE"].to_numpy(),
            group["SIGNED_BIAS"].to_numpy(),
            group["CALIBRATION_ERROR"].to_numpy(),
            group["CUSUM_DRIFT_FLAG"].to_numpy(),
        )
        group["RISK_ALERT"] = (
            group["RELIABILITY_SCORE"] >= 0.50
        ).astype("int8")
        frames.append(group)
    return pd.concat(frames, ignore_index=True) if frames else df
