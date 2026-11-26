import numpy as np
import pandas as pd

from config import (
    WEIGHT_MAE, WEIGHT_BIAS, WEIGHT_CALIBRATION, WEIGHT_DRIFT,
    RISK_ALERT_THRESHOLD,
)


def _rolling_mean(values, window):
    return pd.Series(values, dtype="float64").rolling(window, min_periods=1).mean().to_numpy()


def _past_only_normalize(values):
    """Normalize each observation against its expanding history, never future rows."""
    values = np.asarray(values, dtype=float)
    result = np.zeros(len(values), dtype=float)
    history = []
    for i, value in enumerate(values):
        if not np.isfinite(value):
            result[i] = np.nan
            continue
        scale = max(history) if history else value
        result[i] = value / scale if scale > 0 else 0.0
        history.append(float(value))
    return np.clip(np.nan_to_num(result, nan=0.0, posinf=1.0), 0.0, 1.0)


def _cusum_flags(residuals, k=0.5, threshold=5.0):
    errors = np.abs(np.asarray(residuals, dtype=float))
    flags = np.zeros(len(errors), dtype="int8")
    history = []
    score = 0.0
    for i, error in enumerate(errors):
        if not np.isfinite(error):
            continue
        if len(history) >= 6:
            baseline = float(np.mean(history))
            scale = float(np.std(history)) or 1.0
            score = max(0.0, score + (error - baseline) / scale - k)
            flags[i] = int(score > threshold)
        history.append(float(error))
    return flags


def _adwin_like_flags(values, window=6, threshold=2.0):
    values = np.asarray(values, dtype=float)
    flags = np.zeros(len(values), dtype="int8")
    for i in range(2 * window, len(values)):
        previous = values[i - 2 * window:i - window]
        recent = values[i - window:i]
        if not (np.isfinite(previous).all() and np.isfinite(recent).all()):
            continue
        scale = np.std(np.concatenate([previous, recent]))
        if scale > 0 and abs(recent.mean() - previous.mean()) / scale > threshold:
            flags[i] = 1
    return flags


def add_reliability_columns(df, window=6, nominal_coverage=0.90):
    if not 0 < nominal_coverage < 1:
        raise ValueError("nominal_coverage must be between 0 and 1")
    if window < 2:
        raise ValueError("window must be at least 2")

    frames = []
    for _, source in df.sort_values(
        ["MEDICINE_HOSPITAL_ID", "DATE"]
    ).groupby("MEDICINE_HOSPITAL_ID", sort=False):
        group = source.copy().reset_index(drop=True)
        actual = pd.to_numeric(group["DEMAND_SCENARIO"], errors="coerce").to_numpy(float)
        forecast = pd.Series(actual).shift(1).to_numpy()
        residual = actual - forecast

        # Quantile of prior absolute errors only; current/future errors cannot
        # influence the interval issued for this month.
        prior_abs_error = pd.Series(np.abs(residual)).shift(1)
        error_quantile = prior_abs_error.rolling(
            window, min_periods=3
        ).quantile(nominal_coverage).to_numpy()
        error_quantile = np.nan_to_num(error_quantile, nan=np.nan)
        lower = forecast - error_quantile
        upper = forecast + error_quantile

        valid_error = np.isfinite(residual)
        abs_error = np.where(valid_error, np.abs(residual), np.nan)
        rolling_mae = _rolling_mean(abs_error, window)
        rolling_bias = _rolling_mean(np.where(valid_error, residual, np.nan), window)
        actual_abs = pd.Series(np.where(np.isfinite(actual), np.abs(actual), np.nan))
        error_sum = pd.Series(abs_error).rolling(window, min_periods=1).sum()
        demand_sum = actual_abs.rolling(window, min_periods=1).sum()
        rolling_wape = (error_sum / demand_sum.replace(0, np.nan)).to_numpy()

        valid_interval = np.isfinite(actual) & np.isfinite(lower) & np.isfinite(upper)
        covered = np.where(
            valid_interval, ((actual >= lower) & (actual <= upper)).astype(float), np.nan
        )
        coverage = _rolling_mean(covered, window)
        calibration_error = coverage - nominal_coverage

        cusum = _cusum_flags(residual)
        adwin = _adwin_like_flags(np.abs(residual), window=window)
        drift = np.maximum(cusum, adwin)

        mae_component = _past_only_normalize(rolling_mae)
        bias_component = _past_only_normalize(np.abs(rolling_bias))
        calibration_component = _past_only_normalize(
            np.maximum(0.0, -calibration_error)
        )
        score = (
            WEIGHT_MAE * mae_component
            + WEIGHT_BIAS * bias_component
            + WEIGHT_CALIBRATION * calibration_component
            + WEIGHT_DRIFT * drift
        )

        group["RELIABILITY_FORECAST"] = forecast
        group["PREDICTION_INTERVAL_LOWER"] = lower
        group["PREDICTION_INTERVAL_UPPER"] = upper
        group["FORECAST_RESIDUAL"] = residual
        group["ROLLING_MAE"] = rolling_mae
        group["ROLLING_WAPE"] = rolling_wape
        group["SIGNED_BIAS"] = rolling_bias
        group["ROLLING_COVERAGE"] = coverage
        group["CALIBRATION_ERROR"] = calibration_error
        group["CUSUM_DRIFT_FLAG"] = cusum
        group["ADWIN_LIKE_DRIFT_FLAG"] = adwin
        group["DRIFT_FLAG"] = drift
        group["RELIABILITY_SCORE"] = np.clip(score, 0.0, 1.0)
        group["RISK_ALERT"] = (
            group["RELIABILITY_SCORE"] >= RISK_ALERT_THRESHOLD
        ).astype("int8")
        frames.append(group)

    return pd.concat(frames, ignore_index=True) if frames else df.copy()
