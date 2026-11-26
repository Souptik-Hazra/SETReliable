import numpy as np
import pandas as pd

from config import RISK_ALERT_THRESHOLD

GROUP = "MEDICINE_HOSPITAL_ID"


def _past_scale(values):
    values = np.asarray(values, dtype=float)
    result = np.zeros(len(values), dtype=float)
    history = []
    for i, value in enumerate(values):
        if np.isfinite(value):
            scale = max(history) if history else abs(value)
            result[i] = min(1.0, abs(value) / scale) if scale > 0 else 0.0
            history.append(abs(float(value)))
    return result


def _drift_flags(residuals, window=6, threshold=2.0):
    errors = np.abs(np.asarray(residuals, dtype=float))
    flags = np.zeros(len(errors), dtype="int8")
    for i in range(2 * window, len(errors)):
        prior = errors[i - 2 * window:i - window]
        recent = errors[i - window:i]
        if np.isfinite(prior).all() and np.isfinite(recent).all():
            scale = np.std(np.concatenate([prior, recent]))
            if scale > 0 and abs(recent.mean() - prior.mean()) / scale > threshold:
                flags[i] = 1
    return flags


def add_reliability_columns(df, target="GROSS_POSITIVE_QUANTITY",
                            window=6, nominal_coverage=0.90):
    """Evaluate a one-month-naive baseline using only prior observations.

    Intervals are formed from prior absolute residuals; the row's error is never
    used to construct its own interval. This is a baseline, not a trained model.
    """
    if target not in df:
        raise ValueError(f"Target column not found: {target}")
    if window < 2 or not 0 < nominal_coverage < 1:
        raise ValueError("window must be >= 2 and coverage must be between 0 and 1")

    frames = []
    for _, source in df.sort_values([GROUP, "DATE"]).groupby(GROUP, sort=False):
        group = source.copy().reset_index(drop=True)
        actual = pd.to_numeric(group[target], errors="coerce").to_numpy(float)
        forecast = pd.Series(actual).shift(1).to_numpy()
        residual = actual - forecast
        prior_errors = pd.Series(np.abs(residual)).shift(1)
        radius = prior_errors.rolling(window, min_periods=3).quantile(nominal_coverage).to_numpy()
        lower = forecast - radius
        upper = forecast + radius
        valid = np.isfinite(actual) & np.isfinite(forecast)
        absolute_error = np.where(valid, np.abs(residual), np.nan)
        signed_error = np.where(valid, residual, np.nan)
        rolling_mae = pd.Series(absolute_error).rolling(window, min_periods=1).mean().to_numpy()
        rolling_bias = pd.Series(signed_error).rolling(window, min_periods=1).mean().to_numpy()
        error_sum = pd.Series(absolute_error).rolling(window, min_periods=1).sum()
        actual_sum = pd.Series(np.where(np.isfinite(actual), np.abs(actual), np.nan)).rolling(
            window, min_periods=1
        ).sum()
        rolling_wape = (error_sum / actual_sum.replace(0, np.nan)).to_numpy()
        interval_valid = np.isfinite(actual) & np.isfinite(lower) & np.isfinite(upper)
        covered = np.where(interval_valid, (actual >= lower) & (actual <= upper), np.nan)
        coverage = pd.Series(covered.astype(float)).rolling(window, min_periods=1).mean().to_numpy()
        drift = _drift_flags(residual, window=window)
        mae_score = _past_scale(rolling_mae)
        bias_score = _past_scale(rolling_bias)
        calibration_score = np.nan_to_num(np.maximum(0, nominal_coverage - coverage), nan=0.0)
        score = np.clip(0.35 * mae_score + 0.25 * bias_score +
                        0.25 * calibration_score + 0.15 * drift, 0, 1)

        group["FORECAST_TARGET"] = target
        group["BASELINE_FORECAST"] = forecast
        group["FORECAST_RESIDUAL"] = residual
        group["PREDICTION_INTERVAL_LOWER"] = lower
        group["PREDICTION_INTERVAL_UPPER"] = upper
        group["ROLLING_MAE"] = rolling_mae
        group["ROLLING_WAPE"] = rolling_wape
        group["ROLLING_BIAS"] = rolling_bias
        group["ROLLING_INTERVAL_COVERAGE"] = coverage
        group["DRIFT_FLAG"] = drift
        group["RELIABILITY_SCORE"] = score
        group["RISK_ALERT"] = (score >= RISK_ALERT_THRESHOLD).astype("int8")
        frames.append(group)
    return pd.concat(frames, ignore_index=True) if frames else df.copy()


def evaluate_forecast_metrics(df, target="GROSS_POSITIVE_QUANTITY",
                              forecast_col="BASELINE_FORECAST"):
    """Return pooled one-step-ahead metrics; no random split or future fitting."""
    actual = pd.to_numeric(df[target], errors="coerce")
    forecast = pd.to_numeric(df[forecast_col], errors="coerce")
    valid = actual.notna() & forecast.notna()
    if not valid.any():
        return {"target": target, "n": 0, "mae": np.nan, "rmse": np.nan,
                "wape": np.nan, "bias": np.nan}
    error = actual[valid] - forecast[valid]
    denominator = actual[valid].abs().sum()
    return {
        "target": target,
        "n": int(valid.sum()),
        "mae": float(error.abs().mean()),
        "rmse": float(np.sqrt(np.mean(np.square(error)))),
        "wape": float(error.abs().sum() / denominator) if denominator else np.nan,
        "bias": float(error.mean()),
    }
