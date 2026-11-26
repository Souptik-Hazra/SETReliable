import numpy as np
import pandas as pd
from config import RISK_ALERT_THRESHOLD

GROUP = "MEDICINE_HOSPITAL_ID"


def _cusum_flags(errors, window=6, threshold=2.0):
    values = np.asarray(errors, dtype=float)
    flags = np.zeros(len(values), dtype="int8")
    history = []
    positive = negative = 0.0
    for i, value in enumerate(values):
        if len(history) >= window and np.isfinite(value):
            baseline = np.asarray(history[-window:], dtype=float)
            center = float(np.mean(baseline))
            scale = float(np.std(baseline, ddof=1)) if len(baseline) > 1 else 0.0
            scale = max(scale, np.finfo(float).eps)
            z = (value - center) / scale
            positive = max(0.0, positive + z - 0.5)
            negative = min(0.0, negative + z + 0.5)
            flags[i] = int(positive > threshold or abs(negative) > threshold)
        if np.isfinite(value):
            history.append(float(value))
    return flags


def _adwin_like_flags(errors, min_window=4, delta=0.01):
    """Lightweight adaptive-window mean-shift heuristic; not the ADWIN algorithm."""
    values = np.asarray(errors, dtype=float)
    flags = np.zeros(len(values), dtype="int8")
    history = []
    for i, value in enumerate(values):
        if len(history) >= 2 * min_window and np.isfinite(value):
            found = False
            for cut in range(min_window, len(history) - min_window + 1):
                left, right = np.asarray(history[:cut]), np.asarray(history[cut:])
                n0, n1 = len(left), len(right)
                pooled = np.sqrt(1 / n0 + 1 / n1)
                bound = np.sqrt(0.5 * np.log(4 / delta)) * pooled
                if abs(left.mean() - right.mean()) > bound * max(
                    np.std(np.r_[left, right]), np.finfo(float).eps
                ):
                    found = True
                    break
            flags[i] = int(found)
        if np.isfinite(value):
            history.append(float(value))
    return flags


def add_reliability_columns(df, target="GROSS_POSITIVE_QUANTITY",
                            window=6, nominal_coverage=0.90):
    """Past-only one-step-naive evaluation with rolling calibration and drift signals."""
    if target not in df:
        raise ValueError(f"Target column not found: {target}")
    if window < 2 or not 0 < nominal_coverage < 1:
        raise ValueError("window must be >= 2 and coverage must be between 0 and 1")
    frames = []
    for _, source in df.sort_values([GROUP, "DATE"]).groupby(GROUP, sort=False):
        g = source.copy().reset_index(drop=True)
        actual = pd.to_numeric(g[target], errors="coerce").to_numpy(float)
        forecast = pd.Series(actual).shift(1).to_numpy()
        residual = actual - forecast
        abs_error = np.where(np.isfinite(residual), np.abs(residual), np.nan)
        signed = np.where(np.isfinite(residual), residual, np.nan)
        prior_abs = pd.Series(abs_error).shift(1)
        radius = prior_abs.rolling(window, min_periods=min(3, window)).quantile(
            nominal_coverage).to_numpy()
        lower, upper = forecast - radius, forecast + radius
        rolling_mae_series = pd.Series(abs_error).rolling(window, min_periods=1).mean()
        rolling_bias_series = pd.Series(signed).rolling(window, min_periods=1).mean()
        rolling_mae = rolling_mae_series.to_numpy()
        rolling_bias = rolling_bias_series.to_numpy()
        err_sum = pd.Series(abs_error).rolling(window, min_periods=1).sum()
        actual_sum = pd.Series(np.where(np.isfinite(actual), np.abs(actual), np.nan)).rolling(
            window, min_periods=1).sum()
        rolling_wape = (err_sum / actual_sum.replace(0, np.nan)).to_numpy()
        valid = np.isfinite(actual) & np.isfinite(lower) & np.isfinite(upper)
        covered = np.where(valid, (actual >= lower) & (actual <= upper), np.nan)
        coverage = pd.Series(covered.astype(float)).rolling(window, min_periods=1).mean().to_numpy()
        cusum = _cusum_flags(abs_error, window=window)
        adwin_like = _adwin_like_flags(abs_error)
        drift = np.maximum(cusum, adwin_like)

        # Scale each row using only earlier errors; never use future performance
        # to normalise historical reliability scores.
        prior_scale = rolling_mae_series.shift(1).expanding(min_periods=1).max().to_numpy()
        prior_scale = np.where(np.isfinite(prior_scale), np.maximum(prior_scale, 1e-12), 1.0)
        available = np.isfinite(rolling_mae) & np.isfinite(rolling_bias)
        mae_component = np.nan_to_num(np.clip(rolling_mae / prior_scale, 0, 1), nan=0.0)
        bias_component = np.nan_to_num(
            np.clip(np.abs(rolling_bias) / prior_scale, 0, 1), nan=0.0
        )
        calibration = np.clip(
            (nominal_coverage - np.nan_to_num(coverage, nan=nominal_coverage))
            / nominal_coverage, 0, 1
        )
        score = np.clip(.35 * mae_component + .25 * bias_component +
                        .25 * calibration + .15 * drift, 0, 1)
        score = np.where(available, score, 0.0)

        g["FORECAST_TARGET"] = target
        g["BASELINE_FORECAST"] = forecast
        g["FORECAST_RESIDUAL"] = residual
        g["PREDICTION_INTERVAL_LOWER"] = lower
        g["PREDICTION_INTERVAL_UPPER"] = upper
        g["ROLLING_MAE"] = rolling_mae
        g["ROLLING_WAPE"] = rolling_wape
        g["ROLLING_BIAS"] = rolling_bias
        g["ROLLING_INTERVAL_COVERAGE"] = coverage
        g["CUSUM_DRIFT_FLAG"] = cusum
        g["ADWIN_LIKE_DRIFT_FLAG"] = adwin_like
        g["DRIFT_FLAG"] = drift
        g["RELIABILITY_SCORE_AVAILABLE"] = available.astype("int8")
        g["RELIABILITY_SCORE"] = score
        g["RISK_ALERT"] = ((score >= RISK_ALERT_THRESHOLD) & available).astype("int8")
        frames.append(g)
    return pd.concat(frames, ignore_index=True) if frames else df.copy()


def evaluate_forecast_metrics(df, target="GROSS_POSITIVE_QUANTITY",
                              forecast_col="BASELINE_FORECAST"):
    actual = pd.to_numeric(df[target], errors="coerce")
    forecast = pd.to_numeric(df[forecast_col], errors="coerce")
    valid = actual.notna() & forecast.notna()
    if not valid.any():
        return {"target": target, "n": 0, "mae": np.nan, "rmse": np.nan,
                "wape": np.nan, "bias": np.nan}
    error = actual[valid] - forecast[valid]
    denominator = actual[valid].abs().sum()
    return {"target": target, "n": int(valid.sum()),
            "mae": float(error.abs().mean()),
            "rmse": float(np.sqrt(np.mean(error ** 2))),
            "wape": float(error.abs().sum() / denominator) if denominator else np.nan,
            "bias": float(error.mean())}
