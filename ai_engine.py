"""Demand forecasting and stock optimization logic."""
from datetime import timedelta

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_percentage_error


def evaluate_expiry_and_stock(df_inv):
    """Calculate expiry, stock status, and reorder recommendations."""
    required_columns = {
        "expiry_date",
        "stock_qty",
        "min_threshold",
        "max_capacity",
    }
    missing_columns = required_columns.difference(df_inv.columns)
    if missing_columns:
        raise ValueError(f"Inventory data is missing columns: {sorted(missing_columns)}")

    df = df_inv.copy()
    expiry_dates = pd.to_datetime(df["expiry_date"], errors="raise")
    if expiry_dates.isna().any():
        raise ValueError("Inventory contains a missing expiry date.")

    today = pd.Timestamp.today().normalize()
    df["expiry_date_dt"] = expiry_dates
    df["days_to_expiry"] = (expiry_dates - today).dt.days

    def get_expiry_status(days):
        if days < 0:
            return "🔴 Expired"
        if days <= 30:
            return "🟠 Expiring Soon (≤30d)"
        return "🟢 Safe"

    def get_stock_status(row):
        if row["stock_qty"] < row["min_threshold"]:
            return "⚠️ Critical Shortage"
        if row["stock_qty"] > row["max_capacity"]:
            return "📦 Excess Stock"
        return "✅ Optimal"

    df["expiry_status"] = df["days_to_expiry"].apply(get_expiry_status)
    df["stock_status"] = df.apply(get_stock_status, axis=1)
    df["recommended_reorder"] = df.apply(
        lambda row: max(0, (row["min_threshold"] * 2) - row["stock_qty"])
        if row["stock_qty"] < row["min_threshold"]
        else 0,
        axis=1,
    )
    return df


def forecast_demand(df_history, days_ahead=30):
    """Forecast demand and return a time-ordered holdout accuracy estimate."""
    if isinstance(days_ahead, bool) or not isinstance(days_ahead, int) or days_ahead < 0:
        raise ValueError("days_ahead must be a non-negative integer.")

    required_columns = {"usage_date", "quantity_used"}
    missing_columns = required_columns.difference(df_history.columns)
    if missing_columns:
        raise ValueError(f"Usage history is missing columns: {sorted(missing_columns)}")
    if len(df_history) < 2:
        raise ValueError("At least two usage records are required to forecast demand.")

    df = df_history.copy()
    df["usage_date"] = pd.to_datetime(df["usage_date"], errors="raise")
    df["quantity_used"] = pd.to_numeric(df["quantity_used"], errors="raise")
    if df[["usage_date", "quantity_used"]].isna().any().any():
        raise ValueError("Usage history contains missing dates or quantities.")
    if not np.isfinite(df["quantity_used"]).all() or (df["quantity_used"] < 0).any():
        raise ValueError("Usage quantities must be finite, non-negative numbers.")
    df = df.sort_values("usage_date").reset_index(drop=True)

    df["day_index"] = (df["usage_date"] - df["usage_date"].min()).dt.days
    df["day_of_week"] = df["usage_date"].dt.dayofweek
    df["rolling_7d"] = (
        df["quantity_used"].shift(1).rolling(7, min_periods=1).mean().fillna(0)
    )

    feature_columns = ["day_index", "day_of_week", "rolling_7d"]
    features = df[feature_columns]
    targets = df["quantity_used"]
    split = max(1, min(len(df) - 1, int(len(df) * 0.8)))
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(features.iloc[:split], targets.iloc[:split])
    test_predictions = model.predict(features.iloc[split:])
    mape = mean_absolute_percentage_error(targets.iloc[split:], test_predictions)
    accuracy = round(max(0.0, (1 - mape) * 100), 2)

    model.fit(features, targets)
    last_date = df["usage_date"].max()
    last_index = df["day_index"].max()
    recent_average = df["quantity_used"].tail(7).mean()
    future_dates = [
        last_date + timedelta(days=offset)
        for offset in range(1, days_ahead + 1)
    ]
    future = pd.DataFrame({"usage_date": future_dates})
    if days_ahead:
        future_features = pd.DataFrame({
            "day_index": [last_index + offset for offset in range(1, days_ahead + 1)],
            "day_of_week": [day.dayofweek for day in future_dates],
            "rolling_7d": [recent_average] * days_ahead,
        })
        future["predicted_demand"] = np.maximum(
            0, np.round(model.predict(future_features))
        ).astype(int)
    else:
        future["predicted_demand"] = pd.Series(dtype="int64")

    return df, future, accuracy
