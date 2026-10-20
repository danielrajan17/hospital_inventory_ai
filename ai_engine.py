"""Demand forecasting and stock optimization logic."""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_percentage_error

def evaluate_expiry_and_stock(df_inv):
    """Calculates real-time shelf life, stock optimization states, and smart notifications."""
    today = pd.to_datetime(datetime.today().strftime("%Y-%m-%d"))
    df = df_inv.copy()
    df["expiry_date_dt"] = pd.to_datetime(df["expiry_date"])
    df["days_to_expiry"] = (df["expiry_date_dt"] - today).dt.days

    # Expiry Status Classification
    def get_expiry_status(days):
        if days < 0:
            return "🔴 Expired"
        elif days <= 30:
            return "🟠 Expiring Soon (≤30d)"
        return "🟢 Safe"

    # Stock Optimization Classification
    def get_stock_status(row):
        if row["stock_qty"] < row["min_threshold"]:
            return "⚠️ Critical Shortage"
        elif row["stock_qty"] > row["max_capacity"]:
            return "📦 Excess Stock"
        return "✅ Optimal"

    df["expiry_status"] = df["days_to_expiry"].apply(get_expiry_status)
    df["stock_status"] = df.apply(get_stock_status, axis=1)
    df["recommended_reorder"] = df.apply(
        lambda r: max(0, (r["min_threshold"] * 2) - r["stock_qty"]) if r["stock_qty"] < r["min_threshold"] else 0,
        axis=1
    )
    return df

def forecast_demand(df_history, days_ahead=30):
    """Uses Random Forest Regression to predict future medicine demand and model accuracy."""
    df = df_history.copy()
    df["usage_date"] = pd.to_datetime(df["usage_date"])
    df["day_index"] = (df["usage_date"] - df["usage_date"].min()).dt.days
    df["day_of_week"] = df["usage_date"].dt.dayofweek
    df["rolling_7d"] = df["quantity_used"].rolling(7, min_periods=1).mean()

    X = df[["day_index", "day_of_week", "rolling_7d"]]
    y = df["quantity_used"]

    # Train/Test split for accuracy verification
    split = int(len(df) * 0.8)
    X_train, X_test = X.iloc[:split], X.iloc[split:]
    y_train, y_test = y.iloc[:split], y.iloc[split:]

    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    preds_test = model.predict(X_test)
    mape = mean_absolute_percentage_error(y_test, preds_test)
    accuracy = round(max(0.0, (1 - mape) * 100), 2)

    # Fit on full dataset to forecast future days
    model.fit(X, y)
    last_date = df["usage_date"].max()
    last_index = df["day_index"].max()
    recent_avg = df["quantity_used"].tail(7).mean()

    future_dates = [last_date + timedelta(days=i) for i in range(1, days_ahead + 1)]
    future_X = pd.DataFrame({
        "day_index": [last_index + i for i in range(1, days_ahead + 1)],
        "day_of_week": [d.dayofweek for d in future_dates],
        "rolling_7d": [recent_avg] * days_ahead
    })

    future_preds = np.round(model.predict(future_X)).astype(int)
    df_future = pd.DataFrame({
        "usage_date": future_dates,
        "predicted_demand": future_preds
    })

    return df, df_future, accuracy