"""SQLite database setup and sample hospital data."""
from contextlib import closing
from datetime import date, timedelta
import math
from numbers import Integral
from pathlib import Path
import sqlite3

import numpy as np
import pandas as pd

DB_NAME = Path(__file__).resolve().with_name("inventory.db")


def get_connection():
    return sqlite3.connect(DB_NAME, check_same_thread=False)


def init_db():
    with closing(get_connection()) as conn:
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS inventory (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                medicine_name TEXT NOT NULL,
                category TEXT NOT NULL,
                batch_no TEXT NOT NULL,
                stock_qty INTEGER NOT NULL,
                min_threshold INTEGER NOT NULL,
                max_capacity INTEGER NOT NULL,
                expiry_date TEXT NOT NULL,
                unit_price REAL NOT NULL
            )
        """)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS usage_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                medicine_name TEXT NOT NULL,
                usage_date TEXT NOT NULL,
                quantity_used INTEGER NOT NULL
            )
        """)

        cursor.execute("SELECT COUNT(*) FROM inventory")
        if cursor.fetchone()[0] == 0:
            today = date.today()
            sample_inventory = [
                ("Amoxicillin 500mg", "Antibiotics", "BATCH-101", 120, 150, 600, (today + timedelta(days=15)).isoformat(), 2.50),
                ("Paracetamol IV", "Analgesics", "BATCH-102", 450, 200, 800, (today + timedelta(days=180)).isoformat(), 1.20),
                ("Insulin Glargine", "Endocrinology", "BATCH-103", 45, 60, 200, (today + timedelta(days=8)).isoformat(), 35.00),
                ("Ceftriaxone 1g", "Antibiotics", "BATCH-104", 580, 100, 400, (today + timedelta(days=90)).isoformat(), 8.75),
                ("Atorvastatin 20mg", "Cardiology", "BATCH-105", 310, 120, 500, (today - timedelta(days=3)).isoformat(), 0.85),
                ("Salbutamol Inhaler", "Respiratory", "BATCH-106", 85, 80, 300, (today + timedelta(days=25)).isoformat(), 6.40),
            ]
            cursor.executemany("""
                INSERT INTO inventory (
                    medicine_name, category, batch_no, stock_qty,
                    min_threshold, max_capacity, expiry_date, unit_price
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, sample_inventory)

        cursor.execute("SELECT COUNT(*) FROM usage_history")
        if cursor.fetchone()[0] == 0:
            today = date.today()
            base_rates = {
                "Amoxicillin 500mg": 12,
                "Paracetamol IV": 25,
                "Insulin Glargine": 5,
                "Ceftriaxone 1g": 8,
                "Atorvastatin 20mg": 14,
                "Salbutamol Inhaler": 7,
            }
            rng = np.random.default_rng(42)
            history_records = []
            for medicine, base_rate in base_rates.items():
                for days_ago in range(90, 0, -1):
                    days_elapsed = 90 - days_ago
                    usage_date = (today - timedelta(days=days_ago)).isoformat()
                    trend = days_elapsed * 0.05
                    seasonality = 3 * np.sin(days_elapsed / 7.0)
                    quantity = max(
                        1,
                        int(base_rate + trend + seasonality + rng.normal(0, 2)),
                    )
                    history_records.append((medicine, usage_date, quantity))

            cursor.executemany("""
                INSERT INTO usage_history (medicine_name, usage_date, quantity_used)
                VALUES (?, ?, ?)
            """, history_records)

        conn.commit()


def fetch_inventory():
    with closing(get_connection()) as conn:
        return pd.read_sql_query("SELECT * FROM inventory", conn)


def fetch_usage_history(medicine_name):
    with closing(get_connection()) as conn:
        return pd.read_sql_query(
            """
            SELECT usage_date, quantity_used
            FROM usage_history
            WHERE medicine_name = ?
            ORDER BY usage_date ASC
            """,
            conn,
            params=(medicine_name,),
        )


def add_medicine(
    medicine_name,
    category,
    batch_no,
    stock_qty,
    min_threshold,
    max_capacity,
    expiry_date,
    unit_price,
):
    if not isinstance(medicine_name, str) or not medicine_name.strip():
        raise ValueError("Medicine name is required.")
    if not isinstance(category, str) or not category.strip():
        raise ValueError("Category is required.")
    if not isinstance(batch_no, str) or not batch_no.strip():
        raise ValueError("Batch number is required.")

    quantities = (stock_qty, min_threshold, max_capacity)
    if any(
        isinstance(value, bool) or not isinstance(value, Integral)
        for value in quantities
    ):
        raise ValueError("Stock quantities and thresholds must be whole numbers.")
    if stock_qty < 0 or min_threshold < 0 or max_capacity < min_threshold:
        raise ValueError(
            "Stock and threshold values must be non-negative, and maximum capacity "
            "must be at least the minimum threshold."
        )
    try:
        normalized_price = float(unit_price)
    except (TypeError, ValueError) as exc:
        raise ValueError("Unit price must be a positive finite number.") from exc
    if not math.isfinite(normalized_price) or normalized_price <= 0:
        raise ValueError("Unit price must be a positive finite number.")
    try:
        normalized_expiry = date.fromisoformat(str(expiry_date)).isoformat()
    except (TypeError, ValueError) as exc:
        raise ValueError("Expiry date must be a valid date.") from exc

    with closing(get_connection()) as conn:
        conn.execute("""
            INSERT INTO inventory (
                medicine_name, category, batch_no, stock_qty,
                min_threshold, max_capacity, expiry_date, unit_price
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            medicine_name.strip(),
            category.strip(),
            batch_no.strip(),
            int(stock_qty),
            int(min_threshold),
            int(max_capacity),
            normalized_expiry,
            normalized_price,
        ))
        conn.commit()

def dispense_medicine(batch_no, quantity_dispensed):
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT stock_qty FROM inventory WHERE batch_no = ?", (batch_no,))
    row = cursor.fetchone()
    if row:
        current_stock = row[0]
        new_stock = max(0, current_stock - quantity_dispensed)
        cursor.execute("UPDATE inventory SET stock_qty = ? WHERE batch_no = ?", (new_stock, batch_no))
        conn.commit()
    conn.close()