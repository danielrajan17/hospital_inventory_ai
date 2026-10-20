"""SQLite database setup and sample hospital data."""
import sqlite3
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

DB_NAME = "inventory.db"

def get_connection():
    return sqlite3.connect(DB_NAME, check_same_thread=False)

def init_db():
    conn = get_connection()
    cursor = conn.cursor()

    # Medicine inventory table
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

    # Historical consumption table for AI forecasting
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS usage_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            medicine_name TEXT NOT NULL,
            usage_date TEXT NOT NULL,
            quantity_used INTEGER NOT NULL
        )
    """)

    # Seed initial data if empty
    cursor.execute("SELECT COUNT(*) FROM inventory")
    if cursor.fetchone()[0] == 0:
        today = datetime.today()
        sample_inventory = [
            ("Amoxicillin 500mg", "Antibiotics", "BATCH-101", 120, 150, 600, (today + timedelta(days=15)).strftime("%Y-%m-%d"), 2.50),
            ("Paracetamol IV", "Analgesics", "BATCH-102", 450, 200, 800, (today + timedelta(days=180)).strftime("%Y-%m-%d"), 1.20),
            ("Insulin Glargine", "Endocrinology", "BATCH-103", 45, 60, 200, (today + timedelta(days=8)).strftime("%Y-%m-%d"), 35.00),
            ("Ceftriaxone 1g", "Antibiotics", "BATCH-104", 580, 100, 400, (today + timedelta(days=90)).strftime("%Y-%m-%d"), 8.75),
            ("Atorvastatin 20mg", "Cardiology", "BATCH-105", 310, 120, 500, (today - timedelta(days=3)).strftime("%Y-%m-%d"), 0.85),
            ("Salbutamol Inhaler", "Respiratory", "BATCH-106", 85, 80, 300, (today + timedelta(days=25)).strftime("%Y-%m-%d"), 6.40)
        ]
        cursor.executemany("""
            INSERT INTO inventory (medicine_name, category, batch_no, stock_qty, min_threshold, max_capacity, expiry_date, unit_price)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, sample_inventory)

        # Generate 90 days of realistic historical usage for AI training
        np.random.seed(42)
        history_records = []
        base_rates = {
            "Amoxicillin 500mg": 12,
            "Paracetamol IV": 25,
            "Insulin Glargine": 5,
            "Ceftriaxone 1g": 8,
            "Atorvastatin 20mg": 14,
            "Salbutamol Inhaler": 7
        }

        for med, base in base_rates.items():
            for d in range(90, 0, -1):
                date_str = (today - timedelta(days=d)).strftime("%Y-%m-%d")
                # Simulate slight upward trend + seasonal wave + noise
                trend = (90 - d) * 0.05
                seasonality = 3 * np.sin((90 - d) / 7.0)
                qty = max(1, int(base + trend + seasonality + np.random.normal(0, 2)))
                history_records.append((med, date_str, qty))

        cursor.executemany("""
            INSERT INTO usage_history (medicine_name, usage_date, quantity_used)
            VALUES (?, ?, ?)
        """, history_records)

    conn.commit()
    conn.close()

def fetch_inventory():
    conn = get_connection()
    df = pd.read_sql_query("SELECT * FROM inventory", conn)
    conn.close()
    return df

def fetch_usage_history(medicine_name):
    conn = get_connection()
    df = pd.read_sql_query(
        "SELECT usage_date, quantity_used FROM usage_history WHERE medicine_name = ? ORDER BY usage_date ASC",
        conn, params=(medicine_name,)
    )
    conn.close()
    return df

def add_medicine(medicine_name, category, batch_no, stock_qty, min_threshold, max_capacity, expiry_date, unit_price):
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO inventory (medicine_name, category, batch_no, stock_qty, min_threshold, max_capacity, expiry_date, unit_price)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    """, (medicine_name, category, batch_no, stock_qty, min_threshold, max_capacity, str(expiry_date), unit_price))
    conn.commit()
    conn.close()