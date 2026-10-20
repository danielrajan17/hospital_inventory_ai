from contextlib import closing
import sqlite3
import tempfile
import unittest
from datetime import date, timedelta
from pathlib import Path
from unittest.mock import patch

import pandas as pd

import ai_engine
import database


class DatabaseTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.db_path = Path(self.temp_dir.name) / "test_inventory.db"
        self.database_path = patch.object(database, "DB_NAME", self.db_path)
        self.database_path.start()

    def tearDown(self):
        self.database_path.stop()
        self.temp_dir.cleanup()

    def test_initialization_seeds_data_once(self):
        database.init_db()
        database.init_db()

        self.assertEqual(len(database.fetch_inventory()), 6)
        self.assertEqual(
            len(database.fetch_usage_history("Amoxicillin 500mg")),
            90,
        )

    def test_missing_usage_history_is_seeded_for_existing_inventory(self):
        database.init_db()
        with closing(sqlite3.connect(self.db_path)) as conn:
            conn.execute("DELETE FROM usage_history")
            conn.commit()

        database.init_db()

        self.assertEqual(len(database.fetch_inventory()), 6)
        self.assertEqual(
            len(database.fetch_usage_history("Amoxicillin 500mg")),
            90,
        )

    def test_add_medicine_validates_and_persists_values(self):
        database.init_db()

        with self.assertRaisesRegex(ValueError, "maximum capacity"):
            database.add_medicine(
                "Test Medicine", "Emergency", "TEST-1", 10, 20, 15,
                date.today() + timedelta(days=30), 1.25,
            )

        database.add_medicine(
            " Test Medicine ", "Emergency", " TEST-1 ", 10, 20, 30,
            date.today() + timedelta(days=30), 1.25,
        )
        added = database.fetch_inventory().iloc[-1]

        self.assertEqual(added["medicine_name"], "Test Medicine")
        self.assertEqual(added["batch_no"], "TEST-1")


class AiEngineTests(unittest.TestCase):
    def test_forecast_accepts_two_records_and_zero_horizon(self):
        history = pd.DataFrame({
            "usage_date": ["2026-01-02", "2026-01-01"],
            "quantity_used": [4, 2],
        })

        past, future, accuracy = ai_engine.forecast_demand(history, days_ahead=0)

        self.assertEqual(len(past), 2)
        self.assertTrue(past["usage_date"].is_monotonic_increasing)
        self.assertTrue(future.empty)
        self.assertGreaterEqual(accuracy, 0)

    def test_forecast_rejects_insufficient_history(self):
        with self.assertRaisesRegex(ValueError, "At least two"):
            ai_engine.forecast_demand(
                pd.DataFrame(columns=["usage_date", "quantity_used"])
            )

    def test_forecast_outputs_non_negative_predictions(self):
        history = pd.DataFrame({
            "usage_date": pd.date_range("2026-01-01", periods=20),
            "quantity_used": range(1, 21),
        })

        _, future, _ = ai_engine.forecast_demand(history, days_ahead=3)

        self.assertEqual(len(future), 3)
        self.assertTrue((future["predicted_demand"] >= 0).all())

    def test_inventory_statuses_cover_expired_short_and_excess(self):
        inventory = pd.DataFrame({
            "expiry_date": [
                (date.today() - timedelta(days=1)).isoformat(),
                (date.today() + timedelta(days=60)).isoformat(),
            ],
            "stock_qty": [2, 12],
            "min_threshold": [5, 5],
            "max_capacity": [10, 10],
        })

        evaluated = ai_engine.evaluate_expiry_and_stock(inventory)

        self.assertEqual(evaluated.iloc[0]["expiry_status"], "🔴 Expired")
        self.assertEqual(evaluated.iloc[0]["stock_status"], "⚠️ Critical Shortage")
        self.assertEqual(evaluated.iloc[0]["recommended_reorder"], 8)
        self.assertEqual(evaluated.iloc[1]["stock_status"], "📦 Excess Stock")


if __name__ == "__main__":
    unittest.main()
