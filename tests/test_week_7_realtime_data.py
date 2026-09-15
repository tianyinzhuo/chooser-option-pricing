"""Offline parser and fallback tests for the Week 7 daily snapshot module."""
from __future__ import annotations

import unittest
import tempfile
from http.client import RemoteDisconnected
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

from src import week_7_realtime_data as realtime


class TestWeek7RealtimeData(unittest.TestCase):
    def test_remote_disconnect_is_treated_as_a_data_source_error(self) -> None:
        with patch.object(realtime, "urlopen", side_effect=RemoteDisconnected("closed")):
            with self.assertRaises(realtime.DataSourceError):
                realtime.request_bytes("https://example.invalid", 1.0, 0)

    def test_error_message_redacts_api_key(self) -> None:
        message = realtime._safe_error_message(
            realtime.DataSourceError("We detected your API key as SECRET123")
        )
        self.assertNotIn("SECRET123", message)
        self.assertIn("[REDACTED]", message)

    def test_fred_csv_uses_latest_non_missing_value(self) -> None:
        text = "observation_date,VIXCLS\n2026-09-10,17.84\n2026-09-11,15.84\n2026-09-12,.\n"
        self.assertEqual(
            realtime.parse_fred_csv_latest(text, "VIXCLS"),
            {"date": "2026-09-11", "value": 15.84},
        )

    def test_cboe_vix_history_parser(self) -> None:
        text = "DATE,OPEN,HIGH,LOW,CLOSE\n09/10/2026,18,19,17,17.84\n09/11/2026,16,17,15,15.84\n"
        self.assertEqual(
            realtime.parse_cboe_vix_csv_latest(text),
            {"date": "2026-09-11", "value": 15.84},
        )

    def test_treasury_xml_parser(self) -> None:
        text = """<?xml version="1.0"?>
        <feed xmlns:m="http://schemas.microsoft.com/ado/2007/08/dataservices/metadata"
              xmlns:d="http://schemas.microsoft.com/ado/2007/08/dataservices">
          <entry><content><m:properties><d:NEW_DATE>2026-09-10T00:00:00</d:NEW_DATE><d:BC_10YEAR>4.10</d:BC_10YEAR></m:properties></content></entry>
          <entry><content><m:properties><d:NEW_DATE>2026-09-11T00:00:00</d:NEW_DATE><d:BC_10YEAR>4.08</d:BC_10YEAR></m:properties></content></entry>
        </feed>"""
        self.assertEqual(
            realtime.parse_treasury_10y_xml_latest(text),
            {"date": "2026-09-11", "value": 4.08},
        )

    def test_fred_history_html_parser(self) -> None:
        text = "<table><tr><td>2026-09-10:</td><td>17.84</td></tr><tr><td>2026-09-11:</td><td>15.84</td></tr></table>"
        self.assertEqual(
            realtime.parse_fred_history_html_latest(text, "VIXCLS"),
            {"date": "2026-09-11", "value": 15.84},
        )

    def test_primary_fred_request_is_limited_to_ninety_days(self) -> None:
        payload = b"observation_date,DGS10\n2026-09-11,4.08\n"
        reference = datetime(2026, 9, 15, tzinfo=timezone.utc)
        with patch.object(
            realtime,
            "request_bytes_with_curl_fallback",
            return_value=(payload, "urllib"),
        ) as request:
            result = realtime.collect_fred_latest(
                "DGS10", "10-year yield", "percent", 5.0, 0, "now", reference
            )

        requested_url = request.call_args.args[0]
        self.assertIn("cosd=2026-06-17", requested_url)
        self.assertIn("coed=2026-09-15", requested_url)
        self.assertEqual(result["endpoint"], "fredgraph_csv_recent_window")
        self.assertEqual(result["request_method"], "urllib")
        self.assertFalse(result["fallback_used"])

    def test_local_vix_archive_is_explicitly_stale(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            raw = root / "data" / "raw"
            raw.mkdir(parents=True)
            (raw / "vix_daily_2018_2024.csv").write_text(
                "observation_date,VIXCLS\n2024-12-30,17.40\n2024-12-31,17.35\n",
                encoding="utf-8",
            )
            with patch.object(
                realtime,
                "collect_fred_latest",
                side_effect=realtime.DataSourceError("offline"),
            ):
                result = realtime.collect_vix_with_local_archive(
                    1.0, 0, "2026-09-15T00:00:00Z", root
                )
        self.assertEqual(result["status"], "stale")
        self.assertEqual(result["data"]["date"], "2024-12-31")
        self.assertEqual(result["data"]["value"], 17.35)

    def test_same_day_news_success_is_reused(self) -> None:
        previous = {
            "sources": {
                "jpm_news_sentiment": {
                    "status": "error",
                    "last_successful_data": {"article_count": 50},
                    "last_successful_retrieved_at_utc": "2026-09-15T15:12:53Z",
                }
            }
        }
        cached = realtime._same_day_cached_news(
            previous, datetime(2026, 9, 15, 16, 0, tzinfo=timezone.utc)
        )
        self.assertIsNotNone(cached)
        self.assertEqual(cached["status"], "cached")
        self.assertEqual(cached["data"]["article_count"], 50)


if __name__ == "__main__":
    unittest.main()
