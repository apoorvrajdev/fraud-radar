"""Unit tests for the alerts service (Phase 3H).

Focus: the cursor codec round-trip and rules-JSON parsing edge
cases. The integration suite covers the full endpoint shape and
the bucket-boundary math against real SQLite — these are the
narrow, fast unit tests that don't need a DB.
"""
from __future__ import annotations

from datetime import datetime, timezone
from decimal import Decimal

import pytest

from app.services.alerts import (
    _parse_rules,
    decode_alert_cursor,
    encode_alert_cursor,
)


# ---------------------------------------------------------------------------
# Cursor codec
# ---------------------------------------------------------------------------


def test_cursor_round_trips() -> None:
    score = Decimal("0.7142")
    ts = datetime(2026, 5, 27, 14, 32, 18, tzinfo=timezone.utc)
    id_ = "70f98b5c-7053-4452-a2c6-52d9bcd44420"

    token = encode_alert_cursor(score, ts, id_)
    out_score, out_ts, out_id = decode_alert_cursor(token)

    assert out_score == score
    assert out_ts == ts
    assert out_id == id_


def test_cursor_preserves_decimal_precision() -> None:
    score = Decimal("0.00021934561664238572")
    ts = datetime(2026, 5, 27, 0, 0, 0, tzinfo=timezone.utc)
    token = encode_alert_cursor(score, ts, "x")
    assert decode_alert_cursor(token)[0] == score


def test_malformed_cursor_raises_value_error() -> None:
    with pytest.raises(ValueError, match="invalid cursor"):
        decode_alert_cursor("not-a-real-cursor")


def test_cursor_with_truncated_payload_raises_value_error() -> None:
    # Valid base64 but the decoded JSON is missing the score key.
    import base64
    import json
    bad_payload = base64.urlsafe_b64encode(
        json.dumps({"ts": "2026-01-01T00:00:00+00:00", "id": "x"}).encode()
    ).decode().rstrip("=")
    with pytest.raises(ValueError, match="invalid cursor"):
        decode_alert_cursor(bad_payload)


# ---------------------------------------------------------------------------
# Rules parsing
# ---------------------------------------------------------------------------


def test_parse_rules_none_is_empty_list() -> None:
    assert _parse_rules(None) == []


def test_parse_rules_empty_string_is_empty_list() -> None:
    assert _parse_rules("") == []


def test_parse_rules_happy_path() -> None:
    assert _parse_rules('["a", "b"]') == ["a", "b"]


def test_parse_rules_bad_json_is_empty_list_not_fatal() -> None:
    # The queue is a read surface — bad persisted JSON must never
    # break the analyst's shift.
    assert _parse_rules("not-json") == []


def test_parse_rules_wrong_root_type_is_empty_list() -> None:
    assert _parse_rules('"a single string"') == []
    assert _parse_rules('{"key": "val"}') == []


def test_parse_rules_coerces_non_string_items_to_strings() -> None:
    assert _parse_rules('["a", 42, true]') == ["a", "42", "True"]
