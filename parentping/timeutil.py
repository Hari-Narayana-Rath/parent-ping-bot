"""Attendance times are stored as naive UTC; user-facing strings use the display timezone (default IST)."""

from __future__ import annotations

import datetime as dt
import os
from zoneinfo import ZoneInfo


def display_tz() -> ZoneInfo:
    name = os.getenv("PARENTPING_DISPLAY_TZ", "Asia/Kolkata")
    return ZoneInfo(name)


def utc_now_naive() -> dt.datetime:
    """Store in DB (SQLite-friendly naive datetime = UTC instant)."""
    return dt.datetime.now(dt.timezone.utc).replace(tzinfo=None)


def calendar_today_in_display_tz() -> dt.date:
    """'Today' for attendance date boundaries (IST by default)."""
    return dt.datetime.now(dt.timezone.utc).astimezone(display_tz()).date()


def assume_stored_naive_is_utc(value: dt.datetime) -> dt.datetime:
    if value.tzinfo is not None:
        return value.astimezone(dt.timezone.utc)
    return value.replace(tzinfo=dt.timezone.utc)


def format_time_ist(value: dt.datetime | None) -> str:
    if value is None:
        return "N/A"
    utc = assume_stored_naive_is_utc(value)
    local = utc.astimezone(display_tz())
    return local.strftime("%H:%M:%S")


def format_iso_ist(value: dt.datetime | None) -> str | None:
    """ISO-8601 string with offset, for JSON APIs."""
    if value is None:
        return None
    utc = assume_stored_naive_is_utc(value)
    return utc.astimezone(display_tz()).isoformat()


def format_line_ist(value: dt.datetime | None) -> str:
    """Chatbot / UI line: time with explicit zone label."""
    if value is None:
        return "N/A"
    return f"{format_time_ist(value)} IST"
