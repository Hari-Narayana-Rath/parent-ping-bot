from __future__ import annotations

import datetime as dt
import json
import re
from difflib import SequenceMatcher
from pathlib import Path
from typing import Dict, List, Tuple

from sqlalchemy.orm import Session

from parentping.database.models import Attendance, Student


DEFAULT_INTENT_EXAMPLES: Dict[str, List[str]] = {
    "greeting": ["hi", "hello", "hey", "good morning", "good evening"],
    "thanks": ["thanks", "thank you", "ok thanks", "great thanks"],
    "help": ["help", "what can you do", "what can i ask", "available queries"],
    "status": [
        "did my child attend class today",
        "was my child present today",
        "attendance status yesterday",
        "did my child attend yesterday",
    ],
    "entry_time": [
        "what time did my child enter",
        "entry time today",
        "what time did my child arrive",
    ],
    "exit_time": [
        "what time did my child leave",
        "exit time today",
        "what time did my child go out",
    ],
    "in_class": [
        "is my child in class now",
        "is student currently in classroom",
        "is my child still in class",
    ],
    "weekly_summary": ["show this week attendance", "weekly attendance details"],
    "weekly_count": ["weekly attendance count", "how many days this week", "week total present"],
    "weekly_percentage": ["weekly attendance percentage", "week attendance rate"],
    "monthly_summary": ["show this month attendance", "monthly attendance details"],
    "monthly_count": ["month attendance count", "how many days this month"],
    "monthly_percentage": ["monthly attendance percentage", "month attendance rate"],
    "yearly_summary": ["show this year attendance", "year attendance summary"],
    "yearly_percentage": ["year attendance percentage", "year attendance rate"],
    "latest_update": ["latest attendance update", "latest record", "recent attendance"],
    "last_seen": ["last seen", "when was my child last seen", "latest seen time"],
    "overall_summary": ["overall attendance", "all time attendance summary"],
    "roll_number": ["what is roll number", "student roll number"],
    "student_name": ["what is student name", "child name"],
    "parent_email": ["what is parent email", "registered email"],
    "date_query": ["attendance on 2026-03-12", "show date attendance"],
    "date_range_summary": ["attendance from 2026-03-01 to 2026-03-10 summary"],
    "date_range_list": ["attendance from 2026-03-01 to 2026-03-10"],
}


def _format_date(value: dt.date) -> str:
    return value.strftime("%Y-%m-%d")


def _format_time(value: dt.datetime | None) -> str:
    if value is None:
        return "N/A"
    return value.strftime("%H:%M:%S")


def _normalize(text: str) -> str:
    return " ".join(re.sub(r"[^a-z0-9\s:-]", " ", text.lower()).split())


def _load_intent_examples() -> Dict[str, List[str]]:
    merged: Dict[str, List[str]] = {
        intent: list(values) for intent, values in DEFAULT_INTENT_EXAMPLES.items()
    }
    data_path = Path(__file__).with_name("training_data.json")
    if not data_path.exists():
        return merged
    try:
        payload = json.loads(data_path.read_text(encoding="utf-8"))
    except Exception:
        return merged
    if not isinstance(payload, dict):
        return merged
    for intent, values in payload.items():
        if intent not in merged or not isinstance(values, list):
            continue
        cleaned = [v.strip().lower() for v in values if isinstance(v, str) and v.strip()]
        merged[intent].extend(cleaned)
    return merged


INTENT_EXAMPLES = _load_intent_examples()


def _extract_date(text: str) -> dt.date | None:
    match = re.search(r"\b(\d{4}-\d{2}-\d{2})\b", text)
    if not match:
        return None
    try:
        return dt.date.fromisoformat(match.group(1))
    except ValueError:
        return None


def _extract_date_range(text: str) -> Tuple[dt.date, dt.date] | None:
    matches = re.findall(r"\b(\d{4}-\d{2}-\d{2})\b", text)
    if len(matches) < 2:
        return None
    try:
        start = dt.date.fromisoformat(matches[0])
        end = dt.date.fromisoformat(matches[1])
    except ValueError:
        return None
    if start > end:
        start, end = end, start
    return start, end


def _records_between(student_id: int, start: dt.date, end: dt.date, db: Session) -> List[Attendance]:
    return (
        db.query(Attendance)
        .filter(
            Attendance.student_id == student_id,
            Attendance.date >= start,
            Attendance.date <= end,
        )
        .order_by(Attendance.date.asc(), Attendance.time_in.asc())
        .all()
    )


def _attendance_metrics(records: List[Attendance]) -> tuple[int, int, int]:
    total = len(records)
    present = sum(1 for r in records if r.status.lower() == "present")
    absent = sum(1 for r in records if r.status.lower() == "absent")
    return total, present, absent


def _attendance_percentage(records: List[Attendance]) -> float:
    total, present, _ = _attendance_metrics(records)
    if total == 0:
        return 0.0
    return (present / total) * 100.0


def _latest_record(student_id: int, db: Session) -> Attendance | None:
    return (
        db.query(Attendance)
        .filter(Attendance.student_id == student_id)
        .order_by(Attendance.date.desc(), Attendance.time_in.desc())
        .first()
    )


def _period_attendance(records: List[Attendance], title: str) -> str:
    lines = [title]
    if not records:
        lines.append("No attendance records found for this period.")
        return "\n".join(lines)
    for rec in records:
        lines.append(
            f"- {_format_date(rec.date)}: {rec.status} "
            f"(In {_format_time(rec.time_in)}, Out {_format_time(rec.time_out)})"
        )
    return "\n".join(lines)


def _intent_score(text: str, intent: str) -> float:
    scores = [SequenceMatcher(None, text, ex).ratio() for ex in INTENT_EXAMPLES[intent]]
    return max(scores) if scores else 0.0


def _detect_intent(text: str) -> str:
    # Strong literal triggers first
    date_count = len(re.findall(r"\d{4}-\d{2}-\d{2}", text))
    if (
        date_count >= 2
        and (("from " in text and " to " in text) or ("between" in text and " to " in text))
    ):
        if any(k in text for k in ["summary", "count", "total", "percentage", "rate"]):
            return "date_range_summary"
        return "date_range_list"
    if date_count == 1 and re.search(r"\b\d{4}-\d{2}-\d{2}\b", text):
        return "date_query"

    if any(k in text for k in ["roll number", "admission number"]):
        return "roll_number"
    if "parent email" in text:
        return "parent_email"
    if "student name" in text or "child name" in text:
        return "student_name"

    if "in class" in text or "currently" in text:
        return "in_class"
    if any(k in text for k in ["leave", "exit", "go out"]):
        return "exit_time"
    if any(k in text for k in ["enter", "entry", "arrive"]):
        return "entry_time"

    if "week" in text and any(k in text for k in ["percentage", "percent", "rate", "%"]):
        return "weekly_percentage"
    if "week" in text and any(k in text for k in ["count", "how many", "total"]):
        return "weekly_count"
    if "week" in text:
        return "weekly_summary"

    if "month" in text and any(k in text for k in ["percentage", "percent", "rate", "%"]):
        return "monthly_percentage"
    if "month" in text and any(k in text for k in ["count", "how many", "total"]):
        return "monthly_count"
    if "month" in text:
        return "monthly_summary"

    if "year" in text and any(k in text for k in ["percentage", "percent", "rate", "%"]):
        return "yearly_percentage"
    if "year" in text:
        return "yearly_summary"

    if "overall" in text or "all time" in text:
        return "overall_summary"
    if "last seen" in text:
        return "last_seen"
    if any(k in text for k in ["latest", "recent", "last update"]):
        return "latest_update"

    # Fuzzy fallback against broad set
    intents = list(INTENT_EXAMPLES.keys())
    best_intent = max(intents, key=lambda i: _intent_score(text, i))
    if _intent_score(text, best_intent) >= 0.58:
        return best_intent

    return "help"


def handle_chatbot_query(query: str, student_id: int, db: Session) -> str:
    q = _normalize(query)
    student = db.query(Student).filter(Student.id == student_id).first()
    if not student:
        return "Student record not found for the authenticated parent."

    today = dt.date.today()
    yesterday = today - dt.timedelta(days=1)
    explicit_date = _extract_date(q)
    target_day = today
    day_phrase = f"today ({_format_date(today)})"
    if "yesterday" in q:
        target_day = yesterday
        day_phrase = f"yesterday ({_format_date(yesterday)})"
    elif explicit_date is not None:
        target_day = explicit_date
        day_phrase = _format_date(explicit_date)

    intent = _detect_intent(q)

    if intent == "greeting":
        return (
            f"Hi. I can help with {student.name}'s attendance. "
            "Ask about status, entry/exit time, weekly/monthly summary, or date ranges."
        )
    if intent == "thanks":
        return "You're welcome. I am here whenever you need another update."
    if intent == "help":
        return (
            "You can ask: today/yesterday status, entry or exit time, in-class status, "
            "week/month/year summaries, percentages, latest update, specific date "
            "(YYYY-MM-DD), or range like 2026-03-01 to 2026-03-10."
        )

    if intent == "status":
        record = (
            db.query(Attendance)
            .filter(Attendance.student_id == student_id, Attendance.date == target_day)
            .order_by(Attendance.time_in.desc())
            .first()
        )
        if not record:
            return f"No attendance is marked for {day_phrase}."
        return (
            f"Yes, attendance is marked for {day_phrase}. "
            f"Status: {record.status}. In {_format_time(record.time_in)}, Out {_format_time(record.time_out)}."
        )

    if intent == "entry_time":
        record = (
            db.query(Attendance)
            .filter(Attendance.student_id == student_id, Attendance.date == target_day)
            .order_by(Attendance.time_in.desc())
            .first()
        )
        if not record:
            return f"I couldn't find an entry time for {day_phrase}."
        return f"{student.name} entered at {_format_time(record.time_in)} on {day_phrase}."

    if intent == "exit_time":
        record = (
            db.query(Attendance)
            .filter(Attendance.student_id == student_id, Attendance.date == target_day)
            .order_by(Attendance.time_in.desc())
            .first()
        )
        if not record:
            return f"There is no attendance record for {day_phrase}."
        if record.time_out is None:
            return f"No exit time is marked yet for {day_phrase}."
        return f"{student.name} exited at {_format_time(record.time_out)} on {day_phrase}."

    if intent == "in_class":
        record = (
            db.query(Attendance)
            .filter(Attendance.student_id == student_id, Attendance.date == today)
            .order_by(Attendance.time_in.desc())
            .first()
        )
        if not record:
            return "I can't confirm in-class status because there is no attendance entry today."
        if record.time_out is None:
            return f"Yes. {student.name} is currently marked inside class."
        return f"No. {student.name} is not currently marked in class."

    if intent == "roll_number":
        return f"The roll number is {student.roll_number}."
    if intent == "student_name":
        return f"The registered student name is {student.name}."
    if intent == "parent_email":
        return f"The registered parent email is {student.parent_email}."

    if intent == "latest_update":
        record = _latest_record(student_id, db)
        if not record:
            return "No attendance records are available yet."
        return (
            f"Latest update: {_format_date(record.date)} - {record.status}. "
            f"In {_format_time(record.time_in)}, Out {_format_time(record.time_out)}."
        )

    if intent == "last_seen":
        record = _latest_record(student_id, db)
        if not record:
            return "I don't have a last-seen record yet."
        return f"Last seen on {_format_date(record.date)} at {_format_time(record.time_in)}."

    if intent in {"weekly_summary", "weekly_count", "weekly_percentage"}:
        start = today - dt.timedelta(days=today.weekday())
        records = _records_between(student_id, start, today, db)
        if intent == "weekly_count":
            total, present, absent = _attendance_metrics(records)
            return f"This week: total {total}, present {present}, absent {absent}."
        if intent == "weekly_percentage":
            return f"This week attendance percentage is {_attendance_percentage(records):.1f}%."
        return _period_attendance(records, "Here's this week's attendance:")

    if intent in {"monthly_summary", "monthly_count", "monthly_percentage"}:
        start = today.replace(day=1)
        records = _records_between(student_id, start, today, db)
        if intent == "monthly_count":
            total, present, absent = _attendance_metrics(records)
            return f"This month: total {total}, present {present}, absent {absent}."
        if intent == "monthly_percentage":
            return f"This month attendance percentage is {_attendance_percentage(records):.1f}%."
        return _period_attendance(records, "Here's this month's attendance:")

    if intent in {"yearly_summary", "yearly_percentage"}:
        start = today.replace(month=1, day=1)
        records = _records_between(student_id, start, today, db)
        if intent == "yearly_percentage":
            return f"This year attendance percentage is {_attendance_percentage(records):.1f}%."
        total, present, absent = _attendance_metrics(records)
        return f"This year so far: total {total}, present {present}, absent {absent}."

    if intent == "overall_summary":
        all_records = (
            db.query(Attendance)
            .filter(Attendance.student_id == student_id)
            .order_by(Attendance.date.asc())
            .all()
        )
        total, present, absent = _attendance_metrics(all_records)
        return (
            f"Overall attendance: total {total}, present {present}, absent {absent}, "
            f"attendance {_attendance_percentage(all_records):.1f}%."
        )

    if intent in {"date_range_summary", "date_range_list"}:
        date_range = _extract_date_range(q)
        if not date_range:
            return "Please provide a valid range in YYYY-MM-DD format."
        start, end = date_range
        records = _records_between(student_id, start, end, db)
        if intent == "date_range_summary":
            total, present, absent = _attendance_metrics(records)
            return (
                f"From {_format_date(start)} to {_format_date(end)}: total {total}, "
                f"present {present}, absent {absent}, attendance {_attendance_percentage(records):.1f}%."
            )
        return _period_attendance(records, f"Attendance from {_format_date(start)} to {_format_date(end)}:")

    if intent == "date_query":
        target = explicit_date
        if target is None:
            return "Please provide date in YYYY-MM-DD format."
        record = (
            db.query(Attendance)
            .filter(Attendance.student_id == student_id, Attendance.date == target)
            .order_by(Attendance.time_in.desc())
            .first()
        )
        if not record:
            return f"I couldn't find attendance on {_format_date(target)}."
        return (
            f"On {_format_date(target)}, status was {record.status}. "
            f"Time in: {_format_time(record.time_in)}, time out: {_format_time(record.time_out)}."
        )

    return (
        "I can answer attendance status, entry/exit times, current classroom status, "
        "summaries, percentages, and date or date-range queries."
    )
