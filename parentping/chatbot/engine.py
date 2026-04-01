from __future__ import annotations

import datetime as dt
import json
import re
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path

from sqlalchemy.orm import Session

from parentping.database.models import Attendance, Student


DEFAULT_INTENT_EXAMPLES: dict[str, list[str]] = {
    "greeting": ["hi", "hello", "hey"],
    "thanks": ["thanks", "thank you"],
    "help": ["help", "what can you do", "what can i ask"],
    "status": ["did my child attend class today", "attendance status"],
    "entry_time": ["what time did my child enter", "entry time"],
    "exit_time": ["what time did my child leave", "exit time"],
    "in_class": ["is my child in class now", "currently in class"],
    "weekly_summary": ["show this week attendance", "weekly attendance details"],
    "weekly_count": ["weekly attendance count", "how many days this week"],
    "weekly_percentage": ["weekly attendance percentage", "week attendance rate"],
    "monthly_summary": ["show this month attendance", "monthly attendance details"],
    "monthly_count": ["month attendance count", "how many days this month"],
    "monthly_percentage": ["monthly attendance percentage", "month attendance rate"],
    "yearly_summary": ["show this year attendance", "year attendance summary"],
    "yearly_percentage": ["year attendance percentage", "year attendance rate"],
    "latest_update": ["latest attendance update", "latest record", "recent attendance"],
    "last_seen": ["last seen", "latest seen time"],
    "overall_summary": ["overall attendance", "all time attendance summary"],
    "roll_number": ["what is roll number", "student roll number"],
    "student_name": ["what is student name", "child name"],
    "parent_email": ["what is parent email", "registered email"],
    "date_query": ["attendance on 2026-03-12", "show date attendance"],
    "date_range_summary": ["attendance from 2026-03-01 to 2026-03-10 summary"],
    "date_range_list": ["attendance from 2026-03-01 to 2026-03-10"],
}


@dataclass
class ChatContext:
    student: Student
    today: dt.date
    normalized_query: str
    target_day: dt.date
    day_phrase: str
    explicit_date: dt.date | None
    date_range: tuple[dt.date, dt.date] | None


def _format_date(value: dt.date) -> str:
    return value.strftime("%Y-%m-%d")


def _format_time(value: dt.datetime | None) -> str:
    return value.strftime("%H:%M:%S") if value else "N/A"


def _normalize(text: str) -> str:
    return " ".join(re.sub(r"[^a-z0-9\\s:-]", " ", text.lower()).split())


def _load_examples() -> dict[str, list[str]]:
    merged = {intent: list(values) for intent, values in DEFAULT_INTENT_EXAMPLES.items()}
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
        if intent in merged and isinstance(values, list):
            merged[intent].extend(v.strip().lower() for v in values if isinstance(v, str) and v.strip())
    return merged


INTENT_EXAMPLES = _load_examples()


def _extract_date(text: str) -> dt.date | None:
    match = re.search(r"\b(\d{4}-\d{2}-\d{2})\b", text)
    if not match:
        return None
    try:
        return dt.date.fromisoformat(match.group(1))
    except ValueError:
        return None


def _extract_range(text: str) -> tuple[dt.date, dt.date] | None:
    matches = re.findall(r"\b(\d{4}-\d{2}-\d{2})\b", text)
    if len(matches) < 2:
        return None
    try:
        start = dt.date.fromisoformat(matches[0])
        end = dt.date.fromisoformat(matches[1])
    except ValueError:
        return None
    return (start, end) if start <= end else (end, start)


def _record_for_day(student_id: int, day: dt.date, db: Session) -> Attendance | None:
    return (
        db.query(Attendance)
        .filter(Attendance.student_id == student_id, Attendance.date == day)
        .order_by(Attendance.time_in.desc())
        .first()
    )


def _latest_record(student_id: int, db: Session) -> Attendance | None:
    return (
        db.query(Attendance)
        .filter(Attendance.student_id == student_id)
        .order_by(Attendance.date.desc(), Attendance.time_in.desc())
        .first()
    )


def _records_between(student_id: int, start: dt.date, end: dt.date, db: Session) -> list[Attendance]:
    return (
        db.query(Attendance)
        .filter(Attendance.student_id == student_id, Attendance.date >= start, Attendance.date <= end)
        .order_by(Attendance.date.asc(), Attendance.time_in.asc())
        .all()
    )


def _metrics(records: list[Attendance]) -> tuple[int, int, int]:
    total = len(records)
    present = sum(1 for r in records if r.status.lower() == "present")
    absent = sum(1 for r in records if r.status.lower() == "absent")
    return total, present, absent


def _percentage(records: list[Attendance]) -> float:
    total, present, _ = _metrics(records)
    return 0.0 if total == 0 else (present / total) * 100.0


def _period_details(records: list[Attendance], title: str) -> str:
    if not records:
        return f"{title}\nNo attendance records found for this period."
    lines = [title]
    for rec in records:
        lines.append(
            f"- {_format_date(rec.date)}: {rec.status} "
            f"(In {_format_time(rec.time_in)}, Out {_format_time(rec.time_out)})"
        )
    return "\n".join(lines)


def _intent_score(text: str, intent: str) -> float:
    examples = INTENT_EXAMPLES[intent]
    return max((SequenceMatcher(None, text, ex).ratio() for ex in examples), default=0.0)


def _detect_intent(text: str) -> str:
    date_count = len(re.findall(r"\d{4}-\d{2}-\d{2}", text))
    if date_count >= 2 and (("from " in text and " to " in text) or ("between" in text and " to " in text)):
        if any(k in text for k in ["summary", "count", "total", "percentage", "rate"]):
            return "date_range_summary"
        return "date_range_list"
    if date_count == 1 and re.search(r"\b\d{4}-\d{2}-\d{2}\b", text):
        return "date_query"

    ranked = sorted(INTENT_EXAMPLES.keys(), key=lambda i: _intent_score(text, i), reverse=True)
    best = ranked[0]
    return best if _intent_score(text, best) >= 0.56 else "help"


def _build_context(student: Student, query: str) -> ChatContext:
    normalized = _normalize(query)
    today = dt.date.today()
    yesterday = today - dt.timedelta(days=1)
    explicit_date = _extract_date(normalized)
    target_day = today
    day_phrase = f"today ({_format_date(today)})"
    if "yesterday" in normalized:
        target_day = yesterday
        day_phrase = f"yesterday ({_format_date(yesterday)})"
    elif explicit_date is not None:
        target_day = explicit_date
        day_phrase = _format_date(explicit_date)
    return ChatContext(
        student=student,
        today=today,
        normalized_query=normalized,
        target_day=target_day,
        day_phrase=day_phrase,
        explicit_date=explicit_date,
        date_range=_extract_range(normalized),
    )


def generate_response(query: str, student_id: int, db: Session) -> str:
    student = db.query(Student).filter(Student.id == student_id).first()
    if not student:
        return "Student record not found for the authenticated parent."

    ctx = _build_context(student, query)
    intent = _detect_intent(ctx.normalized_query)

    if intent == "greeting":
        return (
            f"Hi. I can help with {ctx.student.name}'s attendance. "
            "Ask me about attendance status, entry/exit time, in-class state, or summaries."
        )
    if intent == "thanks":
        return "You're welcome. Ask anytime if you need another attendance update."
    if intent == "help":
        return (
            "You can ask: status for today/yesterday, entry or exit time, in-class state, "
            "week/month/year summaries, latest update, specific date (YYYY-MM-DD), "
            "or date range (YYYY-MM-DD to YYYY-MM-DD)."
        )
    if intent == "roll_number":
        return f"The roll number is {ctx.student.roll_number}."
    if intent == "student_name":
        return f"The registered student name is {ctx.student.name}."
    if intent == "parent_email":
        return f"The registered parent email is {ctx.student.parent_email}."

    if intent == "status":
        record = _record_for_day(ctx.student.id, ctx.target_day, db)
        if not record:
            return f"No attendance is marked for {ctx.day_phrase}."
        return (
            f"Attendance is marked for {ctx.day_phrase}. "
            f"Status: {record.status}. In {_format_time(record.time_in)}, Out {_format_time(record.time_out)}."
        )
    if intent == "entry_time":
        record = _record_for_day(ctx.student.id, ctx.target_day, db)
        if not record:
            return f"I couldn't find an entry time for {ctx.day_phrase}."
        return f"{ctx.student.name} entered at {_format_time(record.time_in)} on {ctx.day_phrase}."
    if intent == "exit_time":
        record = _record_for_day(ctx.student.id, ctx.target_day, db)
        if not record:
            return f"There is no attendance record for {ctx.day_phrase}."
        if record.time_out is None:
            return f"No exit time is marked yet for {ctx.day_phrase}."
        return f"{ctx.student.name} exited at {_format_time(record.time_out)} on {ctx.day_phrase}."
    if intent == "in_class":
        record = _record_for_day(ctx.student.id, ctx.today, db)
        if not record:
            return "I can't confirm in-class status because there is no attendance entry today."
        return (
            f"Yes, {ctx.student.name} is currently marked inside class."
            if record.time_out is None
            else f"No, {ctx.student.name} is not currently marked in class."
        )
    if intent == "latest_update":
        record = _latest_record(ctx.student.id, db)
        if not record:
            return "No attendance records are available yet."
        return (
            f"Latest update: {_format_date(record.date)} - {record.status}. "
            f"In {_format_time(record.time_in)}, Out {_format_time(record.time_out)}."
        )
    if intent == "last_seen":
        record = _latest_record(ctx.student.id, db)
        if not record:
            return "I don't have a last-seen record yet."
        return f"Last seen on {_format_date(record.date)} at {_format_time(record.time_in)}."

    if intent in {"weekly_summary", "weekly_count", "weekly_percentage"}:
        start = ctx.today - dt.timedelta(days=ctx.today.weekday())
        records = _records_between(ctx.student.id, start, ctx.today, db)
        if intent == "weekly_count":
            total, present, absent = _metrics(records)
            return f"This week: total {total}, present {present}, absent {absent}."
        if intent == "weekly_percentage":
            return f"This week attendance percentage is {_percentage(records):.1f}%."
        return _period_details(records, "Here is this week's attendance:")

    if intent in {"monthly_summary", "monthly_count", "monthly_percentage"}:
        start = ctx.today.replace(day=1)
        records = _records_between(ctx.student.id, start, ctx.today, db)
        if intent == "monthly_count":
            total, present, absent = _metrics(records)
            return f"This month: total {total}, present {present}, absent {absent}."
        if intent == "monthly_percentage":
            return f"This month attendance percentage is {_percentage(records):.1f}%."
        return _period_details(records, "Here is this month's attendance:")

    if intent in {"yearly_summary", "yearly_percentage"}:
        start = ctx.today.replace(month=1, day=1)
        records = _records_between(ctx.student.id, start, ctx.today, db)
        if intent == "yearly_percentage":
            return f"This year attendance percentage is {_percentage(records):.1f}%."
        total, present, absent = _metrics(records)
        return f"This year so far: total {total}, present {present}, absent {absent}."

    if intent == "overall_summary":
        all_records = (
            db.query(Attendance)
            .filter(Attendance.student_id == ctx.student.id)
            .order_by(Attendance.date.asc())
            .all()
        )
        total, present, absent = _metrics(all_records)
        return (
            f"Overall attendance: total {total}, present {present}, absent {absent}, "
            f"attendance {_percentage(all_records):.1f}%."
        )

    if intent in {"date_range_summary", "date_range_list"}:
        if not ctx.date_range:
            return "Please provide a valid range in YYYY-MM-DD format."
        start, end = ctx.date_range
        records = _records_between(ctx.student.id, start, end, db)
        if intent == "date_range_summary":
            total, present, absent = _metrics(records)
            return (
                f"From {_format_date(start)} to {_format_date(end)}: total {total}, "
                f"present {present}, absent {absent}, attendance {_percentage(records):.1f}%."
            )
        return _period_details(records, f"Attendance from {_format_date(start)} to {_format_date(end)}:")

    if intent == "date_query":
        if ctx.explicit_date is None:
            return "Please provide date in YYYY-MM-DD format."
        record = _record_for_day(ctx.student.id, ctx.explicit_date, db)
        if not record:
            return f"I couldn't find attendance on {_format_date(ctx.explicit_date)}."
        return (
            f"On {_format_date(ctx.explicit_date)}, status was {record.status}. "
            f"Time in: {_format_time(record.time_in)}, time out: {_format_time(record.time_out)}."
        )

    return (
        "I can answer attendance status, entry/exit times, current classroom status, "
        "summaries, percentages, and date or date-range queries."
    )
