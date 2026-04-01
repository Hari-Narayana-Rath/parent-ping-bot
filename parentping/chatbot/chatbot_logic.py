from __future__ import annotations

from sqlalchemy.orm import Session

from parentping.chatbot.engine import generate_response


def handle_chatbot_query(query: str, student_id: int, db: Session) -> str:
    return generate_response(query=query, student_id=student_id, db=db)
