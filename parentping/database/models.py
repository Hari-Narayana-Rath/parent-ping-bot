from __future__ import annotations

import datetime as dt
from typing import List

import numpy as np
from sqlalchemy import Date, DateTime, ForeignKey, Integer, LargeBinary, String
from sqlalchemy.orm import Mapped, mapped_column, relationship

from parentping.database.db import Base


EMBEDDING_SIZE = 512


def serialize_embedding(embedding: np.ndarray) -> bytes:
    vector = embedding.astype(np.float32).reshape(-1)
    if vector.size != EMBEDDING_SIZE:
        raise ValueError(f"Embedding must be {EMBEDDING_SIZE}-dimensional.")
    return vector.tobytes()


def deserialize_embedding(blob: bytes) -> np.ndarray:
    vector = np.frombuffer(blob, dtype=np.float32)
    if vector.size != EMBEDDING_SIZE:
        raise ValueError(f"Stored embedding has invalid dimension: {vector.size}")
    return vector


class Student(Base):
    __tablename__ = "students"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    name: Mapped[str] = mapped_column(String(120), nullable=False)
    roll_number: Mapped[str] = mapped_column(String(64), unique=True, nullable=False, index=True)
    parent_email: Mapped[str] = mapped_column(String(255), nullable=False, index=True)
    embedding_vector: Mapped[bytes] = mapped_column(LargeBinary, nullable=False)

    attendance_records: Mapped[List["Attendance"]] = relationship(
        "Attendance", back_populates="student", cascade="all, delete-orphan"
    )
    parent: Mapped["Parent"] = relationship(
        "Parent", back_populates="student", cascade="all, delete-orphan", uselist=False
    )


class Attendance(Base):
    __tablename__ = "attendance"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    student_id: Mapped[int] = mapped_column(ForeignKey("students.id"), nullable=False, index=True)
    date: Mapped[dt.date] = mapped_column(Date, nullable=False, index=True)
    time_in: Mapped[dt.datetime] = mapped_column(DateTime, nullable=False)
    time_out: Mapped[dt.datetime | None] = mapped_column(DateTime, nullable=True)
    status: Mapped[str] = mapped_column(String(32), nullable=False, default="Present")

    student: Mapped[Student] = relationship("Student", back_populates="attendance_records")


class Parent(Base):
    __tablename__ = "parents"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    student_id: Mapped[int] = mapped_column(ForeignKey("students.id"), nullable=False, unique=True, index=True)
    email: Mapped[str] = mapped_column(String(255), nullable=False, unique=True, index=True)
    password_hash: Mapped[str] = mapped_column(String(255), nullable=False)

    student: Mapped[Student] = relationship("Student", back_populates="parent")

