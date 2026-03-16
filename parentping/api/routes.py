from __future__ import annotations

import datetime as dt
import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any, List

import cv2
import numpy as np
from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, status
from fastapi.security import OAuth2PasswordBearer
from jose import JWTError, jwt
from passlib.context import CryptContext
from pydantic import TypeAdapter
from pydantic import BaseModel, EmailStr, Field
from sqlalchemy.orm import Session

from parentping.chatbot.chatbot_logic import handle_chatbot_query
from parentping.database.db import get_db
from parentping.database.models import Attendance, Parent, Student, serialize_embedding
from parentping.models.embedding_model import load_embedding_model
from parentping.recognition.embedding_extractor import EmbeddingExtractor
from parentping.recognition.face_detector import FaceDetector


router = APIRouter()
pwd_context = CryptContext(schemes=["pbkdf2_sha256"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/login_parent")

SECRET_KEY = os.getenv("PARENTPING_SECRET_KEY", "change-me-in-production")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60
MODEL_PATH = os.getenv("PARENTPING_MODEL_PATH", "best_resnet18_arcface_parentping.pth")
ADMIN_EMAIL = os.getenv("PARENTPING_ADMIN_EMAIL", "")
ADMIN_PASSWORD = os.getenv("PARENTPING_ADMIN_PASSWORD", "")
_detector: FaceDetector | None = None
_extractor: EmbeddingExtractor | None = None


class RegisterStudentRequest(BaseModel):
    name: str = Field(..., min_length=1, max_length=120)
    roll_number: str = Field(..., min_length=1, max_length=64)
    parent_email: EmailStr
    parent_password: str = Field(..., min_length=6, max_length=128)
    embedding: List[float] = Field(..., min_length=512, max_length=512)


class MarkAttendanceRequest(BaseModel):
    student_id: int


class ParentLoginRequest(BaseModel):
    roll_number: str = Field(..., min_length=1, max_length=64)
    password: str


class AdminLoginRequest(BaseModel):
    email: EmailStr
    password: str


class ResetParentPasswordRequest(BaseModel):
    student_id: int
    new_password: str = Field(..., min_length=6, max_length=128)


class ChatbotQueryRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=500)


def _create_access_token(data: dict, expires_delta: dt.timedelta | None = None) -> str:
    to_encode = data.copy()
    expire = dt.datetime.now(dt.timezone.utc) + (
        expires_delta or dt.timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    )
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)


def _verify_password(plain_password: str, hashed_password: str) -> bool:
    return pwd_context.verify(plain_password, hashed_password)


def _hash_password(password: str) -> str:
    return pwd_context.hash(password)


def _get_current_parent(
    token: str = Depends(oauth2_scheme),
    db: Session = Depends(get_db),
) -> Parent:
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        if payload.get("role") != "parent":
            raise credentials_exception
        parent_id = int(payload.get("sub"))
    except (JWTError, TypeError, ValueError):
        raise credentials_exception

    parent = db.query(Parent).filter(Parent.id == parent_id).first()
    if not parent:
        raise credentials_exception
    return parent


def _get_current_admin(token: str = Depends(oauth2_scheme)) -> dict[str, Any]:
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Admin authentication required",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        if payload.get("role") != "admin":
            raise credentials_exception
    except JWTError:
        raise credentials_exception
    return payload


def _register_student_internal(
    *,
    name: str,
    roll_number: str,
    parent_email: str,
    parent_password: str,
    embedding: np.ndarray,
    db: Session,
):
    if db.query(Student).filter(Student.roll_number == roll_number).first():
        raise HTTPException(status_code=400, detail="Roll number already exists.")
    if db.query(Parent).filter(Parent.email == parent_email).first():
        raise HTTPException(status_code=400, detail="Parent email is already registered.")

    student = Student(
        name=name.strip(),
        roll_number=roll_number.strip(),
        parent_email=parent_email,
        embedding_vector=serialize_embedding(embedding),
    )
    db.add(student)
    db.flush()

    parent = Parent(
        student_id=student.id,
        email=parent_email,
        password_hash=_hash_password(parent_password),
    )
    db.add(parent)
    db.commit()
    db.refresh(student)

    return {
        "message": "Student and parent registered successfully.",
        "student_id": student.id,
        "parent_id": parent.id,
    }


def _student_payload(student: Student) -> dict[str, Any]:
    return {
        "id": student.id,
        "name": student.name,
        "roll_number": student.roll_number,
        "parent_email": student.parent_email,
    }


def _get_recognition_components() -> tuple[FaceDetector, EmbeddingExtractor]:
    global _detector, _extractor
    if _detector is None or _extractor is None:
        model, device = load_embedding_model(MODEL_PATH)
        _detector = FaceDetector(use_retinaface=False)
        _extractor = EmbeddingExtractor(model=model, device=device)
    return _detector, _extractor


def _embedding_from_video(video_path: Path) -> np.ndarray:
    detector, extractor = _get_recognition_components()
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise HTTPException(status_code=400, detail="Uploaded video could not be opened.")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    stride = max(total_frames // 80, 1)
    frame_index = 0
    embeddings: List[np.ndarray] = []

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            if frame_index % stride != 0:
                frame_index += 1
                continue

            detections = detector.detect_faces(frame)
            face = detector.crop_largest_face(frame, detections)
            if face is not None and face.size > 0:
                embeddings.append(extractor.extract(face))
                if len(embeddings) >= 20:
                    break
            frame_index += 1
    finally:
        cap.release()

    if len(embeddings) < 5:
        raise HTTPException(
            status_code=400,
            detail="Not enough clear face frames found in the video. Upload a clearer multi-angle clip.",
        )

    merged = np.mean(np.vstack(embeddings), axis=0).astype(np.float32)
    norm = float(np.linalg.norm(merged))
    if norm == 0.0:
        raise HTTPException(status_code=400, detail="Invalid embedding generated from video.")
    return merged / norm


@router.post("/register_student")
def register_student(
    request: RegisterStudentRequest,
    _: dict[str, Any] = Depends(_get_current_admin),
    db: Session = Depends(get_db),
):
    embedding = np.array(request.embedding, dtype=np.float32)
    return _register_student_internal(
        name=request.name,
        roll_number=request.roll_number,
        parent_email=request.parent_email,
        parent_password=request.parent_password,
        embedding=embedding,
        db=db,
    )


@router.post("/register_student_from_video")
def register_student_from_video(
    name: str = Form(...),
    roll_number: str = Form(...),
    parent_email: str = Form(...),
    parent_password: str = Form(...),
    video: UploadFile = File(...),
    _: dict[str, Any] = Depends(_get_current_admin),
    db: Session = Depends(get_db),
):
    if not video.filename:
        raise HTTPException(status_code=400, detail="Video file is required.")
    if video.content_type and not video.content_type.startswith("video/"):
        raise HTTPException(status_code=400, detail="Uploaded file must be a video.")

    try:
        validated_email = TypeAdapter(EmailStr).validate_python(parent_email)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid parent email.")

    suffix = Path(video.filename).suffix or ".mp4"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
        shutil.copyfileobj(video.file, temp_file)
        temp_path = Path(temp_file.name)

    try:
        embedding = _embedding_from_video(temp_path)
    finally:
        try:
            temp_path.unlink(missing_ok=True)
        except Exception:
            pass

    return _register_student_internal(
        name=name,
        roll_number=roll_number,
        parent_email=str(validated_email),
        parent_password=parent_password,
        embedding=embedding,
        db=db,
    )


@router.post("/mark_attendance")
def mark_attendance(request: MarkAttendanceRequest, db: Session = Depends(get_db)):
    student = db.query(Student).filter(Student.id == request.student_id).first()
    if not student:
        raise HTTPException(status_code=404, detail="Student not found.")

    now = dt.datetime.now()
    today = now.date()
    existing = (
        db.query(Attendance)
        .filter(Attendance.student_id == request.student_id, Attendance.date == today)
        .order_by(Attendance.time_in.desc())
        .first()
    )

    if not existing:
        attendance = Attendance(
            student_id=request.student_id,
            date=today,
            time_in=now,
            time_out=None,
            status="Present",
        )
        db.add(attendance)
        db.commit()
        db.refresh(attendance)
        return {"message": "Attendance marked (time_in).", "attendance_id": attendance.id}

    if existing.time_out is None and (now - existing.time_in).total_seconds() >= 60:
        existing.time_out = now
        db.commit()
        return {"message": "Attendance updated (time_out).", "attendance_id": existing.id}

    return {"message": "Attendance already marked recently.", "attendance_id": existing.id}


@router.get("/attendance/{student_id}")
def get_attendance_history(
    student_id: int,
    parent: Parent = Depends(_get_current_parent),
    db: Session = Depends(get_db),
):
    if parent.student_id != student_id:
        raise HTTPException(status_code=403, detail="Not authorized for this student.")

    records = (
        db.query(Attendance)
        .filter(Attendance.student_id == student_id)
        .order_by(Attendance.date.desc(), Attendance.time_in.desc())
        .all()
    )

    return [
        {
            "id": rec.id,
            "student_id": rec.student_id,
            "date": rec.date.isoformat(),
            "time_in": rec.time_in.isoformat(),
            "time_out": rec.time_out.isoformat() if rec.time_out else None,
            "status": rec.status,
        }
        for rec in records
    ]


@router.post("/login_parent")
def login_parent(request: ParentLoginRequest, db: Session = Depends(get_db)):
    student = db.query(Student).filter(Student.roll_number == request.roll_number.strip()).first()
    if not student:
        raise HTTPException(status_code=401, detail="Invalid roll number or password.")

    parent = db.query(Parent).filter(Parent.student_id == student.id).first()
    if not parent or not _verify_password(request.password, parent.password_hash):
        raise HTTPException(status_code=401, detail="Invalid roll number or password.")

    token = _create_access_token({"sub": str(parent.id), "role": "parent"})
    return {
        "access_token": token,
        "token_type": "bearer",
        "student_id": parent.student_id,
        "student_name": student.name if student else None,
        "roll_number": student.roll_number if student else None,
    }


@router.post("/login_admin")
def login_admin(request: AdminLoginRequest):
    if not ADMIN_EMAIL or not ADMIN_PASSWORD:
        raise HTTPException(status_code=503, detail="Admin credentials are not configured.")
    if request.email != ADMIN_EMAIL or request.password != ADMIN_PASSWORD:
        raise HTTPException(status_code=401, detail="Invalid admin email or password.")

    token = _create_access_token({"sub": ADMIN_EMAIL, "role": "admin"})
    return {
        "access_token": token,
        "token_type": "bearer",
        "admin_email": ADMIN_EMAIL,
    }


@router.get("/admin/students")
def list_students(
    _: dict[str, Any] = Depends(_get_current_admin),
    db: Session = Depends(get_db),
):
    students = db.query(Student).order_by(Student.name.asc()).all()
    return [_student_payload(student) for student in students]


@router.post("/admin/reset_parent_password")
def reset_parent_password(
    request: ResetParentPasswordRequest,
    _: dict[str, Any] = Depends(_get_current_admin),
    db: Session = Depends(get_db),
):
    parent = db.query(Parent).filter(Parent.student_id == request.student_id).first()
    if not parent:
        raise HTTPException(status_code=404, detail="Parent record not found for this student.")
    parent.password_hash = _hash_password(request.new_password)
    db.commit()
    return {"message": "Parent password updated successfully."}


@router.delete("/admin/student/{student_id}")
def delete_student(
    student_id: int,
    _: dict[str, Any] = Depends(_get_current_admin),
    db: Session = Depends(get_db),
):
    student = db.query(Student).filter(Student.id == student_id).first()
    if not student:
        raise HTTPException(status_code=404, detail="Student not found.")
    db.delete(student)
    db.commit()
    return {"message": "Student removed successfully."}


@router.post("/admin/upload_model")
def upload_model_file(
    model_file: UploadFile = File(...),
    _: dict[str, Any] = Depends(_get_current_admin),
):
    if not model_file.filename or not model_file.filename.endswith(".pth"):
        raise HTTPException(status_code=400, detail="Upload a valid .pth model file.")

    destination = Path(MODEL_PATH)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("wb") as output:
        shutil.copyfileobj(model_file.file, output)

    global _detector, _extractor
    _detector = None
    _extractor = None
    return {"message": f"Model uploaded successfully to {destination.name}."}


@router.post("/admin/import_data")
def import_private_data(
    data_file: UploadFile = File(...),
    replace_existing: bool = Form(False),
    _: dict[str, Any] = Depends(_get_current_admin),
    db: Session = Depends(get_db),
):
    if not data_file.filename or not data_file.filename.endswith(".json"):
        raise HTTPException(status_code=400, detail="Upload a valid JSON export file.")

    try:
        payload = json.loads(data_file.file.read().decode("utf-8"))
    except Exception:
        raise HTTPException(status_code=400, detail="Could not parse JSON import file.")

    students_payload = payload.get("students", [])
    if not isinstance(students_payload, list):
        raise HTTPException(status_code=400, detail="Import file is missing a valid students list.")

    if replace_existing:
        db.query(Attendance).delete()
        db.query(Parent).delete()
        db.query(Student).delete()
        db.commit()

    imported = 0
    for item in students_payload:
        if not isinstance(item, dict):
            continue

        embedding = np.array(item.get("embedding", []), dtype=np.float32)
        if embedding.size != 512:
            continue

        existing = db.query(Student).filter(Student.roll_number == item.get("roll_number")).first()
        if existing:
            continue

        result = _register_student_internal(
            name=str(item.get("name", "")).strip(),
            roll_number=str(item.get("roll_number", "")).strip(),
            parent_email=str(item.get("parent_email", "")).strip(),
            parent_password=str(item.get("parent_password", "")).strip(),
            embedding=embedding,
            db=db,
        )
        student_id = int(result["student_id"])

        attendance_items = item.get("attendance", [])
        for attendance in attendance_items:
            if not isinstance(attendance, dict):
                continue
            try:
                record = Attendance(
                    student_id=student_id,
                    date=dt.date.fromisoformat(attendance["date"]),
                    time_in=dt.datetime.fromisoformat(attendance["time_in"]),
                    time_out=dt.datetime.fromisoformat(attendance["time_out"])
                    if attendance.get("time_out")
                    else None,
                    status=str(attendance.get("status", "Present")),
                )
                db.add(record)
            except Exception:
                continue
        db.commit()
        imported += 1

    return {"message": f"Imported {imported} student records."}


@router.post("/chatbot_query")
def chatbot_query(
    request: ChatbotQueryRequest,
    parent: Parent = Depends(_get_current_parent),
    db: Session = Depends(get_db),
):
    response = handle_chatbot_query(request.query, parent.student_id, db)
    return {"response": response}
