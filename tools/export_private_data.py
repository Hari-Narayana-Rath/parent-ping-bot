from __future__ import annotations

import json
from pathlib import Path

from parentping.database.db import SessionLocal
from parentping.database.models import Attendance, Parent, Student, deserialize_embedding


def main() -> None:
    output_path = Path("private_data_export.json")
    db = SessionLocal()
    try:
        students = db.query(Student).order_by(Student.id.asc()).all()
        export_students = []
        for student in students:
            parent = db.query(Parent).filter(Parent.student_id == student.id).first()
            attendance_records = (
                db.query(Attendance)
                .filter(Attendance.student_id == student.id)
                .order_by(Attendance.date.asc(), Attendance.time_in.asc())
                .all()
            )
            export_students.append(
                {
                    "name": student.name,
                    "roll_number": student.roll_number,
                    "parent_email": student.parent_email,
                    "parent_password": "CHANGE_ME_BEFORE_IMPORT",
                    "embedding": deserialize_embedding(student.embedding_vector).tolist(),
                    "attendance": [
                        {
                            "date": record.date.isoformat(),
                            "time_in": record.time_in.isoformat(),
                            "time_out": record.time_out.isoformat() if record.time_out else None,
                            "status": record.status,
                        }
                        for record in attendance_records
                    ],
                    "existing_parent_email": parent.email if parent else None,
                }
            )

        output_path.write_text(json.dumps({"students": export_students}, indent=2), encoding="utf-8")
        print(f"Exported private data to {output_path.resolve()}")
        print("Update each parent_password field before importing into the hosted backend.")
    finally:
        db.close()


if __name__ == "__main__":
    main()
