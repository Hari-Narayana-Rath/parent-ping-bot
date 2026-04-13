# ParentPing

ParentPing is a smart classroom attendance system built with FastAPI, Streamlit, PyTorch, OpenCV, and SQLite.

It combines three working parts:

1. A hosted backend API for attendance, authentication, chatbot queries, and student management.
2. Separate Streamlit portals for parents and admin users.
3. A local classroom camera service that performs face recognition and sends live presence updates to the backend.

## Features

- ArcFace-style face embedding inference with a ResNet18 backbone
- Real-time webcam attendance marking
- Multi-frame face confirmation before check-in
- Parent chatbot for attendance queries
- Admin portal for student registration and password reset
- Live classroom presence status in the parent portal
- Automatic student exit when the face leaves the camera view for about 3 seconds
- Render-ready backend and Streamlit-ready frontend apps

## Repository Layout

```text
parent-ping-bot/
├── admin_portal.py
├── camera_portal.py
├── main.py
├── parent_portal.py
├── render.yaml
├── requirements.txt
├── best_resnet18_arcface_parentping.pth
├── parentping.db
└── parentping/
    ├── api/
    │   └── routes.py
    ├── camera/
    │   └── realtime_camera.py
    ├── chatbot/
    │   ├── admin_portal.py
    │   ├── chatbot_logic.py
    │   └── parent_portal.py
    ├── database/
    │   ├── db.py
    │   └── models.py
    ├── models/
    │   └── embedding_model.py
    ├── recognition/
    │   ├── embedding_extractor.py
    │   ├── face_detector.py
    │   └── similarity_matcher.py
    └── timeutil.py
```

## Current Deployment Shape

### Backend API
- Platform: Render
- Expected public URL: `https://parentping-api.onrender.com`

### Parent Portal
- Platform: Streamlit Community Cloud
- Entry file: `parent_portal.py`

### Admin Portal
- Platform: Streamlit Community Cloud
- Entry file: `admin_portal.py`

### Live Camera App
- Runs locally on the classroom PC
- Entry file: `camera_portal.py`
- Do not expect webcam access from Streamlit Cloud

## Core Workflow

### Parent Portal
1. Parent opens the parent portal.
2. Parent logs in using student roll number and password.
3. Parent can ask chatbot questions like:
   - Did my child attend class today?
   - What time did my child enter the classroom?
   - Show this week's attendance.
4. Parent also sees live classroom status.

### Admin Portal
1. Admin logs in using admin email and password.
2. Admin can:
   - register students
   - reset parent passwords
   - remove students
3. Student enrollment from video creates an embedding and stores it in the database.

### Camera App
1. Runs on the classroom laptop/PC.
2. Detects faces from the webcam.
3. Extracts embeddings and matches against stored students.
4. Sends attendance and live presence updates to the Render backend.
5. If a student disappears from the camera view for about 3 seconds, the live status changes to out.

## Tech Stack

- Backend API: FastAPI
- Frontend: Streamlit
- Model inference: PyTorch
- Face detection: OpenCV Haar cascade, optional RetinaFace
- Matching: cosine similarity
- ORM: SQLAlchemy
- Database: SQLite
- Deployment: Render + Streamlit Community Cloud

## Model Details

- Backbone: ResNet18
- Embedding size: 512
- Training concept: ArcFace
- Inference uses only:
  - backbone
  - embedding layer
- Weights file:
  - `best_resnet18_arcface_parentping.pth`

## Environment Variables

### Render API
Set these in Render:

```text
PARENTPING_SECRET_KEY=<long-random-secret>
PARENTPING_ADMIN_EMAIL=<admin-email>
PARENTPING_ADMIN_PASSWORD=<admin-password>
PARENTPING_CAMERA_SECRET=<same-secret-used-by-classroom-camera>
PARENTPING_PRESENCE_TTL_SEC=3
```

Notes:
- `PARENTPING_CAMERA_SECRET` is required for live webcam presence in the parent portal.
- The classroom camera must use the same `PARENTPING_CAMERA_SECRET` value.
- `PARENTPING_PRESENCE_TTL_SEC=3` means the student is treated as out after about 3 seconds without a confirmed face in the webcam view.

### Optional Local Environment
If you want to override defaults locally, you can also set:

```text
PARENTPING_API_BASE_URL=https://parentping-api.onrender.com
PARENTPING_CAMERA_SECRET=<same-secret-as-render>
```

## Local Setup

### 1. Install dependencies

```powershell
pip install -r requirements.txt
```

### 2. Run the backend locally if needed

```powershell
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

### 3. Run the parent portal locally

```powershell
streamlit run parent_portal.py
```

### 4. Run the admin portal locally

```powershell
streamlit run admin_portal.py
```

### 5. Run the classroom camera locally

Recommended command:

```powershell
python -m parentping.camera.realtime_camera --weights "best_resnet18_arcface_parentping.pth" --db "parentping.db" --api "https://parentping-api.onrender.com" --camera-secret "<your-camera-secret>"
```

Important:
- do not use `http://127.0.0.1:8000` if your parent/admin portals are using Render
- use the same camera secret as the Render backend

### 6. Run the camera launcher UI locally

```powershell
streamlit run camera_portal.py
```


## Registering New Students on Render Free Tier

Render free instances have a 512MB memory limit. Processing a face video with PyTorch on Render can crash the service with `HTTP 502` / `Ran out of memory`. For that reason, do not register students from video directly on the hosted API.

Use the local registration tool instead. It extracts the embedding on your PC and sends only the 512D embedding to Render.

```powershell
python tools/register_student_local.py --video "student.mp4" --name "Student Name" --roll-number "ROLL001" --parent-email "parent@example.com" --parent-password "parent_password" --admin-email "your-admin-email"
```

You will be asked for the admin password securely. After it succeeds, refresh the admin portal and the new student will appear.

## Streamlit Deployment

Create two Streamlit apps from the same repository.

### Parent app
- Repository: this repository
- Branch: `main`
- App file: `parent_portal.py`

### Admin app
- Repository: this repository
- Branch: `main`
- App file: `admin_portal.py`

## Render Deployment

This repo includes `render.yaml` for backend deployment.

Steps:
1. Create a new Render Blueprint or Web Service from this repo.
2. Set the required environment variables.
3. Deploy the service.
4. Confirm health endpoint:

```text
https://parentping-api.onrender.com/health
```

Expected response:

```json
{"status":"ok"}
```

## Live Camera Status Logic

The live classroom status shown to parents works like this:

1. The camera service posts currently visible student IDs to the backend.
2. The backend stores the last seen timestamp for each visible student.
3. If a student is not seen again within about 3 seconds, the parent portal shows the student as out of the classroom.
4. If the camera stops sending updates entirely, the parent portal shows camera offline.

## Common Commands

### Parent portal

```powershell
streamlit run parent_portal.py
```

### Admin portal

```powershell
streamlit run admin_portal.py
```

### Camera portal

```powershell
streamlit run camera_portal.py
```

### Direct camera service

```powershell
python -m parentping.camera.realtime_camera --weights "best_resnet18_arcface_parentping.pth" --db "parentping.db" --api "https://parentping-api.onrender.com" --camera-secret "<your-camera-secret>"
```

## Troubleshooting

### Parent portal says camera offline
Check all of these:
1. The classroom camera process is running.
2. The camera is posting to the same API URL as the parent portal.
3. The camera secret matches `PARENTPING_CAMERA_SECRET` on Render.
4. The parent Streamlit app has been rebooted after the latest GitHub update.

### Camera detects a face but parent portal does not update
1. Make sure the camera command is using the Render API URL.
2. Make sure the student exists in the same backend database used by the portals.
3. Make sure the camera secret matches the Render backend secret.
4. Reboot the parent Streamlit app after pushing changes.

### Student does not switch to out quickly enough
1. Confirm Render has `PARENTPING_PRESENCE_TTL_SEC=3`.
2. Confirm the camera is still sending heartbeats while running.
3. Confirm the student actually disappears from the recognized webcam view.

### Webcam window opens but no live updates reach the parent portal
1. Do not use localhost in the camera command if portals are reading from Render.
2. Use:
   - `--api "https://parentping-api.onrender.com"`
3. Use the correct camera secret.

## Notes

- The parent portal and admin portal are cloud-hosted.
- The real-time camera service must run on the classroom PC because webcam access is local hardware access.
- `camera_live.py` is not part of the deployed app flow.

## Latest Fixes Included

- Corrected live camera path to target the hosted backend by default
- Improved live classroom status behavior
- Reduced live presence timeout to about 3 seconds
- Clarified parent portal live-status messaging
- Kept camera status dependent on the same API and secret used by the parent portal
