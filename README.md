# ParentPing

ParentPing is a FastAPI + Streamlit attendance system with face-recognition-backed enrollment and a parent chatbot.

## Local run

Backend:

```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

Streamlit UI:

```bash
streamlit run streamlit_app.py
```

## Streamlit Cloud

Deploy the repository with app file:

```text
streamlit_app.py
```

Add this secret in Streamlit Cloud:

```toml
api_base_url = "https://your-backend-url"
```

## Render backend

This repository includes a `render.yaml` for deploying the FastAPI backend on Render.

Important constraints:

- The repository does not include `parentping.db`.
- The repository does not include `best_resnet18_arcface_parentping.pth`.
- A fresh hosted backend will start with an empty database.
- Video-based student registration on the hosted backend requires the model weights to be present on that server.

For a basic backend deploy:

1. Create a new Blueprint or Web Service in Render from this GitHub repo.
2. Deploy using the included `render.yaml`.
3. Copy the public backend URL.
4. Set that URL as `api_base_url` in Streamlit Cloud secrets.

## Environment variables

- `DATABASE_URL`
- `PARENTPING_SECRET_KEY`
- `PARENTPING_API_BASE_URL` for local or custom Streamlit configuration
