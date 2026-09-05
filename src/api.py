"""API FastAPI légère consommée par le frontend Angular.

Endpoints :

- ``POST /predict``  — deux fichiers (``mlo``, ``cc``) + champs patient →
  prédictions JSON (BI-RADS, densité, manifestations, cas similaires).
- ``POST /report``   — idem + génération du PDF, renvoyé en pièce jointe.
- ``GET  /health``   — sonde de disponibilité.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, Form, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse

from .report_generator import PatientInfo, export_pdf, generate_report, synthesize_with_groq
from .two_view_model import load_model, predict

MODEL_PATH = os.environ.get("MODEL_PATH", "best_two_view_model.pt")
BANK_CSV = os.environ.get("BANK_CSV", "data/processed/combined_hybrid_augmented_weighted.csv")
IMAGE_ROOT = os.environ.get("IMAGE_ROOT")

app = FastAPI(title="Diagnostic Mammographique par IA", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:4200"],
    allow_methods=["*"],
    allow_headers=["*"],
)

_model = None


def get_model():
    """Charge le modèle une seule fois (lazy singleton)."""
    global _model
    if _model is None:
        _model = load_model(MODEL_PATH)
    return _model


def _save_upload(upload: UploadFile) -> str:
    suffix = Path(upload.filename or "img.png").suffix or ".png"
    fd, path = tempfile.mkstemp(suffix=suffix)
    with os.fdopen(fd, "wb") as f:
        f.write(upload.file.read())
    return path


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "model": MODEL_PATH, "model_loaded": _model is not None}


@app.post("/predict")
async def predict_endpoint(
    mlo: UploadFile = File(...),
    cc: UploadFile = File(...),
) -> JSONResponse:
    left = _save_upload(mlo)
    right = _save_upload(cc)
    try:
        prediction = predict(get_model(), left, right)
        payload = {
            k: v for k, v in prediction.__dict__.items() if k != "embedding"
        }
        return JSONResponse(payload)
    finally:
        for p in (left, right):
            Path(p).unlink(missing_ok=True)


@app.post("/report")
async def report_endpoint(
    mlo: UploadFile = File(...),
    cc: UploadFile = File(...),
    patient: str = Form(...),
    age: Optional[int] = Form(None),
    exam_date: Optional[str] = Form(None),
    history: Optional[str] = Form(None),
) -> FileResponse:
    left = _save_upload(mlo)
    right = _save_upload(cc)
    info = PatientInfo(
        name=patient, age=age,
        exam_date=exam_date or PatientInfo(name=patient).exam_date, history=history,
    )
    out_dir = tempfile.mkdtemp()
    try:
        bank = BANK_CSV if Path(BANK_CSV).exists() else None
        pdf_path = generate_report(
            MODEL_PATH, left, right, info,
            bank_csv=bank, image_root=IMAGE_ROOT, out_dir=out_dir,
        )
        return FileResponse(
            pdf_path, media_type="application/pdf", filename=pdf_path.name
        )
    finally:
        for p in (left, right):
            Path(p).unlink(missing_ok=True)
