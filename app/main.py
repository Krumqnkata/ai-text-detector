"""FastAPI бекенд (Стъпка 6).

    uvicorn app.main:app --reload

С DETECTOR_NO_LM=1 приложението работи без езиковите модели (само стиловия модел).
"""
import sys
import threading
from pathlib import Path

import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from detector.model import MODEL_PATH, Detector  # noqa: E402

STATIC = Path(__file__).parent / "static"
MAX_CHARS = 20_000

app = FastAPI(title="Детектор на ИИ текст", version="0.1")
app.mount("/static", StaticFiles(directory=STATIC), name="static")
detector = Detector.load() if MODEL_PATH.exists() else None
if detector is not None and detector.neural:
    # Езиковите модели се зареждат ~15 s — правим го на заден план, за да не чака първата заявка.
    threading.Thread(target=detector.bino._load, daemon=True).start()


class PredictRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=MAX_CHARS)


@app.get("/")
def index():
    return FileResponse(STATIC / "index.html")


@app.get("/health")
def health():
    return {"model_loaded": detector is not None,
            "metrics": detector.metrics.get("features_lr", {}).get("test") if detector else None}


@app.get("/examples")
def examples():
    """По един човешки и един ИИ текст от всеки жанр в test — моделът не ги е виждал."""
    path = ROOT / "data" / "test.csv"
    if not path.exists():
        return []
    test = pd.read_csv(path)
    genre = test["genre"] if "genre" in test else pd.Series("encyclopedia", index=test.index)
    return [{"label": label, "genre": g, "topic": row.topic, "text": row.text}
            for g in sorted(genre.unique())
            for label, cls in (("human", 0), ("ai", 1))
            for row in test[(test.label == cls) & (genre == g)].head(1).itertuples()]


@app.post("/predict")
def predict(req: PredictRequest):
    if detector is None:
        raise HTTPException(503, "Моделът не е обучен. Пусни scripts/train.py.")
    if len(req.text.split()) < 5:
        raise HTTPException(422, "Текстът е твърде кратък за анализ.")
    return detector.predict(req.text)
