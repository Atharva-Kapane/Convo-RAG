from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

from drift.drift_detector import DriftDetector
from rag.pipeline import RAGPipeline
from intent.intent_inference import IntentClassifier

import os


# ==================================================
# APP INIT
# ==================================================

app = FastAPI()


# ==================================================
# CORS
# ==================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ==================================================
# GLOBAL OBJECTS
# ==================================================

pipeline = None

drift_detector = DriftDetector()

intent_classifier = IntentClassifier()


# ==================================================
# REQUEST MODELS
# ==================================================

class Query(BaseModel):
    query: str


class DriftRequest(BaseModel):
    start_day: int
    end_day: int
    selected_user: str


class IntentRequest(BaseModel):
    text: str


# ==================================================
# CHAT API
# ==================================================

@app.post("/chat")
def chat(q: Query):

    global pipeline

    if pipeline is None:
        pipeline = RAGPipeline()

    response = pipeline.run(q.query)

    return {
        "response": response
    }


# ==================================================
# DRIFT API
# ==================================================

@app.post("/drift")
def detect_drift(data: DriftRequest):

    try:

        result = drift_detector.analyze(
            start_day=data.start_day,
            end_day=data.end_day,
            selected_user=data.selected_user
        )

        return result

    except Exception as e:

        return {
            "error": str(e)
        }


# ==================================================
# INTENT API
# ==================================================

@app.post("/intent")
def classify_intent(data: IntentRequest):

    try:

        text = data.text.strip()

        if not text:

            return {
                "error": "Empty text provided"
            }

        result = intent_classifier.predict(text)

        return result

    except Exception as e:

        return {
            "error": str(e)
        }


# ==================================================
# FRONTEND SERVING
# ==================================================

FRONTEND_DIR = "frontend"


@app.get("/")
def serve_frontend():

    return FileResponse(
        os.path.join(
            FRONTEND_DIR,
            "index.html"
        )
    )