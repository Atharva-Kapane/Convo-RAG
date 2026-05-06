from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import os

from rag.pipeline import RAGPipeline

# ---------------- APP INIT ----------------
app = FastAPI()

# ---------------- CORS ----------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # safe for demo; tighten later if needed
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------- PIPELINE ----------------
pipeline = None


# ---------------- API MODEL ----------------
class Query(BaseModel):
    query: str


# ---------------- API ROUTE ----------------
@app.post("/chat")
def chat(q: Query):
    global pipeline
    if pipeline is None:
        pipeline = RAGPipeline()
        
    response = pipeline.run(q.query)
    return {"response": response}


# ---------------- FRONTEND SERVING ----------------
FRONTEND_DIR = "frontend"

@app.get("/")
def serve_frontend():
    return FileResponse(os.path.join(FRONTEND_DIR, "index.html"))
