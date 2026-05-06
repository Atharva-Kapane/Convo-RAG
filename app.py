from fastapi import FastAPI
from pydantic import BaseModel

from rag.pipeline import RAGPipeline
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

pipeline = RAGPipeline()


class Query(BaseModel):
    query: str


@app.post("/chat")
def chat(q: Query):
    response = pipeline.run(q.query)
    return {"response": response}