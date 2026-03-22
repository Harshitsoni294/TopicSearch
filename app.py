import os

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from qdrant_client import QdrantClient
from qdrant_client.http import models as rest
from sentence_transformers import SentenceTransformer

COLLECTION_NAME = "topics"
MODEL_NAME = "BAAI/bge-small-en-v1.5"
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")

app = FastAPI(title="Topic Search API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
model = SentenceTransformer(MODEL_NAME)
# Shared client and model instances to avoid reloading per request.

class SearchRequest(BaseModel):
    query: str

@app.post("/search-topic", response_model=dict)
async def search_topic(request: SearchRequest):
    # Normalize the query before embedding to keep inputs consistent.
    normalized_query = request.query.strip().lower()
    if not normalized_query:
        raise HTTPException(status_code=400, detail="Query must not be empty")

    try:
        embedding = model.encode([normalized_query], normalize_embeddings=True)[0].tolist()
    except Exception as exc:  # pragma: no cover - defensive
        raise HTTPException(status_code=500, detail=f"Failed to embed query: {exc}") from exc

    try:
        results = client.query_points(
            collection_name=COLLECTION_NAME,
            query=embedding,
            limit=3,
            with_payload=True,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Search failed: {exc}") from exc

    if not results.points:
        return {"result": None}

    best = results.points[0]
    payload = best.payload or {}
    response = {
        "title": payload.get("title"),
        "chapter": payload.get("chapter"),
        "topic_id": payload.get("topic_id"),
        "score": float(best.score) if best.score is not None else None,
    }
    return response
