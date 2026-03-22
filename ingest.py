import json
import os
from pathlib import Path
from typing import List, Tuple

from qdrant_client import QdrantClient
from qdrant_client.http import models as rest
from sentence_transformers import SentenceTransformer

COLLECTION_NAME = "topics"
MODEL_NAME = "BAAI/bge-small-en-v1.5"
VECTOR_SIZE = 384
BATCH_SIZE = 128
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")

def load_topics(json_path: Path) -> List[dict]:
    # Load the JSON file into a list of dicts.
    if not json_path.exists():
        raise FileNotFoundError(f"JSON file not found: {json_path}")
    with json_path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, list):
        raise ValueError("Expected a list of topic records in the JSON file")
    return data

def prepare_records(raw_items: List[dict]) -> Tuple[List[int], List[str], List[dict]]:
    # Validate records and shape them into ids, embedding texts, and payloads.
    ids: List[int] = []
    texts: List[str] = []
    payloads: List[dict] = []
    for idx, item in enumerate(raw_items):
        title = item.get("title")
        chapter = item.get("chapter")
        topic_id = (item.get("_id") or {}).get("$oid")
        if not (title and chapter and topic_id):
            print(f"Skipping record with missing fields: {item}")
            continue
        embedding_text = f"{chapter} - {title}".strip()
        ids.append(idx)
        texts.append(embedding_text)
        payloads.append({"title": title, "chapter": chapter, "topic_id": topic_id})
    return ids, texts, payloads

def ensure_collection(client: QdrantClient) -> None:
    # Reset the collection schema to the expected vector size/distance without using deprecated recreate.
    if client.collection_exists(collection_name=COLLECTION_NAME):
        client.delete_collection(collection_name=COLLECTION_NAME)
    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=rest.VectorParams(size=VECTOR_SIZE, distance=rest.Distance.COSINE),
    )
    print(f"Collection '{COLLECTION_NAME}' is ready.")

def batch(iterable: List, size: int):
    for start in range(0, len(iterable), size):
        end = start + size
        yield start, iterable[start:end]

def main() -> None:
    json_path = Path(__file__).parent / "etutor2.topics.json"
    topics_raw = load_topics(json_path)
    if not topics_raw:
        print("No topics found in the JSON file; nothing to ingest.")
        return

    ids, texts, payloads = prepare_records(topics_raw)
    if not ids:
        print("No valid topics to ingest after validation.")
        return

    print(f"Loaded {len(ids)} topics from {json_path.name}.")

    client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
    ensure_collection(client)

    model = SentenceTransformer(MODEL_NAME)
    print(f"Loaded embedding model: {MODEL_NAME}")

    for offset, text_batch in batch(texts, BATCH_SIZE):
        id_batch = ids[offset : offset + len(text_batch)]
        payload_batch = payloads[offset : offset + len(text_batch)]
        embeddings = model.encode(text_batch, batch_size=64, normalize_embeddings=True)
        client.upsert(
            collection_name=COLLECTION_NAME,
            points=rest.Batch(
                ids=id_batch,
                vectors=embeddings.tolist(),
                payloads=payload_batch,
            ),
        )
        print(f"Upserted {len(id_batch)} records (total: {offset + len(text_batch)}/{len(ids)}).")

    print("Ingestion complete.")

if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"Ingestion failed: {exc}")
        raise
