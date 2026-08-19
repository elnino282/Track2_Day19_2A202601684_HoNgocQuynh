"""Minimal hybrid memory POC: Qdrant episodes + Feast user features."""
from __future__ import annotations

import os
import re
from collections import defaultdict, deque
from pathlib import Path
from typing import Any

from qdrant_client import QdrantClient, models
from rank_bm25 import BM25Okapi

from app.embeddings import Embedder

COLLECTION = "bonus_episodic_memory"
FEATURES = [
    "user_profile_features:reading_speed_wpm",
    "user_profile_features:preferred_language",
    "user_profile_features:topic_affinity",
    "query_velocity_features:queries_last_hour",
    "query_velocity_features:distinct_topics_24h",
]
FALLBACK_PROFILE = {
    "reading_speed_wpm": 187,
    "preferred_language": "vi",
    "topic_affinity": "cloud",
    "queries_last_hour": 11,
    "distinct_topics_24h": 4,
}
class HybridMemoryAgent:
    """Combine user-filtered episodic search with online profile features."""

    def __init__(
        self,
        client: QdrantClient | None = None,
        embedder: Embedder | None = None,
        feature_store: Any | None = None,
        top_k: int = 3,
    ) -> None:
        self.client = client or QdrantClient(":memory:")
        self.embedder = embedder or Embedder()
        self.feature_store = feature_store or self._load_feature_store()
        self.top_k = top_k
        self._records: list[dict[str, Any]] = []
        self._recent: dict[str, deque[str]] = defaultdict(lambda: deque(maxlen=20))
        self._next_id = 0
        existing = {item.name for item in self.client.get_collections().collections}
        if COLLECTION not in existing:
            config = models.VectorParams(size=self.embedder.dim, distance=models.Distance.COSINE)
            self.client.create_collection(collection_name=COLLECTION, vectors_config=config)

    @staticmethod
    def _load_feature_store() -> Any | None:
        """Use the lab's materialized Feast store; keep the POC portable."""
        try:
            os.environ.setdefault("POLARS_SKIP_CPU_CHECK", "1")
            from feast import FeatureStore

            repo = Path(__file__).resolve().parents[1] / "app" / "feast_repo"
            return FeatureStore(repo_path=str(repo))
        except Exception:
            return None
    @staticmethod
    def _chunks(text: str, max_words: int = 120) -> list[str]:
        sentences = re.split(r"(?<=[.!?])\s+", " ".join(text.split()))
        chunks: list[str] = []
        current: list[str] = []
        for sentence in sentences:
            if current and len(" ".join(current + [sentence]).split()) > max_words:
                chunks.append(" ".join(current))
                current = []
            current.append(sentence)
        if current:
            chunks.append(" ".join(current))
        return chunks
    def remember(self, text: str, user_id: str = "u_001") -> None:
        """Chunk, embed, and upsert one user's episodic memory."""
        if not text.strip() or not user_id.strip():
            raise ValueError("text and user_id must be non-empty")
        chunks = self._chunks(text)
        vectors = list(self.embedder.embed(chunks))
        points: list[models.PointStruct] = []
        for chunk, vector in zip(chunks, vectors):
            record = {"id": self._next_id, "user_id": user_id, "text": chunk}
            self._records.append(record)
            points.append(models.PointStruct(
                id=self._next_id, vector=vector.tolist(),
                payload={"user_id": user_id, "text": chunk},
            ))
            self._next_id += 1
        self.client.upsert(collection_name=COLLECTION, points=points)
    def _profile(self, user_id: str) -> dict[str, Any]:
        if self.feature_store is None:
            return dict(FALLBACK_PROFILE)
        try:
            raw = self.feature_store.get_online_features(
                features=FEATURES, entity_rows=[{"user_id": user_id}]
            ).to_dict()
            profile = {name: values[0] for name, values in raw.items() if name != "user_id"}
            return {**FALLBACK_PROFILE, **{k: v for k, v in profile.items() if v is not None}}
        except Exception:
            return dict(FALLBACK_PROFILE)
    def _retrieve(self, query: str, user_id: str) -> list[str]:
        records = [record for record in self._records if record["user_id"] == user_id]
        if not records:
            return []
        depth = min(max(self.top_k * 3, 10), len(records))
        bm25 = BM25Okapi([r["text"].lower().split() for r in records])
        kw_scores = bm25.get_scores(query.lower().split())
        # Zero-score documents add arbitrary BM25 ranks and can pollute RRF.
        order = sorted(
            (i for i, score in enumerate(kw_scores) if score > 0),
            key=lambda i: -kw_scores[i],
        )[:depth]
        kw_ids = [records[i]["id"] for i in order]
        query_vector = next(self.embedder.embed([query])).tolist()
        vector_hits = self.client.query_points(
            collection_name=COLLECTION,
            query=query_vector,
            query_filter=models.Filter(
                must=[models.FieldCondition(key="user_id", match=models.MatchValue(value=user_id))]
            ),
            limit=depth,
        ).points
        scores: dict[int, float] = defaultdict(float)
        for ranked_ids in (kw_ids, [int(hit.id) for hit in vector_hits]):
            for rank, memory_id in enumerate(ranked_ids, start=1):
                scores[memory_id] += 1.0 / (60 + rank)
        by_id = {record["id"]: record["text"] for record in records}
        return [by_id[i] for i, _ in sorted(scores.items(), key=lambda x: -x[1])[: self.top_k]]
    def recall(self, query: str, user_id: str = "u_001") -> str:
        """Retrieve memories and profile/activity features; assemble LLM context."""
        if not query.strip() or not user_id.strip():
            raise ValueError("query and user_id must be non-empty")
        profile = self._profile(user_id)
        memories = self._retrieve(query, user_id)
        self._recent[user_id].append(query)
        recent = list(self._recent[user_id])[-3:]
        memory_lines = "\n".join(f"  {i}. {text}" for i, text in enumerate(memories, 1))
        return (
            f"User {user_id}: language={profile['preferred_language']}, "
            f"reading_speed={profile['reading_speed_wpm']} wpm, "
            f"topic_affinity={profile['topic_affinity']}.\n"
            f"Recent activity: queries_last_hour={profile['queries_last_hour']}, "
            f"distinct_topics_24h={profile['distinct_topics_24h']}; "
            f"session_queries={recent}.\n"
            f"Top memories (hybrid RRF, user-filtered):\n{memory_lines or '  (none)'}"
        )
