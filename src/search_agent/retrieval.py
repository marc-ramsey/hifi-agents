import logging
from typing import List, Dict, Any
import numpy as np

logger = logging.getLogger(__name__)


class RetrievalEngine:
    """Preference mode (memory-first) and parallel mode (memory-only)."""

    def __init__(self, memory_store, ingestion_engine, settings):
        self.store = memory_store
        self.ingestion = ingestion_engine
        self.settings = settings

    async def retrieve(self, query: str, mode: str = "preference",
                       top_k: int = None) -> List[Dict]:
        """
        Retrieve relevant memory chunks.

        preference: query memory → if confidence too low, do live SearXNG search
                    and re-query memory with new results.
        parallel:   just query memory (no live search).
        """
        k = top_k or self.settings.retention.top_k

        # Embed the query
        query_vec = (await self.ingestion.embed([query])).astype(np.float32)

        # Query vector store
        results = self.store.query(query_vec, top_k=k)

        if mode == "preference":
            if not results:
                logger.info("Memory empty — triggering live search")
                live_results = await self.ingestion.fetch(query)
                if live_results:
                    await self.ingestion.ingest(query, live_results)
                    results = self.store.query(query_vec, top_k=k)
            else:
                max_score = max(r["score"] for r in results)
                if max_score < self.settings.retention.similarity_threshold:
                    logger.info(
                        "Memory confidence %.2f below threshold %.2f — triggering live search",
                        max_score, self.settings.retention.similarity_threshold
                    )
                    live_results = await self.ingestion.fetch(query)
                    if live_results:
                        await self.ingestion.ingest(query, live_results)
                        results = self.store.query(query_vec, top_k=k)

        # Sort by score descending
        results.sort(key=lambda r: r["score"], reverse=True)
        return results
