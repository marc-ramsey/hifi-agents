import asyncio
import logging
from typing import List, Dict, Any
import aiohttp
import numpy as np

logger = logging.getLogger(__name__)


class IngestionEngine:
    """Fetch from SearXNG, chunk, embed, and store."""

    def __init__(self, settings, memory_store):
        self.settings = settings
        self.store = memory_store
        self._session: aiohttp.ClientSession | None = None

    async def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession()
        return self._session

    async def close(self):
        if self._session and not self._session.closed:
            await self._session.close()

    async def fetch(self, query: str, categories: str = None,
                    language: str = None) -> List[Dict]:
        """Query SearXNG and return raw results."""
        session = await self._get_session()
        url = f"{self.settings.searxng.base_url}/search"
        params = {
            "q": query,
            "format": "json",
            "categories": categories or self.settings.searxng.default_categories,
            "language": language or self.settings.searxng.default_language,
            "pageno": 1,
        }
        async with session.get(url, params=params, timeout=aiohttp.ClientTimeout(total=15)) as resp:
            resp.raise_for_status()
            data = await resp.json()
            results = data.get("results", [])[:self.settings.searxng.max_results]
            logger.info("SearXNG returned %d results for '%s'", len(results), query)
            return results

    @staticmethod
    def chunk(text: str, title: str, url: str,
              max_words: int = 200, overlap: int = 30) -> List[Dict]:
        """Split text into overlapping word chunks."""
        if not text or len(text.strip()) < 50:
            return []

        words = text.split()
        chunks = []
        step = max_words - overlap
        for i in range(0, max(1, len(words) - overlap), step):
            chunk_words = words[i:i + max_words]
            if not chunk_words:
                break
            chunks.append({
                "text": " ".join(chunk_words),
                "title": title,
                "url": url,
                "metadata": {"chunk_index": len(chunks), "total_words": len(words)}
            })
        return chunks

    async def embed(self, texts: List[str]) -> np.ndarray:
        """Batch embed texts via OpenAI-compatible API."""
        session = await self._get_session()
        url = f"{self.settings.embedding.base_url}/embeddings"
        payload = {
            "model": self.settings.embedding.model,
            "input": texts,
        }
        async with session.post(url, json=payload,
                                timeout=aiohttp.ClientTimeout(total=30)) as resp:
            resp.raise_for_status()
            data = await resp.json()

        # Handle both OpenAI format ({"data": [...]}) and single-object format
        # ({"embedding": [...], "index": 0})
        if isinstance(data, dict) and "data" in data and isinstance(data["data"], list):
            embeddings = data["data"]
        elif isinstance(data, dict) and "embedding" in data:
            embeddings = [data]
        elif isinstance(data, list):
            embeddings = data
        else:
            raise ValueError(f"Unexpected embedding response format: {type(data)}")

        vectors = np.array([e["embedding"] for e in embeddings], dtype=np.float32)
        logger.info("Embedded %d texts → shape %s", len(vectors), vectors.shape)
        return vectors

    async def ingest(self, query: str, results: List[Dict]) -> int:
        """Fetch, chunk, embed, and store. Returns count of new chunks."""
        new_count = 0
        seen_urls = set()

        for res in results:
            url = res.get("url", "")
            if not url or url in seen_urls:
                continue
            seen_urls.add(url)

            title = res.get("title", "")
            content = res.get("content", "") or res.get("snippet", "")
            if not content:
                continue

            chunks = self.chunk(content, title, url)
            if not chunks:
                continue

            # Batch embed all chunks for this result
            chunk_texts = [c["text"] for c in chunks]
            vectors = await self.embed(chunk_texts)

            for i, chunk in enumerate(chunks):
                self.store.upsert(
                    text=chunk["text"],
                    vector=vectors[i],
                    url=url,
                    title=title,
                    metadata={**chunk["metadata"], "query": query}
                )
                new_count += 1

        # Prune if exceeding limit
        self.store.prune_oldest(10000)
        logger.info("Ingested %d chunks from %d URLs", new_count, len(seen_urls))
        return new_count
