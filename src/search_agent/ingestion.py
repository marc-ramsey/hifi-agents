import hashlib
import json
import logging
from typing import List, Dict, Any
import asyncio
import aiohttp
import numpy as np
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

class SearXNGIngestionEngine:
    def __init__(self, config, vector_store, graph_store, cache):
        self.config = config
        self.vector_store = vector_store
        self.graph_store = graph_store
        self.cache = cache
        self.model = SentenceTransformer(config.vector.embedding_model)
        self.session = None

    async def _get_session(self) -> aiohttp.ClientSession:
        if self.session is None or self.session.closed:
            headers = {"Authorization": f"Bearer {self.config.searxng.api_key}"} if self.config.searxng.api_key else {}
            self.session = aiohttp.ClientSession(headers=headers)
        return self.session

    async def fetch_results(self, query: str, categories: str = None, language: str = None) -> List[Dict]:
        session = await self._get_session()
        url = f"{self.config.searxng.base_url}/search"
        params = {
            "q": query,
            "format": "json",
            "categories": categories or self.config.searxng.default_categories,
            "language": language or self.config.searxng.default_language
        }
        async with session.get(url, params=params) as resp:
            resp.raise_for_status()
            data = await resp.json()
            return data.get("results", [])

    def chunk_content(self, content: str, title: str, url: str, chunk_size: int = 512, overlap: int = 50) -> List[Dict]:
        words = content.split()
        chunks = []
        for i in range(0, len(words), chunk_size - overlap):
            chunk_words = words[i:i + chunk_size]
            chunks.append({
                "text": " ".join(chunk_words),
                "title": title,
                "url": url,
                "metadata": {
                    "source_engine": "searxng",
                    "ingestion_timestamp": asyncio.get_event_loop().time(),
                    "decay_factor": 1.0
                }
            })
        return chunks

    def deduplicate(self, new_chunks: List[Dict], existing_chunks: List[Dict]) -> List[Dict]:
        valid_chunks = []
        for chunk in new_chunks:
            url_hash = hashlib.sha256(f"{chunk['url']}_{chunk['title']}".encode()).hexdigest()
            # Semantic deduplication would query vector store here for cosine similarity > 0.92
            valid_chunks.append(chunk)
        return valid_chunks

    async def ingest(self, query: str, results: List[Dict]):
        for res in results:
            chunks = self.chunk_content(res.get("content", ""), res.get("title", ""), res.get("url", ""))
            chunks = self.deduplicate(chunks, [])
            for chunk in chunks:
                embedding = self.model.encode(chunk["text"])
                await self.vector_store.upsert(chunk["text"], embedding, chunk["metadata"])
                await self.graph_store.upsert_entity(chunk["url"], chunk["title"], chunk["metadata"])
                await self.cache.set(f"chunk:{chunk['url']}", json.dumps(chunk), ttl=self.config.cache.ttl)
