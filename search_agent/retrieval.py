import asyncio
from typing import List, Dict, Any
import numpy as np

class RetrievalEngine:
    def __init__(self, memory_store, config, ingestion_engine):
        self.memory_store = memory_store
        self.config = config
        self.ingestion_engine = ingestion_engine

    async def retrieve(self, query: str, mode: str = "preference") -> List[Dict]:
        query_vector = self.ingestion_engine.model.encode(query)
        
        if mode == "preference":
            results = await self.memory_store.vector.query(query_vector, top_k=self.config.retrieval.top_k)
            max_sim = max([r.get("score", 0) for r in results]) if results else 0
            
            if max_sim < self.config.retrieval.preference_mode_threshold:
                live_results = await self.ingestion_engine.fetch_results(query)
                await self.ingestion_engine.ingest(query, live_results)
                results = await self.memory_store.vector.query(query_vector, top_k=self.config.retrieval.top_k)
                
        elif mode == "parallel":
            results = await self.memory_store.vector.query(query_vector, top_k=self.config.retrieval.top_k)
            # Background scheduler would trigger ingestion independently
            
        return await self.memory_store.enrich_with_context(results)
