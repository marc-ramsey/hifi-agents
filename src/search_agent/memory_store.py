import asyncio
import json
from typing import List, Dict, Any, Tuple
import numpy as np

class VectorStoreInterface:
    async def upsert(self, text: str, vector: np.ndarray, metadata: Dict): ...
    async def query(self, query_vector: np.ndarray, top_k: int) -> List[Dict]: ...

class GraphStoreInterface:
    async def upsert_entity(self, url: str, title: str, metadata: Dict): ...
    async def get_entities(self, query: str) -> List[Dict]: ...

class CacheInterface:
    async def get(self, key: str) -> Any: ...
    async def set(self, key: str, value: Any, ttl: int): ...

class MemoryStore:
    def __init__(self, vector_cfg, graph_cfg, cache_cfg):
        # Initialize actual implementations based on config providers
        # e.g., self.vector = ChromaStore(vector_cfg) or QdrantStore(vector_cfg)
        self.vector = None 
        self.graph = None  
        self.cache = None  

    async def enrich_with_context(self, vec_results: List[Dict]) -> List[Dict]:
        enriched = []
        for res in vec_results:
            ctx = await self.cache.get(f"ctx:{res.get('id', '')}")
            res["context"] = ctx if ctx else "No additional context"
            enriched.append(res)
        return enriched
