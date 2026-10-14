import os
from pydantic import BaseModel, Field

class SearXNGConfig(BaseModel):
    base_url: str = "http://localhost:8888"
    api_key: str | None = None
    default_categories: str = "general"
    default_language: str = "en"
    rate_limit_delay: float = 1.0
    max_retries: int = 3

class DatabaseConfig(BaseModel):
    raw_db_url: str = "sqlite:///raw_memory.db"
    graph_db_url: str = "bolt://localhost:7687"
    graph_db_user: str = "neo4j"
    graph_db_password: str = "password"

class VectorDBConfig(BaseModel):
    provider: str = "chroma"  # chroma, qdrant, faiss
    collection_name: str = "search_memory"
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    similarity_threshold: float = 0.92

class CacheConfig(BaseModel):
    redis_url: str = "redis://localhost:6379/0"
    ttl: int = 3600

class RetrievalConfig(BaseModel):
    preference_mode_threshold: float = 0.75
    min_coverage_entities: int = 3
    top_k: int = 5
    parallel_interval_minutes: int = 30

class Settings(BaseModel):
    searxng: SearXNGConfig = Field(default_factory=SearXNGConfig)
    db: DatabaseConfig = Field(default_factory=DatabaseConfig)
    vector: VectorDBConfig = Field(default_factory=VectorDBConfig)
    cache: CacheConfig = Field(default_factory=CacheConfig)
    retrieval: RetrievalConfig = Field(default_factory=RetrievalConfig)
