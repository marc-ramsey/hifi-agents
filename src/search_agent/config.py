import os
from pydantic import BaseModel, Field


class SearXNGConfig(BaseModel):
    base_url: str = "http://maceo.local:8888"
    default_categories: str = "general"
    default_language: str = "en"
    max_results: int = 10


class EmbeddingConfig(BaseModel):
    base_url: str = "http://localhost:8080/v1"
    model: str = "nomic-embed-text-v2"
    dimensions: int = 768  # nomic-embed-text-v2 dimension


class RetentionConfig(BaseModel):
    max_chunks: int = 10000  # prune oldest when exceeded
    similarity_threshold: float = 0.75  # min confidence to skip live search
    top_k: int = 5


class Settings(BaseModel):
    searxng: SearXNGConfig = Field(default_factory=SearXNGConfig)
    embedding: EmbeddingConfig = Field(default_factory=EmbeddingConfig)
    retention: RetentionConfig = Field(default_factory=RetentionConfig)
    db_path: str = Field(default_factory=lambda: os.environ.get(
        "SEARCH_AGENT_DB", "search_memory.db"))
