import os
from dataclasses import dataclass

@dataclass
class RAGConfig:
    """Configuration for multimodal RAG system"""
    api_key: str = None
    model: str = "gpt-4o"
    max_tokens: int = 2000
    temperature: float = 0.3
    chunk_size: int = 2000
    max_image_size: tuple = (512, 512)
    enable_image_processing: bool = True
    vector_store_path: str = "vector_store.pkl"
    
    def __post_init__(self):
        if self.api_key is None:
            self.api_key = os.getenv("OPENAI_API_KEY")