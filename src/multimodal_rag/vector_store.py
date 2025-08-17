import numpy as np
import faiss
import pickle
import logging
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Any

logger = logging.getLogger(__name__)

class VectorStore:
    """Handles vector storage and retrieval"""
    
    def __init__(self, config):
        self.config = config
        self.embedder = SentenceTransformer('all-MiniLM-L6-v2')
        self.index = None
        self.items = []
    
    def add_items(self, items: List[Dict[str, Any]]):
        """Add text items to the vector store"""
        # Only add text items (images are handled separately)
        text_items = [item for item in items if item["type"] == "text"]
        
        if not text_items:
            return
        
        # Extract contents
        contents = [item["content"] for item in text_items]
        
        # Generate embeddings
        embeddings = self.embedder.encode(contents, normalize_embeddings=True)
        
        # Initialize index if needed
        if self.index is None:
            self.index = faiss.IndexFlatL2(embeddings.shape[1])
        
        # Add to index
        self.index.add(embeddings)
        self.items.extend(text_items)
        logger.info(f"Added {len(text_items)} text items to vector store")
    
    def retrieve_text(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """Retrieve relevant text items"""
        if not self.index or len(self.items) == 0:
            return []
        
        # Embed query
        query_embedding = self.embedder.encode([query], normalize_embeddings=True)
        
        # Search index
        distances, indices = self.index.search(query_embedding, k)
        
        # Return top results
        return [self.items[i] for i in indices[0] if i < len(self.items)]
    
    def save(self, filepath: str):
        """Save vector store to disk"""
        with open(filepath, 'wb') as f:
            pickle.dump({
                "items": self.items,
                "index": faiss.serialize_index(self.index) if self.index else None
            }, f)
        logger.info(f"Vector store saved to {filepath}")
    
    def load(self, filepath: str):
        """Load vector store from disk"""
        try:
            with open(filepath, 'rb') as f:
                data = pickle.load(f)
            self.items = data["items"]
            if data["index"]:
                self.index = faiss.deserialize_index(data["index"])
            logger.info(f"Vector store loaded from {filepath}")
        except Exception as e:
            logger.error(f"Error loading vector store: {e}")