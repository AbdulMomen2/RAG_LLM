import pytest
import numpy as np
from unittest.mock import patch
from src.multimodal_rag.vector_store import VectorStore
from src.multimodal_rag.config import RAGConfig

@pytest.fixture
def vector_store():
    config = RAGConfig()
    return VectorStore(config)

def test_add_items(vector_store):
    # Test adding text items
    items = [
        {"type": "text", "content": "First text chunk", "metadata": {}},
        {"type": "text", "content": "Second text chunk", "metadata": {}},
        {"type": "image", "content": "base64_image", "metadata": {}}  # Should be ignored
    ]
    
    vector_store.add_items(items)
    assert len(vector_store.items) == 2
    assert vector_store.items[0]["content"] == "First text chunk"

def test_retrieve_text(vector_store):
    # Test retrieval functionality
    vector_store.add_items([
        {"type": "text", "content": "AI systems", "metadata": {}},
        {"type": "text", "content": "Machine learning", "metadata": {}},
    ])
    
    results = vector_store.retrieve_text("artificial intelligence", k=1)
    assert len(results) == 1
    assert "AI" in results[0]["content"] or "learning" in results[0]["content"]

@patch('faiss.write_index')
def test_save_load(vector_store, mock_faiss):
    # Test persistence
    vector_store.add_items([{"type": "text", "content": "Test save", "metadata": {}}])
    
    vector_store.save("test.pkl")
    assert mock_faiss.called
    
    # Reset and load
    new_store = VectorStore(RAGConfig())
    new_store.load("test.pkl")
    assert len(new_store.items) == 1
    assert new_store.items[0]["content"] == "Test save"