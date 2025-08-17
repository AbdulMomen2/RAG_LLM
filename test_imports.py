# Simple test to verify module imports
def test_imports():
    from src.multimodal_rag import MultimodalRAGSystem, RAGConfig
    from src.multimodal_rag.rag_system import MultimodalRAGSystem
    from src.multimodal_rag.config import RAGConfig
    from src.multimodal_rag.pdf_processor import PDFProcessor
    from src.multimodal_rag.vector_store import VectorStore
    print("All imports successful!")