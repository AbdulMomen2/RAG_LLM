import pytest
from unittest.mock import patch, MagicMock
from src.multimodal_rag.pdf_processor import PDFProcessor
from src.multimodal_rag.config import RAGConfig

@pytest.fixture
def processor():
    config = RAGConfig(enable_image_processing=True)
    return PDFProcessor(config)

def test_chunk_text(processor):
    # Test text chunking logic
    text = "A" * 5000  # 5000-character text
    chunks = processor._chunk_text(text)
    
    assert len(chunks) > 1
    assert all(len(chunk) <= processor.config.chunk_size + 100 for chunk in chunks)
    assert ''.join(chunks) == text

@patch('src.multimodal_rag.pdf_processor.fitz.open')
def test_process_pdf(mock_fitz, processor):
    # Mock PDF processing
    mock_page = MagicMock()
    mock_page.get_text.return_value = "Sample text\n[Image: test_image]"
    mock_page.get_images.return_value = [(1,)]
    
    mock_doc = MagicMock()
    mock_doc.__len__.return_value = 1
    mock_doc.load_page.return_value = mock_page
    mock_doc.extract_image.return_value = {"image": b"dummy_image"}
    mock_fitz.return_value.__enter__.return_value = mock_doc
    
    text_chunks, images = processor.process_pdf(b"dummy_pdf", "test.pdf")
    
    assert len(text_chunks) > 0
    assert text_chunks[0]["content"] == "Sample text"
    assert len(images) == 1
    assert images[0]["type"] == "image"

def test_process_image(processor):
    # Test image processing
    img_bytes = b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01\x08\x06\x00\x00\x00\x1f\x15\xc4\x89\x00\x00\x00\rIDATx\xdac\xf8\x0f\x00\x01\x05\x01\x02\xcf\xcc\x10\x1e\x00\x00\x00\x00IEND\xaeB`\x82"
    result = processor._process_image(img_bytes, "test.pdf", 0, 0)
    
    assert result["type"] == "image"
    assert "base64" in result["content"]
    assert result["metadata"]["file"] == "test.pdf"