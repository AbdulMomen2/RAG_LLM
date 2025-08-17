import pytest
from unittest.mock import patch, MagicMock, ANY
from src.multimodal_rag.rag_system import MultimodalRAGSystem
from src.multimodal_rag.config import RAGConfig

@pytest.fixture
def rag_system():
    config = RAGConfig(api_key="test_key", model="gpt-4-turbo")
    system = MultimodalRAGSystem(config)
    
    # Initialize with dummy vector store
    system.vector_store.items = [
        {"type": "text", "content": "Sample text", "metadata": {}},
        {"type": "text", "content": "Text with [Image: img1] reference", "metadata": {}}
    ]
    return system

def test_process_pdf(rag_system):
    # Mock PDF processing
    with patch.object(rag_system.processor, 'process_pdf') as mock_process:
        mock_process.return_value = (
            [{"type": "text", "content": "chunk"}],
            [{"type": "image", "content": "base64_img"}]
        )
        result = rag_system.process_pdf(b"dummy", "test.pdf")
        
        assert result["text_chunks"] == 1
        assert result["images"] == 1
        mock_process.assert_called_once_with(b"dummy", "test.pdf")

@patch('src.multimodal_rag.rag_system.openai.OpenAI', autospec=True)
def test_query(mock_openai, rag_system):
    # Setup mock responses
    mock_client = MagicMock()
    mock_openai.return_value = mock_client
    
    # Create response generators
    image_response = MagicMock()
    image_response.choices = [MagicMock(message=MagicMock(content="Image description"))]
    
    final_response = MagicMock()
    final_response.choices = [MagicMock(message=MagicMock(content="Final answer"))]
    
    mock_client.chat.completions.create.side_effect = [image_response, final_response]
    
    # Setup image store
    rag_system.image_store = {
        "img1": {
            "content": "base64_img", 
            "metadata": {"file": "test.pdf", "page": 1}
        }
    }
    
    # Execute query
    result = rag_system.query("Test question")
    
    # Verify results
    assert result["answer"] == "Final answer"
    assert result["text_context"] == ["Text with [Image: img1] reference"]
    assert "data:image/png;base64,base64_img" in result["image_context"]
    
    # Verify API calls
    assert mock_client.chat.completions.create.call_count == 2
    
    # Verify image description call
    image_call = mock_client.chat.completions.create.call_args_list[0]
    assert image_call[1]["model"] == "gpt-4-turbo"
    assert "Describe this image in detail" in image_call[1]["messages"][0]["content"][0]["text"]
    
    # Verify final query call
    final_call = mock_client.chat.completions.create.call_args_list[1]
    assert "Test question" in final_call[1]["messages"][-1]["content"][0]["text"]

def test_clear_data(rag_system):
    # Add dummy data
    rag_system.vector_store.items = [{"type": "text", "content": "test"}]
    rag_system.image_store = {"img1": "data"}
    
    # Clear data
    rag_system.clear_data()
    
    # Verify clearing
    assert len(rag_system.vector_store.items) == 0
    assert len(rag_system.image_store) == 0
    assert rag_system.vector_store.index is None

def test_describe_image(rag_system):
    # Mock OpenAI response
    with patch.object(rag_system.client.chat.completions, 'create') as mock_create:
        mock_response = MagicMock()
        mock_response.choices = [MagicMock(message=MagicMock(content="Description"))]
        mock_create.return_value = mock_response
        
        # Test image description
        result = rag_system._describe_image("base64_img")
        assert result == "Description"
        
        # Verify API call
        mock_create.assert_called_once()
        call_args = mock_create.call_args[1]
        assert call_args["model"] == "gpt-4-turbo"
        assert "Describe this image in detail" in call_args["messages"][0]["content"][0]["text"]