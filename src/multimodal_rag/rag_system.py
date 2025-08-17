import openai
import logging
import base64
import os
from typing import List, Dict, Any
from .config import RAGConfig
from .pdf_processor import PDFProcessor
from .vector_store import VectorStore

logger = logging.getLogger(__name__)

class MultimodalRAGSystem:
    """Multimodal RAG system using LLM for processing"""
    
    def __init__(self, config: RAGConfig = None):
        self.config = config or RAGConfig()
        self.processor = PDFProcessor(self.config)
        self.vector_store = VectorStore(self.config)
        self.image_store = {}
        self.client = openai.OpenAI(api_key=self.config.api_key)
        
        # Load existing vector store if available
        if os.path.exists(self.config.vector_store_path):
            self.vector_store.load(self.config.vector_store_path)
    
    def process_pdf(self, pdf_bytes: bytes, file_name: str) -> Dict[str, Any]:
        """Process a PDF file"""
        text_chunks, images = self.processor.process_pdf(pdf_bytes, file_name)
        
        # Store text in vector store
        self.vector_store.add_items(text_chunks)
        
        # Store images
        for img in images:
            img_id = f"{file_name}_page{img['metadata']['page']}_img{img['metadata']['image_idx']}"
            self.image_store[img_id] = img
            img["metadata"]["image_id"] = img_id
        
        # Save vector store
        self.vector_store.save(self.config.vector_store_path)
        
        return {
            "text_chunks": len(text_chunks),
            "images": len(images)
        }
    
    def query(self, query: str, max_images: int = 2) -> Dict[str, Any]:
        """Query the system with multimodal support"""
        # Retrieve relevant text
        text_results = self.vector_store.retrieve_text(query, k=3)
        context_text = "\n\n".join([item["content"] for item in text_results])
        
        # Prepare image context
        image_context = []
        image_urls = []
        
        # Add images from text results that reference images
        for result in text_results:
            if "[Image:" in result["content"]:
                start = result["content"].find("[Image:") + 7
                end = result["content"].find("]", start)
                img_id = result["content"][start:end].strip()
                if img_id in self.image_store:
                    img = self.image_store[img_id]
                    image_context.append(
                        f"Image from {img['metadata']['file']} page {img['metadata']['page']}: " + 
                        self._describe_image(img['content'])
                    )
                    image_urls.append({
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{img['content']}"
                        }
                    })
        
        # Add up to max_images additional images
        remaining = max_images - len(image_urls)
        if remaining > 0:
            for img_id, img in list(self.image_store.items())[:remaining]:
                if img_id not in [i['metadata']['image_id'] for i in image_urls]:
                    image_context.append(
                        f"Image from {img['metadata']['file']} page {img['metadata']['page']}: " + 
                        self._describe_image(img['content'])
                    )
                    image_urls.append({
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{img['content']}"
                        }
                    })
        
        # Prepare messages for LLM
        messages = [
            {
                "role": "system",
                "content": "You are a helpful assistant that answers questions based on document context."
            }
        ]
        
        # Add text context if available
        if context_text:
            messages.append({
                "role": "user",
                "content": [
                    {"type": "text", "text": "Document context:"},
                    {"type": "text", "text": context_text}
                ]
            })
        
        # Add image context if available
        if image_context:
            messages.append({
                "role": "user",
                "content": [
                    {"type": "text", "text": "Image context:"},
                    {"type": "text", "text": "\n".join(image_context)}
                ]
            })
        
        # Add images
        for img in image_urls:
            messages.append({
                "role": "user",
                "content": [img]
            })
        
        # Add the actual query
        messages.append({
            "role": "user",
            "content": [
                {"type": "text", "text": f"Question: {query}"}
            ]
        })
        
        # Generate response
        response = self.client.chat.completions.create(
            model=self.config.model,
            messages=messages,
            max_tokens=self.config.max_tokens,
            temperature=self.config.temperature
        )
        
        return {
            "answer": response.choices[0].message.content,
            "text_context": [item["content"] for item in text_results],
            "image_context": [img["image_url"]["url"] for img in image_urls]
        }
    
    def _describe_image(self, base64_image: str) -> str:
        """Get text description of an image"""
        response = self.client.chat.completions.create(
            model=self.config.model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Describe this image in detail:"},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{base64_image}"
                            }
                        }
                    ]
                }
            ],
            max_tokens=300
        )
        return response.choices[0].message.content
    
    def clear_data(self):
        """Clear all stored data"""
        self.vector_store = VectorStore(self.config)
        self.image_store = {}
        if os.path.exists(self.config.vector_store_path):
            os.remove(self.config.vector_store_path)