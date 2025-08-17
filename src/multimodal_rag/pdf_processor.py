import fitz  # PyMuPDF
import base64
import logging
import io
from PIL import Image
from typing import List, Tuple, Dict, Any
from .config import RAGConfig

logger = logging.getLogger(__name__)

class PDFProcessor:
    """Handles PDF processing and content extraction"""
    
    def __init__(self, config: RAGConfig):
        self.config = config
    
    def process_pdf(self, pdf_bytes: bytes, file_name: str) -> Tuple[List[Dict], List[Dict]]:
        """Process a PDF file and return text chunks and images"""
        text_chunks = []
        images = []
        
        try:
            with fitz.open(stream=pdf_bytes, filetype="pdf") as doc:
                for page_num in range(len(doc)):
                    page = doc.load_page(page_num)
                    
                    # Extract text
                    text = page.get_text("text", sort=True)
                    if text.strip():
                        chunks = self._chunk_text(text.strip())
                        for chunk_idx, chunk in enumerate(chunks):
                            text_chunks.append({
                                "type": "text",
                                "content": chunk,
                                "metadata": {
                                    "file": file_name,
                                    "page": page_num,
                                    "chunk": chunk_idx
                                }
                            })
                    
                    # Extract images
                    if self.config.enable_image_processing:
                        img_list = page.get_images(full=True)
                        if img_list:
                            for img_idx, img_info in enumerate(img_list):
                                try:
                                    xref = img_info[0]
                                    base_image = doc.extract_image(xref)
                                    if base_image:
                                        # Process and store image
                                        image_data = self._process_image(
                                            base_image["image"], 
                                            file_name, 
                                            page_num, 
                                            img_idx
                                        )
                                        images.append(image_data)
                                except Exception as e:
                                    logger.error(f"Error processing image: {e}")
        
        except Exception as e:
            logger.error(f"Error processing PDF: {e}")
            raise
        
        return text_chunks, images
    
    def _chunk_text(self, text: str) -> List[str]:
        """Split text into manageable chunks"""
        if len(text) <= self.config.chunk_size:
            return [text]
        
        chunks = []
        start = 0
        while start < len(text):
            end = min(start + self.config.chunk_size, len(text))
            chunk = text[start:end]
            
            # Find a natural break point
            break_point = max(
                chunk.rfind('.'), 
                chunk.rfind('\n'), 
                chunk.rfind(' '),
                end - 100  # Fallback
            )
            
            if break_point > start:
                end = break_point + 1
                chunk = text[start:end]
            
            chunks.append(chunk.strip())
            start = end
        
        return chunks
    
    def _process_image(self, image_bytes: bytes, file_name: str, page_num: int, img_idx: int) -> Dict[str, Any]:
        """Process and format image data"""
        # Create PIL image
        img = Image.open(io.BytesIO(image_bytes))
        
        # Resize if needed
        if max(img.size) > max(self.config.max_image_size):
            img.thumbnail(self.config.max_image_size)
        
        # Convert to base64
        buffered = io.BytesIO()
        img.save(buffered, format="PNG")
        img_base64 = base64.b64encode(buffered.getvalue()).decode()
        
        return {
            "type": "image",
            "content": img_base64,
            "metadata": {
                "file": file_name,
                "page": page_num,
                "image_idx": img_idx,
                "dimensions": img.size
            }
        }