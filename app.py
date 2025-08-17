import streamlit as st
import base64
from PIL import Image
import io
import os
from pathlib import Path
import sys
from src.multimodal_rag import MultimodalRAGSystem, RAGConfig

# Add the src directory to Python path
sys.path.append(str(Path(__file__).parent))

# Page config
st.set_page_config(
    page_title="Multimodal RAG System",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

def main():
    # Initialize session state
    if 'rag_system' not in st.session_state:
        st.session_state.rag_system = None
    if 'processing_stats' not in st.session_state:
        st.session_state.processing_stats = {}
    if 'image_store' not in st.session_state:
        st.session_state.image_store = {}
    
    # Title
    st.title("🧠 Multimodal RAG System")
    st.markdown("Upload PDFs and ask questions about both text and images")
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # API Key
        api_key = st.text_input("OpenAI API Key", type="password")
        if api_key:
            os.environ["OPENAI_API_KEY"] = api_key
            try:
                config = RAGConfig(api_key=api_key)
                st.session_state.rag_system = MultimodalRAGSystem(config)
                st.success("System initialized!")
            except Exception as e:
                st.error(f"Initialization failed: {e}")
        
        if not api_key:
            st.warning("Please enter your OpenAI API key to continue")
            st.stop()
        
        # File upload
        st.header("📁 Upload PDFs")
        uploaded_files = st.file_uploader(
            "Choose PDF files", 
            type="pdf", 
            accept_multiple_files=True
        )
        
        if st.button("Process Documents") and uploaded_files:
            for file in uploaded_files:
                content = file.read()
                stats = st.session_state.rag_system.process_pdf(content, file.name)
                st.session_state.processing_stats[file.name] = stats
                st.success(f"Processed {file.name}: {stats['text_chunks']} text chunks, {stats['images']} images")
        
        st.divider()
        
        # System management
        if st.button("Clear All Data"):
            st.session_state.rag_system.clear_data()
            st.session_state.processing_stats = {}
            st.session_state.image_store = {}
            st.success("All data cleared!")
    
    # Main content
    if st.session_state.rag_system:
        # Display processing stats
        if st.session_state.processing_stats:
            st.header("📊 Processing Statistics")
            for file_name, stats in st.session_state.processing_stats.items():
                with st.expander(f"📄 {file_name}"):
                    st.metric("Text Chunks", stats['text_chunks'])
                    st.metric("Images Extracted", stats['images'])
        
        # Query interface
        st.header("💬 Ask Questions")
        query = st.text_input("Enter your question:", placeholder="What is this document about?")
        
        if st.button("Submit"):
            with st.spinner("Processing..."):
                try:
                    result = st.session_state.rag_system.query(query)
                    
                    # Display answer
                    st.subheader("🤖 Answer")
                    st.markdown(result['answer'])
                    
                    # Display text context
                    if result['text_context']:
                        st.subheader("📖 Relevant Text")
                        for i, text in enumerate(result['text_context']):
                            with st.expander(f"Text excerpt {i+1}"):
                                st.text(text)
                    
                    # Display image context
                    if result['image_context']:
                        st.subheader("🖼️ Relevant Images")
                        cols = st.columns(min(3, len(result['image_context'])))
                        for i, img_data in enumerate(result['image_context']):
                            if img_data.startswith("data:image/png;base64,"):
                                base64_str = img_data.split(",")[1]
                                img_bytes = base64.b64decode(base64_str)
                                img = Image.open(io.BytesIO(img_bytes))
                                with cols[i % len(cols)]:
                                    st.image(img, caption=f"Image {i+1}", use_column_width=True)
                except Exception as e:
                    st.error(f"Error processing query: {e}")

if __name__ == "__main__":
    main()