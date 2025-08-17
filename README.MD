# Multimodal RAG System

A comprehensive system for processing PDF documents and answering questions about both text and image content.

[![Python Version](https://img.shields.io/badge/python-3.9%2B-blue)]()
[![License](https://img.shields.io/badge/license-MIT-green)]()

## Features

- PDF text extraction with intelligent chunking
- Image extraction and processing
- Multimodal search (text + images)
- GPT-powered question answering
- Streamlit web interface
- Persistent index storage

## Installation

```bash
git clone https://github.com/yourusername/multimodal-rag.git
cd multimodal-rag

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate    # Windows

# Install dependencies
pip install -r requirements.txt

# Set up environment variables
cp .env.example .env

