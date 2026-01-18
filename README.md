**PDF Agent — Document Question Answering System**

**Live Application:**  
https://pdf-agent-x8rdrmuqixui52rmnwmpvi.streamlit.app/

**Overview**

PDF Agent is an end-to-end document question answering web application that enables users to upload PDF documents and query them using natural language. The system retrieves the most relevant document segments through semantic search and generates precise, context-aware answers strictly grounded in the uploaded document.

This project demonstrates a real-world implementation of a **Retrieval-Augmented Generation (RAG)** pipeline by integrating document parsing, vector embeddings, similarity search, and a Large Language Model (LLM) into a unified AI system.

**Key Features**

- Upload and process large PDF documents directly through a web interface  
- Automatic text extraction from PDF files  
- Intelligent document chunking for semantic understanding  
- Semantic embedding generation for contextual search  
- Vector-based similarity search using FAISS  
- Context-aware answer generation using a Large Language Model  
- Answers restricted strictly to the document content (no hallucination)  
- Cached vector indexing for faster repeated queries  
- Clean, responsive, and user-friendly Streamlit interface  

## System Architecture (RAG Pipeline)
PDF Upload
↓
Text Extraction (PyPDF2)
↓
Text Chunking (RecursiveCharacterTextSplitter)
↓
Embedding Generation
↓
FAISS Vector Database
↓
Similarity Search (Top-k Context Retrieval)
↓
LLM Answer Generation

## Technology Stack

- **Python** — Core programming language  
- **Streamlit** — Web application framework  
- **FAISS** — Vector database for similarity search  
- **LangChain** — Text splitting and vector store interface  
- **PyPDF2** — PDF text extraction  
- **dotenv** — Environment variable management  
- **Google Gemini API / Groq LLaMA-3.1** — Large Language Model backend  
- **HuggingFace Sentence Transformers** — Local embedding generation  


