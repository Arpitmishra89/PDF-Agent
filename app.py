import os
import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from typing import List

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.embeddings import Embeddings

import google.generativeai as genai


# =========================
# ENV + GEMINI SETUP
# =========================

load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY not found in environment")

genai.configure(api_key=GEMINI_API_KEY)

# Gemini LLM (for answers)
llm = genai.GenerativeModel("gemini-1.5-flash")


# =========================
# STREAMLIT CONFIG
# =========================

st.set_page_config(page_title="PDF Agent", layout="wide")
st.title("📄 PDF Agent – Document Question Answering")


# =========================
# PDF LOADING
# =========================

def load_pdf(file) -> str:
    reader = PdfReader(file)
    text = ""

    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text + "\n"

    return text


# =========================
# TEXT CHUNKING
# =========================

def chunk_text(text: str) -> List[str]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=900,
        chunk_overlap=120
    )
    return splitter.split_text(text)


# =========================
# GEMINI EMBEDDINGS
# =========================

class GeminiEmbeddings(Embeddings):

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        embeddings = []
        for text in texts:
            response = genai.embed_content(
                model="models/text-embedding-004",
                content=text
            )
            embeddings.append(response["embedding"])
        return embeddings

    def embed_query(self, text: str) -> List[float]:
        response = genai.embed_content(
            model="models/text-embedding-004",
            content=text
        )
        return response["embedding"]


# =========================
# FAISS INDEX
# =========================

def create_faiss_index(chunks: List[str]):
    if not chunks:
        raise ValueError("No text chunks created from PDF")

    embeddings = GeminiEmbeddings()
    return FAISS.from_texts(texts=chunks, embedding=embeddings)


# =========================
# RETRIEVAL
# =========================

def retrieve_context(query: str, vectorstore, k: int = 4) -> str:
    docs = vectorstore.similarity_search(query, k=k)
    return "\n\n".join(d.page_content for d in docs) if docs else ""


# =========================
# ANSWER GENERATION
# =========================

def generate_answer(query: str, context: str) -> str:
    if not context.strip():
        return "Answer not found in the document."

    prompt = f"""
You are a document-based question answering assistant.

Rules:
- Use ONLY the given context
- Do NOT add external knowledge
- If the answer is not in the context, say:
  "Answer not found in the document."

Context:
{context}

Question:
{query}
"""

    response = llm.generate_content(prompt)
    return response.text.strip()


# =========================
# PDF PROCESSING (CACHED)
# =========================

@st.cache_resource(show_spinner=False)
def process_pdf(file):
    text = load_pdf(file)

    if not text.strip():
        st.error("This PDF appears to be scanned or empty.")
        st.stop()

    chunks = chunk_text(text)
    return create_faiss_index(chunks)


# =========================
# STREAMLIT UI
# =========================

uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"])

if uploaded_file:
    with st.spinner("Processing PDF..."):
        vectorstore = process_pdf(uploaded_file)

    st.success("✅ PDF processed successfully")

    query = st.text_input("Ask a question from the document")

    if query:
        with st.spinner("Generating answer..."):
            context = retrieve_context(query, vectorstore)
            answer = generate_answer(query, context)

        st.subheader("Answer")
        st.write(answer)


