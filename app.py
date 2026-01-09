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

# Gemini LLM
llm = genai.GenerativeModel("gemini-pro")


# =========================
# STREAMLIT CONFIG
# =========================

st.set_page_config(page_title="PDF Agent", layout="wide")
st.title("📄 PDF Agent - Document Question Answering")


# =========================
# PDF LOADING
# =========================

def load_pdf(file) -> str:
    reader = PdfReader(file)
    text = ""

    for page in reader.pages:
        try:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
        except:
            continue

    return text.strip()


# =========================
# TEXT CHUNKING
# =========================

def chunk_text(text: str) -> List[str]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=700,
        chunk_overlap=100
    )

    chunks = splitter.split_text(text)

    # Clean chunks
    clean_chunks = []
    for chunk in chunks:
        chunk = chunk.strip()

        if not chunk:
            continue

        if len(chunk) < 40:   # skip tiny garbage chunks
            continue

        clean_chunks.append(chunk)

    return clean_chunks


# =========================
# GEMINI EMBEDDINGS (PRODUCTION SAFE)
# =========================

class GeminiEmbeddings(Embeddings):

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        embeddings = []

        for text in texts:
            text = text.strip()

            # Skip empty chunks
            if not text:
                continue

            try:
                response = genai.embed_content(
                    model="models/text-embedding-004",
                    content=text
                )

                vector = response["embedding"]

                # Validate embedding vector
                if vector and len(vector) > 10:
                    embeddings.append(vector)

            except Exception as e:
                print("Embedding error:", e)
                continue

        if not embeddings:
            raise ValueError("No valid embeddings could be created from the document.")

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

def create_faiss_index(chunks):
    # Remove empty chunks
    clean_chunks = [c.strip() for c in chunks if c.strip()]

    if not clean_chunks:
        raise ValueError("No readable text found in this PDF.")

    embeddings = GeminiEmbeddings()

    return FAISS.from_texts(
        texts=clean_chunks,
        embedding=embeddings
    )



# =========================
# RETRIEVAL
# =========================

def retrieve_context(query: str, vectorstore, k: int = 4) -> str:
    docs = vectorstore.similarity_search(query, k=k)
    if not docs:
        return ""
    return "\n\n".join(d.page_content for d in docs)


# =========================
# ANSWER GENERATION
# =========================

def generate_answer(query: str, context: str) -> str:
    if not context.strip():
        return "Answer not found in the document."

    prompt = f"""
You are a document question answering system.

Rules:
- Use ONLY the provided context
- Do NOT use prior knowledge
- If the answer is not explicitly present, reply exactly:
  "Answer not found in the document."

Context:
{context}

Question:
{query}
"""

    response = llm.generate_content(prompt)
    return response.text


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

    if not chunks:
        st.error("No text chunks could be created.")
        st.stop()

    return create_faiss_index(chunks)



# =========================
# STREAMLIT UI
# =========================

uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"])

if uploaded_file:
    with st.spinner("Processing PDF..."):
        vectorstore = process_pdf(uploaded_file)

    st.success("PDF processed successfully")

    query = st.text_input("Ask a question from the document")

    if query:
        with st.spinner("Generating answer..."):
            context = retrieve_context(query, vectorstore)
            answer = generate_answer(query, context)

        st.subheader("Answer")
        st.write(answer)
