import os
import streamlit as st
from PyPDF2 import PdfReader
from typing import List
from dotenv import load_dotenv

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.embeddings import Embeddings

from sentence_transformers import SentenceTransformer
from openai import OpenAI


# =========================
# ENV SETUP
# =========================

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY not found in .env file")

client = OpenAI(
    api_key=GROQ_API_KEY,
    base_url="https://api.groq.com/openai/v1"
)


# =========================
# STREAMLIT CONFIG
# =========================

st.set_page_config(page_title="PDF AI Agent (Groq)", layout="wide")
st.title("📄 PDF AI Agent")


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

    clean_chunks = []
    for chunk in chunks:
        chunk = chunk.strip()
        if not chunk:
            continue
        if len(chunk) < 40:
            continue
        clean_chunks.append(chunk)

    return clean_chunks


# =========================
# EMBEDDINGS (LOCAL + FREE)
# =========================

class HFEmbeddings(Embeddings):
    def __init__(self):
        self.model = SentenceTransformer("all-MiniLM-L6-v2")

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return self.model.encode(texts).tolist()

    def embed_query(self, text: str) -> List[float]:
        return self.model.encode(text).tolist()


# =========================
# FAISS INDEX
# =========================

def create_faiss_index(chunks):
    embeddings = HFEmbeddings()
    return FAISS.from_texts(texts=chunks, embedding=embeddings)


# =========================
# RETRIEVAL
# =========================

def retrieve_context(query: str, vectorstore, k: int = 4) -> str:
    docs = vectorstore.similarity_search(query, k=k)

    if not docs:
        return ""

    return "\n\n".join(d.page_content for d in docs)


# =========================
# ANSWER GENERATION (Groq)
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

    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",   # ✅ latest Groq model
        messages=[
            {"role": "system", "content": "You are a helpful document assistant"},
            {"role": "user", "content": prompt}
        ],
        temperature=0.2
    )

    return response.choices[0].message.content


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
