import os
import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from typing import List

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.embeddings import Embeddings

from google import genai

# ENV + GEMINI CLIENT

load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY not found in environment")

client = genai.Client(api_key=GEMINI_API_KEY)

# STREAMLIT CONFIG

st.set_page_config(page_title="PDF Agent", layout="wide")
st.title("PDF Agent - Document Question Answering")

# PDF LOADING

def load_pdf(file):
    reader = PdfReader(file)
    text = ""

    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text + "\n"

    return text

# CHUNKING

def chunk_text(text):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=900,
        chunk_overlap=120
    )
    return splitter.split_text(text)

# PROPER GEMINI EMBEDDINGS (CRITICAL)


class GeminiEmbeddings(Embeddings):

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        response = client.models.embed_content(
            model="models/text-embedding-004",
            contents=texts
        )
        return [emb.values for emb in response.embeddings]

    def embed_query(self, text: str) -> List[float]:
        response = client.models.embed_content(
            model="models/text-embedding-004",
            contents=text
        )
        return response.embeddings[0].values


# FAISS INDEX CREATION


def create_faiss_index(chunks):
    embeddings = GeminiEmbeddings()
    return FAISS.from_texts(
        texts=chunks,
        embedding=embeddings
    )


# RETRIEVAL


def retrieve_context(query, vectorstore, k=4):
    docs = vectorstore.similarity_search(query, k=k)
    if not docs:
        return ""
    return "\n\n".join(d.page_content for d in docs)


# ANSWER GENERATION


def generate_answer(query, context):
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

    response = client.models.generate_content(
        model="models/gemini-flash-lite-latest",
        contents=prompt
    )

    return response.text


# PDF PROCESSING (CACHED)


@st.cache_resource(show_spinner=False)
def process_pdf(file):
    text = load_pdf(file)

    if not text.strip():
        st.error("This PDF appears to be scanned. Text extraction failed.")
        st.stop()

    chunks = chunk_text(text)
    return create_faiss_index(chunks)


# STREAMLIT UI


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
