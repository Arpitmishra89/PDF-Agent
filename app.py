import os
import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from typing import List

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from openai import OpenAI

# =========================
# ENV CONFIG
# =========================

load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY not found in environment")

client = OpenAI(
    api_key=GROQ_API_KEY,
    base_url="https://api.groq.com/openai/v1"
)

# =========================
# STREAMLIT CONFIG
# =========================

st.set_page_config(page_title="PDF AI Agent", layout="wide")
st.title("📄 PDF AI Agent")

# =========================
# PDF LOADING
# =========================

def load_pdf(file):
    reader = PdfReader(file)
    documents = []

    for page_num, page in enumerate(reader.pages, start=1):
        try:
            text = page.extract_text()
            if text:
                documents.append(
                    Document(
                        page_content=text,
                        metadata={"page": page_num}
                    )
                )
        except:
            continue

    return documents


# =========================
# TEXT CHUNKING
# =========================

def chunk_documents(documents):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=150
    )

    chunks = []

    for doc in documents:
        split_texts = splitter.split_text(doc.page_content)
        for text in split_texts:
            chunks.append(
                Document(
                    page_content=text,
                    metadata={"page": doc.metadata["page"]}
                )
            )

    return chunks


# =========================
# EMBEDDINGS (Local Sentence Transformer)
# =========================

from sentence_transformers import SentenceTransformer

class LocalEmbeddings(Embeddings):
    def __init__(self):
        self.model = SentenceTransformer("all-MiniLM-L6-v2")

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return self.model.encode(texts).tolist()

    def embed_query(self, text: str) -> List[float]:
        return self.model.encode(text).tolist()


# =========================
# VECTOR STORE
# =========================

def create_faiss_index(chunks):
    texts = [c.page_content for c in chunks]
    metadatas = [c.metadata for c in chunks]

    embeddings = LocalEmbeddings()

    return FAISS.from_texts(
        texts=texts,
        embedding=embeddings,
        metadatas=metadatas
    )


# =========================
# RETRIEVAL
# =========================

def retrieve_docs(query, vectorstore, k=4):
    return vectorstore.similarity_search(query, k=k)


# =========================
# ANSWER GENERATION WITH PAGE CITATION
# =========================

def generate_answer(query, docs):
    if not docs:
        return "Answer not found in the document."

    context = ""
    pages = set()

    for doc in docs:
        context += doc.page_content + "\n\n"
        pages.add(str(doc.metadata["page"]))

    pages_str = ", ".join(sorted(pages))

    prompt = f"""
You are a document-based AI assistant.

Rules:
- Use ONLY the provided document content.
- Do NOT use external knowledge.
- Write ONE descriptive paragraph.
- Do NOT use bullet points or headings.
- If answer is missing, say: Answer not found in the document.

Document Content:
{context}

Question:
{query}

Write a descriptive paragraph answer:
"""

    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {"role": "system", "content": "You are a helpful academic assistant."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.3,
        max_tokens=400
    )

    answer = response.choices[0].message.content.strip()

    return f"{answer}\n\nSource: Pages {pages_str}"


# =========================
# PDF PROCESSING (CACHED)
# =========================

@st.cache_resource(show_spinner=False)
def process_pdf(file):
    documents = load_pdf(file)

    if not documents:
        st.error("This PDF appears to be scanned or empty.")
        st.stop()

    chunks = chunk_documents(documents)

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
            docs = retrieve_docs(query, vectorstore)
            answer = generate_answer(query, docs)

        st.subheader("Answer")
        st.write(answer)
