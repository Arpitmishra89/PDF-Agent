import os
import re
import time
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
# STREAMLIT CONFIG
# =========================

st.set_page_config(page_title="PDF AI Agent", layout="wide")
st.title("📄 PDF AI Agent")

# =========================
# MODEL CONFIG
# =========================

DEFAULT_MODEL = "openai/gpt-oss-20b"

# =========================
# ENV CONFIG
# =========================

load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY and hasattr(st, "secrets") and "GROQ_API_KEY" in st.secrets:
    GROQ_API_KEY = st.secrets["GROQ_API_KEY"]

if not GROQ_API_KEY:
    st.error("⚠️ GROQ_API_KEY not found. Please add it to your .env file locally or to Streamlit Secrets.")
    st.stop()

client = OpenAI(
    api_key=GROQ_API_KEY,
    base_url="https://api.groq.com/openai/v1"
)

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
# EMBEDDINGS (Cached Singleton)
# =========================

from sentence_transformers import SentenceTransformer

@st.cache_resource(show_spinner=False)
def get_embedding_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

class LocalEmbeddings(Embeddings):
    def __init__(self):
        self.model = get_embedding_model()

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
# RETRIEVAL (Multi-Query Aware)
# =========================

def retrieve_docs(query, vectorstore, k=6):
    # Primary search with the full question
    all_docs = vectorstore.similarity_search(query, k=k)

    # Sub-query decomposition for multi-part questions
    sub_queries = [
        q.strip()
        for q in re.split(r"\?|\band\b|;|\balso\b|\bas well as\b", query, flags=re.IGNORECASE)
        if len(q.strip()) > 8
    ]

    # Retrieve chunks for distinct sub-topics
    if len(sub_queries) > 1:
        for sub_q in sub_queries[:4]:
            sub_docs = vectorstore.similarity_search(sub_q, k=3)
            all_docs.extend(sub_docs)

    # Deduplicate while preserving relevance order
    seen = set()
    unique_docs = []
    for doc in all_docs:
        sig = (doc.metadata.get("page", 0), doc.page_content[:80])
        if sig not in seen:
            seen.add(sig)
            unique_docs.append(doc)

    return unique_docs[:8]


# =========================
# ANSWER GENERATION WITH PAGE CITATION
# =========================

def format_sources(docs):
    pages = sorted(set(str(d.metadata.get("page", "?")) for d in docs), key=lambda x: int(x) if x.isdigit() else x)
    pages_str = ", ".join(pages) if pages else "N/A"
    return f"**Source:** Pages {pages_str}"


def stream_answer(query, docs, model=DEFAULT_MODEL, chat_history=None):
    if not docs:
        yield "Answer not found in the document."
        return

    context = ""
    for doc in docs:
        context += doc.page_content + "\n\n"

    system_instruction = (
        "You are an expert document-based AI assistant.\n\n"
        "Instructions:\n"
        "- Answer the user's question thoroughly and accurately using ONLY the provided Document Content.\n"
        "- Do NOT use external knowledge or invent information.\n"
        "- For multi-part questions, address each part that is supported by the document content.\n"
        "- If certain specific parts are not covered in the document, answer the parts you can and briefly clarify which specific part could not be found.\n"
        "- If none of the requested information exists in the document, say: 'Answer not found in the document.'\n"
        "- Keep the answer clear, structured, and easy to read.\n\n"
        f"Document Content:\n{context}"
    )

    messages = [{"role": "system", "content": system_instruction}]

    # Include recent conversation turns for multi-turn context
    if chat_history:
        for msg in chat_history[-4:]:
            messages.append({"role": msg["role"], "content": msg["content"]})

    messages.append({"role": "user", "content": query})

    try:
        response_stream = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0.3,
            max_tokens=800,
            stream=True
        )
        for chunk in response_stream:
            if chunk.choices and chunk.choices[0].delta.content:
                content = chunk.choices[0].delta.content
                yield content
                time.sleep(0.015)  # ChatGPT-style smooth typewriter pacing
    except Exception as e:
        yield f"\n\n⚠️ Error generating answer from LLM: {e}"


def generate_answer(query, docs, model=DEFAULT_MODEL):
    """Non-streaming fallback helper."""
    chunks = list(stream_answer(query, docs, model=model))
    answer = "".join(chunks)
    citation = format_sources(docs)
    return f"{answer}\n\n{citation}"


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
# STREAMLIT UI & CHAT INTERFACE
# =========================

# Initialize chat session states
if "messages" not in st.session_state:
    st.session_state.messages = []

if "current_file" not in st.session_state:
    st.session_state.current_file = None

with st.sidebar:
    st.header("⚙️ Options")
    if st.button("🗑️ Clear Chat History", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"])

if uploaded_file:
    # Reset conversation when a new file is uploaded
    if st.session_state.current_file != uploaded_file.name:
        st.session_state.current_file = uploaded_file.name
        st.session_state.messages = []

    with st.spinner("Processing PDF..."):
        vectorstore = process_pdf(uploaded_file)

    # Temporary toast notification that automatically disappears after a few seconds
    if st.session_state.get("toast_shown_for") != uploaded_file.name:
        st.toast(f"PDF '{uploaded_file.name}' processed successfully!", icon="✅")
        st.session_state.toast_shown_for = uploaded_file.name

    # Display existing chat history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # User chat input
    if prompt := st.chat_input("Ask a question about the document..."):
        # Display user message in chat
        with st.chat_message("user"):
            st.markdown(prompt)

        # Retrieve relevant chunks
        docs = retrieve_docs(prompt, vectorstore)
        citation = format_sources(docs)

        # Stream assistant response
        with st.chat_message("assistant"):
            full_response = st.write_stream(
                stream_answer(prompt, docs, model=DEFAULT_MODEL, chat_history=st.session_state.messages)
            )
            st.markdown(f"\n\n{citation}")

        # Save to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.session_state.messages.append({
            "role": "assistant",
            "content": f"{full_response}\n\n{citation}"
        })
else:
    st.info("👆 Please upload a PDF document above to start chatting.")
