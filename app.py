import os
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

# Custom styling to replace red focus ring with modern blue
st.markdown("""
<style>
    /* Chat input container & textarea focus border */
    [data-testid="stChatInput"] textarea:focus,
    [data-testid="stChatInput"] > div:focus-within {
        border-color: #3B82F6 !important;
        box-shadow: 0 0 0 1px #3B82F6 !important;
    }
</style>
""", unsafe_allow_html=True)

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

def format_sources(docs):
    pages = sorted(set(str(d.metadata.get("page", "?")) for d in docs), key=lambda x: int(x) if x.isdigit() else x)
    pages_str = ", ".join(pages) if pages else "N/A"
    return f"**Source:** Pages {pages_str}"


def stream_answer(query, docs, model="llama-3.1-8b-instant", chat_history=None):
    if not docs:
        yield "Answer not found in the document."
        return

    context = ""
    for doc in docs:
        context += doc.page_content + "\n\n"

    system_instruction = (
        "You are a document-based AI assistant.\n\n"
        "Rules:\n"
        "- Use ONLY the provided document content to answer the question.\n"
        "- Do NOT use external knowledge.\n"
        "- Be clear, factual, and concise.\n"
        "- If the answer is not present in the document content, say: 'Answer not found in the document.'\n\n"
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
            max_tokens=500,
            stream=True
        )
        for chunk in response_stream:
            if chunk.choices and chunk.choices[0].delta.content:
                content = chunk.choices[0].delta.content
                yield content
                time.sleep(0.015)  # ChatGPT-style smooth typewriter pacing
    except Exception as e:
        yield f"\n\n⚠️ Error generating answer from LLM ({model}): {e}\n\nPlease check your Groq API key or try choosing another model from the sidebar."


def generate_answer(query, docs, model="llama-3.1-8b-instant"):
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
    st.header("⚙️ Settings")
    model_choice = st.selectbox(
        "LLM Model (Groq)",
        [
            "llama-3.1-8b-instant",
            "llama-3.3-70b-versatile",
            "openai/gpt-oss-20b",
            "openai/gpt-oss-120b"
        ],
        index=0,
        help="Select the Groq model to use for generating answers."
    )

    st.markdown("---")
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

    st.success(f"✅ PDF '{uploaded_file.name}' processed successfully!")

    # Display existing chat history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("sources"):
                with st.expander("🔍 View Retrieved Document Chunks"):
                    for i, src in enumerate(msg["sources"], start=1):
                        st.markdown(f"**Chunk {i} (Page {src['page']}):**\n\n{src['content']}")

    # User chat input
    if prompt := st.chat_input("Ask a question about the document..."):
        # Display user message in chat
        with st.chat_message("user"):
            st.markdown(prompt)

        # Retrieve relevant chunks
        docs = retrieve_docs(prompt, vectorstore)

        sources_info = [
            {"page": d.metadata.get("page", "?"), "content": d.page_content}
            for d in docs
        ]
        citation = format_sources(docs)

        # Stream assistant response
        with st.chat_message("assistant"):
            full_response = st.write_stream(
                stream_answer(prompt, docs, model=model_choice, chat_history=st.session_state.messages)
            )
            st.markdown(f"\n\n{citation}")

            with st.expander("🔍 View Retrieved Document Chunks"):
                for i, src in enumerate(sources_info, start=1):
                    st.markdown(f"**Chunk {i} (Page {src['page']}):**\n\n{src['content']}")

        # Save to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.session_state.messages.append({
            "role": "assistant",
            "content": f"{full_response}\n\n{citation}",
            "sources": sources_info
        })
else:
    st.info("👆 Please upload a PDF document above to start chatting.")
