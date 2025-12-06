import os
import tempfile
import streamlit as st
from dotenv import load_dotenv
from datetime import datetime

# LangChain primitives
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS   # <-- switched to FAISS

# ── Setup ──────────────────────────────────────────────────────────────────────
load_dotenv()
st.set_page_config(page_title="📝 RAG Q&A", layout="wide")
st.title("📝 RAG Q&A with Multiple PDFs + Chat History")

# Sidebar
with st.sidebar:
    st.header("⚙️ Config")
    api_key_input = st.text_input("Groq API Key", type="password")
    st.caption("Upload PDFs → Ask questions → Get answers")

# Accept key from input OR .env
api_key = api_key_input or os.getenv("GROQ_API_KEY")
if not api_key:
    st.warning("🔑 Please enter your Groq API Key (or set GROQ_API_KEY in .env).")

# ── Initialize Models ──────────────────────────────────────────────────────────
embeddings, llm = None, None
if api_key:
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"}   # force CPU for Streamlit Cloud
    )
    llm = ChatGroq(groq_api_key=api_key, model_name="openai/gpt-oss-20b")

# ── Upload PDFs ────────────────────────────────────────────────────────────────
uploaded_files = st.file_uploader("📚 Upload PDF files", type="pdf", accept_multiple_files=True)

all_docs = []
if uploaded_files and embeddings:
    tmp_paths = []
    for pdf in uploaded_files:
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
        tmp.write(pdf.getvalue())
        tmp.close()
        tmp_paths.append(tmp.name)

        loader = PyPDFLoader(tmp.name)
        docs = loader.load()
        for d in docs:
            d.metadata["source_file"] = pdf.name
        all_docs.extend(docs)

    # Clean up temp files
    for p in tmp_paths:
        try:
            os.unlink(p)
        except Exception:
            pass

    st.success(f"✅ Loaded {len(all_docs)} pages from {len(uploaded_files)} PDFs")

    # ── Chunking ──────────────────────────────────────────────────────────────
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    splits = text_splitter.split_documents(all_docs)

    # ── Vectorstore (FAISS instead of Chroma) ──────────────────────────────────
    vectorstore = FAISS.from_documents(splits, embeddings)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

    st.sidebar.write(f"🔍 Indexed {len(splits)} chunks for retrieval")

    # ── Helper ────────────────────────────────────────────────────────────────
    def _join_docs(docs, max_chars=7000):
        chunks, total = [], 0
        for d in docs:
            piece = d.page_content
            if total + len(piece) > max_chars:
                break
            chunks.append(piece)
            total += len(piece)
        return "\n\n---\n\n".join(chunks)

    # ── Prompts ───────────────────────────────────────────────────────────────
    contextualize_q_prompt = ChatPromptTemplate.from_messages([
        ("system",
         "You are a helpful assistant that rewrites the user's latest question into a "
         "standalone search query, using the chat history for context. "
         "Return only the rewritten query, no preamble."),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}")
    ])

    qa_prompt = ChatPromptTemplate.from_messages([
        ("system",
         "You are a STRICT RAG assistant. You MUST answer using only the provided context.\n"
         "If the context does not contain the answer, then reply exactly: "
         "'Out of scope - not found in provided documents.'\n"
         "Do NOT use outside knowledge.\n\n"
         "Context:\n{context}"),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}")
    ])

    # ── Session state for chat history ────────────────────────────────────────
    if "chathistory" not in st.session_state:
        st.session_state.chathistory = {}

    def get_history(session_id: str):
        if session_id not in st.session_state.chathistory:
            st.session_state.chathistory[session_id] = ChatMessageHistory()
        return st.session_state.chathistory[session_id]

    # ── Chat UI ───────────────────────────────────────────────────────────────
    session_id = st.text_input("🆔 Session ID", value="default_session")
    user_q = st.chat_input("💬 Ask a question...")

    if user_q:
        history = get_history(session_id)

        # 1) Rewrite question with history
        rewrite_msgs = contextualize_q_prompt.format_messages(
            chat_history=history.messages,
            input=user_q
        )
        try:
            standalone_q = llm.invoke(rewrite_msgs).content.strip()
        except Exception as e:
            st.error(f"LLM rewrite error: {e}")
            standalone_q = user_q

        # 2) Retrieve docs
        docs = retriever.get_relevant_documents(standalone_q)

        if not docs:
            answer = "Out of scope - not found in provided documents."
        else:
            # 3) Build context
            context_str = _join_docs(docs)

            # 4) Get answer
            qa_msgs = qa_prompt.format_messages(
                chat_history=history.messages,
                input=user_q,
                context=context_str
            )
            try:
                answer = llm.invoke(qa_msgs).content
            except Exception as e:
                st.error(f"LLM answer error: {e}")
                answer = "Error generating answer."

        # 5) Display + save
        st.chat_message("user").write(user_q)
        st.chat_message("assistant").write(answer)
        history.add_user_message(user_q)
        history.add_ai_message(answer)

        # Sources
        if docs:
            with st.expander("📑 Sources"):
                for i, doc in enumerate(docs, 1):
                    source_file = doc.metadata.get('source_file', 'Unknown')
                    page_num = doc.metadata.get('page', '?')
                    st.markdown(f"**Source {i}: {source_file} (Page {page_num})**")
                    st.text(doc.page_content[:400] + "..." if len(doc.page_content) > 400 else doc.page_content)

else:
    st.info("Enter your API key and upload PDFs to begin.")
