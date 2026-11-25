"""Minimal, beginner-friendly PDF Chatbot using RAG (LangChain + FAISS + Gemini)."""

import os
import tempfile

import streamlit as st
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Load environment variables from a .env file if present
load_dotenv()

EMBEDDING_MODEL = "models/gemini-embedding-001"
LLM_MODELS = [
    "gemini-3.6-flash",
    "gemini-3.5-flash",
    "gemini-3.5-flash-lite",
    "gemini-3.8-flash",
]

SYSTEM_PROMPT = """You are a helpful assistant answering questions strictly from the provided PDF context.
Use only the context below to answer the user's question.
If the answer is not contained in the context, reply: "I couldn't find that information in the uploaded PDF."
Do not make up facts or use outside knowledge.

Context:
{context}"""


def init_session_state() -> None:
    """Initialize Streamlit session state variables once per session."""
    if "vector_store" not in st.session_state:
        st.session_state.vector_store = None
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "pdf_name" not in st.session_state:
        st.session_state.pdf_name = None


def build_vector_store(uploaded_file, api_key: str) -> FAISS:
    """Extract text from an uploaded PDF, split it into chunks, and index it in FAISS."""
    # PyPDFLoader requires a file path on disk, so write the upload to a temp file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.getbuffer())
        tmp_path = tmp_file.name

    try:
        loader = PyPDFLoader(tmp_path)
        documents = loader.load()
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

    if not documents or all(not doc.page_content.strip() for doc in documents):
        raise ValueError(
            "No extractable text was found in this PDF. "
            "Scanned or image-only PDFs are not supported."
        )

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
    )
    chunks = splitter.split_documents(documents)

    embeddings = GoogleGenerativeAIEmbeddings(
        model=EMBEDDING_MODEL,
        google_api_key=api_key,
    )
    return FAISS.from_documents(chunks, embeddings)


def answer_question(question: str, vector_store: FAISS, api_key: str) -> str:
    """Retrieve relevant PDF chunks and generate an answer grounded strictly in that context."""
    retriever = vector_store.as_retriever(search_kwargs={"k": 4})
    relevant_docs = retriever.invoke(question)

    context = "\n\n".join(
        f"[Page {doc.metadata.get('page', 0) + 1}]\n{doc.page_content}"
        for doc in relevant_docs
    )

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", SYSTEM_PROMPT),
            ("human", "{question}"),
        ]
    )

    last_error: Exception | None = None
    for model_name in LLM_MODELS:
        try:
            llm = ChatGoogleGenerativeAI(
                model=model_name,
                google_api_key=api_key,
                temperature=0.2,
                max_retries=1,
            )
            chain = prompt | llm | StrOutputParser()
            return chain.invoke({"context": context, "question": question})
        except Exception as exc:
            last_error = exc
            error_text = str(exc)
            # Fall back to the next model on transient overload/availability errors
            if any(code in error_text for code in ("503", "UNAVAILABLE", "404", "NOT_FOUND", "429")):
                continue
            raise

    assert last_error is not None
    raise last_error


def main() -> None:
    """Render the Streamlit sidebar and chat interface."""
    st.set_page_config(page_title="PDF Chatbot", page_icon="📄")
    init_session_state()

    st.title("📄 PDF Chatbot")
    st.caption("Ask questions about your PDF using LangChain, FAISS, and Gemini.")

    # --- Sidebar: API Key & PDF Upload ---
    with st.sidebar:
        st.header("Setup")

        env_api_key = os.getenv("GOOGLE_API_KEY", "")
        api_key_input = st.text_input(
            "Google API Key",
            value="",
            type="password",
            placeholder="Loaded from .env" if env_api_key else "Enter your Gemini API key",
            help="Optional if GOOGLE_API_KEY is already set in your .env file.",
        )
        api_key = api_key_input.strip() or env_api_key.strip()

        uploaded_pdf = st.file_uploader("Upload a PDF", type=["pdf"])

        if st.button("Process PDF", use_container_width=True, disabled=uploaded_pdf is None):
            if not api_key:
                st.error("Please provide a Google API key in `.env` or the sidebar.")
            else:
                with st.spinner("Reading and indexing PDF..."):
                    try:
                        st.session_state.vector_store = build_vector_store(uploaded_pdf, api_key)
                        st.session_state.pdf_name = uploaded_pdf.name
                        st.session_state.messages = []
                        st.success(f"Indexed `{uploaded_pdf.name}`!")
                    except Exception as exc:
                        st.error(f"Failed to process PDF: {exc}")

        if st.session_state.pdf_name:
            st.info(f"Active PDF: **{st.session_state.pdf_name}**")

    # --- Main Area: Chat History & Input ---
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("Ask a question about your PDF..."):
        if not api_key:
            st.warning("Please set your Google API key in `.env` or the sidebar first.")
            return
        if st.session_state.vector_store is None:
            st.warning("Please upload and process a PDF in the sidebar first.")
            return

        # Display and store user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Generate and display assistant response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    reply = answer_question(prompt, st.session_state.vector_store, api_key)
                except Exception as exc:
                    reply = f"Error generating response: {exc}"
                st.markdown(reply)

        st.session_state.messages.append({"role": "assistant", "content": reply})


if __name__ == "__main__":
    main()
