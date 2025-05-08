import os
import time
import threading
import re
import uuid
from pathlib import Path
from typing import Any, Dict

from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv

# Suppress unnecessary warnings
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# LangChain and model imports
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI

# Load environment variables
load_dotenv(dotenv_path="/var/www/html/chattingbot/.env")

# Initialize Flask app and enable CORS
app = Flask(__name__)
CORS(app)

# Define paths for storage
BASE_DIR = Path(__file__).parent
UPLOAD_DIR = BASE_DIR / "upload"
CHROMA_DIR = BASE_DIR / "chroma_db"

# Create directories if they don't exist
UPLOAD_DIR.mkdir(exist_ok=True)
CHROMA_DIR.mkdir(exist_ok=True)

# Storage for RAG chains per session
SESSION_CHAINS: Dict[str, Any] = {}
SESSION_DB_PATHS: Dict[str, Path] = {}

class LoadingIndicator:
    """Simple loading indicator with spinner animation"""
    def __init__(self, message="Processing"):
        self.message = message
        self.running = False
        self.spinner_thread = None

    def start(self):
        self.running = True
        self.spinner_thread = threading.Thread(target=self._spin)
        self.spinner_thread.start()

    def stop(self):
        self.running = False
        if self.spinner_thread:
            self.spinner_thread.join()
        print("\r" + " " * (len(self.message) + 10) + "\r", end="", flush=True)

    def _spin(self):
        spinner = ["|", "/", "-", "\\"]
        i = 0
        while self.running:
            print(f"\r{self.message} {spinner[i % len(spinner)]}", end="", flush=True)
            i += 1
            time.sleep(0.1)


def load_gemini_model() -> Any:
    """Load the Gemini model through Google's API."""
    loader = LoadingIndicator("Initializing Gemini model")
    loader.start()
    try:
        llm = ChatGoogleGenerativeAI(
            model="gemini-1.5-flash",
            temperature=0.3,
            top_p=0.95,
            max_output_tokens=2048,
        )
        return llm
    finally:
        loader.stop()


def load_and_process_pdf(file_path: str, session_id: str) -> Any:
    """
    Load PDF and create a persistent retriever from it.
    Returns a retriever object.
    """
    loader = LoadingIndicator("Loading and processing PDF")
    loader.start()
    try:
        # Create a session-specific database path
        db_path = CHROMA_DIR / session_id
        db_path.mkdir(exist_ok=True)
        SESSION_DB_PATHS[session_id] = db_path
        
        # Load and split the document
        pdf_loader = PyPDFLoader(file_path)
        docs = pdf_loader.load()
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=100,
            separators=["\n\n", "\n", " ", ""],
        )
        chunks = splitter.split_documents(docs)
        print(f"Processed PDF into {len(chunks)} chunks")

        # Create embeddings and vector store with persistence
        embedding = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={"device": "cpu"},
        )
        
        # Create persistent Chroma database
        # The persist_directory parameter ensures data is saved to disk automatically
        vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=embedding,
            persist_directory=str(db_path),  # Add persistence
        )
        
        # Return retriever
        return vectorstore.as_retriever(
            search_type="mmr",
            search_kwargs={"k": 5, "fetch_k": 10, "lambda_mult": 0.7},
        )
    finally:
        loader.stop()


def setup_rag_chain(retriever: Any, llm: Any) -> Any:
    """
    Set up the RAG chain with a simple prompt.
    """
    loader = LoadingIndicator("Configuring RAG pipeline")
    loader.start()
    try:
        system_prompt = (
            "You are an expert document assistant analyzing a PDF. Use the context to provide:\n"
            "- Precise, factual answers\n"
            "- Clear headings for key points\n"
            "- Bullet points when appropriate\n"
            "- Direct quotes from the context when possible\n\n"
            "Context: {context}\n\nQuestion: {input}\nAnswer: "
        )
        prompt = ChatPromptTemplate.from_template(system_prompt)
        qa_chain = create_stuff_documents_chain(llm, prompt)
        return create_retrieval_chain(retriever, qa_chain)
    finally:
        loader.stop()


def clean_response(response: str) -> str:
    """Clean up formatting artifacts and AI phrases"""
    r = re.sub(r"\*\*(.*?)\*\*", r"\1", response)
    r = re.sub(r"\*(\*?)(.*?)\1\*", r"\2", r)
    r = re.sub(r"^-\s*", "• ", r, flags=re.MULTILINE)
    for phrase in ["based on the provided context", "according to the document"]:
        r = r.replace(phrase, "")
    return re.sub(r"\n{3,}", "\n\n", r).strip()


def process_response(chain_output: Dict) -> str:
    """Process chain output to extract and clean answer"""
    if not chain_output or "answer" not in chain_output:
        return "Error: Unable to generate response."
    raw = chain_output["answer"]
    if "**Note:" in raw:
        raw = raw.split("**Note:")[0]
    return clean_response(raw)


def get_or_create_chain(session_id: str) -> Any:
    """Get an existing chain or create a new one if it doesn't exist"""
    if session_id in SESSION_CHAINS:
        return SESSION_CHAINS[session_id]
    
    # If the chain doesn't exist but we have a database path, we can rebuild it
    if session_id in SESSION_DB_PATHS:
        db_path = SESSION_DB_PATHS[session_id]
        if db_path.exists():
            embedding = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2",
                model_kwargs={"device": "cpu"},
            )
            vectorstore = Chroma(
                persist_directory=str(db_path),
                embedding_function=embedding
            )
            retriever = vectorstore.as_retriever(
                search_type="mmr",
                search_kwargs={"k": 5, "fetch_k": 10, "lambda_mult": 0.7},
            )
            llm = load_gemini_model()
            chain = setup_rag_chain(retriever, llm)
            SESSION_CHAINS[session_id] = chain
            return chain
    
    return None


@app.route("/upload", methods=["POST"])
def upload_pdf():
    file = request.files.get("file")
    if not file or not file.filename.lower().endswith(".pdf"):
        return jsonify({"error": "Please upload a PDF file."}), 400

    # Create session and save PDF
    session_id = str(uuid.uuid4())
    session_dir = UPLOAD_DIR / session_id
    session_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = session_dir / f"{session_id}{Path(file.filename).suffix}"
    file.save(pdf_path)

    try:
        # Build chain
        llm = load_gemini_model()
        retriever = load_and_process_pdf(str(pdf_path), session_id)
        SESSION_CHAINS[session_id] = setup_rag_chain(retriever, llm)
        return jsonify({"status": "PDF processed", "session_id": session_id}), 200
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route("/chat", methods=["POST"])
def chat_pdf():
    data = request.get_json(force=True)
    sid = data.get("session_id")
    question = data.get("question", "").strip()

    if not sid:
        return jsonify({"error": "Session ID is required."}), 400
    if not question:
        return jsonify({"error": "Question cannot be empty."}), 400

    try:
        # Get or rebuild the chain
        chain = get_or_create_chain(sid)
        if not chain:
            return jsonify({"error": "Invalid session_id or session expired."}), 400
            
        resp = chain.invoke({"input": question})
        answer = process_response(resp)
        return jsonify({"answer": answer}), 200
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)