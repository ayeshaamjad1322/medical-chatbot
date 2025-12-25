from dotenv import load_dotenv
import os

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma


def setup_rag_chain(chroma_path):
    # Load environment variables (if any .env present)
    load_dotenv()

    # Initialize Hugging Face embeddings
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    # Check if the vector DB exists
    if not os.path.exists(chroma_path):
        raise FileNotFoundError(f"Chroma path '{chroma_path}' does not exist. Run your RAG script first to create it.")

    # Load the Chroma vector store
    db = Chroma(
        persist_directory=chroma_path,
        embedding_function=embeddings
    )

    # Return retriever to be used in your chatbot or flask app
    retriever = db.as_retriever()

    return retriever
