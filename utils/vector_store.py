import os
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from config.env import OPENAI_API_KEY

os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY


def load_faiss_vectorstore() -> FAISS:
    index_path = "faiss_index"
    if not os.path.isdir(index_path):
        raise FileNotFoundError(f"FAISS index directory not found at {index_path}")
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.load_local(index_path, embeddings, allow_dangerous_deserialization=True)
    return vectorstore

# Load once globally for reuse
vectorstore = load_faiss_vectorstore()
