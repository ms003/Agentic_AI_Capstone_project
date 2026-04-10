import os
import csv
import pandas as pd

from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain.docstore.document import Document
from config.env import OPENAI_API_KEY
import os

os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

# Initialize embeddings
embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)

# Path to the CSV file
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(BASE_DIR, "../data", "qa.csv")

def load_documents_from_csv():
    if not os.path.exists(CSV_PATH):
        raise FileNotFoundError(f"{CSV_PATH} not found.")

    docs = []

    try:
        # Try reading using pandas first (safe fallback with strings)
        df = pd.read_csv(CSV_PATH, dtype=str, encoding='utf-8', engine='python')
        print("Loaded using pandas:", df.shape)
    except Exception as e:
        print(f"Pandas failed to read CSV: {e}")
        print("Falling back to csv module...")

        with open(CSV_PATH, newline='', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        df = pd.DataFrame(rows)

    for _, row in df.iterrows():
        question = str(row.get("Question", "")).strip()
        answer = str(row.get("Answer", "")).strip()
        department = str(row.get("Department", "General")).strip()

        if question and answer:
            content = f"Q: {question}\nA: {answer}"
            metadata = {"department": department}
            docs.append(Document(page_content=content, metadata=metadata))

    return docs

def create_vector_db():
    docs = load_documents_from_csv()
    return FAISS.from_documents(docs, embeddings)
