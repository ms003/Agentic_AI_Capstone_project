from create_rag.vector_store import create_vector_db  # Replace with your actual filename without .py


def main():

    vector_db = create_vector_db()
    vector_db.save_local("faiss_index")

    print("RAG vector database created and saved as 'faiss_index'.")


if __name__ == "__main__":
    main()
