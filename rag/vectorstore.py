from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings

from rag.loader import load_football_manual, split_documents


CHROMA_PATH = "chroma_db"


def create_vector_store():

    # 1. Load the football manual
    documents = load_football_manual()

    # 2. Split the manual into chunks
    chunks = split_documents(documents)

    # 3. Create the embedding model
    embeddings = OllamaEmbeddings(
        model="nomic-embed-text"
    )

    # 4. Convert chunks to vectors and store them in Chroma
    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=CHROMA_PATH
    )

    return vector_store


if __name__ == "__main__":
    vector_store = create_vector_store()

    print("Vector database created successfully.")