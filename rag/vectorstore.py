from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings

from rag.loader import load_football_manual, split_documents


CHROMA_PATH = "chroma_db"


def get_embeddings():
    return OllamaEmbeddings(
        model="nomic-embed-text"
    )


def create_vector_store():
    # 1. Load the football manual
    documents = load_football_manual()

    # 2. Split the manual into chunks
    chunks = split_documents(documents)

    # 3. Create the embedding model
    embeddings = get_embeddings()

    # 4. Convert chunks to vectors and store them in Chroma
    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=CHROMA_PATH,
        collection_name="football_rules"
    )

    return vector_store


def load_vector_store():
    # Load the existing Chroma database
    embeddings = get_embeddings()

    vector_store = Chroma(
        collection_name="football_rules",
        embedding_function=embeddings,
        persist_directory=CHROMA_PATH
    )

    return vector_store


if __name__ == "__main__":
    create_vector_store()

    print("Vector database created successfully.")