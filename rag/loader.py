from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

PDF_PATH = "documents/2425_football_sport_manual_update_12_3_24.pdf"


def load_football_manual():
    loader = PyPDFLoader(PDF_PATH)
    documents = loader.load()

    return documents


def split_documents(documents):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=150
    )

    chunks = text_splitter.split_documents(documents)

    return chunks


if __name__ == "__main__":
    docs = load_football_manual()
    chunks = split_documents(docs)

    print("Number of pages loaded:", len(docs))
    print("Number of chunks created:", len(chunks))

    print("\nFIRST CHUNK:")
    print(chunks[0].page_content)

    print("\nMETADATA:")
    print(chunks[0].metadata)