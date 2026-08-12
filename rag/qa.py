from langchain_ollama import ChatOllama
from rag.vectorstore import load_vector_store


def ask_rulebook(question: str):
    # Load the existing Chroma vector database
    vector_store = load_vector_store()

    # Search for the 3 most relevant chunks
    results = vector_store.similarity_search(
        question,
        k=3
    )

    # Combine the retrieved rule book chunks into one context
    context = "\n\n".join(
        result.page_content for result in results
    )

    # Connect to the local Qwen model through Ollama
    llm = ChatOllama(
        model="qwen2.5:latest",
        temperature=0
    )

    # Instructions for Qwen
    prompt = f"""
You are a football rules assistant.

Your job is to answer questions using ONLY the rule book context provided below.
Before answering:

1. Read all retrieved context carefully.
2. Identify every rule or condition that directly applies to the user's question.
3. Include all applicable limits, exceptions, restrictions, and requirements.
4. Do not omit a rule simply because another retrieved rule already answers part of the question.
5. Do not state that information is missing if the answer appears anywhere in the provided context.
6. Do not invent, add, or infer rules that are not explicitly supported.
7. Check the final answer against the retrieved context before returning it.
RULE BOOK CONTEXT:
{context}

QUESTION:
{question}

ANSWER:
"""

    # Send the prompt to Qwen
    response = llm.invoke(prompt)

    # Return the answer and retrieved source chunks
    return response.content, results

if __name__ == "__main__":
    print("\nCoach Rulebook RAG")
    print("Type 'exit' to stop.\n")

    while True:
        question = input("Enter your question: ")

        if question.lower() in ["exit", "quit"]:
            print("Exiting Coach Rulebook RAG.")
            break

        answer, sources = ask_rulebook(question)

        print("\nANSWER:")
        print(answer)

        print("\nSOURCES:")

        for i, source in enumerate(sources, start=1):
            print(f"\n--- SOURCE {i} ---")
            print("Page:", source.metadata.get("page_label"))
            print(source.page_content)

        print("\n" + "-" * 60 + "\n")