from fastapi import FastAPI
from pydantic import BaseModel

from rag.qa import ask_rulebook


app = FastAPI(
    title="Coach Rulebook RAG API",
    description="API for asking questions about coaching rule books",
    version="1.0.0"
)


class QuestionRequest(BaseModel):
    question: str


@app.get("/")
def home():
    return {
        "message": "Coach Rulebook RAG API is running"
    }


@app.post("/ask")
def ask_question(request: QuestionRequest):
    answer, sources = ask_rulebook(request.question)

    source_list = []

    for source in sources:
        source_list.append(
            {
                "page": source.metadata.get("page_label"),
                "text": source.page_content
            }
        )

    return {
        "question": request.question,
        "answer": answer,
        "sources": source_list
    }