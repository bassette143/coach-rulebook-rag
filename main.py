from fastapi import FastAPI

app = FastAPI(
    title="Coach Rulebook RAG API",
    description="API for asking questions about coaching rule books",
    version="1.0.0"
)

@app.get("/")
def home():
    return {
        "message": "Coach Rulebook RAG API is running"
    }