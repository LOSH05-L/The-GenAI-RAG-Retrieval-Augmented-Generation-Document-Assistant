import os
import traceback
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from langchain_text_splitters import RecursiveCharacterTextSplitter
from google import genai

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize Gemini Client using environment variable
client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))

# In-memory document store
document_chunks = []

class QueryRequest(BaseModel):
    question: str

@app.post("/upload-doc")
async def upload_document(file: UploadFile = File(...)):
    global document_chunks
    try:
        content = await file.read()
        text = content.decode("utf-8", errors="ignore")
        
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=50)
        chunks = text_splitter.split_text(text)
        
        document_chunks = chunks
        return {"status": "success", "message": f"Successfully indexed {len(chunks)} chunks from {file.filename}!"}
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/chat")
def chat_with_docs(request: QueryRequest):
    global document_chunks
    try:
        context = "No documents uploaded yet."
        if document_chunks:
            context = "\n\n".join(document_chunks)
        
        # Updated to active production model endpoint
        response = client.models.generate_content(
            model='gemini-3.8-flash',
            contents=f"Context:\n{context}\n\nQuestion: {request.question}"
        )
        return {"answer": response.text}
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
def read_root():
    return {"message": "RAG Document Assistant Backend is live!"}
