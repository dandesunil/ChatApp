from fastapi import FastAPI, Request, UploadFile, File
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from services.ingest import ingest_document
from services.retriever import Answerer
from fastapi.encoders import jsonable_encoder
import os
from models.models import *
from config import *

# Directories
os.makedirs(CODE_DIR, exist_ok=True)

# Initialize Answerer
answerer = Answerer(chroma_dir=CHROMA_DIR)

# FastAPI app
app = FastAPI()

ingest_document(DEFAULT_PDF, chroma_dir=CHROMA_DIR) # Ingest on startup



@app.post("/chat")
async def chat(request_data: Chat):
    payload = jsonable_encoder(request_data)
    q = payload.get("query")
    if not q:
        return JSONResponse({"error": "no query"}, status_code=400)
    resp = answerer.answer(q)
    print(f'resp:   {resp}')
    return JSONResponse({"answer": resp})

# Run server
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host=HOST, port=PORT, reload=True)
