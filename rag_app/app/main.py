import os
from typing import List, Optional

import uvicorn # For running the app directly for testing
from fastapi import FastAPI, File, UploadFile, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

# Adjust the import path based on how the app is run.
# If run from `rag_app` directory using `python -m app.main`: from .rag_handler import RAGHandler
# If run from project root using `uvicorn rag_app.app.main:app`: from rag_app.app.rag_handler import RAGHandler
try:
    from .rag_handler import RAGHandler
    from .rag_handler import UPLOAD_DIRECTORY as RAG_UPLOAD_DIRECTORY
    from .rag_handler import VECTOR_STORE_PATH as RAG_VECTOR_STORE_PATH
except ImportError:
    from rag_handler import RAGHandler # Fallback for simpler local runs from app directory
    from rag_handler import UPLOAD_DIRECTORY as RAG_UPLOAD_DIRECTORY
    from rag_handler import VECTOR_STORE_PATH as RAG_VECTOR_STORE_PATH


# --- Configuration & Initialization ---

# Define base directory for paths, assuming main.py is in rag_app/app/
APP_DIR = os.path.dirname(__file__) # rag_app/app
RAG_APP_ROOT_DIR = os.path.dirname(APP_DIR) # rag_app/

STATIC_FILES_DIR = os.path.join(RAG_APP_ROOT_DIR, "static")
TEMPLATES_DIR = os.path.join(RAG_APP_ROOT_DIR, "templates")
UPLOAD_DIR_MAIN = os.path.join(RAG_APP_ROOT_DIR, RAG_UPLOAD_DIRECTORY) # rag_app/uploads

# Ensure upload directory exists
os.makedirs(UPLOAD_DIR_MAIN, exist_ok=True)

# --- Pydantic Models ---
class QueryRequest(BaseModel):
    query: str

class QueryResponse(BaseModel):
    answer: str

class UploadResponse(BaseModel):
    message: str
    filenames: List[str]

# --- FastAPI App Initialization ---
app = FastAPI()

# Mount static files
# This assumes your static files (CSS, JS) are in a 'static' directory at the same level as 'templates'
# and 'app' directories inside 'rag_app'.
app.mount("/static", StaticFiles(directory=STATIC_FILES_DIR), name="static")

# Initialize Jinja2Templates
templates = Jinja2Templates(directory=TEMPLATES_DIR)

# Initialize RAGHandler
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    # For production, you might want to log this and exit,
    # or have endpoints return a specific error if rag_handler is None.
    raise ImportError(
        "OPENAI_API_KEY environment variable not set. This is required to initialize RAGHandler."
    )

try:
    rag_handler = RAGHandler(openai_api_key=OPENAI_API_KEY)
except Exception as e:
    # Catch any other initialization errors from RAGHandler
    raise RuntimeError(f"Failed to initialize RAGHandler: {e}")


# --- FastAPI Endpoints ---

@app.get("/", response_class=HTMLResponse)
async def get_index(request: Request):
    """
    Serves the main HTML page.
    """
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/upload/", response_model=UploadResponse)
async def upload_documents(files: List[UploadFile] = File(...)):
    """
    Handles document uploads, saves them, and processes them using RAGHandler.
    """
    if not rag_handler:
        raise HTTPException(status_code=503, detail="RAG service is not available due to missing API key.")

    saved_file_paths = []
    processed_filenames = []

    for file in files:
        if not file.filename:
            # Should not happen with FastAPI's File(...) but good to check
            print("Warning: Received a file without a filename.")
            continue
        
        file_path = os.path.join(UPLOAD_DIR_MAIN, file.filename)
        
        try:
            with open(file_path, "wb") as buffer:
                buffer.write(await file.read())
            saved_file_paths.append(file_path)
            processed_filenames.append(file.filename)
        except Exception as e:
            # If one file fails, we might want to continue with others or stop
            # For now, let's report error for this file and continue
            print(f"Error saving file {file.filename}: {e}")
            # Optionally, raise HTTPException here to stop all uploads on first error
            # raise HTTPException(status_code=500, detail=f"Error saving file {file.filename}: {e}")

    if not saved_file_paths:
        raise HTTPException(status_code=400, detail="No files were successfully saved.")

    try:
        # Pass absolute paths to the handler
        rag_handler.load_and_process_documents(saved_file_paths)
    except Exception as e:
        # This could be a more specific exception from RAGHandler
        print(f"Error processing documents with RAGHandler: {e}")
        raise HTTPException(status_code=500, detail=f"Error processing documents: {e}")

    return UploadResponse(
        message=f"Successfully uploaded and initiated processing for {len(processed_filenames)} files.",
        filenames=processed_filenames,
    )

@app.post("/chat/", response_model=QueryResponse)
async def chat_with_rag(query_request: QueryRequest):
    """
    Receives a query, gets an answer from RAGHandler, and returns it.
    """
    if not rag_handler:
        raise HTTPException(status_code=503, detail="RAG service is not available due to missing API key.")

    if not query_request.query or not query_request.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty.")

    try:
        answer = rag_handler.get_answer(query_request.query)
    except Exception as e:
        # Catch-all for unexpected errors in get_answer
        print(f"Error in RAGHandler get_answer: {e}")
        raise HTTPException(status_code=500, detail="Error retrieving answer from RAG system.")

    if answer is None:
        # This could mean the vector store is not ready, or no relevant answer was found.
        # The RAGHandler's get_answer prints "Vector store not initialized..."
        # or "No answer found..."
        # We can be more specific if RAGHandler raises custom exceptions.
        raise HTTPException(
            status_code=404,
            detail="Could not retrieve an answer. The document store might not be ready or relevant information not found."
        )

    return QueryResponse(answer=answer)


# --- For local testing ---
if __name__ == "__main__":
    # This allows running the app with `python rag_app/app/main.py`
    # Make sure OPENAI_API_KEY is set in your environment.
    print(f"Attempting to run FastAPI server from {__file__}...")
    print(f"Static files directory: {STATIC_FILES_DIR}")
    print(f"Templates directory: {TEMPLATES_DIR}")
    print(f"Upload directory: {UPLOAD_DIR_MAIN}")
    print(f"RAG Upload directory (from handler): {RAG_UPLOAD_DIRECTORY}")
    print(f"RAG Vector Store path (from handler): {RAG_VECTOR_STORE_PATH}")

    if not os.path.exists(STATIC_FILES_DIR):
        print(f"Warning: Static files directory does not exist: {STATIC_FILES_DIR}")
    if not os.path.exists(TEMPLATES_DIR):
        print(f"Warning: Templates directory does not exist: {TEMPLATES_DIR}")
        print("Make sure you have 'rag_app/templates/index.html'.")
    
    if not OPENAI_API_KEY:
        print("ERROR: OPENAI_API_KEY is not set. The application will not work.")
        print("Please set it in your environment before running.")
    else:
        print("OPENAI_API_KEY is set.")
        uvicorn.run(app, host="0.0.0.0", port=8000)

# To run from project root:
# uvicorn rag_app.app.main:app --reload
# Ensure OPENAI_API_KEY is set.
# Ensure rag_app/static, rag_app/templates, rag_app/uploads directories exist.
# And rag_app/templates/index.html is present.
# The RAGHandler will create rag_app/vector_store/faiss_index
