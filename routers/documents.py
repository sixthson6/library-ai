import logging
from typing import Optional
from pydantic import BaseModel, Field
from fastapi import APIRouter, Depends, UploadFile, File, HTTPException
from sqlalchemy.orm import Session
from database import get_db
from models import DocumentChunk
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/documents",
    tags=["documents"]
)

# Load the embedding model
# This will be downloaded on the first run, and cached locally.
logger.info("Loading sentence transformer model 'all-MiniLM-L6-v2'...")
model = SentenceTransformer("all-MiniLM-L6-v2")
logger.info("Model loaded successfully.")

def chunk_text(text: str, chunk_size: int = 1000, overlap: int = 100):
    """A simple fixed-size character chunker with overlap."""
    chunks = []
    start = 0
    text_len = len(text)
    while start < text_len:
        end = min(start + chunk_size, text_len)
        chunks.append(text[start:end])
        if end == text_len:
            break
        start += (chunk_size - overlap)
    return chunks

@router.get("/")
def list_documents():
    return {"documents": []}

class SearchRequest(BaseModel):
    payload: str
    filename: Optional[str] = None
    limit: int = Field(default=3, ge=1, le=20)
    threshold: float = Field(default=0.5, description="Maximum cosine distance threshold (0.0 to 2.0). Lower is stricter.")

@router.post("/search/")
def search_documents(
    request: SearchRequest,
    db: Session = Depends(get_db)
):
    # 1. Embed the query
    query_vector = model.encode(request.payload).tolist()
    
    # 2. Build the database query
    # calculate distance
    distance_col = DocumentChunk.embedding.cosine_distance(query_vector).label("distance")
    
    base_query = db.query(DocumentChunk, distance_col)
    
    # Apply threshold filter (lower distance means more similar)
    base_query = base_query.filter(DocumentChunk.embedding.cosine_distance(query_vector) < request.threshold)
    
    # Apply filename filter if provided
    if request.filename:
        base_query = base_query.filter(DocumentChunk.filename == request.filename)
        
    # Order by distance and limit
    results = base_query.order_by(distance_col).limit(request.limit).all()
    
    # 3. Format response
    formatted_results = []
    for chunk, distance in results:
        formatted_results.append({
            "id": chunk.id,
            "filename": chunk.filename,
            "chunk_index": chunk.chunk_index,
            "content": chunk.content,
            "distance": round(distance, 4)
        })
        
    return {
        "payload": request.payload,
        "results_count": len(formatted_results),
        "results": formatted_results
    }

@router.post("/upload/")
async def upload_document(file: UploadFile = File(...), db: Session = Depends(get_db)):
    if not file.filename.endswith(".txt"):
        raise HTTPException(status_code=400, detail="Only .txt files are supported currently.")
    
    # 1. Extraction
    content_bytes = await file.read()
    try:
        content_str = content_bytes.decode("utf-8")
    except UnicodeDecodeError:
        raise HTTPException(status_code=400, detail="File must be valid UTF-8 text.")
    
    # 2. Chunking
    chunks = chunk_text(content_str)
    
    if not chunks:
        raise HTTPException(status_code=400, detail="Document is empty.")
    
    # 3. Embedding Generation
    # model.encode() accepts a list of strings and returns a numpy array
    embeddings = model.encode(chunks)
    
    # 4. Storage in vector database
    db_chunks = []
    for i, (chunk, emb) in enumerate(zip(chunks, embeddings)):
        db_chunk = DocumentChunk(
            filename=file.filename,
            chunk_index=i,
            content=chunk,
            embedding=emb.tolist() # Convert numpy array to list for pgvector
        )
        db_chunks.append(db_chunk)
        
    db.add_all(db_chunks)
    db.commit()
    
    return {
        "message": "Document uploaded and processed successfully.",
        "filename": file.filename,
        "chunks_created": len(db_chunks)
    }
