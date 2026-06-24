from sqlalchemy import Column, Integer, String, Text
from pgvector.sqlalchemy import Vector
from database import Base

class DocumentChunk(Base):
    __tablename__ = "document_chunks"

    id = Column(Integer, primary_key=True, index=True)
    filename = Column(String, index=True)
    chunk_index = Column(Integer)
    content = Column(Text)
    # all-MiniLM-L6-v2 produces 384-dimensional embeddings
    embedding = Column(Vector(384))
