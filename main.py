from fastapi import FastAPI
from routers import documents
from database import engine, Base
from sqlalchemy import text
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Ensure vector extension exists before creating tables
logger.info("Initializing database and ensuring pgvector extension exists...")
with engine.connect() as conn:
    conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector;"))
    conn.commit()

# Create tables based on models
logger.info("Creating database tables...")
import models # ensure models are imported before create_all
Base.metadata.create_all(bind=engine)
logger.info("Database initialized successfully.")

app = FastAPI(title="Library-AI", description="Context aware document API")

app.include_router(documents.router)

@app.get("/")
def read_root():
    return {"message": "Welcome to Library-AI"}
