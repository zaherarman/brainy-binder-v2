import logging
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

import pytesseract

def configure_tesseract() -> None:
    """Configure pytesseract binary path from environment or OS defaults."""
    
    env_cmd = os.getenv("TESSERACT_CMD")
    
    if env_cmd:
        pytesseract.pytesseract.tesseract_cmd = env_cmd
        return
    
    if os.name == "nt":
        default = Path(r"C:\Program Files\Tesseract-OCR\tesseract.exe")
        
        if default.exists():
            pytesseract.pytesseract.tesseract_cmd = str(default)

configure_tesseract()

from fastmcp import FastMCP  
from .config import settings  
from .ingestion.pipeline import IngestionPipeline  
from .llm.answer_engine import AnswerEngine  
from .store.neo4j import Neo4jStore  

logger = logging.getLogger(__name__)

store: "Neo4jStore | None" = None
engine: "AnswerEngine | None" = None

def get_store() -> "Neo4jStore":
    global store
    if store is None:
        store = Neo4jStore()
    return store

def get_engine() -> "AnswerEngine":
    global engine
    
    if engine is None:
        engine = AnswerEngine(get_store())
    return engine

mcp = FastMCP(
    name="kb",
    instructions=(
        "Knowledge-base assistant backed by a Neo4j graph store. "
        "Use ingest_documents to load new documents, then rag_search or "
        "hybrid_search to answer questions."
    )
)

@mcp.tool()
def ingest_documents(data_dir: str | None = None, reset_index: bool = False) -> dict[str, int]:
    """Ingest documents from a directory into the knowledge base."""
    
    allowed_root = settings.data_dir.resolve()

    if data_dir is not None:
        resolved = Path(data_dir).resolve()
        
        if not resolved.is_relative_to(allowed_root):
            raise ValueError(
                f"data_dir must be within the configured data root ({allowed_root})."
            )
        data_directory = resolved
    
    else:
        data_directory = allowed_root

    if not data_directory.exists():
        raise ValueError(
            f"Data directory does not exist: {data_directory}. "
            "Create the directory and add documents before ingesting."
        )

    logger.info("Starting ingestion from %s (reset_index=%s)", data_directory, reset_index)
    
    pipeline = IngestionPipeline(
        data_dir=data_directory,
        reset_index=reset_index,
        neo4j_store=get_store(),
    )
    
    stats = pipeline.run()
    logger.info("Ingestion complete: %s", stats)
    return stats

@mcp.tool()
def rag_search(question: str) -> str:
    """Answer a question using pure vector (RAG) search over ingested documents."""
    
    if not question.strip():
        raise ValueError("question must not be empty.")

    logger.info("rag_search: %s", question)
    return get_engine().rag_search(question)

@mcp.tool()
def hybrid_search(question: str) -> str:
    """
    Answer a question using hybrid search: vector retrieval augmented with
    knowledge-graph context (entity relationships).

    Embeds the question, retrieves the top-k chunks, fetches related graph
    entities, and generates an answer with the LLM.
    """
    
    if not question.strip():
        raise ValueError("question must not be empty.")

    logger.info("hybrid_search: %s", question)
    return get_engine().hybrid_search(question)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    mcp.run(transport="stdio")
