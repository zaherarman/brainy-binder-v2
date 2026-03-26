import asyncio
import logging
import os
import shutil
import uuid
import aiofiles
import json

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # python-dotenv not installed; rely on pydantic-settings env_file

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from src.config import settings
from src.ingestion.pipeline import IngestionPipeline
from src.store.neo4j import Neo4jStore
from .agent import agent
from .conversation import conversation_manager
from pathlib import Path
from contextlib import asynccontextmanager

# Allowed extensions for ingestion 
ALLOWED_EXTENSIONS = frozenset({".txt", ".md", ".pdf", ".docx", ".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".gif", ".webp", ".heic"})
MAX_FILE_SIZE_BYTES = int(os.getenv("INGEST_MAX_FILE_SIZE", "52428800"))  # 50 MB default 

def sanitize_filename(name: str) -> str:
    """Use basename only to prevent path traversal; replace unsafe chars."""
    base = Path(name).name
    
    if not base:
        return "unnamed"
    
    safe = "".join(c if c.isalnum() or c in "._-" else "_" for c in base)
    return safe or "unnamed"

logger = logging.getLogger(__name__)

class ChatRequest(BaseModel):
    """Chat request body."""

    message: str = Field(..., min_length=1, max_length=8_000)
    session_id: str | None = Field(None, max_length=64)

class ChatResponse(BaseModel):
    """Chat response body."""

    response: str
    session_id: str

async def _periodic_cleanup(interval_seconds: int = 300) -> None:
    """Background task that evicts stale sessions every *interval_seconds*."""
    
    while True:
        await asyncio.sleep(interval_seconds)
        
        try:
            removed = conversation_manager.cleanup_old_sessions()
            if removed:
                logger.info("Periodic cleanup: removed %d stale session(s)", removed)
        
        except Exception:
            logger.exception("Periodic session cleanup failed")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown: connect agent to MCP server, then disconnect."""
    
    try:
        await agent.startup()
        logger.info("Agent started, MCP client connected")
    
    except Exception as e:
        logger.exception("Agent startup failed: %s", e)
        raise

    app.state.neo4j_store = Neo4jStore()
    logger.info("Shared Neo4jStore initialised")
    cleanup_task = asyncio.create_task(_periodic_cleanup())
    yield

    cleanup_task.cancel()
    
    try:
        await cleanup_task
    
    except asyncio.CancelledError:
        pass

    await asyncio.to_thread(app.state.neo4j_store.neo4j_driver.close)
    logger.info("Shared Neo4jStore closed")

    try:
        await agent.shutdown()
        logger.info("Agent shutdown, MCP client disconnected")
    
    except Exception as e:
        logger.exception("Agent shutdown failed: %s", e)

app = FastAPI(
    title="Knowledge-Base Chatbot",
    description="AI-powered chatbot backed by Neo4j knowledge graph and MCP tools",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:3001"],
    allow_credentials=True,
    allow_methods=["GET", "POST", "DELETE"],
    allow_headers=["Content-Type", "Accept", "Authorization"],
)

@app.get("/")
async def root():
    """Root endpoint with service information."""
    
    return {
        "service": "Knowledge-Base Chatbot",
        "status": "running",
        "version": "1.0.0",
        "features": ["tool_calling", "conversation_history", "streaming"],
    }

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    
    return {
        "status": "healthy",
        "service": "chatbot",
        "active_sessions": conversation_manager.get_session_count(),
    }

@app.get("/api/health")
async def api_health_check():
    """API health check endpoint."""
    
    return {
        "status": "OK",
        "message": "Chatbot API is running",
        "version": "1.0.0",
    }

@app.post("/api/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Non-streaming chat endpoint."""
    
    try:
        session_id = request.session_id or str(uuid.uuid4())

        response = await agent.chat(
            message=request.message,
            session_id=session_id,
        )

        return ChatResponse(response=response, session_id=session_id)
    
    except RuntimeError as e:
        
        if "not started" in str(e).lower():
            raise HTTPException(status_code=503, detail="Service unavailable") from e
        logger.exception("Chat runtime error: %s", e)
        
        raise HTTPException(status_code=500, detail="Internal server error") from e
    
    except Exception as e:
        logger.exception("Chat error: %s", e)
        raise HTTPException(status_code=500, detail="Internal server error") from e

@app.post("/api/chat/stream")
async def chat_stream(request: ChatRequest):
    """Streaming chat endpoint using Server-Sent Events."""
    
    try:
        session_id = request.session_id or str(uuid.uuid4())

        async def event_generator():
            yield f"data: {json.dumps({'session_id': session_id})}\n\n"
            async for chunk in agent.chat_stream(
                message=request.message,
                session_id=session_id,
            ):
                yield chunk

        return StreamingResponse(
            event_generator(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )
    
    except RuntimeError as e:
        
        if "not started" in str(e).lower():
            raise HTTPException(status_code=503, detail="Service unavailable") from e
        
        logger.exception("Chat stream runtime error: %s", e)
        
        raise HTTPException(status_code=500, detail="Internal server error") from e
    
    except Exception as e:
        logger.exception("Chat stream error: %s", e)
        raise HTTPException(status_code=500, detail="Internal server error") from e

@app.delete("/api/chat/session/{session_id}")
async def clear_session(session_id: str):
    """
    Clear conversation history for a session."""
    
    try:
        
        conversation_manager.clear_session(session_id)
        return {
            "status": "success",
            "message": f"Session {session_id} cleared",
            "session_id": session_id,
        }
    
    except Exception as e:
        logger.exception("Error clearing session: %s", e)
        raise HTTPException(status_code=500, detail="Internal server error") from e

@app.get("/api/chat/session/{session_id}/history")
async def get_session_history(session_id: str):
    """Get conversation history for a session."""
    
    try:
        history = conversation_manager.get_history(session_id)
        return {
            "session_id": session_id,
            "message_count": len(history),
            "messages": history,
        }
    
    except Exception as e:
        logger.exception("Error getting history: %s", e)
        raise HTTPException(status_code=500, detail="Internal server error") from e

@app.post("/api/cleanup")
async def cleanup_sessions():
    """
    Manually trigger cleanup of old sessions from cache."""
    
    try:
        before_count = conversation_manager.get_session_count()
        removed = conversation_manager.cleanup_old_sessions()
        after_count = conversation_manager.get_session_count()

        return {
            "status": "success",
            "sessions_before": before_count,
            "sessions_after": after_count,
            "sessions_cleaned": removed,
        }
        
    except Exception as e:
        logger.exception("Error during cleanup: %s", e)
        raise HTTPException(status_code=500, detail="Internal server error") from e

@app.post("/api/ingest")
async def ingest_files(files: list[UploadFile] = File(...),reset_index: bool = Form(False)):
    """
    Upload files for ingestion into the knowledge base.

    Saves files to data/uploads/{upload_id}/ and runs the ingestion pipeline
    on that directory. Returns ingestion statistics.
    """
    
    if not files:
        raise HTTPException(status_code=400, detail="At least one file is required")

    upload_id = str(uuid.uuid4())
    upload_dir = settings.data_dir / "uploads" / upload_id
    files_accepted: list[str] = []
    files_rejected: list[dict[str, str]] = []

    for file in files:
        filename = sanitize_filename(file.filename or "")
        ext = Path(filename).suffix.lower()
        
        if ext not in ALLOWED_EXTENSIONS:
            files_rejected.append({"name": file.filename or filename, "reason": "unsupported file type"})
            continue

        dest = upload_dir / filename
        bytes_read = 0
        size_exceeded = False
        
        try:
            upload_dir.mkdir(parents=True, exist_ok=True)
            
            async with aiofiles.open(dest, "wb") as f:
                while chunk := await file.read(65536):
                    bytes_read += len(chunk)
                    
                    if bytes_read > MAX_FILE_SIZE_BYTES:
                        size_exceeded = True
                        break
                    
                    await f.write(chunk)
                
                else:
                    files_accepted.append(filename)
            
            if size_exceeded:
                dest.unlink(missing_ok=True)
                files_rejected.append(
                    {"name": file.filename or filename, "reason": "file exceeds size limit"}
                )
        
        except Exception as e:
            dest.unlink(missing_ok=True)
            files_rejected.append({"name": file.filename or filename, "reason": str(e)})

    if not files_accepted:
        return {
            "upload_id": upload_id,
            "files_accepted": [],
            "files_rejected": files_rejected,
            "ingestion": None,
        }

    pipeline = IngestionPipeline(
        reset_index=reset_index,
        data_dir=upload_dir,
        neo4j_store=app.state.neo4j_store,
    )
    
    try:
        stats = await asyncio.to_thread(pipeline.run)
    
    finally:
        await asyncio.to_thread(shutil.rmtree, upload_dir, True)

    return {
        "upload_id": upload_id,
        "files_accepted": files_accepted,
        "files_rejected": files_rejected,
        "ingestion": stats,
    }


if __name__ == "__main__":
    import uvicorn

    port = int(os.getenv("PORT", "8000"))
    uvicorn.run(
        "src.api.main:app",
        host="0.0.0.0",
        port=port,
        reload=False,
    )
