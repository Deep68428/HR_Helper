import sys
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from loguru import logger

parent_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(parent_dir))

from backend.core.config import get_settings
from backend.core.deps import get_base_llm, get_ocr, get_thinking_llm
from backend.core.logging import setup_logging
from backend.middlewares.logging import logging_middleware
from backend.middlewares.request_id import request_id_middleware
from backend.middlewares.timing import timing_middleware
from backend.routes.resume import route


@asynccontextmanager
async def lifespan(app: FastAPI):
    # -------------------------
    # Startup
    # -------------------------
    setup_logging()
    settings = get_settings()

    logger.info("🚀 Starting AI Agent App")
    logger.info(f"Ollama URL: {settings.OLLAMA_URL}")
    logger.info(f"Base model: {settings.MODEL_NAME}")
    logger.info(f"Thinking model: {settings.THINKING_MODEL}")

    # Force-load heavy resources (fail fast)
    logger.info("🔄 Loading models...")
    get_base_llm()
    get_thinking_llm()
    get_ocr()
    logger.info("✅ Models loaded")

    yield  # -------- App runs here -------- #

    # -------------------------
    # Shutdown
    # -------------------------
    logger.warning("🛑 Shutting down AI Agent App")


app = FastAPI(
    title="AI Agent App",
    version="0.1.0",
    lifespan=lifespan,
)


app.middleware("http")(request_id_middleware)
app.middleware("http")(logging_middleware)
app.middleware("http")(timing_middleware)


app.include_router(route, prefix="/v1")


@app.get("/health", tags=["system"])
async def health():
    return {
        "status": "ok",
        "service": app.title,
        "version": app.version,
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
