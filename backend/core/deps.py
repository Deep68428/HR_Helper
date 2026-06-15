from functools import lru_cache

from langchain_community.llms import Ollama
from loguru import logger
from paddleocr import PaddleOCR

from backend.core.config import get_settings

s = get_settings()


@lru_cache
def get_base_llm():
    logger.info(f"🤖 Loading base LLM: {s.MODEL_NAME} from {s.OLLAMA_URL}")
    try:
        llm = Ollama(
            base_url=s.OLLAMA_URL,
            model=s.MODEL_NAME,
            temperature=0,
            format="json",
        )
        logger.info("✅ Base LLM loaded successfully")
        return llm
    except Exception as e:
        logger.error(f"❌ Failed to load base LLM {s.MODEL_NAME}: {e}")
        raise


@lru_cache
def get_thinking_llm():
    logger.info(f"🧠 Loading thinking LLM: {s.THINKING_MODEL} from {s.OLLAMA_URL}")
    try:
        llm = Ollama(
            base_url=s.OLLAMA_URL,
            model=s.THINKING_MODEL,
            temperature=0,
            format="json",
        )
        logger.info("✅ Thinking LLM loaded successfully")
        return llm
    except Exception as e:
        logger.error(f"❌ Failed to load thinking LLM {s.THINKING_MODEL}: {e}")
        raise


@lru_cache
def get_ocr():
    logger.info(f"📷 Loading OCR model with language: {s.OCR_LANG}")
    try:
        ocr = PaddleOCR(
            lang=s.OCR_LANG,
            use_textline_orientation=True,
        )
        logger.info("✅ OCR model loaded successfully")
        return ocr
    except Exception as e:
        logger.error(f"❌ Failed to load OCR model: {e}")
        raise


base_llm = get_base_llm()
thinking_llm = get_thinking_llm()
