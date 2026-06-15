import tempfile
import time
from pathlib import Path

from fastapi import HTTPException, UploadFile
from loguru import logger

from backend.utils.experience_calculator import enrich_with_experience_llm
from backend.utils.parsers import parse_cv_with_ollama, parse_jd_with_ollama
from backend.utils.suitability import calculate_suitability_with_llm
from backend.utils.text_extraction import extract_text

log = logger.bind()


def save_temp_file(upload: UploadFile) -> str:
    if not upload.filename:
        logger.error("No filename provided for uploaded file")
        raise Exception("File name is required")

    suffix = Path(upload.filename).suffix
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    tmp.write(upload.file.read())
    tmp.close()

    logger.debug(f"Saved temp file: {upload.filename} -> {tmp.name}")
    return tmp.name


def process_cv(jd_file, cv_file):
    logger.info("🚀 Starting CV analysis pipeline")
    logger.info(f"📄 Processing files - JD: {jd_file.filename}, CV: {cv_file.filename}")

    if not jd_file.filename or not cv_file.filename:
        logger.error(
            "Missing required files - JD: {}, CV: {}",
            jd_file.filename,
            cv_file.filename,
        )
        raise HTTPException(status_code=400, detail="Both JD and CV files are required")

    start = time.time()

    jd_path = save_temp_file(jd_file)
    cv_path = save_temp_file(cv_file)

    try:
        # STEP 1: Extract text
        logger.info("📖 Step 1/5: Extracting text from documents")
        jd_text = extract_text(jd_path)
        cv_text = extract_text(cv_path)

        # STEP 2: Parse JD
        logger.info("📋 Step 2/5: Parsing job description")
        jd = parse_jd_with_ollama(jd_text)

        # STEP 3: Parse CV
        logger.info("👤 Step 3/5: Parsing CV profile")
        cv_profile = parse_cv_with_ollama(cv_text)

        # STEP 4: Enrich experience
        position = jd["jd_profile"]["job_title"]
        logger.info(f"⚡ Step 4/5: Enriching experience for position: {position}")
        final_cv = enrich_with_experience_llm(cv_profile, position)

        # STEP 5: Suitability
        logger.info("🎯 Step 5/5: Calculating suitability score")
        suitability = calculate_suitability_with_llm(jd["jd_profile"], final_cv)

        execution_time = round(time.time() - start, 2)
        response = {
            "summary": {
                "candidate_file": cv_file.filename,
                "position": position,
                "overall_score": suitability.get("total_score"),
                "classification": suitability.get("classification"),
                "experience_score": suitability.get("breakdown", {}).get(
                    "experience_score"
                ),
                "keyword_score": suitability.get("breakdown", {}).get("keyword_score"),
                "education_score": suitability.get("breakdown", {}).get(
                    "education_score"
                ),
            },
            "execution_time_sec": execution_time,
        }

        logger.info(
            f"✅ CV analysis complete in {execution_time}s - Score: {suitability.get('total_score')}/100 ({suitability.get('classification')})"
        )

        # Clean up temp files
        try:
            Path(jd_path).unlink(missing_ok=True)
            Path(cv_path).unlink(missing_ok=True)
            logger.debug("Cleaned up temporary files")
        except Exception as cleanup_error:
            logger.warning(f"Failed to clean up temp files: {cleanup_error}")

        return response
    except Exception as e:
        execution_time = round(time.time() - start, 2)
        logger.error(f"❌ CV analysis failed after {execution_time}s: {e}")

        # Clean up temp files on error
        try:
            Path(jd_path).unlink(missing_ok=True)
            Path(cv_path).unlink(missing_ok=True)
        except Exception:
            pass  # Ignore cleanup errors during exception handling

        raise HTTPException(status_code=500, detail=str(e))
