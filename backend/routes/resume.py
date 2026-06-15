from fastapi import APIRouter, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from loguru import logger

from backend.agents.resume_agent import process_cv
from backend.schemas.common import ErrorResponse

route = APIRouter()
log = logger.bind()


@route.post("/analyze", responses={500: {"model": ErrorResponse}})
async def analyze_cv_jd(
    jd_file: UploadFile = File(...), cv_file: UploadFile = File(...)
):
    log.info("📨 Received CV analysis request")
    log.debug(
        f"Files - JD: {jd_file.filename} ({jd_file.content_type}), CV: {cv_file.filename} ({cv_file.content_type})"
    )

    try:
        response = process_cv(jd_file, cv_file)
        log.info("📤 CV analysis request completed successfully")
        return JSONResponse(response)
    except HTTPException:
        raise
    except Exception as e:
        log.error(f"💥 Unexpected error in CV analysis endpoint: {e}")
        raise HTTPException(status_code=500, detail=str(e))
