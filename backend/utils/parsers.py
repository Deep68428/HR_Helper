import json
from datetime import datetime

from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import PromptTemplate
from loguru import logger

from backend.core.deps import base_llm

# ================= LANGCHAIN CHAINS =================


def parse_jd_with_ollama(jd_text: str) -> dict:
    logger.info("📋 Parsing job description with LLM")
    logger.debug(f"JD text length: {len(jd_text)} characters")

    prompt_template = PromptTemplate(
        input_variables=["jd_text"],
        template="""You are a Job Description parsing engine.

GENERAL RULES:
- Extract data ONLY from the given JD text
- Do NOT invent information
- Controlled normalization is allowed ONLY where explicitly defined
- Output VALID JSON ONLY
- Do NOT infer or normalize or assume.

JOB TITLE RULES:
- job_title is the role being hired for
- If a line starts with a known title prefix and has text after it,
  you MUST extract the substring after the prefix
- Known prefixes (case-insensitive):
  "Job description", "Job Description", "Position", "Role"
- Ignore section headers like "Roles and Responsibilities"

LOCATION RULES:
- Extract location names exactly as written

EXPERIENCE RULES:
- If Fresher / Entry Level / Intern then put 0.0
- Extract explicitly stated experience
- Convert months to years (12 months = 1.0)
- If multiple experience values exist, choose the HIGHEST value
- "2+ years" → 2.0

EDUCATION RULES:
- Extract the highest education mentioned verbatim

KEYWORD RULES:
- Prefer MAXIMUM keyword extraction that are relevant to job title
- Keywords must be ATOMIC (1–3 words)
- Split tool lists into individual tools
- Get keywords from statements as well

INPUT JD:
\"\"\"
{jd_text}
\"\"\"

RETURN JSON ONLY:
{{
  "jd_profile": {{
    "job_title": "",
    "locations": [],
    "required_experience": Float,
    "education_requirements": "",
    "keywords": []
  }}
}}""",
    )

    parser = JsonOutputParser()
    chain = prompt_template | base_llm | parser

    try:
        result = chain.invoke({"jd_text": jd_text})
        logger.debug("JD parsing result: {}", json.dumps(result, indent=2))
        logger.info("✅ Successfully parsed job description")
        return result
    except Exception as e:
        logger.error("❌ Failed to parse job description: {}", e)
        raise


def parse_cv_with_ollama(cv_text: str) -> dict:
    logger.info("👤 Parsing CV with LLM")
    logger.debug(f"CV text length: {len(cv_text)} characters")
    today = datetime.today().strftime("%B %Y")

    prompt_template = PromptTemplate(
        input_variables=["cv_text", "today"],
        template="""You are a STRICT CV INFORMATION EXTRACTION ENGINE.

GLOBAL RULES:
- Extract ONLY verbatim text from the CV.
- Do NOT calculate months or years.
- Do NOT infer or normalize or assume.
- Output ONLY valid JSON.

EXPERIENCE RULES:
- Treat Present as {today}
- Extract ALL roles under Professional Experience (including internships).
- For each role extract EXACT text:
  job_title, company, start, end, description (brief summary of responsibilities/achievements)
- If end date is current/present, use "{today}" for end only for current/present
- Process date ranges intelligently
- Extract job description/responsibilities if available

KEYWORDS:
- Prefer MAXIMUM keyword extraction that are relevant to job title
- Keywords must be ATOMIC (1–3 words)
- Split tool lists into individual tools
- Get keywords from statements as well
- you can get it from skills area project description or job description

EDUCATION:
- Extract highest education EXACTLY as written.

NOTE:
- Don't confuse the cv title with role

INPUT CV:
\"\"\"
{cv_text}
\"\"\"

RETURN JSON ONLY:

{{
  "cv_profile": {{
    "roles": [
      {{
        "job_title": "",
        "company": "",
        "start": "",
        "end": "",
        "description": ""
      }}
    ],
    "keywords": [],
    "highest_education": ""
  }}
}}""",
    )

    parser = JsonOutputParser()
    chain = prompt_template | base_llm | parser

    try:
        result = chain.invoke({"cv_text": cv_text, "today": today})
        logger.debug("CV parsing result: {}", json.dumps(result, indent=2))
        logger.info("✅ Successfully parsed CV")
        return result["cv_profile"]
    except Exception as e:
        logger.error("❌ Failed to parse CV: {}", e)
        raise
