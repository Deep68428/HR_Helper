import json

from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import PromptTemplate
from loguru import logger

from backend.core.deps import base_llm, thinking_llm


def match_keywords_with_llm(jd_keywords: list, cv_keywords: list) -> dict:
    prompt_template = PromptTemplate(
        input_variables=["jd_keywords", "cv_keywords"],
        template="""You are a keyword matching expert for job recruitment.

TASK:
Match keywords from the Job Description with keywords from the CV.
Consider synonyms, related terms, and semantic similarity.

MATCHING RULES:
- Exact matches (e.g., "Python" = "Python")
- Synonyms (e.g., "QA" = "Quality Assurance", "JS" = "JavaScript")
- Related terms (e.g., "Selenium" matches "Automation Testing")
- Tool versions count as matches (e.g., "React" = "React.js" = "ReactJS")
- Different names for same technology (e.g., "SQL Server" = "MS SQL")

JD KEYWORDS:
{jd_keywords}

CV KEYWORDS:
{cv_keywords}

RETURN JSON ONLY (no markdown, no code blocks):
{{
  "matched_keywords": [
    {{
      "jd_keyword": "keyword from JD",
      "cv_keyword": "matched keyword from CV",
      "match_type": "exact|synonym|related"
    }}
  ],
  "unmatched_jd_keywords": ["keywords from JD that have no match in CV"],
  "total_jd_keywords": 0,
  "total_matched": 0
}}""",
    )

    try:
        parser = JsonOutputParser()
        chain = prompt_template | base_llm | parser

        result = chain.invoke(
            {
                "jd_keywords": json.dumps(jd_keywords),
                "cv_keywords": json.dumps(cv_keywords),
            }
        )

        logger.debug("Keyword match result: {}", json.dumps(result, indent=2))
        logger.info(
            f"✅ Keyword matching complete: {result.get('total_matched', 0)}/{result.get('total_jd_keywords', 0)} matched"
        )

        return result

    except Exception as e:
        logger.error("❌ Failed in keyword matching: {}", e)
        return {
            "matched_keywords": [],
            "unmatched_jd_keywords": jd_keywords,
            "total_jd_keywords": len(jd_keywords),
            "total_matched": 0,
        }


def match_education_with_llm(jd_education: str, cv_education: str) -> dict:
    prompt_template = PromptTemplate(
        input_variables=["jd_education", "cv_education"],
        template="""You are an education requirement matcher for job recruitment.

EDUCATION HIERARCHY (lowest to highest):
1. High School / 12th
2. Diploma
3. Bachelor's (B.Tech, B.E, B.Sc, B.A, BS, BCA, B.Com)
4. Master's (M.Tech, M.Sc, MBA, MS, MA, MCA, M.Com)
5. PhD / Doctorate (Ph.D)

MATCHING RULES:
- Matched percentage should not go more than 100% STRICT
- If CV education == JD education: FULL MATCH (100%)
- If CV education > JD education: FULL MATCH (100%)
- If CV education < JD education but close (1 level below): PARTIAL MATCH (70%)
- If CV education < JD education by 2+ levels: WEAK MATCH (30%)
- If no JD education specified: DEFAULT MATCH (100%)

NOTE:
- If CV education is higher give 100%

JD EDUCATION REQUIREMENT:
{jd_education}

CV EDUCATION:
{cv_education}

RETURN JSON ONLY:
{{
  "jd_education_level": "level name from hierarchy or null",
  "cv_education_level": "level name from hierarchy or null",
  "match_status": "full|partial|weak|default",
  "match_percentage": 0,
  "explanation": "brief explanation of the match"
}}""",
    )

    try:
        parser = JsonOutputParser()
        chain = prompt_template | thinking_llm | parser

        result = chain.invoke(
            {
                "jd_education": jd_education if jd_education else "Not specified",
                "cv_education": cv_education if cv_education else "Not specified",
            }
        )

        logger.debug("Education match result: {}", json.dumps(result, indent=2))
        logger.info(
            f"✅ Education matching complete: {result.get('match_percentage', 0)}% match"
        )

        return result

    except Exception as e:
        logger.error("❌ Failed in education matching: {}", e)
        return {
            "jd_education_level": "Unknown",
            "cv_education_level": "Unknown",
            "match_status": "default",
            "match_percentage": 100,
            "explanation": "Unable to parse education levels, defaulting to 100%",
        }


def match_experience_with_llm(required_exp: float, relevant_exp: float) -> dict:
    logger.debug(
        f"Matching experience - Required: {required_exp}, Relevant: {relevant_exp}"
    )

    prompt_template = PromptTemplate(
        input_variables=["required_exp", "relevant_exp"],
        template="""You are an experience requirement matcher for job recruitment. Think step-by-step.

EXPERIENCE MATCHING RULES:
- Match percentage CANNOT exceed 100%
- If candidate relevant experience >= required experience: FULL MATCH (100%)
- If candidate relevant experience < required experience: Calculate (relevant_exp / required_exp) × 100
- If no required experience specified but candidate has experience: FULL MATCH (100%)
- If no required experience and no candidate experience: DEFAULT (50%)

CALCULATION STEPS:
1. Compare relevant_exp vs required_exp
2. Determine if >= (full match) or < (partial match)
3. Calculate percentage
4. Cap at 100% maximum

EXAMPLES:
- Required: 3.0, Relevant: 5.0 → 5.0 >= 3.0 → 100%
- Required: 3.0, Relevant: 2.0 → 2.0 < 3.0 → (2.0/3.0) × 100 = 66.67%
- Required: 2.0, Relevant: 2.5 → 2.5 >= 2.0 → 100%
- Required: 0.0, Relevant: 3.0 → No requirement → 100%

JD REQUIRED EXPERIENCE: {required_exp} years
CV RELEVANT EXPERIENCE: {relevant_exp} years

SHOW YOUR CALCULATION IN THINKING, THEN RETURN JSON ONLY (no markdown, no code blocks):
{{
  "required_experience": {required_exp},
  "relevant_experience": {relevant_exp},
  "match_percentage": 0,
  "explanation": "brief explanation with calculation"
}}""",
    )

    try:
        parser = JsonOutputParser()
        chain = prompt_template | thinking_llm | parser

        result = chain.invoke(
            {"required_exp": required_exp, "relevant_exp": relevant_exp}
        )

        logger.debug("Experience match result: {}", json.dumps(result, indent=2))
        logger.info(
            f"✅ Experience matching complete: {result.get('match_percentage', 0)}% match"
        )

        # Validate result has required fields
        if not result or "match_percentage" not in result:
            logger.warning("Invalid experience match result, calculating manually")
            return calculate_experience_match_fallback(required_exp, relevant_exp)

        return result

    except Exception as e:
        logger.error("❌ Failed in experience matching: {}, using fallback", e)
        return calculate_experience_match_fallback(required_exp, relevant_exp)


def calculate_experience_match_fallback(
    required_exp: float, relevant_exp: float
) -> dict:
    """Fallback function to calculate experience match without LLM"""
    if required_exp == 0.0:
        if relevant_exp > 0:
            match_percentage = 100.0
            explanation = (
                f"No specific experience required, candidate has {relevant_exp} years"
            )
        else:
            match_percentage = 50.0
            explanation = "No experience required or provided"
    elif relevant_exp >= required_exp:
        match_percentage = 100.0
        explanation = f"Candidate has {relevant_exp} years, meets requirement of {required_exp} years"
    else:
        match_percentage = round((relevant_exp / required_exp) * 100, 2)
        explanation = (
            f"Candidate has {relevant_exp} years out of required {required_exp} years"
        )

    return {
        "required_experience": required_exp,
        "relevant_experience": relevant_exp,
        "match_percentage": match_percentage,
        "explanation": explanation,
    }
