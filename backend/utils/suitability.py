from loguru import logger

from backend.utils.matchers import (
    match_education_with_llm,
    match_experience_with_llm,
    match_keywords_with_llm,
)


def calculate_suitability_with_llm(jd_profile: dict, cv_profile: dict) -> dict:
    """
    Calculate suitability score using LLM for intelligent matching
    - Work Experience (50%)
    - Job-specific Keywords & Skills (40%)
    - Education (10%)
    """
    logger.info("🎯 Starting suitability calculation")
    logger.debug(
        f"JD keywords: {len(jd_profile.get('keywords', []))}, CV keywords: {len(cv_profile.get('keywords', []))}"
    )

    logger.info("🔍 [1/3] Matching keywords with LLM...")
    keyword_match = match_keywords_with_llm(
        jd_profile.get("keywords", []), cv_profile.get("keywords", [])
    )

    logger.info("🎓 [2/3] Matching education with LLM...")
    education_match = match_education_with_llm(
        jd_profile.get("education_requirements", ""),
        cv_profile.get("highest_education", ""),
    )

    logger.info("💼 [3/3] Matching experience with thinking model...")
    experience_match = match_experience_with_llm(
        jd_profile.get("required_experience", 0.0),
        cv_profile.get("total_years_of_relevant_experience", 0.0),
    )

    # Cap experience match at 100%
    if experience_match.get("match_percentage", 0) > 100:
        experience_match["match_percentage"] = 100

    # Cap education match at 100%
    if education_match.get("match_percentage", 0) > 100:
        education_match["match_percentage"] = 100

    # Calculate weighted scores
    keyword_score = (
        (
            keyword_match.get("total_matched", 0)
            / keyword_match.get("total_jd_keywords", 1)
        )
        * 40
        if keyword_match.get("total_jd_keywords", 0) > 0
        else 0
    )
    education_score = (education_match.get("match_percentage", 0) / 100) * 10
    experience_score = (experience_match.get("match_percentage", 0) / 100) * 50

    total_score = round(keyword_score + education_score + experience_score, 2)

    # Classification
    if total_score >= 75:
        classification = "Perfect Match"
    elif total_score >= 55:
        classification = "Suitable Match"
    elif total_score >= 35:
        classification = "Medium Match"
    else:
        classification = "Weak Match"

    logger.info(
        f"✅ Suitability calculation complete: {total_score}/100 ({classification})"
    )
    logger.debug(
        f"Scores - Experience: {experience_score}, Keywords: {keyword_score}, Education: {education_score}"
    )

    return {
        "total_score": total_score,
        "classification": classification,
        "breakdown": {
            "experience_score": round(experience_score, 2),
            "experience_weight": "50%",
            "experience_details": experience_match,
            "keyword_score": round(keyword_score, 2),
            "keyword_weight": "40%",
            "keyword_details": keyword_match,
            "education_score": round(education_score, 2),
            "education_weight": "10%",
            "education_details": education_match,
        },
    }
