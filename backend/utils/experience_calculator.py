import json
from datetime import datetime

from dateutil import parser
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import PromptTemplate
from loguru import logger

from backend.core.deps import thinking_llm


def calculate_months_between(start_str: str, end_str: str, today_str: str) -> int:
    """Calculate months between two dates, handling 'Present'"""
    try:
        # Handle end date
        if not end_str or end_str.lower() in ["present", "current", "now", ""]:
            end_str = today_str
            logger.debug(f"Using today ({today_str}) for 'present' end date")

        # Parse dates
        start_date = parser.parse(start_str, fuzzy=True)
        end_date = parser.parse(end_str, fuzzy=True)

        # Calculate months (inclusive of both start and end month)
        months = (
            (end_date.year - start_date.year) * 12
            + (end_date.month - start_date.month)
            + 1
        )
        result = max(months, 1)  # Minimum 1 month
        logger.debug(f"Calculated {result} months for {start_str} to {end_str}")
        return result
    except Exception as e:
        logger.warning(f"Failed to calculate months for {start_str} to {end_str}: {e}")
        return 0


def validate_and_recalculate_experience(cv_profile: dict, today: str) -> dict:
    """Recalculate experience to ensure accuracy"""
    total_months = 0
    relevant_months = 0

    all_ranges = cv_profile.get("total_experience_ranges", [])

    for exp in all_ranges:
        # Extract date range
        range_str = exp.get("range", "")

        # Try to parse from range string first
        if " to " in range_str:
            start_str, end_str = range_str.split(" to ")
        else:
            # Fallback to roles data
            roles = cv_profile.get("roles", [])
            matching_role = next(
                (r for r in roles if r["job_title"] == exp.get("role")), None
            )
            if matching_role:
                start_str = matching_role.get("start", "")
                end_str = matching_role.get("end", today)
            else:
                continue

        # Calculate actual months
        months = calculate_months_between(start_str, end_str, today)
        years = round(months / 12, 2)

        # Update the experience entry
        exp["years"] = years
        exp["calculated_months"] = months

        total_months += months
        if exp.get("is_relevant"):
            relevant_months += months

    # Update totals
    cv_profile["total_years_of_experience"] = round(total_months / 12, 2)
    cv_profile["total_years_of_relevant_experience"] = round(relevant_months / 12, 2)

    # Update relevant ranges
    cv_profile["relevant_experience_ranges"] = [
        exp for exp in all_ranges if exp.get("is_relevant") is True
    ]

    return cv_profile


def enrich_with_experience_llm(cv_profile: dict, jd_position: str) -> dict:
    logger.info("⚡ Starting experience enrichment with LLM")
    logger.debug(
        f"Position: {jd_position}, Roles count: {len(cv_profile.get('roles', []))}"
    )
    today = datetime.today().strftime("%B %Y")

    prompt_template = PromptTemplate(
        input_variables=["today", "jd_position", "roles"],
        template="""You are an experience calculation expert for recruitment. You must think step-by-step and show your calculations.

TODAY'S DATE: {today}

TASK:
From the CV roles, calculate:
1. Total experience (sum of ALL roles in years, 2 decimals)
2. Relevant experience for the given JD position (sum of ONLY relevant roles in years, 2 decimals)
3. Structured role-wise experience ranges

STEP-BY-STEP CALCULATION PROCESS:
1. For each role:
   a. Parse start date (extract month and year)
   b. Parse end date (if "Present/Current/Now", use {today})
   c. Calculate duration in months: (end_year - start_year) × 12 + (end_month - start_month) + 1
   d. **CRITICAL: Convert to years by dividing months by 12**
   e. Round to 2 decimal places
   f. Check if role is relevant to JD position (check BOTH job title AND description)

2. Sum all role durations for total experience
3. Sum only relevant role durations for relevant experience

**EXAMPLE CALCULATION:**
- Start: Aug 2023 (2023.67)
- End: Jan 2026 (2026.08)
- Months = (2026 - 2023) × 12 + (1 - 8) + 1 = 36 + (-7) + 1 = 30 months
- **Years = 30 ÷ 12 = 2.50 years** ← THIS IS THE CORRECT FORMAT

RELEVANCE CRITERIA (IMPORTANT - Check BOTH title AND description):
- Job title contains keywords from JD position
- Job description contains tasks/responsibilities related to JD position
- Similar domain (e.g., "Security Analyst" is relevant to "Cyber Security Analyst")
- Similar role (e.g., "QA Tester" is relevant to "Software Tester")
- If job title is different BUT description shows relevant work, mark as RELEVANT
  Example: "Software Developer" with description mentioning "AI model development, machine learning"
  is RELEVANT to "AI/ML Engineer"
- NOT relevant: completely different fields with no overlap in description
  Example: "Web Developer" doing only frontend work vs "Security Analyst"

STRICT RULES:
- **YEARS VALUE MUST BE IN YEARS (DECIMAL), NOT MONTHS**
- Show your calculation steps in thinking
- Use ONLY the given CV data
- Parse month/year text intelligently (e.g., "Aug 2023", "August 2023", "2023-08", "2021-2026")
- If end date is "Present/Current/Now", use {today}
- ALWAYS analyze job description content for relevance, not just title
- If no relevant experience found, return 0.0 and empty relevant_experience_ranges list only
- total_years_of_experience means calculation of total_experience_ranges
- total_years_of_relevant_experience means calculation of relevant_experience_ranges
- Be precise with date calculations
- Output ONLY valid JSON

JD POSITION:
{jd_position}

CV ROLES:
{roles}

CALCULATION STEPS (show in your thinking):
For each role, show:
- Start: [month year] = [year].[month decimal]
- End: [month year] = [year].[month decimal]
- Months = calculation showing the formula
- **Years = months ÷ 12 = X.XX**
- Job Title: [title]
- Description keywords: [key terms from description]
- Is relevant? [yes/no and WHY - explain based on title AND description]

RETURN JSON ONLY (no markdown, no code blocks):
{{
  "total_years_of_experience": Float,
  "total_years_of_relevant_experience": Float,
  "total_experience_ranges": [
    {{
      "role": "",
      "company": "",
      "range": "Month-Year to Month-Year",
      "years": Float,
      "is_relevant": true,
      "relevance_reason": "Brief explanation why this role is/isn't relevant"
    }}
  ],
  "relevant_experience_ranges": [
    {{
      "role": "",
      "company": "",
      "range": "Month-Year to Month-Year",
      "years": Float,
      "relevance_reason": "Brief explanation of relevance"
    }}
  ]
}}""",
    )

    parser = JsonOutputParser()
    chain = prompt_template | thinking_llm | parser

    llm_result = chain.invoke(
        {
            "today": today,
            "jd_position": jd_position,
            "roles": json.dumps(cv_profile.get("roles", []), indent=2),
        }
    )

    # Store LLM results first
    cv_profile["total_experience_ranges"] = llm_result.get(
        "total_experience_ranges", []
    )

    # CRITICAL: Validate and recalculate experience with accurate math
    cv_profile = validate_and_recalculate_experience(cv_profile, today)

    logger.debug("Experience enrichment result: {}", json.dumps(cv_profile, indent=2))
    logger.info("✅ Successfully enriched CV with experience data")
    return cv_profile
