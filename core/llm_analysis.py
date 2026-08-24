"""
llm_analysis.py
Uses Google Gemini (free tier) to (1) extract structured skills/requirements
from a job description, (2) diff those against the resume to find gaps, and
(3) suggest concrete resume edits. All calls request strict JSON so the
output can be rendered directly in the UI.
"""

from __future__ import annotations
import json
import os

from google import genai
from google.genai import types

MODEL = "gemini-3.6-flash"

_client = None


def _get_client() -> genai.Client:
    global _client
    if _client is None:
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise RuntimeError(
                "GEMINI_API_KEY is not set. Add it to your environment "
                "or .env file before running the app."
            )
        _client = genai.Client(api_key=api_key)
    return _client


def _call_json(system: str, user: str, max_tokens: int = 1000) -> dict:
    client = _get_client()
    response = client.models.generate_content(
        model=MODEL,
        contents=user,
        config=types.GenerateContentConfig(
            system_instruction=system,
            response_mime_type="application/json",
            max_output_tokens=max_tokens,
        ),
    )
    raw = response.text or ""
    cleaned = raw.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
    return json.loads(cleaned)


def extract_requirements(jd_text: str) -> dict:
    system = (
        "You extract structured hiring requirements from job descriptions. "
        "Respond with ONLY valid JSON, no preamble, no markdown fences, "
        "matching this exact schema: "
        '{"required_skills": [string], "nice_to_have_skills": [string], '
        '"min_years_experience": number|null, "key_responsibilities": [string]}'
    )
    return _call_json(system, jd_text)


def find_gaps(resume_text: str, requirements: dict) -> dict:
    system = (
        "You compare a resume against a list of job requirements. For each "
        "skill in required_skills and nice_to_have_skills, decide if the "
        "resume provides clear evidence of that skill (not just keyword "
        "presence -- look for real usage/experience). Respond with ONLY "
        "valid JSON matching this schema: "
        '{"present_required": [string], "missing_required": [string], '
        '"present_nice_to_have": [string], "missing_nice_to_have": [string]}'
    )
    user = (
        f"REQUIREMENTS:\n{json.dumps(requirements)}\n\n"
        f"RESUME:\n{resume_text}"
    )
    return _call_json(system, user)


def suggest_edits(resume_text: str, jd_text: str, gaps: dict) -> dict:
    system = (
        "You are a career coach helping a candidate tailor their resume to "
        "a specific job description without fabricating experience. Given "
        "the resume, the job description, and a list of missing skills, "
        "suggest specific ways the candidate could reframe EXISTING resume "
        "content to better surface relevant experience, plus honest notes "
        "on genuine gaps they cannot paper over. Respond with ONLY valid "
        "JSON matching this schema: "
        '{"suggested_edits": [{"original_or_area": string, "suggested_rewrite": string, '
        '"reason": string}], "honest_gap_notes": [string]}'
    )
    user = (
        f"JOB DESCRIPTION:\n{jd_text}\n\n"
        f"RESUME:\n{resume_text}\n\n"
        f"IDENTIFIED GAPS:\n{json.dumps(gaps)}"
    )
    return _call_json(system, user, max_tokens=3000)


def full_analysis(resume_text: str, jd_text: str) -> dict:
    requirements = extract_requirements(jd_text)
    gaps = find_gaps(resume_text, requirements)
    edits = suggest_edits(resume_text, jd_text, gaps)
    return {
        "requirements": requirements,
        "gaps": gaps,
        "edits": edits,
    }