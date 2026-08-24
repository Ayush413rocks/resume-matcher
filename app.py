"""
Resume ↔ Job Match Assistant
Streamlit app: upload a resume + paste a job description, get a
quantitative match score, a skill gap breakdown, and specific
rewrite suggestions.

Run with:  streamlit run app.py
"""

import tempfile
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

import streamlit as st

from core.parser import load_document, clean_text
from core.embedder import similarity_score
from core.llm_analysis import full_analysis

st.set_page_config(page_title="Resume ↔ Job Match Assistant", page_icon="🎯", layout="wide")

st.title("🎯 Resume ↔ Job Match Assistant")
st.caption(
    "Upload your resume and paste a job description. Get a match score, "
    "a skill-gap breakdown, and concrete edit suggestions -- grounded in "
    "your real experience, not fabricated."
)

col1, col2 = st.columns(2)

with col1:
    st.subheader("Your resume")
    resume_file = st.file_uploader("Upload PDF or .txt", type=["pdf", "txt"])
    resume_text_input = st.text_area(
        "...or paste resume text directly", height=200, placeholder="Paste resume text here"
    )

with col2:
    st.subheader("Job description")
    jd_text_input = st.text_area(
        "Paste the job description", height=280, placeholder="Paste job description here"
    )

run = st.button("Analyze match", type="primary", use_container_width=True)

if run:
    # Resolve resume text from either upload or pasted text
    resume_text = ""
    if resume_file is not None:
        with tempfile.NamedTemporaryFile(delete=False, suffix=Path(resume_file.name).suffix) as tmp:
            tmp.write(resume_file.getvalue())
            tmp_path = tmp.name
        resume_text = load_document(tmp_path)
    elif resume_text_input.strip():
        resume_text = clean_text(resume_text_input)

    jd_text = clean_text(jd_text_input) if jd_text_input.strip() else ""

    if not resume_text or not jd_text:
        st.error("Please provide both a resume and a job description.")
        st.stop()

    with st.spinner("Scoring semantic match..."):
        score = similarity_score(resume_text, jd_text)

    with st.spinner("Extracting requirements and analyzing gaps with Claude..."):
        try:
            analysis = full_analysis(resume_text, jd_text)
        except Exception as e:
            st.error(f"LLM analysis failed: {e}")
            st.stop()

    st.divider()

    # --- Score ---
    st.subheader("Match score")
    score_col, bar_col = st.columns([1, 3])
    with score_col:
       st.metric("Semantic similarity", f"{round(score, 1)}%")
    with bar_col:
        st.progress(min(int(score), 100) / 100)
    st.caption(
        "Score is based on embedding similarity between your full resume and "
        "the job description. Treat it as a coarse signal -- the breakdown "
        "below is the more actionable part."
    )

    # --- Gaps ---
    gaps = analysis["gaps"]
    req = analysis["requirements"]

    st.subheader("Skill breakdown")
    g1, g2 = st.columns(2)
    with g1:
        st.markdown("**✅ Required skills you demonstrate**")
        for s in gaps.get("present_required", []):
            st.markdown(f"- {s}")
        if not gaps.get("present_required"):
            st.markdown("_None detected_")

        st.markdown("**❌ Required skills missing / not evidenced**")
        for s in gaps.get("missing_required", []):
            st.markdown(f"- {s}")
        if not gaps.get("missing_required"):
            st.markdown("_None -- strong match on required skills_")

    with g2:
        st.markdown("**✅ Nice-to-have skills you demonstrate**")
        for s in gaps.get("present_nice_to_have", []):
            st.markdown(f"- {s}")
        if not gaps.get("present_nice_to_have"):
            st.markdown("_None detected_")

        st.markdown("**⚠️ Nice-to-have skills missing**")
        for s in gaps.get("missing_nice_to_have", []):
            st.markdown(f"- {s}")
        if not gaps.get("missing_nice_to_have"):
            st.markdown("_None_")

    if req.get("min_years_experience"):
        st.info(f"Job appears to require ~{req['min_years_experience']}+ years of experience.")

    # --- Suggested edits ---
    st.subheader("Suggested resume edits")
    edits = analysis["edits"].get("suggested_edits", [])
    if edits:
        for i, e in enumerate(edits, 1):
            with st.expander(f"{i}. {e.get('original_or_area', 'Suggestion')}"):
                st.markdown(f"**Rewrite:** {e.get('suggested_rewrite', '')}")
                st.markdown(f"**Why:** {e.get('reason', '')}")
    else:
        st.markdown("_No specific edits suggested._")

    honest_notes = analysis["edits"].get("honest_gap_notes", [])
    if honest_notes:
        st.subheader("Honest gaps (can't be papered over)")
        for note in honest_notes:
            st.markdown(f"- {note}")

st.divider()
st.caption(
    "Built with sentence-transformers (local embeddings) + Gemini (structured "
    "gap analysis). See eval/run_eval.py for score validation against human judgment."
)
