"""
embedder.py
Computes a quantitative resume <-> job description match score using
sentence embeddings. Local model, no API cost, fast enough for a live demo.
"""

from __future__ import annotations
from functools import lru_cache

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


MODEL_NAME = "all-MiniLM-L6-v2"


@lru_cache(maxsize=1)
def _get_model() -> SentenceTransformer:
    # Cached so the model loads once per process, not once per request.
    return SentenceTransformer(MODEL_NAME)


def embed(text: str):
    model = _get_model()
    return model.encode([text], normalize_embeddings=True)


def similarity_score(resume_text: str, jd_text: str) -> float:
    """
    Returns a 0-100 match score based on cosine similarity of the
    full-document embeddings. This is a coarse signal -- pair it with
    the LLM-based skill-gap analysis for something explainable.
    """
    resume_vec = embed(resume_text)
    jd_vec = embed(jd_text)
    sim = cosine_similarity(resume_vec, jd_vec)[0][0]
    # cosine similarity for MiniLM embeddings on real text tends to sit
    # in a compressed ~0.2-0.8 band; rescale so the score uses the full
    # 0-100 range in a way that roughly matches human intuition.
    rescaled = max(0.0, min(1.0, (sim - 0.2) / 0.6))
    return round(rescaled * 100, 1)


def section_scores(resume_sections: dict[str, str], jd_text: str) -> dict[str, float]:
    """
    Optional finer-grained view: score individual resume sections
    (e.g. 'experience', 'skills') against the JD separately.
    """
    return {
        name: similarity_score(section_text, jd_text)
        for name, section_text in resume_sections.items()
        if section_text.strip()
    }
