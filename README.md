# Resume ↔ Job Match Assistant

Upload a resume and paste a job description. Get back:
- A quantitative **match score** (sentence-embedding cosine similarity)
- A structured **skill gap breakdown** (present vs. missing, required vs. nice-to-have)
- **Concrete rewrite suggestions** grounded in your actual experience — not fabricated bullet points

Built to be genuinely useful for your own job search, and to demonstrate an end-to-end ML pipeline: embeddings, structured LLM extraction, and a small evaluation harness that checks the score against human judgment.

## Architecture

```
resume/JD text
      │
      ├─► core/embedder.py    → sentence-transformers cosine similarity → 0-100 match score
      │
      └─► core/llm_analysis.py → Claude (3 calls, structured JSON):
              1. extract_requirements(jd)      → required/nice-to-have skills, years exp
              2. find_gaps(resume, requirements) → present vs. missing skills
              3. suggest_edits(resume, jd, gaps) → specific rewrite suggestions
      │
      └─► app.py (Streamlit UI) → renders score, gap table, suggested edits
```

`core/parser.py` handles PDF/txt extraction and text cleanup so the rest of the pipeline just works with plain strings.

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env   # then add your ANTHROPIC_API_KEY
streamlit run app.py
```

The first run will download the `all-MiniLM-L6-v2` embedding model (~90MB) from Hugging Face — needs an open internet connection once, then it's cached locally.

## Evaluation

`eval/` contains a small labeled set of synthetic resume/JD pairs (`eval_set.csv`) with human-judged relevance scores from 0–100. `eval/run_eval.py` computes the embedding-based score for each pair and reports Pearson/Spearman correlation against the human labels:

```bash
python -m eval.run_eval
```

This is the piece that turns the project from "a demo" into "a demo with evidence it works." Before using any correlation number in something like a resume, **expand the eval set** with real resume/JD pairs (ideally 30+, scored blind by you or a friend) rather than relying on the 10 synthetic pairs shipped here.

## Deploying

Fastest path: [Streamlit Community Cloud](https://streamlit.io/cloud) or a [Hugging Face Space](https://huggingface.co/spaces) — both have free tiers and deploy directly from a GitHub repo. Add `ANTHROPIC_API_KEY` as a secret in the deployment settings rather than committing `.env`.

## Resume framing

Once deployed with a real eval number, a bullet like this is accurate and specific:

> Built and deployed an ML-powered resume-matching tool combining sentence embeddings and LLM-based structured gap analysis; validated match scores against human relevance judgments (Pearson r = X.XX on a N-pair eval set).

Fill in the real X.XX and N from your own expanded eval run — don't reuse the placeholder numbers from the shipped sample set.

## Project layout

```
resume-matcher/
├── app.py                  # Streamlit UI
├── core/
│   ├── parser.py            # PDF/txt extraction + cleaning
│   ├── embedder.py           # embedding similarity score
│   └── llm_analysis.py       # Claude-based requirement extraction, gap analysis, edit suggestions
├── eval/
│   ├── eval_set.csv          # labeled resume/JD pairs
│   └── run_eval.py           # correlation of model score vs. human judgment
├── sample_data/
│   ├── resumes.py            # synthetic resumes used in eval
│   └── jobs.py                # synthetic job descriptions used in eval
├── requirements.txt
└── .env.example
```

## Possible extensions (good "what would you improve" interview answers)

- Swap the single full-document embedding for **section-level** scoring (`embedder.section_scores` is stubbed in) so "Skills" and "Experience" sections are weighted separately
- Add a **cost/latency comparison** between using Claude for gap analysis vs. a cheaper keyword-matching baseline
- Expand the eval set and track score **drift** as you tweak prompts, so prompt changes are validated rather than vibes-based
- Add **caching** of JD requirement extraction so re-running against multiple resumes for the same job doesn't re-call the LLM each time
