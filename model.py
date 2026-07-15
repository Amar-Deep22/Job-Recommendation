

import re
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from difflib import SequenceMatcher

from utils import (
    parse_skill_list, extract_skills_from_text,
    clean_text, detect_experience_level
)

# ─── Hybrid scoring weights ───────────────────────────────
SKILL_WEIGHT = 0.70
DESC_WEIGHT  = 0.30

# ─── Category cap: no category > this fraction of results ─
CATEGORY_CAP = 0.45

# ─── Title similarity threshold (dedup near-dupes) ────────
TITLE_SIMILARITY_THRESHOLD = 0.82

# ─── Confidence tier thresholds ───────────────────────────
TIERS = [
    (70, "Strong Match",  "🟢", "#10B981"),
    (45, "Good Match",    "🟡", "#F59E0B"),
    (25, "Partial Match", "🟠", "#F97316"),
    ( 0, "Weak Match",    "🔴", "#EF4444"),
]

# ─── Experience-level job keywords for soft-filter ────────
EXP_KEYWORDS = {
    "Senior": ["senior", "lead", "principal", "head", "director", "manager", "vp"],
    "Entry":  ["junior", "trainee", "intern", "graduate", "fresher", "entry", "associate"],
}


# ═══════════════════════════════════════════════════════════
# STEP 1 — LOAD & CLEAN DATA
# ═══════════════════════════════════════════════════════════
def load_and_clean_data(csv_path: str) -> pd.DataFrame:
    """
    Load CSV, normalize skills, clean descriptions,
    deduplicate on (title + skills), and reset index.
    """
    df = pd.read_csv(csv_path)

    # Keep only needed columns
    keep = [c for c in ["job_id", "category", "job_title",
                         "job_description", "job_skill_set"] if c in df.columns]
    df = df[keep].copy()

    # Drop rows without title or skills
    df.dropna(subset=["job_title", "job_skill_set"], inplace=True)

    # Normalize job titles
    df["job_title"] = df["job_title"].str.strip().str.title()

    # Parse and normalize skill lists
    df["skills_list"] = df["job_skill_set"].apply(parse_skill_list)
    df = df[df["skills_list"].map(len) > 0].copy()

    # Clean description for description-TF-IDF
    if "job_description" in df.columns:
        df["desc_clean"] = df["job_description"].apply(
            lambda t: clean_text(t, remove_stopwords=True)
        )
    else:
        df["desc_clean"] = ""

    # Skills text for skill-TF-IDF
    df["skills_text"] = df["skills_list"].apply(lambda lst: " ".join(lst))

    # Deduplicate: same title + same skill set = duplicate
    df["_dedup_key"] = df.apply(
        lambda r: r["job_title"].lower() + "|" + ",".join(sorted(r["skills_list"])),
        axis=1,
    )
    df.drop_duplicates(subset="_dedup_key", inplace=True)
    df.drop(columns=["_dedup_key", "job_skill_set"], errors="ignore", inplace=True)
    df.reset_index(drop=True, inplace=True)

    return df


# ═══════════════════════════════════════════════════════════
# STEP 2 — BUILD TF-IDF MODELS
# ═══════════════════════════════════════════════════════════
def build_models(df: pd.DataFrame):
    """
    Build two TF-IDF vectorizers:
      skill_vec  → trained on skill strings (n-grams 1-2)
      desc_vec   → trained on cleaned descriptions (n-grams 1-2)

    Returns: skill_vec, skill_matrix, desc_vec, desc_matrix
    """
    # Skill TF-IDF — sublinear_tf reduces impact of very common skills
    skill_vec = TfidfVectorizer(
        ngram_range=(1, 2),
        min_df=1,
        max_df=0.90,
        sublinear_tf=True,
    )
    skill_matrix = skill_vec.fit_transform(df["skills_text"])

    # Description TF-IDF — stopwords already removed in clean_text
    desc_vec = TfidfVectorizer(
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.80,
        sublinear_tf=True,
        max_features=10000,
    )
    # If description is too short, fall back to skills_text
    desc_corpus = df["desc_clean"].where(
        df["desc_clean"].str.len() > 30, df["skills_text"]
    )
    desc_matrix = desc_vec.fit_transform(desc_corpus)

    return skill_vec, skill_matrix, desc_vec, desc_matrix


# ═══════════════════════════════════════════════════════════
# STEP 3 — BUILD SKILL VOCABULARY
# ═══════════════════════════════════════════════════════════
def build_skill_vocabulary(df: pd.DataFrame):
    """
    Build a set of all known skills and a frequency dict.
    Frequency = how many jobs require each skill (market demand proxy).
    """
    skill_vocab: set = set()
    skill_freq: dict = {}
    for skill_list in df["skills_list"]:
        for skill in skill_list:
            skill_vocab.add(skill)
            skill_freq[skill] = skill_freq.get(skill, 0) + 1
    return skill_vocab, skill_freq


# ═══════════════════════════════════════════════════════════
# STEP 4 — RECOMMEND JOBS
# ═══════════════════════════════════════════════════════════
def recommend_jobs(
    resume_text: str,
    df: pd.DataFrame,
    skill_vec, skill_matrix,
    desc_vec, desc_matrix,
    skill_vocab: set,
    skill_freq: dict,
    category_filter: str = "All",
    top_n: int = 5,
) -> list:
    """
    Hybrid recommendation engine:
      hybrid_score = SKILL_WEIGHT × cos(skill) + DESC_WEIGHT × cos(desc)

    Features:
      - Experience-level soft penalty
      - Category dominance cap
      - Title deduplication
      - Skill gap analysis
      - Confidence tier tagging
    """

    # ── Extract resume info ───────────────────────────────
    resume_skills  = extract_skills_from_text(resume_text, skill_vocab)
    exp_level      = detect_experience_level(resume_text)
    resume_skill_s = set(resume_skills)

    # Build input vectors
    skill_input = " ".join(resume_skills) if resume_skills else clean_text(resume_text)
    desc_input  = clean_text(resume_text, remove_stopwords=True)

    if not skill_input.strip():
        return []

    try:
        resume_sv = skill_vec.transform([skill_input])
        resume_dv = desc_vec.transform([desc_input])
    except Exception:
        return []

    # ── Apply category filter ─────────────────────────────
    if category_filter and category_filter != "All":
        mask        = df["category"] == category_filter
        filtered_df = df[mask].copy()
        idx_arr     = filtered_df.index.to_numpy()
    else:
        filtered_df = df
        idx_arr     = np.arange(len(df))

    if filtered_df.empty:
        return []

    f_skill_mat = skill_matrix[idx_arr]
    f_desc_mat  = desc_matrix[idx_arr]

    # ── Compute cosine similarity ─────────────────────────
    skill_sim = cosine_similarity(resume_sv, f_skill_mat).flatten()
    desc_sim  = cosine_similarity(resume_dv, f_desc_mat).flatten()

    # ── Hybrid score ──────────────────────────────────────
    hybrid = SKILL_WEIGHT * skill_sim + DESC_WEIGHT * desc_sim

    # ── Experience-level soft penalty ─────────────────────
    # Penalise jobs clearly outside the detected level by 15%
    hybrid = _apply_exp_penalty(hybrid, filtered_df, exp_level)

    # ── Candidate pool — take 6× top_n for diversity pass ─
    n_cand   = min(top_n * 6, len(hybrid))
    cand_idx = np.argsort(hybrid)[::-1][:n_cand]

    # ── Build results with diversity filtering ─────────────
    results        = []
    seen_titles    = []
    category_count = {}

    for idx in cand_idx:
        score = float(hybrid[idx])
        if score <= 0.0:
            break

        row        = filtered_df.iloc[int(idx)]
        job_title  = str(row["job_title"])
        category   = str(row["category"])
        job_skills = set(row["skills_list"])

        if not job_skills:
            continue

        # ── Title similarity dedup ────────────────────────
        if _is_duplicate_title(job_title, seen_titles):
            continue

        # ── Category cap ─────────────────────────────────
        cat_max = max(1, int(top_n * CATEGORY_CAP))
        if category_count.get(category, 0) >= cat_max:
            continue

        # ── Skill gap analysis ────────────────────────────
        matched     = sorted(job_skills & resume_skill_s)
        missing     = sorted(job_skills - resume_skill_s)
        recommended = _rank_by_demand(missing, skill_freq, max_n=8)

        match_pct   = round(len(matched) / len(job_skills) * 100, 1) if job_skills else 0.0
        tier_label, tier_icon, tier_color = _get_tier(match_pct)

        # ── Record result ─────────────────────────────────
        results.append({
            "job_title":          job_title,
            "category":           category,
            "match_pct":          match_pct,
            "hybrid_score":       round(score * 100, 2),
            "skill_score":        round(float(skill_sim[int(idx)]) * 100, 2),
            "desc_score":         round(float(desc_sim[int(idx)]) * 100, 2),
            "tier":               tier_label,
            "tier_icon":          tier_icon,
            "tier_color":         tier_color,
            "experience_level":   exp_level,
            "matched_skills":     matched,
            "missing_skills":     missing,
            "recommended_skills": recommended,
            "total_job_skills":   len(job_skills),
            "skill_coverage_pct": match_pct,
        })

        seen_titles.append(job_title.lower())
        category_count[category] = category_count.get(category, 0) + 1

        if len(results) >= top_n:
            break

    # Final sort: match_pct first, hybrid_score as tiebreak
    results.sort(key=lambda x: (x["match_pct"], x["hybrid_score"]), reverse=True)
    return results[:top_n]


# ═══════════════════════════════════════════════════════════
# HELPER FUNCTIONS
# ═══════════════════════════════════════════════════════════
def _rank_by_demand(missing: list, skill_freq: dict, max_n: int = 8) -> list:
    """Sort missing skills by market demand (higher freq = more in-demand)."""
    return sorted(missing, key=lambda s: skill_freq.get(s, 0), reverse=True)[:max_n]


def _get_tier(pct: float):
    """Return (label, icon, color) tier for a match percentage."""
    for threshold, label, icon, color in TIERS:
        if pct >= threshold:
            return label, icon, color
    return "Weak Match", "🔴", "#EF4444"


def _is_duplicate_title(title: str, seen: list, threshold: float = TITLE_SIMILARITY_THRESHOLD) -> bool:
    """Check if a job title is too similar to one already selected."""
    title_lower = title.lower()
    for s in seen:
        ratio = SequenceMatcher(None, title_lower, s).ratio()
        if ratio >= threshold:
            return True
    return False


def _apply_exp_penalty(scores: np.ndarray, df: pd.DataFrame, exp_level: str) -> np.ndarray:
    """
    Apply a 15% penalty to jobs that are clearly outside the detected experience level.
    'Senior' resume → penalise Entry-only jobs.
    'Entry' resume  → penalise Senior-only jobs.
    'Mid' → no penalty.
    """
    if exp_level == "Mid":
        return scores

    scores = scores.copy()
    titles = df["job_title"].str.lower().to_numpy()

    if exp_level == "Senior":
        # Penalise jobs with entry-level keywords
        entry_kw = EXP_KEYWORDS["Entry"]
        for i, title in enumerate(titles):
            if any(kw in title for kw in entry_kw):
                scores[i] *= 0.85

    elif exp_level == "Entry":
        # Penalise jobs with senior-level keywords
        senior_kw = EXP_KEYWORDS["Senior"]
        for i, title in enumerate(titles):
            if any(kw in title for kw in senior_kw):
                scores[i] *= 0.85

    return scores
