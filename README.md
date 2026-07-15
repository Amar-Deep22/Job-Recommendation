<div align="center">

# ✦ JobDhundho — AI Resume ↔ Job Matcher

**A hybrid ML-powered job recommendation engine** that reads a resume (PDF), extracts skills and experience level, and matches it against real-world job postings using TF-IDF + cosine similarity — with full skill-gap analysis and a premium dark-theme Streamlit UI.

[![Python](https://img.shields.io/badge/Python-3.9%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-F7931E?logo=scikitlearn&logoColor=white)](https://scikit-learn.org/)
[![License](https://img.shields.io/badge/License-MIT-green)](#-license)

*"JobDhundho" ("job dhundho" = "search for a job" in Hindi) helps job seekers instantly see how well their resume fits open roles, what skills they're missing, and what to learn next.*

</div>

---

## 📑 Table of Contents

- [Features](#-features)
- [Project Structure](#-project-structure)
- [How It Works](#-how-it-works)
- [Setup](#️-setup)
- [Tech Stack](#-tech-stack)
- [Notes & Limitations](#-notes--limitations)
- [Possible Improvements](#-possible-improvements)
- [Connect With Me](#-connect-with-me)
- [License](#-license)

---

## ✨ Features

- **Resume parsing** — extracts raw text from PDF resumes via PyMuPDF (`fitz`)
- **Skill extraction** — matches resume text against a normalized skill vocabulary built from the job dataset, using 150+ alias mappings (e.g. `js` → `javascript`, `ml` → `machine learning`)
- **Experience-level detection** — classifies resumes as `Entry`, `Mid`, or `Senior` using regex patterns over years-of-experience and title keywords
- **Hybrid scoring engine**
  - `Skill Score` (70%) — cosine similarity between resume skills and job skill sets
  - `Description Score` (30%) — cosine similarity between resume text and job descriptions
  - `Hybrid Score` = weighted combination of the two
- **Smart ranking & diversity**
  - Experience-level soft penalty (15%) for clearly mismatched seniority
  - Category dominance cap so one category can't flood the results
  - Near-duplicate job title filtering (`difflib.SequenceMatcher`)
- **Skill gap analysis** — per job, shows **matched**, **missing**, and **recommended-to-learn** skills (missing skills ranked by market demand/frequency)
- **Confidence tiers** — 🟢 Strong / 🟡 Good / 🟠 Partial / 🔴 Weak Match, based on skill coverage %
- **Interactive dashboard** (Streamlit)
  - Resume analysis panel (detected skills + experience badge)
  - Recommendations tab — ranked job cards with score breakdowns and skill chips
  - Skill Gap tab — top missing/matched skills, match % bar chart
  - Analysis tab — full results table + resume stats

---

## 🗂 Project Structure

```
JobDhundho/
├── app.py                 # Streamlit frontend (UI, styling, page flow)
├── model.py                # Recommendation engine (TF-IDF, scoring, ranking)
├── utils.py                 # Text cleaning, skill parsing/aliasing, PDF extraction
├── all_job_post.csv          # Job postings dataset
├── requirements.txt
└── README.md
```

---

## 🧠 How It Works

```
Resume PDF
    │
    ▼
extract_text_from_pdf()          (utils.py — PyMuPDF)
    │
    ▼
extract_skills_from_text()       → resume skills (matched against skill vocabulary)
detect_experience_level()        → Entry / Mid / Senior
    │
    ▼
recommend_jobs()                 (model.py)
    ├── Skill TF-IDF cosine similarity   × 0.70
    ├── Description TF-IDF cosine similarity × 0.30
    ├── Experience-level penalty (±15%)
    ├── Category cap + title dedup
    └── Skill gap analysis (matched / missing / recommended)
    │
    ▼
Top-N ranked jobs with match %, tier badge, and skill breakdown
```

### Dataset schema (`all_job_post.csv`)

| Column | Description |
|---|---|
| `job_id` | Unique job identifier |
| `category` | Job category/industry |
| `job_title` | Job title |
| `job_description` | Full job description text |
| `job_skill_set` | List/string of required skills |

---

## ⚙️ Setup

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Run the app
```bash
python -m streamlit run app.py
```

The app will open at `http://localhost:8501`. Make sure `all_job_post.csv` is in the same directory as `app.py` (the app loads it via `Path(__file__).parent / "all_job_post.csv"`).

### 3. Use it
1. Upload a **text-based** PDF resume (scanned/image-only PDFs won't parse) in the sidebar
2. Optionally filter by **Category** and choose the **number of results** (3/5/7/10)
3. Toggle **Show Score Breakdown** to see Skill/Description/Hybrid scores per job
4. Review your matches across the **Recommendations**, **Skill Gap**, and **Analysis** tabs

---

## 🛠 Tech Stack

| Layer | Tool |
|---|---|
| Frontend | Streamlit (custom dark-theme CSS) |
| ML / NLP | scikit-learn (TF-IDF, cosine similarity) |
| PDF parsing | PyMuPDF (`fitz`) |
| Data handling | pandas, numpy |
| Text matching | Python `re`, `difflib` |

---

## 📌 Notes & Limitations

- Skill extraction quality depends heavily on the resume having a clear **Skills** section — free-form prose resumes may under-match.
- Only **text-based PDFs** are supported; scanned/image resumes will raise a parsing error.
- Experience-level detection uses heuristic regex patterns, not a trained classifier — edge cases (e.g. career changers) may be misclassified.
- The model and vectorizers are rebuilt from `all_job_post.csv` on first load and cached via `st.cache_resource` for the session.

---

## 🚀 Possible Improvements

- Replace TF-IDF with sentence embeddings (e.g. `sentence-transformers`) for semantic matching
- Fine-tune experience-level detection with a trained classifier
- Add multi-resume comparison / batch mode
- Persist and version the skill vocabulary instead of rebuilding per session

---

## 🤝 Connect With Me

Built by **Amardeep Ranjan** — B.Tech CSE student, currently exploring roles in IoT, embedded systems, and applied ML.

Feel free to connect, fork the repo, raise issues, or suggest improvements — always happy to discuss the project or collaborate!

<p align="left">
  <a href="https://www.linkedin.com/in/amardeep-ranjan05"><img src="https://img.shields.io/badge/LinkedIn-Connect-0A66C2?logo=linkedin&logoColor=white" alt="LinkedIn"/></a>
  <a href="mailto:amardeepranjansingh@gmail.com"><img src="https://img.shields.io/badge/Email-Reach%20Out-D14836?logo=gmail&logoColor=white" alt="Email"/></a>
</p>

---

## 📄 License

This project is open-sourced under the [MIT License](LICENSE) — feel free to use, modify, and build on it with attribution.

<div align="center">

⭐ If you found this project useful, consider giving it a star!

</div>
