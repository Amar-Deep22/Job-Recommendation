
import re
import ast
import fitz  # PyMuPDF


# ═══════════════════════════════════════════════════════════
# STOPWORDS  — removed before TF-IDF
# ═══════════════════════════════════════════════════════════
STOPWORDS = {
    "a","an","the","and","or","but","in","on","at","to","for","of","with",
    "is","are","was","were","be","been","being","have","has","had","do",
    "does","did","will","would","could","should","may","might","shall",
    "not","no","nor","so","yet","both","either","each","few","more",
    "most","other","some","such","than","too","very","just","as","if",
    "then","that","this","these","those","i","you","he","she","it","we",
    "they","what","which","who","whom","how","when","where","why",
    "all","any","both","each","every","few","more","most","other","some",
    "such","about","above","after","before","between","into","through",
    "during","including","until","against","among","throughout","despite",
    "towards","upon","concerning","per","via","well","also","however",
    "therefore","thus","hence","whereas","while","although","because",
    "since","unless","until","whether","within","without","across",
    "along","following","across","behind","beyond","plus","except",
    "up","out","around","down","off","over","under","again","further",
    "there","once","must","need","get","use","using","used","make",
    "made","work","working","ability","strong","good","excellent",
    "experience","years","year","knowledge","skills","skill","team",
    "management","business","company","role","position","job",
}


# ═══════════════════════════════════════════════════════════
# SKILL ALIAS MAP  — 150+ normalisations
# ═══════════════════════════════════════════════════════════
SKILL_ALIASES = {
    # Programming languages
    "python3": "python", "py": "python", "python2": "python",
    "js": "javascript", "javascript es6": "javascript", "es6": "javascript",
    "node": "node.js", "nodejs": "node.js", "node js": "node.js",
    "ts": "typescript", "typescript js": "typescript",
    "c sharp": "c#", "csharp": "c#",
    "cplusplus": "c++", "c plus plus": "c++",
    "golang": "go",
    "ruby on rails": "rails",
    "shell scripting": "bash", "shell script": "bash",
    # Office / Productivity
    "ms excel": "excel", "microsoft excel": "excel",
    "ms word": "word", "microsoft word": "word",
    "ms office": "microsoft office", "office 365": "microsoft office",
    "ms powerpoint": "powerpoint", "microsoft powerpoint": "powerpoint",
    "google sheets": "google sheets", "g-suite": "google workspace",
    "gsuite": "google workspace",
    # ML / AI / Data Science
    "ml": "machine learning", "dl": "deep learning",
    "nlp": "natural language processing", "natural language proc": "natural language processing",
    "ai": "artificial intelligence",
    "data analytics": "data analysis", "analytics": "data analysis",
    "powerbi": "power bi", "power-bi": "power bi",
    "tableau desktop": "tableau", "tableau public": "tableau",
    "sci-kit learn": "scikit-learn", "sklearn": "scikit-learn",
    "tensorflow2": "tensorflow", "tf": "tensorflow",
    "pytorch": "pytorch", "torch": "pytorch",
    "keras": "keras",
    "numpy": "numpy", "np": "numpy",
    "pandas": "pandas", "pd": "pandas",
    "matplotlib": "matplotlib",
    "seaborn": "seaborn",
    "opencv": "opencv", "cv2": "opencv",
    "huggingface": "hugging face", "hf transformers": "hugging face",
    "langchain": "langchain",
    # Cloud / DevOps
    "amazon web services": "aws", "amazon aws": "aws",
    "gcp": "google cloud platform", "google cloud": "google cloud platform",
    "azure": "microsoft azure", "ms azure": "microsoft azure",
    "cicd": "ci/cd", "ci cd": "ci/cd", "continuous integration": "ci/cd",
    "k8s": "kubernetes", "kube": "kubernetes",
    "docker container": "docker", "dockerize": "docker",
    "terraform": "terraform",
    "jenkins": "jenkins", "github actions": "github actions",
    "linux": "linux", "unix": "unix",
    "bash scripting": "bash",
    # Databases
    "ms sql": "sql server", "mssql": "sql server",
    "microsoft sql server": "sql server", "t-sql": "sql server",
    "postgres": "postgresql", "psql": "postgresql",
    "mongo": "mongodb", "mongoose": "mongodb",
    "mysql": "mysql", "mariadb": "mysql",
    "nosql": "nosql", "no-sql": "nosql",
    "redis": "redis", "memcache": "memcached",
    "elasticsearch": "elasticsearch", "elastic search": "elasticsearch",
    "snowflake": "snowflake", "bigquery": "bigquery", "redshift": "redshift",
    # Web Frameworks
    "flask framework": "flask", "django framework": "django",
    "fastapi": "fastapi", "fast api": "fastapi",
    "reactjs": "react", "react.js": "react", "react js": "react",
    "vue.js": "vue", "vuejs": "vue",
    "angular": "angular", "angular.js": "angularjs",
    "spring boot": "spring boot", "spring framework": "spring",
    "express.js": "express", "expressjs": "express",
    "nextjs": "next.js", "next js": "next.js",
    # Finance
    "financial modelling": "financial modeling",
    "financial model": "financial modeling",
    "p&l": "profit and loss", "pl management": "profit and loss",
    "fp&a": "financial planning and analysis",
    "ifrs": "ifrs", "gaap": "gaap",
    "accounts payable": "accounts payable", "accounts receivable": "accounts receivable",
    "tally erp": "tally", "tally erp 9": "tally",
    "quickbooks": "quickbooks",
    # HR / Business
    "hris": "hr information systems", "hrms": "hr information systems",
    "hr software": "hr information systems",
    "kpi": "key performance indicators", "kpis": "key performance indicators",
    "crm tools": "crm", "salesforce crm": "salesforce",
    "hubspot crm": "hubspot",
    "sap hr": "sap", "sap erp": "sap",
    "workday": "workday",
    "payroll processing": "payroll",
    # Soft skills
    "team work": "teamwork", "team player": "teamwork",
    "problem-solving": "problem solving", "problemsolving": "problem solving",
    "time-management": "time management", "time mgmt": "time management",
    "project-management": "project management", "project mgmt": "project management",
    "cust service": "customer service",
    "interpersonal skill": "interpersonal skills",
    "comm": "communication", "comms": "communication",
    "org skills": "organizational skills",
    "attention-to-detail": "attention to detail",
    "relationship mgmt": "relationship management",
    "relationship building": "relationship building",
    "adaptable": "adaptability", "adaptable": "adaptability",
    "multi-tasking": "multitasking",
    "conflict resolution": "conflict resolution",
    "decision-making": "decision making",
    "critical-thinking": "critical thinking",
    "emotional intelligence": "emotional intelligence",
    "negotiation skills": "negotiation",
    "presentation skills": "presentation",
    "public speaking": "public speaking",
    "analytical skills": "analytical thinking",
    "strategic thinking": "strategic planning",
}


# ═══════════════════════════════════════════════════════════
# EXPERIENCE LEVEL DETECTION
# ═══════════════════════════════════════════════════════════
EXPERIENCE_PATTERNS = {
    "Senior": [
        r"\b(1[0-9]|20|\d{2})\+?\s*(?:years?|yrs?)\b",
        r"\b(senior|lead|principal|staff|head|director|vp|vice\s*president|cto|cfo|coo|ceo|manager)\b",
        r"\bmanag(?:ed|ing)\s+(?:team|people|staff|engineers|developers)\b",
        r"\b(?:8|9)\+?\s*(?:years?|yrs?)\b",
    ],
    "Mid": [
        r"\b([4-7])\+?\s*(?:years?|yrs?)\b",
        r"\b(mid.level|intermediate|experienced|associate)\b",
    ],
    "Entry": [
        r"\b([0-3])\+?\s*(?:years?|yrs?)\b",
        r"\b(fresher|graduate|intern|entry.level|junior|trainee|recent\s*grad|new\s*grad)\b",
    ],
}

def detect_experience_level(text: str) -> str:
    """Return 'Senior', 'Mid', or 'Entry' based on resume text patterns."""
    text_lower = text.lower()
    for level in ["Senior", "Mid", "Entry"]:
        for pat in EXPERIENCE_PATTERNS[level]:
            if re.search(pat, text_lower):
                return level
    return "Mid"  # safe default


# ═══════════════════════════════════════════════════════════
# SKILL NORMALISATION
# ═══════════════════════════════════════════════════════════
def normalize_skill(skill: str) -> str:
    """Lowercase, strip, apply aliases, clean special chars."""
    if not isinstance(skill, str):
        return ""
    skill = skill.lower().strip()
    if not skill:
        return ""
    # Direct alias lookup
    if skill in SKILL_ALIASES:
        return SKILL_ALIASES[skill]
    # Clean then re-check
    skill_clean = re.sub(r"[^a-z0-9 /\+#\.\-]", "", skill)
    skill_clean = re.sub(r"\s+", " ", skill_clean).strip()
    return SKILL_ALIASES.get(skill_clean, skill_clean)


def parse_skill_list(skill_str: str) -> list:
    """Parse a skill string (CSV or Python list literal) into a clean list."""
    if not isinstance(skill_str, str) or not skill_str.strip():
        return []
    try:
        raw = ast.literal_eval(skill_str)
        items = raw if isinstance(raw, list) else []
    except (ValueError, SyntaxError):
        items = [s.strip() for s in skill_str.split(",")]
    normalized = [normalize_skill(s) for s in items if isinstance(s, str) and s.strip()]
    return [s for s in normalized if s and len(s) >= 2]


# ═══════════════════════════════════════════════════════════
# TEXT CLEANING  (for TF-IDF corpus)
# ═══════════════════════════════════════════════════════════
def clean_text(text: str, remove_stopwords: bool = True) -> str:
    """
    Full NLP preprocessing pipeline:
    1. Lowercase
    2. Remove URLs
    3. Remove HTML tags
    4. Remove special characters (keep alphanumeric + space)
    5. Remove extra whitespace
    6. Optionally remove stopwords
    """
    if not isinstance(text, str):
        return ""
    # Lowercase
    text = text.lower()
    # Remove URLs
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)
    # Remove HTML tags
    text = re.sub(r"<[^>]+>", " ", text)
    # Remove email addresses
    text = re.sub(r"\S+@\S+", " ", text)
    # Remove special characters — keep alphanumeric and spaces
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    # Remove standalone numbers (years etc. not useful for skill matching)
    text = re.sub(r"\b\d+\b", " ", text)
    # Collapse whitespace
    text = re.sub(r"\s+", " ", text).strip()
    # Remove stopwords
    if remove_stopwords:
        tokens = text.split()
        tokens = [t for t in tokens if t not in STOPWORDS and len(t) > 1]
        text = " ".join(tokens)
    return text


# ═══════════════════════════════════════════════════════════
# PDF TEXT EXTRACTION
# ═══════════════════════════════════════════════════════════
def extract_text_from_pdf(uploaded_file) -> str:
    """Extract text from a PDF upload (PyMuPDF). Returns empty string on failure."""
    try:
        pdf_bytes = uploaded_file.read()
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        if len(doc) == 0:
            return ""
        pages = [page.get_text("text").strip() for page in doc]
        doc.close()
        text = "\n".join(p for p in pages if p)
        return text
    except Exception as exc:
        raise RuntimeError(
            f"Could not read PDF: {exc}. Please upload a text-based PDF (not a scanned image)."
        ) from exc


# ═══════════════════════════════════════════════════════════
# SKILL EXTRACTION FROM RESUME TEXT
# ═══════════════════════════════════════════════════════════
def extract_skills_from_text(resume_text: str, known_skills: set) -> list:
    """
    Match known skill vocabulary against resume text.
    Applies alias normalisation to resume text first for better coverage.
    """
    if not resume_text or not known_skills:
        return []
    text = resume_text.lower()

    # Apply all aliases to the text so variants match
    for alias, canonical in SKILL_ALIASES.items():
        pattern = r"(?<![a-z0-9])" + re.escape(alias) + r"(?![a-z0-9])"
        text = re.sub(pattern, f" {canonical} ", text)

    matched = set()
    for skill in known_skills:
        if not skill or len(skill) < 2:
            continue
        # Escape skill for regex; handle special chars like c++ or c#
        escaped = re.escape(skill)
        pattern = r"(?<![a-z0-9])" + escaped + r"(?![a-z0-9])"
        if re.search(pattern, text):
            matched.add(skill)

    return sorted(matched)
