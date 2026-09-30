import re
import numpy as np

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


# ============================================================
# 1. JD SECTION SPLITTING
# ============================================================

NICE_TO_HAVE_HEADERS = [
    "nice to have",
    "nice-to-have",
    "preferred qualifications",
    "preferred",
    "bonus",
    "good to have",
    "good-to-have",
    "optional",
]


def split_jd_sections(jd_text):
    """
    Split JD into:
        required section
        nice-to-have section

    If no nice-to-have heading exists, the complete JD
    is treated as required.
    """

    if not isinstance(jd_text, str) or not jd_text.strip():
        return "", ""

    lower = jd_text.lower()

    nice_start = None

    for header in NICE_TO_HAVE_HEADERS:

        idx = lower.find(header)

        if idx != -1:
            if nice_start is None or idx < nice_start:
                nice_start = idx

    if nice_start is None:
        return jd_text, ""

    return jd_text[:nice_start], jd_text[nice_start:]


# ============================================================
# 2. SKILL / REQUIREMENT HEADER DETECTION
# ============================================================

SKILL_HEADER_PATTERN = re.compile(
    r'(required skills|technical skills|key skills|must[- ]have|'
    r'skills required|skills needed|skills|requirements)\s*:\s*(.*)',
    re.IGNORECASE
)


FILLER_PREFIXES = [
    r'^(basic |working |hands[- ]on |practical )?(knowledge|understanding) of\s+',
    r'^experience (with|in|using|working with|creating|building|developing)\s+',
    r'^familiarity with\s+',
    r'^proficiency (in|with)\s+',
    r'^exposure to\s+',
]


def _clean_candidate(phrase):

    phrase = phrase.strip()

    for pattern in FILLER_PREFIXES:

        phrase = re.sub(
            pattern,
            '',
            phrase,
            flags=re.IGNORECASE
        )

    return phrase.strip(" .")


# ============================================================
# 3. SPLIT REQUIREMENTS INTO INDIVIDUAL PHRASES
# ============================================================

def _split_candidates(text):

    results = []

    parens = re.findall(r'\(([^)]+)\)', text)

    text_without_parens = re.sub(
        r'\([^)]+\)',
        '',
        text
    )

    for chunk in [text_without_parens] + parens:

        for part in re.split(r',|;', chunk):

            part = part.strip().rstrip('.').strip()

            if not part:
                continue

            for alternative in re.split(
                r'\bor\b|/',
                part
            ):

                alternative = alternative.strip()

                if (
                    alternative
                    and not re.match(
                        r'^(similar|other|etc)\b',
                        alternative,
                        re.IGNORECASE
                    )
                ):

                    results.append(alternative)

    return results


# ============================================================
# 4. EXTRACT SKILLS FROM A JD SECTION
# ============================================================

def extract_skill_phrases_from_section(section_text):

    lines = section_text.split('\n')

    list_chunks = []

    capturing = False

    for line in lines:

        stripped = line.strip()

        if not stripped:

            capturing = False
            continue

        match = SKILL_HEADER_PATTERN.match(stripped)

        if match:

            after_header = match.group(2).strip()

            if after_header:
                list_chunks.append(after_header)

            capturing = True

            continue

        if capturing:

            list_chunks.append(
                re.sub(
                    r'^[-•*]\s*',
                    '',
                    stripped
                )
            )

    phrases = []
    seen = set()

    for chunk in list_chunks:

        for phrase in _split_candidates(chunk):

            phrase = _clean_candidate(phrase)

            key = phrase.lower()

            if (
                phrase
                and len(phrase) > 1
                and key not in seen
                and not _is_soft_skill(phrase)
            ):

                seen.add(key)

                phrases.append(phrase)

    return phrases


# ============================================================
# 5. NICE-TO-HAVE PHRASES IN NORMAL SENTENCES
# ============================================================

NICE_TO_HAVE_PHRASES = [
    "nice to have",
    "nice-to-have",
    "preferred",
    "bonus",
    "good to have",
    "good-to-have",
    "optional",
    "additional advantage",
    "added advantage",
    "an advantage",
    "advantage",
    "preferred but not required",
    "would be a plus",
    "is a plus",
    "are a plus",
    "plus if",
    "desired",
    "desirable",
]


LEADIN_PATTERNS = [
    r'^candidates?\s+(with|who have|having)\s+',
    r'^(experience|knowledge|familiarity)\s+(in|with|of)\s+',
    r'^(those|applicants?)\s+(with|who have)\s+',
]


TRAILING_FILLER_PATTERN = re.compile(
    r'\b(will|would)\s+(have|be|get|receive)\b.*$',
    re.IGNORECASE
)


def _split_sentences(text):

    parts = re.split(
        r'(?<=[.!?])\s+|\n+',
        text
    )

    return [
        p.strip()
        for p in parts
        if p.strip()
    ]


def extract_bonus_skills_from_sentence(sentence):

    lower = sentence.lower()

    idx = None
    matched_len = 0

    for phrase in NICE_TO_HAVE_PHRASES:

        i = lower.find(phrase)

        if i != -1:

            if idx is None or i < idx:

                idx = i
                matched_len = len(phrase)

    if idx is None:
        return []

    before = sentence[:idx].strip()

    text = (
        before
        if before
        else sentence[idx + matched_len:].strip()
    )
    text = re.sub(r'^[:\-\u2013\u2014]\s*', '', text).strip()
    for pattern in LEADIN_PATTERNS:

        text = re.sub(
            pattern,
            '',
            text,
            flags=re.IGNORECASE
        )

    text = TRAILING_FILLER_PATTERN.sub(
        '',
        text
    ).strip()

    return [
        p
        for p in (
            _clean_candidate(c)
            for c in _split_candidates(text)
        )
        if p and not _is_soft_skill(p)
    ]


# ============================================================
# 5b. REQUIRED PHRASES HIDDEN IN NORMAL SENTENCES (prose JDs)
# ============================================================

REQUIRED_TRIGGER_PHRASES = [
    "should have",
    "must have",
    "required to have",
    "experience with",
    "experience in",
    "experience using",
    "knowledge of",
    "proficiency in",
    "proficiency with",
    "familiarity with",
    "expected",
]


LEADING_FILLER_PATTERN = re.compile(
    r'^(strong|good|solid|excellent|advanced|basic|'
    r'along with|as well as|together with)\s+',
    re.IGNORECASE
)


REQUIRED_TRAILING_FILLER_PATTERN = re.compile(
    r'\b(is|are|was|were)\s+(also\s+)?'
    r'(important|beneficial|necessary|essential|required|needed|'
    r'expected|desirable|preferred|helpful|valuable|useful)\b.*$',
    re.IGNORECASE
)

# Phrases that are technically "requirements" in a JD but are not
# technical skills — soft skills and education/qualification lines.
# Filtered out so the Skill Gap Analysis only shows actual skills.
NON_SKILL_MARKERS = [
    # soft skills
    "attention to detail",
    "problem-solving",
    "problem solving",
    "ability to work",
    "communication skill",
    "interpersonal skill",
    "team player",
    "work effectively in a team",
    "good communication",
    "strong communication",
    # education / qualifications
    "degree",
    "bachelor",
    "master",
    "b.e",
    "b.tech",
    "m.tech",
    "bsc",
    "msc",
    "cgpa",
    "gpa",
    "university",
    "college",
    "diploma",
    "graduate",
    "undergraduate",
    "phd",
    "related field",
    "similar field",
    "relevant field",
    "or equivalent",
]


def _is_soft_skill(phrase):
    lower = phrase.lower()
    return any(marker in lower for marker in NON_SKILL_MARKERS)


def extract_required_skills_from_sentence(sentence):
    """
    Fallback for JDs written in prose with no 'Required Skills:'
    header — pulls skills out of sentences using common requirement
    phrasing instead (mirrors extract_bonus_skills_from_sentence).
    """

    lower = sentence.lower()

    idx = None
    matched_len = 0

    for phrase in REQUIRED_TRIGGER_PHRASES:

        i = lower.find(phrase)

        if i != -1:
            if idx is None or i < idx:
                idx = i
                matched_len = len(phrase)

    if idx is None:
        return []

    text = sentence[idx + matched_len:].strip()
    text = re.sub(r'^[:\-\u2013\u2014]\s*', '', text).strip()
    candidates = []

    for chunk in _split_candidates(text):
        for sub in re.split(r'\band\b', chunk):
            sub = _clean_candidate(sub)
            sub = LEADING_FILLER_PATTERN.sub('', sub).strip()
            sub = REQUIRED_TRAILING_FILLER_PATTERN.sub('', sub).strip()
            sub = _clean_candidate(sub)
            if sub and not _is_soft_skill(sub):
                candidates.append(sub)

    return candidates


# ============================================================
# 6. EXTRACT ALL JD REQUIREMENTS
# ============================================================

def extract_jd_requirements(jd_text):

    required_chunk, nice_chunk = split_jd_sections(
        jd_text
    )

    # Required skills
    required_phrases = extract_skill_phrases_from_section(
        required_chunk
    )

    # Explicit nice-to-have skills
    nice_phrases = extract_skill_phrases_from_section(
        nice_chunk
    )

    # Nice-to-have skills hidden inside sentences
    bonus_phrases = []

    for sentence in _split_sentences(jd_text):

        if any(
            phrase in sentence.lower()
            for phrase in NICE_TO_HAVE_PHRASES
        ):

            bonus_phrases.extend(
                extract_bonus_skills_from_sentence(sentence)
            )

    # Remove bonus skills from required list
    bonus_keys = {
        skill.lower()
        for skill in bonus_phrases
    }

    required_phrases = [
        skill
        for skill in required_phrases
        if skill.lower() not in bonus_keys
    ]

    # Fallback: many JDs describe required skills in prose sentences
    # instead of under a "Required Skills:" header. Only runs if the
    # header-based extraction above found nothing at all.
    if not required_phrases:

        for sentence in _split_sentences(required_chunk):

            if any(
                phrase in sentence.lower()
                for phrase in NICE_TO_HAVE_PHRASES
            ):
                continue  # already handled as a bonus/nice-to-have sentence

            for skill in extract_required_skills_from_sentence(sentence):

                if skill.lower() not in bonus_keys:
                    required_phrases.append(skill)

        seen_req = set()
        deduped = []
        for skill in required_phrases:
            key = skill.lower()
            if key not in seen_req:
                seen_req.add(key)
                deduped.append(skill)
        required_phrases = deduped

    # Add bonus skills to nice-to-have
    seen_nice = {
        skill.lower()
        for skill in nice_phrases
    }

    for skill in bonus_phrases:

        if skill.lower() not in seen_nice:

            nice_phrases.append(skill)

            seen_nice.add(skill.lower())

    # Remove duplicates that appear in required
    required_keys = {
        skill.lower()
        for skill in required_phrases
    }

    nice_phrases = [
        skill
        for skill in nice_phrases
        if skill.lower() not in required_keys
    ]

    return {
        "required": required_phrases,
        "nice_to_have": nice_phrases
    }


# ============================================================
# 7. RESUME CHUNKING
# ============================================================

def split_resume_into_chunks(resume_text):

    if not isinstance(resume_text, str):
        return []

    chunks = re.split(
        r'\n+|[.!?]+',
        resume_text
    )

    chunks = [
        chunk.strip()
        for chunk in chunks
        if chunk.strip()
        and len(chunk.strip()) > 2
    ]

    return chunks


# ============================================================
# 8. SEMANTIC SKILL MATCHING USING BERT
# ============================================================

def semantic_skill_match(
    skill,
    resume_text,
    bert_model,
    match_threshold=0.72,
    partial_threshold=0.55
):
    """
    Compare one JD requirement against the resume.

    Returns:
        Matched
        Partial
        Missing
    """

    # Direct literal match first — catches exact skill mentions (e.g.
    # "Python" listed inside a long comma-separated skills line) that
    # BERT similarity underrates because the chunk it's compared
    # against is a whole list, not a sentence about just that skill.
    skill_norm = skill.strip().lower()
    if skill_norm and isinstance(resume_text, str):
        if re.search(
            r'\b' + re.escape(skill_norm) + r'\b',
            resume_text.lower()
        ):
            return {
                "status": "Matched",
                "similarity": 1.0,
                "evidence": skill.strip()
            }

    resume_chunks = split_resume_into_chunks(
        resume_text
    )

    if not resume_chunks:

        return {
            "status": "Missing",
            "similarity": 0.0,
            "evidence": ""
        }

    skill_embedding = bert_model.encode(
        [skill],
        normalize_embeddings=True
    )

    resume_embeddings = bert_model.encode(
        resume_chunks,
        normalize_embeddings=True
    )

    similarities = cosine_similarity(
        skill_embedding,
        resume_embeddings
    )[0]

    best_index = int(
        np.argmax(similarities)
    )

    best_similarity = float(
        similarities[best_index]
    )

    best_evidence = resume_chunks[
        best_index
    ]

    if best_similarity >= match_threshold:

        status = "Matched"

    elif best_similarity >= partial_threshold:

        status = "Partial"

    else:

        status = "Missing"
        best_evidence = ""

    return {
        "status": status,
        "similarity": round(
            best_similarity,
            4
        ),
        "evidence": best_evidence
    }


# ============================================================
# 9. ANALYZE ONE CATEGORY
# ============================================================

def analyze_requirement_category(
    requirements,
    resume_text,
    bert_model,
    match_threshold=0.72,
    partial_threshold=0.55
):

    results = []

    for requirement in requirements:

        result = semantic_skill_match(
            requirement,
            resume_text,
            bert_model,
            match_threshold,
            partial_threshold
        )

        results.append({
            "skill": requirement,
            "status": result["status"],
            "similarity": result["similarity"],
            "evidence": result["evidence"]
        })

    return results


# ============================================================
# 10. COMPLETE SKILL-GAP ANALYSIS
# ============================================================

def analyze_skill_gap(
    jd_text,
    resume_text,
    bert_model,
    match_threshold=0.72,
    partial_threshold=0.55
):
    """
    Complete JD-driven semantic skill-gap analysis.
    """

    requirements = extract_jd_requirements(
        jd_text
    )

    required_results = analyze_requirement_category(
        requirements["required"],
        resume_text,
        bert_model,
        match_threshold,
        partial_threshold
    )

    nice_results = analyze_requirement_category(
        requirements["nice_to_have"],
        resume_text,
        bert_model,
        match_threshold,
        partial_threshold
    )

    matched_required = [
        r for r in required_results
        if r["status"] == "Matched"
    ]

    partial_required = [
        r for r in required_results
        if r["status"] == "Partial"
    ]

    missing_required = [
        r for r in required_results
        if r["status"] == "Missing"
    ]

    matched_nice = [
        r for r in nice_results
        if r["status"] == "Matched"
    ]

    partial_nice = [
        r for r in nice_results
        if r["status"] == "Partial"
    ]

    missing_nice = [
        r for r in nice_results
        if r["status"] == "Missing"
    ]

    total_required = len(required_results)

    if total_required > 0:

        matched_score = len(matched_required)
        partial_score = 0.5 * len(partial_required)

        required_coverage = (
            (matched_score + partial_score)
            / total_required
        )

    else:

        required_coverage = 0.0

    return {

        "required": required_results,
        "nice_to_have": nice_results,

        "matched_required": matched_required,
        "partial_required": partial_required,
        "missing_required": missing_required,

        "matched_nice_to_have": matched_nice,
        "partial_nice_to_have": partial_nice,
        "missing_nice_to_have": missing_nice,

        "required_skill_coverage": round(
            required_coverage,
            4
        )
    }


# ============================================================
# 11. LOAD BERT MODEL
# ============================================================

def load_skill_gap_model(
    model_name="all-MiniLM-L6-v2"
):

    print(
        f"Loading Sentence-BERT model: {model_name}"
    )

    model = SentenceTransformer(
        model_name
    )

    print(
        "Sentence-BERT loaded successfully!"
    )

    return model


# ============================================================
# 12. TEST
# ============================================================

if __name__ == "__main__":

    bert_model = load_skill_gap_model()

    jd = """
    We are looking for a Python Developer.

    Required Skills:
    Python, SQL, Machine Learning, AWS

    Nice to Have:
    Docker, Kubernetes
    """

    resume = """
    Software developer with experience in Python
    and SQL.

    Developed predictive machine learning models
    using TensorFlow and scikit-learn.

    Built and deployed applications using Docker.
    """

    result = analyze_skill_gap(
        jd,
        resume,
        bert_model
    )

    print("\n========== REQUIRED SKILLS ==========")

    for item in result["required"]:

        print(
            f"{item['skill']} -> "
            f"{item['status']} "
            f"({item['similarity']})"
        )

        if item["evidence"]:

            print(
                f"   Evidence: {item['evidence']}"
            )

    print("\n========== NICE TO HAVE ==========")

    for item in result["nice_to_have"]:

        print(
            f"{item['skill']} -> "
            f"{item['status']} "
            f"({item['similarity']})"
        )

    print(
        "\nRequired Skill Coverage:",
        result["required_skill_coverage"]
    )