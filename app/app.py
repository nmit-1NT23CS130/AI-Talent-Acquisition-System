import streamlit as st
import importlib.util
import os
import pandas as pd
import time
import joblib

# ---------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------
st.set_page_config(
    page_title="AI Talent Acquisition System",
    page_icon="🚀",
    layout="wide"
)

# ---------------------------------------------------
# CUSTOM CSS
# ---------------------------------------------------
st.markdown("""
<style>

.main {
    background-color: #0E1117;
}

h1, h2, h3 {
    color: white;
}

.stTextArea textarea {
    background-color: #1E1E1E;
    color: white;
    border-radius: 12px;
}

.stFileUploader {
    background-color: #1E1E1E;
    padding: 15px;
    border-radius: 12px;
}

.metric-card {
    background-color: #1E1E1E;
    padding: 20px;
    border-radius: 15px;
    text-align: center;
    box-shadow: 0px 0px 10px rgba(255,255,255,0.05);
}

.section-card {
    background-color: #161B22;
    padding: 20px;
    border-radius: 18px;
    margin-bottom: 20px;
}

.top-card {
    background: linear-gradient(135deg, #1F6FEB, #8250DF);
    padding: 25px;
    border-radius: 20px;
    color: white;
}

.small-text {
    color: #D0D0D0;
}

</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------
# DYNAMIC IMPORTS
# ---------------------------------------------------
def load_module(module_name, filename):

    path = os.path.join(
        os.path.dirname(__file__),
        filename
    )

    spec = importlib.util.spec_from_file_location(
        module_name,
        path
    )

    module = importlib.util.module_from_spec(spec)

    spec.loader.exec_module(module)

    return module

# ---------------------------------------------------
# LOAD EXISTING FILES
# ---------------------------------------------------
preprocessing = load_module(
    "preprocessing",
    "02_preprocessing.py"
)

semantic_matching = load_module(
    "semantic_matching",
    "04_semantic_matching.py"
)

ranking_module = load_module(
    "ranking_module",
    "06_ranking.py"
)

skill_gap_module = load_module(
    "skill_gap_module",
    "07_skill_gap.py"
)

explainability_module = load_module(
    "explainability_module",
    "08_explainability.py"
)

# ---------------------------------------------------
# FUNCTIONS
# ---------------------------------------------------
preprocess_text = preprocessing.preprocess_text

load_bert_model = semantic_matching.load_bert_model

rank_live_candidates = ranking_module.rank_live_candidates

build_explainer = explainability_module.build_explainer
explain_candidate = explainability_module.explain_candidate
analyze_skill_gap = skill_gap_module.analyze_skill_gap

# ---------------------------------------------------
# LOAD MODELS
# ---------------------------------------------------
@st.cache_resource
def load_all_models():

    bert_model = load_bert_model()

    BASE_DIR = os.path.dirname(os.path.dirname(__file__))

    model_path = os.path.join(
        BASE_DIR,
        "models",
        "best_model.pkl"
    )

    le_path = os.path.join(
        BASE_DIR,
        "models",
        "label_encoder.pkl"
    )

    tfidf_path = os.path.join(
        BASE_DIR,
        "models",
        "tfidf_vectorizer.pkl"
    )

    model = joblib.load(model_path)
    le = joblib.load(le_path)
    tfidf = joblib.load(tfidf_path)

    try:
        explainer = build_explainer(model)
    except Exception:
        explainer = None

    return model, le, tfidf, bert_model, explainer


model, le, tfidf, bert_model, explainer = load_all_models()


# ---------------------------------------------------
# HEADER
# ---------------------------------------------------
st.markdown("""
<div class="top-card">
    <h1>🚀 AI Talent Acquisition System</h1>
    <p class="small-text">
        Intelligent Resume Ranking & Candidate Screening Dashboard
    </p>
</div>
""", unsafe_allow_html=True)

st.write("")

# ---------------------------------------------------
# SIDEBAR
# ---------------------------------------------------
st.sidebar.title("👨‍💼 HR Manager Panel")

st.sidebar.success("System Status: Online")

st.sidebar.markdown("""
### Workflow

1️⃣ Post Job Description  
2️⃣ Upload Candidate Resumes  
3️⃣ Rank Candidates  
4️⃣ Review AI Insights  
5️⃣ Shortlist Top Applicants
""")

# ---------------------------------------------------
# METRICS
# ---------------------------------------------------
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown("""
    <div class="metric-card">
        <h2>TF-IDF</h2>
        <p>Feature Extraction</p>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="metric-card">
        <h2>BERT</h2>
        <p>Semantic Matching</p>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("""
    <div class="metric-card">
        <h2>XGBoost</h2>
        <p>AI Classification</p>
    </div>
    """, unsafe_allow_html=True)

with col4:
    st.markdown("""
    <div class="metric-card">
        <h2>Top 5</h2>
        <p>Auto Shortlisting</p>
    </div>
    """, unsafe_allow_html=True)

st.write("")

# ---------------------------------------------------
# JOB DESCRIPTION
# ---------------------------------------------------
st.markdown('<div class="section-card">', unsafe_allow_html=True)

st.header("📄 Step 1: Post Job Description")

jd_text = st.text_area(
    "Enter Job Description",
    height=220,
    placeholder="""
Example:

We are looking for a Python Developer with experience in:
- Machine Learning
- NLP
- Streamlit
- SQL
- Data Analytics
- Scikit-learn
"""
)

st.markdown('</div>', unsafe_allow_html=True)

# ---------------------------------------------------
# RESUME UPLOAD
# ---------------------------------------------------
st.markdown('<div class="section-card">', unsafe_allow_html=True)

st.header("📂 Step 2: Upload Candidate Resumes")

uploaded_files = st.file_uploader(
    "Upload Multiple Resume Files (.pdf, .docx, .txt)",
    type=["txt", "pdf", "docx"],
    accept_multiple_files=True
)

st.markdown('</div>', unsafe_allow_html=True)

# ---------------------------------------------------
# BUTTON
# ---------------------------------------------------
rank_button = st.button("🚀 Rank Candidates")

# ---------------------------------------------------
# PROCESSING
# ---------------------------------------------------
if rank_button:

    if not jd_text:
        st.error("Please enter a Job Description.")
        st.stop()

    if not uploaded_files:
        st.error("Please upload resumes.")
        st.stop()

    resumes = []

    for file in uploaded_files:

        try:
            from file_reader import extract_text

            resume_text = extract_text(file)

            resumes.append({
                "name": file.name,
                "text": resume_text
            })

        except Exception as e:
            st.warning(f"Could not process {file.name}: {e}")

    if not resumes:
        st.error("Could not extract text from any uploaded resume. Please check the files and try again.")
        st.stop()

    with st.spinner("🤖 AI is analyzing resumes using TF-IDF + BERT + XGBoost..."):

        time.sleep(2)

        df = rank_live_candidates(
            resumes,
            jd_text,
            model,
            le,
            tfidf,
            bert_model,
            preprocess_text,
            explainer=explainer,
            explain_fn=explain_candidate,
            skill_gap_fn=analyze_skill_gap
        )

    # ---------------------------------------------------
    # RESULTS
    # ---------------------------------------------------
    st.markdown('<div class="section-card">', unsafe_allow_html=True)

    st.header("📊 Ranked Candidates")

    st.dataframe(
        df.drop(columns=['explanation'], errors='ignore'),
        use_container_width=True
    )

    st.markdown('</div>', unsafe_allow_html=True)

    # ---------------------------------------------------
    # BEST CANDIDATE
    # ---------------------------------------------------
    top_candidate = df.iloc[0]

    st.markdown('<div class="section-card">', unsafe_allow_html=True)

    st.header("🏆 Best Candidate")

    st.success(f"""
Top Candidate: {top_candidate['candidate']}

Final Score: {top_candidate['final_score']}

Predicted Fit: {top_candidate['predicted_label']}
""")

    st.markdown('</div>', unsafe_allow_html=True)

    # ---------------------------------------------------
    # TOP 5
    # ---------------------------------------------------
    st.markdown('<div class="section-card">', unsafe_allow_html=True)

    st.header("📋 Top 5 Shortlisted Candidates")

    top5 = df.head(5)

    st.dataframe(
        top5.drop(columns=['explanation'], errors='ignore'),
        use_container_width=True
    )

    st.markdown('</div>', unsafe_allow_html=True)
    # ---------------------------------------------------
    # EXPLAINABLE AI
    # ---------------------------------------------------
    st.markdown('<div class="section-card">', unsafe_allow_html=True)

    st.header("🧩 Explainable AI — Why This Candidate?")

    def _label(skill):
        """Skill list entries may be plain strings or dicts like
        {'skill': 'Python', 'status': 'Matched', ...} — always show
        just the readable skill name."""
        return skill['skill'] if isinstance(skill, dict) else skill

    for _, row in top5.iterrows():
        explanation = row.get('explanation')
        with st.expander(f"Explain: {row['candidate']} — {row['predicted_label']}"):
            if not explanation:
                st.write("No explanation available for this candidate.")
                continue

            semantic_note = explanation['semantic_note']

            # One clean category — no "model said X but changed to Y"
            # messaging, and no raw BERT/TF-IDF decimal values shown.
            st.markdown(f"**Category:** {row['predicted_label']}")
            st.markdown(f"**Overall Match:** {semantic_note['reading']}")

            skill_gap = explanation.get('skill_gap')

            st.markdown("#### 🎯 Skill Gap Analysis")

            gcol1, gcol2 = st.columns(2)
            with gcol1:
                st.markdown("✅ **Matched Required Skills**")
                if skill_gap and skill_gap['matched_required']:
                    for skill in skill_gap['matched_required']:
                        st.write(f"- {_label(skill)}")
                else:
                    st.write("None")
            with gcol2:
                st.markdown("⚠️ **Missing Required Skills**")
                if skill_gap and skill_gap['missing_required']:
                    for skill in skill_gap['missing_required']:
                        st.write(f"- {_label(skill)}")
                else:
                    st.write("None")

            gcol3, gcol4 = st.columns(2)
            with gcol3:
                st.markdown("✅ **Matched Nice-to-Have**")
                if skill_gap and skill_gap['matched_nice_to_have']:
                    for skill in skill_gap['matched_nice_to_have']:
                        st.write(f"- {_label(skill)}")
                else:
                    st.write("None")
            with gcol4:
                st.markdown("💡 **Missing Nice-to-Have**")
                if skill_gap and skill_gap['missing_nice_to_have']:
                    for skill in skill_gap['missing_nice_to_have']:
                        st.write(f"- {_label(skill)}")
                else:
                    st.write("None")

    st.markdown('</div>', unsafe_allow_html=True)

    # ---------------------------------------------------
    # FAIRNESS REPORT
    # ---------------------------------------------------
    st.markdown('<div class="section-card">', unsafe_allow_html=True)

    st.header("⚖️ Fairness Report")

    st.success("""
✔ Gender-identifying words are masked

✔ Ranking is based on semantic similarity

✔ AI evaluates skills and qualifications

✔ Bias-aware preprocessing is applied
""")

    st.markdown('</div>', unsafe_allow_html=True)

    # ---------------------------------------------------
    # HR RECOMMENDATION
    # ---------------------------------------------------
    st.markdown('<div class="section-card">', unsafe_allow_html=True)

    st.header("👨‍💼 HR Recommendation")

    st.write("""
The HR Manager can now:

✅ Review ranked candidates  
✅ Shortlist top applicants  
✅ Schedule interviews  
✅ Reduce manual screening effort
""")

    st.markdown('</div>', unsafe_allow_html=True)