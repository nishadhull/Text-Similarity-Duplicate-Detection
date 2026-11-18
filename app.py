import streamlit as st
import pandas as pd
import numpy as np
import joblib
import re
import os


from pathlib import Path
from scipy.sparse import hstack, csr_matrix
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer


# ==========================================
# 1. PAGE CONFIGURATION
# ==========================================

st.set_page_config(
    page_title="Text Similarity Detector",
    page_icon="🔍",
    layout="wide"
)
# =====================================
# LOAD SENTENCE-BERT MODEL
# =====================================

@st.cache_resource
def load_sbert_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

sbert_model = load_sbert_model()

st.success("Sentence-BERT Model Loaded Successfully!")


# ==========================================
# 2. CUSTOM CSS
# ==========================================

st.markdown("""
<style>

.main {
    background-color: #f5f7fb;
}

.title {
    font-size: 38px;
    font-weight: bold;
    color: #263b80;
}

.subtitle {
    font-size: 17px;
    color: gray;
}

.result-box {
    padding: 20px;
    border-radius: 12px;
    background-color: #eaf2ff;
}

</style>
""", unsafe_allow_html=True)


# ==========================================
# 3. MODEL PATHS
# ==========================================

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "model"


# ==========================================
# 4. LOAD TRAINED MODELS
# ==========================================

@st.cache_resource
def load_models():

    vectorizer = joblib.load(
        MODEL_DIR / "tfidf_vectorizer.pkl"
    )

    model_path = MODEL_DIR / "best_model.pkl"

    if not model_path.exists():
        model_path = MODEL_DIR / "linear_svm.pkl"

    model = joblib.load(model_path)

    comparison_path = MODEL_DIR / "model_comparison.csv"

    if comparison_path.exists():
        comparison = pd.read_csv(comparison_path)
    else:
        comparison = pd.DataFrame()

    return vectorizer, model, comparison


try:

    vectorizer, model, comparison = load_models()

    st.sidebar.success("Application Status: Running")

except Exception as e:

    st.error(f"Model loading error: {e}")
    st.stop()


# ==========================================
# 5. NUMERIC FEATURE EXTRACTION
# ==========================================

def create_numeric_features(q1, q2, v1, v2):

    features = []

    for text1, text2, vec1, vec2 in zip(q1, q2, v1, v2):

        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())

        common = words1.intersection(words2)

        union = words1.union(words2)

        overlap = (
            len(common) /
            max(len(words1), len(words2), 1)
        )

        jaccard = (
            len(common) /
            max(len(union), 1)
        )

        length1 = len(words1)
        length2 = len(words2)

        length_difference = abs(length1 - length2)

        length_ratio = (
            min(length1, length2) /
            max(length1, length2, 1)
        )

        cosine = float(vec1.multiply(vec2).sum())

        features.append([
            cosine,
            overlap,
            jaccard,
            np.log1p(length1),
            np.log1p(length2),
            length_difference,
            length_ratio
        ])

    return csr_matrix(
        np.array(features, dtype=np.float32)
    )


# ==========================================
# 6. PAIRWISE FEATURE EXTRACTION
# ==========================================

def create_pair_features(v1, v2, numeric):

    difference = abs(v1 - v2)

    product = v1.multiply(v2)

    return hstack(
        [
            difference,
            product,
            numeric
        ],
        format="csr"
    )


# ==========================================
# 7. TEXT PREPARATION
# ==========================================

def prepare_text(text):
    if pd.isna(text):
        return ""

    text = str(text).lower()

    # Expand common abbreviations
    replacements = {
        "nlp": "natural language processing",
        "ml": "machine learning",
        "ai": "artificial intelligence",
        "dl": "deep learning",
        "svm": "support vector machine"
    }

    for short, full in replacements.items():
        text = re.sub(
            r'\b' + re.escape(short) + r'\b',
            full,
            text
        )

    # Remove special characters
    text = re.sub(r'[^a-zA-Z0-9\s]', ' ', text)

    # Remove extra spaces
    text = re.sub(r'\s+', ' ', text)

    return text.strip()

    # Handle missing values
    if pd.isna(text):
        return ""

    # Convert text into lowercase
    text = str(text).lower()

    # Remove punctuation and special characters
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)

    # Remove extra spaces
    text = re.sub(r'\s+', ' ', text)

    return text.strip()

# ==========================================
# 8. PREDICTION FUNCTION
# ==========================================

def predict_similarity(question1, question2):

    # Prepare questions
    q1 = prepare_text(question1)
    q2 = prepare_text(question2)

    if not q1 or not q2:
        raise ValueError("Please enter both questions.")

    # Generate TF-IDF vectors
    v1 = vectorizer.transform([q1])
    v2 = vectorizer.transform([q2])
    # Generate numeric features
    numeric = create_numeric_features(
        [q1],
        [q2],
        [v1],
        [v2]
    )

    # Generate pairwise features
    features = create_pair_features(
        v1,
        v2,
        numeric
    )

    # Check feature dimensions
    expected = getattr(model, "n_features_in_", None)

    if expected is not None and features.shape[1] != expected:
        raise ValueError(
            f"Feature mismatch: App generated {features.shape[1]} "
            f"features, but model expects {expected}."
        )

    # Predict
    prediction = int(model.predict(features)[0])

    # Calculate cosine similarity
    similarity = float(cosine_similarity(v1, v2)[0][0])

    # Generate semantic embeddings
    embeddings = sbert_model.encode(
        [q1, q2],
        convert_to_numpy=True
    )

    # Calculate semantic similarity
    semantic_similarity = float(
        cosine_similarity(
            [embeddings[0]],
            [embeddings[1]]
        )[0][0]
    )

    # Convert into percentage
    semantic_similarity = max(
        0,
        semantic_similarity * 100
    )

    return prediction, similarity, semantic_similarity

# ==========================================
# 9. SIDEBAR NAVIGATION
# ==========================================

st.sidebar.title("🔍 Similarity AI")

st.sidebar.markdown("---")

page = st.sidebar.radio(
    "Navigation",
    [
        "Home",
        "Prediction History",
        "Model Information"
    ]
)

st.sidebar.markdown("---")

st.sidebar.info(
    "NLP & Machine Learning Project"
)


# ==========================================
# 10. SESSION HISTORY
# ==========================================

if "history" not in st.session_state:

    st.session_state.history = []


# ==========================================
# 11. HOME PAGE
# ==========================================

if page == "Home":

    st.markdown(
        '<p class="title">🔍 Text Similarity & Duplicate Detection</p>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<p class="subtitle">An NLP and Machine Learning based application to identify duplicate questions.</p>',
        unsafe_allow_html=True
    )

    st.divider()

    st.subheader("Compare Two Questions")

    col1, col2 = st.columns(2)

    with col1:

        question1 = st.text_area(
            "Enter First Question",
            placeholder="Example: What is machine learning?",
            height=130,
            key="q1"
        )

    with col2:

        question2 = st.text_area(
            "Enter Second Question",
            placeholder="Example: Explain machine learning.",
            height=130,
            key="q2"
        )

    st.write("")

    if st.button(
        "🔍 Check Similarity",
        use_container_width=True
    ):

        if not question1.strip() or not question2.strip():

            st.warning("Please enter both questions.")

        else:

            try:

                with st.spinner("Analyzing questions..."):

                    prediction, similarity, semantic_similarity = predict_similarity(
                        question1, question2
                    )
                st.divider()

                st.subheader("Analysis Results")

                # Display three analysis metrics
                col1, col2, col3 = st.columns(3)

                # Cosine Similarity
                with col1:
                    st.metric(
                        "Cosine Similarity",
                        f"{similarity * 100:.2f}%"
                    )

                # Semantic Similarity
                with col2:
                    st.metric(
                        "Semantic Similarity",
                        f"{semantic_similarity * 100:.2f}%"
                    )

                # ML Prediction
                with col3:
                    st.metric(
                        "Prediction",
                        "Duplicate" if prediction == 1 else "Non-Duplicate"
                    ) 
                            # Display prediction result
                if prediction == 1:
                    st.success(
                        "✅ These questions are predicted as DUPLICATES."
                    )
                else:
                    st.error(
                        "❌ These questions are predicted as NON-DUPLICATES."
                    )

                # Display similarity progress
                st.progress(
                    min(max(similarity, 0.0), 1.0)
                )


                # Save history

                st.session_state.history.append({

                    "Question 1": question1,
                    "Question 2": question2,

                    "Prediction": (
                        "Duplicate"
                        if prediction == 1
                        else "Non-Duplicate"
                    ),

                    "Similarity": round(
                        similarity * 100, 2
                    )

                })

            except Exception as e:

                st.error(f"Prediction Error: {e}")

                st.info(
                    "Please verify that app.py and train.py "
                    "use the same preprocessing and feature extraction."
                )

    st.divider()

    st.subheader("Project Technologies")

    st.write("""
    - Python
    - Pandas
    - NumPy
    - Scikit-learn
    - NLP
    - TF-IDF Vectorization
    - Cosine Similarity
    - Linear SVM
    - Streamlit
    """)


# ==========================================
# 12. PREDICTION HISTORY
# ==========================================

elif page == "Prediction History":

    st.title("📋 Prediction History")

    if len(st.session_state.history) > 0:

        history_df = pd.DataFrame(
            st.session_state.history
        )

        st.dataframe(
            history_df,
            width="stretch"
        )

        csv = history_df.to_csv(index=False).encode("utf-8")

        st.download_button(
            "Download History CSV",
            data=csv,
            file_name="prediction_history.csv",
            mime="text/csv"
        )

        if st.button("Clear History"):

            st.session_state.history = []

            st.rerun()

    else:

        st.info("No prediction history available yet.")


# ==========================================
# 13. MODEL INFORMATION
# ==========================================

elif page == "Model Information":

    st.title("🤖 Model Information")

    st.subheader("Selected Algorithm")

    st.success(
        model.__class__.__name__
    )

    st.subheader("Algorithms Used")

    st.write("""
    1. Logistic Regression
    2. Random Forest
    3. Linear Support Vector Machine
    4. TF-IDF Vectorization
    5. Cosine Similarity
    """)

    st.subheader("Model Performance Comparison")

    if not comparison.empty:

        st.dataframe(
            comparison,
            width="stretch"
        )

    else:

        st.warning(
            "Model comparison file is not available."
        )

    st.subheader("Feature Engineering")

    st.write("""
    The application uses:

    1. TF-IDF Vectorization
    2. Difference between TF-IDF vectors
    3. Element-wise product of TF-IDF vectors
    4. Cosine similarity
    5. Word overlap
    6. Jaccard similarity
    7. Question length features
    """)

    st.subheader("Project Technologies")

    st.write("""
    - Python
    - Pandas
    - NumPy
    - Scikit-learn
    - NLTK
    - Streamlit
    - Matplotlib
    - Seaborn
    """)

    st.success("Application is running successfully!")