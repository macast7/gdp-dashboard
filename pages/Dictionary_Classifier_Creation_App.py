import json
import io
import re
from typing import List, Dict, Tuple

import numpy as np
import pandas as pd
import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt

st.set_page_config(page_title="Dictionary Classifier", page_icon="📚", layout="wide")

st.title("📚 Dictionary Classifier Creation")
st.markdown("Turn keyword lists into **0/1** classifiers and evaluate performance on your labelled data.")

# -----------------------------------------------------------------------------
# Helper functions
# -----------------------------------------------------------------------------

def detect_columns(headers: List[str], file_type: str) -> Dict[str, str]:
    """Attempt to infer sensible default column mappings."""
    id_patterns = ["id", "identifier", "key"]
    text_patterns = ["statement", "text", "sentence", "content", "message"]
    label_patterns = ["mode_researcher", "label", "target", "class", "ground_truth"]

    def find(patterns: List[str]):
        for header in headers:
            for pattern in patterns:
                if pattern.lower() in header.lower():
                    return header
        return headers[0] if headers else ""

    if file_type == "sentences":
        return {
            "id": find(id_patterns),
            "text": find(text_patterns),
        }
    elif file_type == "labels":
        return {
            "id": find(id_patterns),
            "label": find(label_patterns),
        }
    return {}


def classify_text(text: str, keywords: List[str]) -> Tuple[int, List[str]]:
    """Return prediction (0/1) and matched keywords."""
    matched = []
    for kw in keywords:
        if re.search(rf"\b{re.escape(kw.lower())}\b", text.lower()):
            matched.append(kw)
    return (1 if matched else 0), matched


def calculate_metrics(pred: List[int], actual: List[int]):
    tp = sum((p == 1 and a == 1) for p, a in zip(pred, actual))
    tn = sum((p == 0 and a == 0) for p, a in zip(pred, actual))
    fp = sum((p == 1 and a == 0) for p, a in zip(pred, actual))
    fn = sum((p == 0 and a == 1) for p, a in zip(pred, actual))

    accuracy = (tp + tn) / max(len(pred), 1)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "matrix": {"tp": tp, "tn": tn, "fp": fp, "fn": fn},
    }


def render_confusion_matrix(matrix: Dict[str, int]):
    data = np.array([[matrix["tn"], matrix["fp"]], [matrix["fn"], matrix["tp"]]])
    fig, ax = plt.subplots(figsize=(4, 3))
    sns.heatmap(data, annot=True, fmt="d", cmap="Blues", cbar=False,
                xticklabels=["Pred 0", "Pred 1"],
                yticklabels=["Actual 0", "Actual 1"], ax=ax)
    ax.set_title("Confusion Matrix")
    st.pyplot(fig)


# -----------------------------------------------------------------------------
# Sidebar – file uploads & keyword editing
# -----------------------------------------------------------------------------

st.sidebar.header("📂 Upload Files")

sentences_file = st.sidebar.file_uploader("Sentences CSV", type="csv")
labels_file = st.sidebar.file_uploader("Labels CSV", type="csv")
keywords_file = st.sidebar.file_uploader("Keywords JSON", type="json")

# Initialise state variables
if "kw_edit" not in st.session_state:
    st.session_state.kw_edit = False
if "keywords_text" not in st.session_state:
    st.session_state.keywords_text = ""

# -----------------------------------------------------------------------------
# Parse uploaded files
# -----------------------------------------------------------------------------

sentences_df, labels_df, keywords_list = pd.DataFrame(), pd.DataFrame(), []

if sentences_file:
    sentences_df = pd.read_csv(sentences_file)
    st.sidebar.success(f"Loaded {len(sentences_df):,} sentence rows")

if labels_file:
    labels_df = pd.read_csv(labels_file)
    st.sidebar.success(f"Loaded {len(labels_df):,} label rows")

if keywords_file:
    try:
        raw = json.load(keywords_file)
        if isinstance(raw, list):
            keywords_list = raw
        elif isinstance(raw, dict):
            # Flatten dict values
            keywords_list = [kw for v in raw.values() for kw in (v if isinstance(v, list) else [v])]
        else:
            st.sidebar.error("JSON should contain a list or object of lists")
        st.session_state.keywords_text = "\n".join(keywords_list)
        st.sidebar.success(f"Loaded {len(keywords_list):,} keywords")
    except Exception as e:
        st.sidebar.error(f"Failed to parse keywords: {e}")

# -----------------------------------------------------------------------------
# Column mapping UI
# -----------------------------------------------------------------------------

col1, col2 = st.columns(2)

with col1:
    if not sentences_df.empty:
        default_sentence_map = detect_columns(sentences_df.columns.tolist(), "sentences")
        st.subheader("📝 Sentences – Column Mapping")
        sent_id_col = st.selectbox("Sentence ID column", sentences_df.columns, index=sentences_df.columns.get_loc(default_sentence_map["id"]))
        sent_text_col = st.selectbox("Sentence text column", sentences_df.columns, index=sentences_df.columns.get_loc(default_sentence_map["text"]))

with col2:
    if not labels_df.empty:
        default_label_map = detect_columns(labels_df.columns.tolist(), "labels")
        st.subheader("🎯 Labels – Column Mapping")
        label_id_col = st.selectbox("Label ID column", labels_df.columns, index=labels_df.columns.get_loc(default_label_map["id"]))
        label_value_col = st.selectbox("Label value column (0/1)", labels_df.columns, index=labels_df.columns.get_loc(default_label_map["label"]))

# -----------------------------------------------------------------------------
# Keyword editor
# -----------------------------------------------------------------------------

if keywords_list:
    with st.expander("✏️ Edit keywords"):
        st.session_state.kw_edit = st.checkbox("Enable edit", value=st.session_state.kw_edit, help="Toggle to edit keywords manually")
        textarea_disabled = not st.session_state.kw_edit
        st.session_state.keywords_text = st.text_area(
            "One keyword per line", value=st.session_state.keywords_text, height=150, disabled=textarea_disabled
        )

# Use edited or original keywords
final_keywords = [kw.strip() for kw in st.session_state.keywords_text.split("\n") if kw.strip()] if st.session_state.keywords_text else keywords_list

# -----------------------------------------------------------------------------
# Data previews
# -----------------------------------------------------------------------------

if not sentences_df.empty:
    st.subheader("Preview – Sentences")
    st.dataframe(sentences_df.head())

if not labels_df.empty:
    st.subheader("Preview – Labels")
    st.dataframe(labels_df.head())

# -----------------------------------------------------------------------------
# Run classification
# -----------------------------------------------------------------------------

run_button = st.button("🚀 Run Classification", disabled=sentences_df.empty or labels_df.empty or not final_keywords)

if run_button:
    with st.spinner("Running classification…"):
        # Build look‑up maps
        sentence_map = sentences_df.set_index(sent_id_col)[sent_text_col].dropna().to_dict()
        labels_map = labels_df.set_index(label_id_col)[label_value_col].dropna().astype(int).to_dict()

        predictions, ground_truth, rows = [], [], []
        for sid, text in sentence_map.items():
            if sid in labels_map:
                pred, matched = classify_text(text, final_keywords)
                predictions.append(pred)
                ground_truth.append(labels_map[sid])
                rows.append({
                    "ID": sid,
                    "Statement": text,
                    "Prediction": pred,
                    "GroundTruth": labels_map[sid],
                    "MatchedKeywords": ", ".join(matched),
                })

        if not rows:
            st.error("No overlapping IDs between sentences and labels.")
            st.stop()

        metrics = calculate_metrics(predictions, ground_truth)

        # ---------------------------------------------------------------------
        # Results display
        # ---------------------------------------------------------------------

        mcol1, mcol2, mcol3, mcol4 = st.columns(4)
        mcol1.metric("Accuracy", f"{metrics['accuracy']*100:.2f}%")
        mcol2.metric("Precision", f"{metrics['precision']*100:.2f}%")
        mcol3.metric("Recall", f"{metrics['recall']*100:.2f}%")
        mcol4.metric("F1 Score", f"{metrics['f1']*100:.2f}%")
        st.caption(f"Total samples: {len(rows):,}")

        render_confusion_matrix(metrics["matrix"])

        st.subheader("Sample Predictions")
        st.dataframe(pd.DataFrame(rows).head(10))

        # ------------------------------------------------------------------
        # Download button
        # ------------------------------------------------------------------

        csv_buffer = io.StringIO()
        pd.DataFrame(rows).to_csv(csv_buffer, index=False)
        st.download_button(
            label="💾 Download full predictions as CSV",
            data=csv_buffer.getvalue(),
            file_name="statement_level_predictions.csv",
            mime="text/csv",
        )
