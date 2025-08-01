import io
from pathlib import Path
from typing import Dict, Set, List

import pandas as pd
import streamlit as st

###############################################################################
# Streamlit – Marketing Keyword Classifier                                   #
###############################################################################
st.set_page_config(page_title="Marketing Keyword Classifier", layout="wide")
st.title("📈 Marketing Keyword Classifier")

# ---------------------------------------------------------------------------
# 🛠️ Sidebar – Upload & Configuration
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("🗂️ 1. Upload Your CSV")
    uploaded_file = st.file_uploader("CSV file with a 'Statement' column", type=["csv"])

    st.markdown("---")
    st.header("🔧 2. Configure Dictionaries")

    # Default marketing keyword dictionaries
    default_dicts: Dict[str, Set[str]] = {
        "urgency_marketing": {
            "limited", "limited time", "limited run", "limited edition", "order now",
            "last chance", "hurry", "while supplies last", "before they're gone",
            "selling out", "selling fast", "act now", "don't wait", "today only",
            "expires soon", "final hours", "almost gone",
        },
        "exclusive_marketing": {
            "exclusive", "exclusively", "exclusive offer", "exclusive deal",
            "members only", "vip", "special access", "invitation only",
            "premium", "privileged", "limited access", "select customers",
            "insider", "private sale", "early access",
        },
    }

    # Load edited or new dictionaries into this object
    current_dicts: Dict[str, Set[str]] = {}

    for label, keywords in default_dicts.items():
        kw_text = "\n".join(sorted(keywords))
        new_kw_text = st.text_area(
            f"Keywords for **{label}** (one per line)", kw_text, key=label
        )
        kw_set = {kw.strip().lower() for kw in new_kw_text.split("\n") if kw.strip()}
        if kw_set:
            current_dicts[label] = kw_set

    # Section to add a completely new category
    st.markdown("---")
    st.subheader("➕ Add New Category")
    new_label = st.text_input("New category name (alphanumeric and underscores)")
    new_kw_input = st.text_area("Keywords for new category (one per line)")
    if new_label and new_kw_input:
        new_kw_set = {kw.strip().lower() for kw in new_kw_input.split("\n") if kw.strip()}
        if new_kw_set:
            current_dicts[new_label.strip().lower()] = new_kw_set

    st.markdown("---")
    one_hot = st.checkbox("Add one‑hot encoded columns", value=True)

###############################################################################
# Helper – Classification Function
###############################################################################

def classify_statement(text: str, dictionaries: Dict[str, Set[str]]) -> List[str]:
    """Return list of dictionary names whose keywords appear in *text*."""
    text_lower = text.lower()
    matched: List[str] = []
    for label, keywords in dictionaries.items():
        if any(kw in text_lower for kw in keywords):
            matched.append(label)
    return matched

###############################################################################
# 🚀 Main – Run Classification & Display Results
###############################################################################

def run_classifier(file_buffer: io.BytesIO, dictionaries: Dict[str, Set[str]]):
    df = pd.read_csv(file_buffer)

    if "Statement" not in df.columns:
        st.error("❌ The uploaded CSV must contain a column named 'Statement'.")
        return

    # Classify
    with st.spinner("Classifying statements…"):
        df["labels"] = df["Statement"].astype(str).apply(classify_statement, dictionaries=dictionaries)
        if one_hot:
            for label in dictionaries:
                df[label] = df["labels"].apply(lambda cats, lbl=label: lbl in cats)

    st.success("✅ Classification complete!")

    # Preview
    st.subheader("🔍 Preview (first 10 rows)")
    st.dataframe(df.head(10), use_container_width=True)

    # Download
    csv_bytes = df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="Download classified CSV",
        data=csv_bytes,
        file_name="classified_output.csv",
        mime="text/csv",
    )

###############################################################################
# 🏁 App Execution
###############################################################################
if uploaded_file is not None:
    try:
        run_classifier(uploaded_file, current_dicts)
    except Exception as e:
        st.exception(e)
else:
    st.info("👆 Upload a CSV file to get started.")
