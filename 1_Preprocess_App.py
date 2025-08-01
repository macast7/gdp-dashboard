import json
from io import StringIO
from pathlib import Path

import nltk
import pandas as pd
import streamlit as st

# Ensure the Punkt tokenizer is available (suppresses download log noise)
nltk.download("punkt", quiet=True)

st.set_page_config(page_title="Instagram Caption Pre‑processor", page_icon="📸", layout="centered")

st.title("📸 Instagram Caption Pre‑processor")
st.markdown(
    """
Transform your raw Instagram post data into a sentence‑level CSV that’s ready for downstream analysis. 

**How it works**
1. Upload a CSV containing `shortcode` and `caption` columns (the defaults expected by CrowdTangle exports).
2. (Optional) Adjust the column‑renaming dictionary if your CSV uses different names.
3. Download the transformed file—each sentence gets its own row with incremental sentence IDs.
"""
)

# --- Sidebar: dictionary controls ------------------------------------------------------
with st.sidebar:
    st.header("⚙️ Rename Columns")
    st.write("Modify the mapping if your CSV uses different column names.")

    DEFAULT_MAPPING = {"shortcode": "ID", "caption": "Context"}

    mapping_str = st.text_area(
        "Column mapping (JSON)",
        value=json.dumps(DEFAULT_MAPPING, indent=2),
        height=120,
        help="Keys are your source column names; values are the desired output names.",
    )

    # Parse mapping safely
    try:
        rename_dict = json.loads(mapping_str)
        if not isinstance(rename_dict, dict):
            st.warning("Mapping must be a JSON object/dictionary. Reverting to default.")
            rename_dict = DEFAULT_MAPPING
    except json.JSONDecodeError as e:
        st.warning(f"Invalid JSON: {e}. Reverting to default mapping.")
        rename_dict = DEFAULT_MAPPING

    st.write("Current mapping:")
    st.code(json.dumps(rename_dict, indent=2), language="json")

# --- Utility functions ----------------------------------------------------------------

def sentence_tokenize(text: str):
    """Return a clean list of sentences for a single caption."""
    if pd.isna(text):
        return []
    # Normalise fancy apostrophes so they don't split words unexpectedly
    text = text.replace("’", "'")
    return [s.strip() for s in nltk.sent_tokenize(text) if s.strip()]


def transform(df_raw: pd.DataFrame, mapping: dict) -> pd.DataFrame:
    """Rename columns according to mapping and explode captions into sentences."""
    # Validate required columns
    missing = [col for col in mapping.keys() if col not in df_raw.columns]
    if missing:
        raise ValueError(f"Missing required column(s) in uploaded file: {', '.join(missing)}")

    df = df_raw.rename(columns=mapping)

    records = []
    for _, row in df.iterrows():
        for idx, sent in enumerate(sentence_tokenize(row[mapping.get('caption', 'Context')]), start=1):
            records.append({
                mapping.get('shortcode', 'ID'): row[mapping.get('shortcode', 'ID')],
                "Sentence ID": idx,
                mapping.get('caption', 'Context'): row[mapping.get('caption', 'Context')],
                "Statement": sent,
            })
    return pd.DataFrame(records)

# --- Main layout ----------------------------------------------------------------------

uploaded_file = st.file_uploader("Upload CSV", type="csv")

if uploaded_file:
    try:
        df_raw = pd.read_csv(uploaded_file)
        st.success(f"Loaded {len(df_raw):,} rows from {uploaded_file.name}")

        with st.expander("Preview raw data"):
            st.dataframe(df_raw.head())

        # Perform transformation
        try:
            df_out = transform(df_raw, rename_dict)
        except Exception as e:
            st.error(f"❌ Transformation failed: {e}")
        else:
            st.subheader("🔍 Transformed Data Preview")
            st.dataframe(df_out.head())
            st.info(f"Transformation produced {len(df_out):,} rows.")

            # Offer download
            csv_buffer = StringIO()
            df_out.to_csv(csv_buffer, index=False)
            st.download_button(
                label="💾 Download transformed CSV",
                data=csv_buffer.getvalue(),
                file_name=f"{Path(uploaded_file.name).stem}_transformed.csv",
                mime="text/csv",
            )

    except Exception as e:
        st.error(f"❌ Could not read CSV: {e}")
else:
    st.info("👆 Upload a CSV file to begin.")
