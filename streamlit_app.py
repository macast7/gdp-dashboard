import io
import re
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="Marketing Term Classifier",
    page_icon="📈",
    layout="wide",
)

st.title("📈 Marketing Term Classifier")
st.markdown(
    """
Upload a CSV file containing statements, **customize the keyword dictionaries**, and classify each row on‑the‑fly.

* The app adds a `categories` column (semicolon‑separated labels).
* One boolean column is created **per dictionary**.
* Download the enriched dataset when you’re done.
    """
)

# ---------------------------------------------------------------------------
# 1️⃣  Default dictionaries (copied from your notebook)
# ---------------------------------------------------------------------------
DEFAULT_DICTIONARIES: dict[str, set[str]] = {
    "urgency_marketing": {
        "limited",
        "limited time",
        "limited run",
        "limited edition",
        "order now",
        "last chance",
        "hurry",
        "while supplies last",
        "before they're gone",
        "selling out",
        "selling fast",
        "act now",
        "don't wait",
        "today only",
        "expires soon",
        "final hours",
        "almost gone",
    },
    "exclusive_marketing": {
        "exclusive",
        "exclusively",
        "exclusive offer",
        "exclusive deal",
        "members only",
        "vip",
        "special access",
        "invitation only",
        "premium",
        "privileged",
        "limited access",
        "select customers",
        "insider",
        "private sale",
        "early access",
    },
}

# ---------------------------------------------------------------------------
# 2️⃣  Persist dictionaries in session_state so the UI is reactive
# ---------------------------------------------------------------------------
if "dictionaries" not in st.session_state:
    # Deep‑copy so edits don’t mutate the original constant
    st.session_state["dictionaries"] = {
        k: set(v) for k, v in DEFAULT_DICTIONARIES.items()
    }

dictionaries: dict[str, set[str]] = st.session_state["dictionaries"]

# ---------------------------------------------------------------------------
# 3️⃣  File uploader + preview
# ---------------------------------------------------------------------------
uploaded_file = st.file_uploader("📤 Upload a CSV file", type=["csv"])

if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
    except Exception as e:
        st.error(f"⚠️ Could not read the file: {e}")
        st.stop()

    st.subheader("👀 Data preview")
    st.dataframe(df.head(), use_container_width=True)

    # Let user choose which column contains the text
    text_columns = df.select_dtypes(include=["object"]).columns.tolist()
    if not text_columns:
        st.error("⚠️ No text columns found in your dataset.")
        st.stop()

    statement_col = st.selectbox(
        "Select the column that contains the statements to classify:",
        options=text_columns,
        index=text_columns.index("Statement") if "Statement" in text_columns else 0,
        key="statement_col",
    )

    # -----------------------------------------------------------------------
    # 4️⃣  Dictionary editor
    # -----------------------------------------------------------------------
    st.subheader("📝 Edit keyword dictionaries")

    for cat in sorted(dictionaries):
        with st.expander(f"{cat} ({len(dictionaries[cat])} terms)", expanded=False):
            terms_str = st.text_area(
                "Comma‑separated terms", value=", ".join(sorted(dictionaries[cat])), key=f"ta_{cat}"
            )
            # Update the dictionary live
            dictionaries[cat] = {t.strip() for t in terms_str.split(",") if t.strip()}

    # Add new category UI
    with st.expander("➕ Add a new category"):
        new_cat = st.text_input("Category name", key="new_cat")
        new_terms = st.text_area("Comma‑separated terms", key="new_terms")
        if st.button("Add category", key="btn_add") and new_cat:
            if new_cat in dictionaries:
                st.warning("Category already exists.")
            else:
                dictionaries[new_cat] = {
                    t.strip() for t in new_terms.split(",") if t.strip()
                }
                st.success(f"Added new category **{new_cat}** ✓")

    # -----------------------------------------------------------------------
    # 5️⃣  Run classification
    # -----------------------------------------------------------------------
    if st.button("🚀 Run classification"):
        st.info("Compiling patterns and classifying… this may take a moment for large datasets.")

        # Pre‑compile regex patterns for all categories
        compiled_patterns: dict[str, re.Pattern[str]] = {
            cat: re.compile(
                r"\b(?:" + "|".join(re.escape(term) for term in terms) + r")\b",
                re.I,
            )
            for cat, terms in dictionaries.items()
            if terms  # skip empty term lists
        }

        def classify(text: str) -> list[str]:
            if pd.isna(text):
                return []
            return [cat for cat, pat in compiled_patterns.items() if pat.search(str(text))]

        # Apply
        results = df[statement_col].apply(classify)
        df["categories"] = results.map("; ".join)

        for cat in dictionaries:
            df[cat] = results.map(lambda lst, c=cat: c in lst)

        st.success("✅ Classification complete!")
        st.subheader("🔎 Results preview")
        st.dataframe(df.head(), use_container_width=True)

        # -------------------------------------------------------------------
        # 6️⃣  Download enriched dataset
        # -------------------------------------------------------------------
        out_csv = df.to_csv(index=False)
        st.download_button(
            label="💾 Download full results as CSV",
            data=out_csv,
            file_name="classified_output.csv",
            mime="text/csv",
        )
else:
    st.info("⬆️ Upload a CSV file to get started.")
