import os
import json
import re
import datetime as dt
from io import StringIO

import streamlit as st
import pandas as pd
import requests

# ------------------------------
# CONFIGURATION
# ------------------------------
# Expect the Anthropic API key as an env var. You can rename the env var if you prefer.
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
ANTHROPIC_MODEL   = "claude-sonnet-4-20250514"

st.set_page_config(
    page_title="Dictionary Refinement",
    page_icon="📖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ------------------------------
# SESSION STATE INITIALISATION
# ------------------------------
if "keywords" not in st.session_state:
    st.session_state.keywords = []  # list[str]
if "uploaded_df" not in st.session_state:
    st.session_state.uploaded_df = None  # pd.DataFrame | None
if "preview_df" not in st.session_state:
    st.session_state.preview_df = pd.DataFrame()

# ------------------------------
# HELPER FUNCTIONS
# ------------------------------

def call_anthropic(prompt: str, max_tokens: int = 1000):
    """Call the Anthropic Messages API and return text content."""
    if not ANTHROPIC_API_KEY:
        raise RuntimeError("Missing ANTHROPIC_API_KEY env var")

    url = "https://api.anthropic.com/v1/messages"
    headers = {
        "x-api-key": ANTHROPIC_API_KEY,
        "anthropic-version": "2023-06-01",
        "content-type": "application/json",
    }
    payload = {
        "model": ANTHROPIC_MODEL,
        "max_tokens": max_tokens,
        "messages": [{"role": "user", "content": prompt}],
    }
    r = requests.post(url, headers=headers, json=payload, timeout=90)
    r.raise_for_status()
    ans = r.json()
    # Anthropic returns a list in "content" field
    text = ans["content"][0]["text"] if isinstance(ans.get("content"), list) else ans.get("content", "")
    return text.strip()


def generate_keywords(definition: str) -> list[str]:
    prompt = (
        f"Generate 30 one-word (unigram) keywords that signal the following tactic: {definition}.\n\n"
        "Return as a JSON array of strings. Focus on words that would clearly indicate this tactic when found in text.\n\n"
        "Example format: [\"word1\", \"word2\", \"word3\", ...]\n\n"
        "IMPORTANT: Your entire response must be a single, valid JSON array. Do not include any text outside of the JSON structure."
    )
    raw = call_anthropic(prompt)
    # Clean common formatting issues (backticks, triple‑code fences, etc.)
    cleaned = re.sub(r"```json|```", "", raw).strip()
    keywords = json.loads(cleaned)
    # normalise: lower‑case & strip
    return sorted({kw.strip().lower() for kw in keywords if kw.strip()})


def parse_csv(file_bytes: bytes) -> pd.DataFrame:
    """Load CSV bytes into DataFrame and validate required columns."""
    df = pd.read_csv(StringIO(file_bytes.decode("utf-8")))
    required = {"ID", "Statement"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(missing)}")
    return df


def find_matches(df: pd.DataFrame, kw: list[str]) -> pd.DataFrame:
    if df is None or not kw:
        return pd.DataFrame()
    pattern = re.compile(r"\\b(" + "|".join(re.escape(k) for k in kw) + r")\\b", re.IGNORECASE)

    def _extract(statement: str):
        found = pattern.findall(statement or "")
        found = [f.lower() for f in found]
        uniq  = sorted(set(found))
        return uniq, len(uniq)

    results = []
    for _, row in df.iterrows():
        uniq, cnt = _extract(str(row["Statement"]))
        if cnt > 0:
            results.append({
                "ID": row.get("ID"),
                "Statement": row.get("Statement"),
                "MatchedKeywords": ", ".join(uniq),
                "Count": cnt,
            })
    return pd.DataFrame(results)

# ------------------------------
# UI LAYOUT
# ------------------------------
# Sidebar – Configuration
st.sidebar.header("🎯 Configuration")

tactic_def = st.sidebar.text_area(
    "Tactic Definition", value="", height=120,
    placeholder="Enter the tactic you want to create keywords for…",
)

col_gk, col_prev, col_dl = st.sidebar.columns(3)

with col_gk:
    if st.button("🔮 Generate", use_container_width=True, disabled=not tactic_def.strip()):
        try:
            with st.spinner("Calling Anthropic…"):
                st.session_state.keywords = generate_keywords(tactic_def)
            st.success(f"Generated {len(st.session_state.keywords)} keywords ✅")
        except Exception as e:
            st.error(f"Keyword generation failed: {e}")

with col_prev:
    prev_disabled = not (st.session_state.keywords and st.session_state.uploaded_df is not None)
    if st.button("🔍 Preview", use_container_width=True, disabled=prev_disabled):
        st.session_state.preview_df = find_matches(
            st.session_state.uploaded_df, st.session_state.keywords
        ).head(10)
        if not st.session_state.preview_df.empty:
            st.success(f"Found {len(st.session_state.preview_df)} sample matches")
        else:
            st.info("No matches found in sample data")

with col_dl:
    if st.session_state.keywords:
        dict_data = {
            "keywords": st.session_state.keywords,
            "count": len(st.session_state.keywords),
            "created_at": dt.datetime.utcnow().isoformat(),
            "tactic": tactic_def.strip(),
        }
        json_bytes = json.dumps(dict_data, indent=2).encode("utf-8")
        st.download_button(
            "⬇️ Download", data=json_bytes, file_name="dictionary.json", mime="application/json",
            use_container_width=True,
        )
    else:
        st.button("⬇️ Download", use_container_width=True, disabled=True)

# File uploader
uploaded_file = st.sidebar.file_uploader("Upload Sample CSV (ID, Statement columns)", type="csv")
if uploaded_file is not None:
    try:
        df = parse_csv(uploaded_file.read())
        st.session_state.uploaded_df = df
        st.sidebar.success(f"Loaded {len(df)} rows ✅")
    except Exception as e:
        st.session_state.uploaded_df = None
        st.sidebar.error(f"Error parsing CSV: {e}")

st.sidebar.markdown("---")
st.sidebar.write("Made with ❤️ in Streamlit")

# ------------------------------
# MAIN PANEL
# ------------------------------

st.title("📖 Dictionary Refinement")
st.caption("Craft high‑quality keyword lists with AI assistance.")

# Show tactic definition summary
if tactic_def.strip() and st.session_state.keywords:
    with st.expander("Current Tactic Definition", expanded=True):
        st.markdown(f"**{tactic_def.strip()}**")

# Keyword Editor
st.subheader("🏷️ Keyword Editor")

kw_col1, kw_col2 = st.columns([3, 1])
with kw_col1:
    new_kw = st.text_input("Add new keyword", placeholder="Type a keyword and hit Enter")
    if new_kw:
        norm = new_kw.strip().lower()
        if norm and norm not in st.session_state.keywords:
            st.session_state.keywords.append(norm)
            st.session_state.keywords.sort()
            st.success(f"Added '{norm}' to dictionary")
with kw_col2:
    if st.button("➕ Add", disabled=not new_kw):
        pass  # handled above automatically on text input

if st.session_state.keywords:
    # Display as chips with remove buttons
    cols = st.columns(6)
    for i, kw in enumerate(st.session_state.keywords):
        with cols[i % 6]:
            remove = st.button("❌", key=f"rm_{kw}")
            st.write(kw)
        if remove:
            st.session_state.keywords.remove(kw)
            st.experimental_rerun()
    st.info(f"Dictionary contains **{len(st.session_state.keywords)}** keywords.")
else:
    st.warning("No keywords yet – generate or add manually.")

# Statistics
if st.session_state.keywords or st.session_state.uploaded_df is not None:
    st.subheader("📊 Statistics")
    stat1, stat2 = st.columns(2)
    stat1.metric("Keywords", f"{len(st.session_state.keywords)}")
    n_rows = len(st.session_state.uploaded_df) if st.session_state.uploaded_df is not None else 0
    stat2.metric("Sample Rows", f"{n_rows}")

# Preview Results
if not st.session_state.preview_df.empty:
    st.subheader("🎯 Sample Matches (first 10)")
    st.dataframe(st.session_state.preview_df, hide_index=True, use_container_width=True)

    total_matches = find_matches(
        st.session_state.uploaded_df, st.session_state.keywords
    ) if st.session_state.uploaded_df is not None else pd.DataFrame()

    if not total_matches.empty and len(total_matches) > len(st.session_state.preview_df):
        st.caption(f"Showing first 10 of {len(total_matches)} total matches.")

# End of app

