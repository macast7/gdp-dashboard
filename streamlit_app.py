"""
Streamlit App – Conversational Context ↔ Statement Pair Generator
-----------------------------------------------------------------
A drop‑in replacement for the original *preprocess_app.py* (Gradio/CLI)
that lets users:
  • upload a CSV conversational dataset
  • tweak column names & preprocessing parameters (the “default dictionary”)
  • preview the first rows of generated (context, statement) pairs
  • download the three resulting CSVs

Run with:
    streamlit run streamlit_preprocess_app.py

Author: ChatGPT – Streamlit App Creator · 2025‑08‑01
"""
from __future__ import annotations

import io
import datetime as _dt
from pathlib import Path
from typing import List, Tuple

import pandas as pd
import nltk
from nltk.tokenize import sent_tokenize
import streamlit as st

# ──────────────────────────────────────────────────────────────────────────────
# Constants & helpers – these constitute the "default dictionary" that the user
# can override via the sidebar controls.
# ──────────────────────────────────────────────────────────────────────────────
DEFAULT_ID_COL = "ID"
DEFAULT_TURN_COL = "Turn"
DEFAULT_STATEMENT_COL = "Statement"
DEFAULT_SPEAKER_COL = "Speaker"

STATEMENT_UNITS = ["sentence", "turn", "post"]
CONTEXT_SCOPES = ["rolling", "whole"]
SPEAKER_CHOICES = ["customer", "salesperson"]

_TIMESTAMP = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")


def _clean_text(t: str | float) -> str:
    """Normalize quotes, strip, and handle NaN values."""
    if isinstance(t, float):
        return ""
    return (
        str(t)
        .replace("’", "'")
        .replace("“", '"')
        .replace("”", '"')
        .replace("\u200b", "")
        .strip()
    )


def _ensure_punkt():
    """Download NLTK Punkt tokenizer quietly if it isn't present."""
    try:
        nltk.data.find("tokenizers/punkt")
    except LookupError:
        nltk.download("punkt", quiet=True)


# ──────────────────────────────────────────────────────────────────────────────
# Core preprocessing – copied (lightly refactored) from the original script
# ──────────────────────────────────────────────────────────────────────────────

def preprocess_pairs(
    df: pd.DataFrame,
    *,
    id_col: str = DEFAULT_ID_COL,
    turn_col: str | None = DEFAULT_TURN_COL,
    statement_col: str = DEFAULT_STATEMENT_COL,
    speaker_col: str | None = DEFAULT_SPEAKER_COL,
    statement_unit: str = "sentence",
    context_scope: str = "rolling",
    statement_speaker: str = "customer",
    include_both_speakers_in_context: bool = True,
    window_n: int = 0,
    assume_odd_even: bool = True,
    language: str = "english",
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """See original docstring in *preprocess_app.py*."""
    if id_col not in df.columns:
        raise ValueError(f"Column '{id_col}' not found in data.")
    if statement_col not in df.columns:
        raise ValueError(f"Column '{statement_col}' not found in data.")
    if statement_unit in ("sentence", "turn") and (turn_col is None or turn_col not in df.columns):
        raise ValueError(
            f"Column '{turn_col}' required for statement_unit='{statement_unit}'."
        )
    if speaker_col and speaker_col not in df.columns:
        speaker_col = None  # treat as missing

    _ensure_punkt()

    # Normalize statement text
    df = df.copy()
    df[statement_col] = df[statement_col].apply(_clean_text)

    # Resolve / create speaker column
    if speaker_col:
        df["Speaker_resolved"] = df[speaker_col].fillna("unknown").astype(str).str.lower()
    else:
        if assume_odd_even and turn_col and turn_col in df.columns:
            df["Speaker_resolved"] = df[turn_col].apply(lambda x: "customer" if int(x) % 2 == 1 else "salesperson")
        else:
            df["Speaker_resolved"] = "unknown"

    # Cast Turn to int for sorting
    if turn_col and turn_col in df.columns:
        df[turn_col] = pd.to_numeric(df[turn_col], errors="coerce").fillna(0).astype(int)

    # Sort rows
    sort_cols = [id_col]
    if turn_col and turn_col in df.columns:
        sort_cols.append(turn_col)
    df = df.sort_values(sort_cols).reset_index(drop=True)

    # Filter to statement speaker rows
    filtered_df = df[df["Speaker_resolved"] == statement_speaker].copy()

    # Build units list
    units: List[dict] = []
    idx_counter: dict[str, int] = {}

    if statement_unit == "post":
        for _id, grp in filtered_df.groupby(id_col, sort=False):
            text_concat = " ".join(grp[statement_col].dropna().astype(str).tolist()).strip()
            if text_concat:
                units.append({"ID": _id, "Turn": "", "Speaker": statement_speaker, "unit_level": "post", "unit_index": 1, "text": text_concat})
    else:  # sentence / turn
        for _, row in filtered_df.iterrows():
            _id = row[id_col]
            idx_counter.setdefault(_id, 0)

            if statement_unit == "turn":
                text = row[statement_col]
                if text:
                    idx_counter[_id] += 1
                    units.append({"ID": _id, "Turn": row[turn_col], "Speaker": statement_speaker, "unit_level": "turn", "unit_index": idx_counter[_id], "text": text})
            else:  # sentence
                sentences = [s.strip() for s in sent_tokenize(row[statement_col], language=language) if s.strip()]
                for sent in sentences:
                    idx_counter[_id] += 1
                    units.append({"ID": _id, "Turn": row[turn_col], "Speaker": statement_speaker, "unit_level": "sentence", "unit_index": idx_counter[_id], "text": sent})

    if not units:
        raise ValueError("No statement units generated – check your parameters.")

    units_df = pd.DataFrame(units)
    units_df["text"] = units_df["text"].apply(_clean_text)

    # Build context & pairs
    pair_rows: List[dict] = []
    for _id, grp in units_df.groupby("ID", sort=False):
        grp = grp.reset_index(drop=True)
        texts_full_order = grp["text"].tolist()
        speakers_full_order = grp["Speaker"].tolist()

        for i, unit_row in grp.iterrows():
            if context_scope == "rolling":
                candidate_idx = list(range(i))
                if window_n > 0:
                    candidate_idx = candidate_idx[-window_n:]
            else:  # whole
                candidate_idx = list(range(len(grp)))
                candidate_idx.remove(i)

            if not include_both_speakers_in_context:
                candidate_idx = [j for j in candidate_idx if speakers_full_order[j] == statement_speaker]

            context_text = " ".join([texts_full_order[j] for j in candidate_idx]).strip()
            pair_rows.append({
                "ID": unit_row["ID"],
                "Turn": unit_row["Turn"],
                "Speaker": unit_row["Speaker"],
                "unit_level": unit_row["unit_level"],
                "unit_index": unit_row["unit_index"],
                "context_mode": context_scope,
                "context_includes_both_speakers": include_both_speakers_in_context,
                "window_N": window_n,
                "statement_text": unit_row["text"],
                "context_text": context_text,
            })

    df_pairs = pd.DataFrame(pair_rows)

    # Lightweight mapping files
    pairs_to_turn_df = df_pairs[["ID", "Turn"]].reset_index().rename(columns={"index": "pair_index"})
    pairs_to_post_df = df_pairs[["ID"]].reset_index().rename(columns={"index": "pair_index"})

    return df_pairs, pairs_to_turn_df, pairs_to_post_df


# ──────────────────────────────────────────────────────────────────────────────
# Streamlit UI
# ──────────────────────────────────────────────────────────────────────────────

def _df_to_csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8")


def main() -> None:
    st.set_page_config(
        page_title="Conversation Pre‑processing",
        page_icon="⚙️",
        layout="wide",
    )

    st.title("⚙️ Conversational Context ↔ Statement Pair Generator")
    st.markdown(
        "Upload a **CSV** conversational dataset and tweak the parameters in the sidebar to generate (context, statement) pairs."
    )

    # Sidebar – parameters (the editable "default dictionary")
    st.sidebar.header("Parameters")

    id_col = st.sidebar.text_input("ID column", value=DEFAULT_ID_COL)
    turn_col = st.sidebar.text_input("Turn column (blank if none)", value=DEFAULT_TURN_COL)
    stmt_col = st.sidebar.text_input("Statement column", value=DEFAULT_STATEMENT_COL)
    speaker_col = st.sidebar.text_input("Speaker column (blank if none)", value=DEFAULT_SPEAKER_COL)

    st.sidebar.divider()

    statement_unit = st.sidebar.selectbox("Statement unit", STATEMENT_UNITS, index=0)
    context_scope = st.sidebar.selectbox("Context scope", CONTEXT_SCOPES, index=0)
    stmt_speaker = st.sidebar.selectbox("Speaker for statements", SPEAKER_CHOICES, index=0)
    include_both = st.sidebar.checkbox("Include both speakers in context", value=True)
    window_n = st.sidebar.number_input("Rolling window N (0 = unlimited)", value=0, step=1)
    assume_odd_even = st.sidebar.checkbox("Assume odd Turn = customer", value=True)

    st.sidebar.markdown("---")
    st.sidebar.markdown(
        "**Tip:** leave *Turn* or *Speaker* column blank if your CSV doesn't have them."
    )

    # Main area – file upload & run button
    uploaded_file = st.file_uploader("📄 Upload CSV file", type=["csv"])

    if "run_clicked" not in st.session_state:
        st.session_state.run_clicked = False

    if st.button("Run Pre‑processing", disabled=uploaded_file is None, type="primary"):
        st.session_state.run_clicked = True

    if st.session_state.get("run_clicked"):
        if uploaded_file is None:
            st.error("Please upload a CSV file first.")
            st.stop()

        try:
            df_in = pd.read_csv(uploaded_file)
        except Exception as e:
            st.error(f"Failed to read CSV – {e}")
            st.stop()

        try:
            df_pairs, p2turn, p2post = preprocess_pairs(
                df_in,
                id_col=id_col,
                turn_col=turn_col or None,
                statement_col=stmt_col,
                speaker_col=speaker_col or None,
                statement_unit=statement_unit,
                context_scope=context_scope,
                statement_speaker=stmt_speaker,
                include_both_speakers_in_context=include_both,
                window_n=int(window_n),
                assume_odd_even=assume_odd_even,
            )
        except Exception as e:
            st.error(f"⚠️ Pre‑processing failed: {e}")
            st.stop()

        st.success("✅ Pre‑processing complete!")

        # Preview
        st.subheader("Preview of pairs (first 20 rows)")
        st.dataframe(df_pairs.head(20), use_container_width=True, hide_index=True)

        # Downloads
        st.subheader("Download outputs")
        col1, col2, col3 = st.columns(3)
        csv_bytes_pairs = _df_to_csv_bytes(df_pairs)
        csv_bytes_turn = _df_to_csv_bytes(p2turn)
        csv_bytes_post = _df_to_csv_bytes(p2post)
        col1.download_button("Download pairs", csv_bytes_pairs, file_name=f"preprocessed_pairs_{_TIMESTAMP}.csv", mime="text/csv")
        col2.download_button("pairs ↔ turn map", csv_bytes_turn, file_name=f"pairs_to_turn_{_TIMESTAMP}.csv", mime="text/csv")
        col3.download_button("pairs ↔ post map", csv_bytes_post, file_name=f"pairs_to_post_{_TIMESTAMP}.csv", mime="text/csv")


if __name__ == "__main__":
    main()


