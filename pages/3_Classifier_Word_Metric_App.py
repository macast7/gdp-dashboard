import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
import io
import re
from typing import List

st.set_page_config(page_title="Classifier Word Metrics", page_icon="📊", layout="wide")

##############################################################################
# Helper utilities
##############################################################################

def count_words(text: str) -> int:
    """Return the number of whitespace‑separated tokens in *text*."""
    if not isinstance(text, str):
        return 0
    return len(text.strip().split())


def count_keywords(matched: str, split_regex: str) -> int:
    """Return count of keywords inside *matched* according to *split_regex*."""
    if not isinstance(matched, str):
        return 0
    cleaned = matched.strip()
    if cleaned == "" or cleaned.lower() == "none":
        return 0
    return len([k for k in re.split(split_regex, cleaned) if k])


def validate_dataframe(df: pd.DataFrame) -> None:
    """Raise *ValueError* if *df* does not contain required columns."""
    required = {"ID", "Statement", "Prediction", "MatchedKeyword(s)"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(missing)}")


def generate_statement_metrics(df: pd.DataFrame, split_regex: str) -> pd.DataFrame:
    """Return dataframe with per‑statement metrics added."""
    df = df.copy()
    df["KeywordCount"] = df["MatchedKeyword(s)"].apply(lambda x: count_keywords(x, split_regex))
    df["WordCount"] = df["Statement"].apply(count_words)
    df["KeywordDensity"] = df.apply(
        lambda r: r.KeywordCount / r.WordCount if r.WordCount else 0, axis=1
    )
    df["KeywordDensity"] = df["KeywordDensity"].round(4)
    df["Prediction"] = pd.to_numeric(df["Prediction"], errors="coerce").fillna(0).astype(int)
    return df


def aggregate_post_level(df_stmt: pd.DataFrame) -> pd.DataFrame:
    """Aggregate per‑post metrics from statement‑level dataframe."""
    grouped = df_stmt.groupby("ID")
    agg_df = grouped.apply(
        lambda g: pd.Series(
            {
                "SentencesWithTactic": int(g["Prediction"].sum()),
                "TotalSentences": int(len(g)),
                "PctSentencesTactic": round(g["Prediction"].mean(), 4),
                "TotalKeywords": int(g["KeywordCount"].sum()),
                "TotalWords": int(g["WordCount"].sum()),
                "PctWordsTactic": round(g["KeywordCount"].sum() / g["WordCount"].sum() if g["WordCount"].sum() else 0, 4),
            }
        )
    ).reset_index()
    return agg_df


def compute_summary(stmt_df: pd.DataFrame, agg_df: pd.DataFrame) -> dict:
    """Return overall summary statistics."""
    return {
        "total_posts": len(agg_df),
        "total_sentences": len(stmt_df),
        "avg_sentences_per_post": round(len(stmt_df) / len(agg_df), 2) if len(agg_df) else 0,
        "avg_keyword_density": round(stmt_df["KeywordDensity"].mean(), 4) if not stmt_df.empty else 0,
        "avg_pct_words_tactic": round(agg_df["PctWordsTactic"].mean(), 4) if not agg_df.empty else 0,
        "avg_pct_sentences_tactic": round(agg_df["PctSentencesTactic"].mean(), 4) if not agg_df.empty else 0,
    }


def to_csv_download(df: pd.DataFrame) -> bytes:
    """Return CSV bytes suitable for st.download_button."""
    return df.to_csv(index=False).encode("utf-8")

##############################################################################
# UI
##############################################################################

st.title("📊 Classifier Word Metrics")
st.write(
    "Convert binary classifier predictions into continuous scores at the sentence and post levels."
)

with st.expander("ℹ️ Required CSV Columns"):
    st.markdown(
        """
        * `ID` – Post identifier  
        * `Statement` – Sentence text  
        * `Prediction` – Binary classifier result (0/1)  
        * `MatchedKeyword(s)` – Keywords found in sentence
        """
    )

uploaded_file = st.file_uploader("Upload **statement_level_predictions.csv**", type="csv")

# Allow user to customise keyword splitting
split_regex_default = "[,;|]"
split_regex = st.text_input(
    "Keyword delimiter regex (advanced)",
    value=split_regex_default,
    help="Advanced: customise how 'MatchedKeyword(s)' is split into individual keywords.",
)

if uploaded_file is not None:
    with st.spinner("Processing your data …"):
        try:
            df_raw = pd.read_csv(uploaded_file)
            validate_dataframe(df_raw)
            df_stmt = generate_statement_metrics(df_raw, split_regex)
            df_agg = aggregate_post_level(df_stmt)
            summary = compute_summary(df_stmt, df_agg)
        except Exception as ex:
            st.error(f"❌ {ex}")
            st.stop()

    # ───────────────────────────────── Summary metrics ──────────────────────────────────
    st.success("✅ Processing complete!")
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("Total posts", summary["total_posts"])
    c2.metric("Total sentences", summary["total_sentences"])
    c3.metric("Avg sentences/post", summary["avg_sentences_per_post"])
    c4.metric("Avg keyword density", summary["avg_keyword_density"])
    c5.metric("Avg % words tactic", summary["avg_pct_words_tactic"])
    c6.metric("Avg % sentences tactic", summary["avg_pct_sentences_tactic"])

    # ───────────────────────────────── Downloads ───────────────────────────────────────
    st.markdown("### 📥 Download processed data")
    btn1, btn2 = st.columns(2)
    with btn1:
        st.download_button(
            "Download statement‑level CSV",
            data=to_csv_download(df_stmt),
            file_name="statement_level_word_metrics.csv",
            mime="text/csv",
        )
    with btn2:
        st.download_button(
            "Download ID‑level CSV",
            data=to_csv_download(df_agg),
            file_name="id_level_aggregated_metrics.csv",
            mime="text/csv",
        )

    # ───────────────────────────────── Visualisations ───────────────────────────────────
    st.markdown("## 📊 Visualisations")

    # Histogram of % Words Tactic
    st.markdown("#### Distribution of % Words Tactic")
    hist_chart = (
        alt.Chart(df_agg)
        .transform_bin("bin", field="PctWordsTactic", bin=alt.Bin(maxbins=20))
        .mark_bar(color="#3B82F6")
        .encode(x="bin:O", y="count()", tooltip=["count()"])
        .properties(height=400)
    )
    st.altair_chart(hist_chart, use_container_width=True)

    # Scatter plot: % Words vs % Sentences Tactic
    st.markdown("#### % Words vs % Sentences Tactic by Post")
    scatter = (
        alt.Chart(df_agg)
        .mark_circle(size=60, color="#10B981")
        .encode(
            x=alt.X("PctWordsTactic", title="% Words Tactic"),
            y=alt.Y("PctSentencesTactic", title="% Sentences Tactic"),
            tooltip=["ID", "PctWordsTactic", "PctSentencesTactic"],
        )
        .properties(height=400)
    )
    st.altair_chart(scatter, use_container_width=True)

    # ───────────────────────────────── Data preview ─────────────────────────────────────
    with st.expander("🔍 Preview processed data"):
        st.dataframe(df_stmt.head(50))

else:
    st.info("⬆️ Upload a CSV file to begin.")
