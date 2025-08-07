import streamlit as st
import pandas as pd
import numpy as np
import json
import altair as alt
from io import StringIO

st.set_page_config(page_title="Join Table – Data Merger", layout="wide")

# -----------------------
# Sidebar – Upload files
# -----------------------
st.sidebar.header("📂 Upload Your CSV Files")
file_metrics = st.sidebar.file_uploader(
    "Upload Metrics File (e.g. id_level_aggregated_metrics.csv)",
    type=["csv"],
    key="metrics",
)
file_instagram = st.sidebar.file_uploader(
    "Upload Instagram File (e.g. ig_posts_shi_new.csv)",
    type=["csv"],
    key="instagram",
)

# -----------------------
# Helper functions
# -----------------------

def read_csv(file) -> pd.DataFrame:
    """Read CSV to DataFrame with stripped header white‑space and trimmed string columns."""
    df = pd.read_csv(file)
    df.columns = df.columns.str.strip()
    # Optionally trim whitespace on string cols
    str_cols = df.select_dtypes(include=["object"]).columns
    df[str_cols] = df[str_cols].apply(lambda s: s.str.strip())
    return df

@st.cache_data(show_spinner=False)
def join_frames(df1: pd.DataFrame, key1: str, df2: pd.DataFrame, key2: str) -> pd.DataFrame:
    """Perform an inner join between df1 and df2 on the specified keys."""
    joined = pd.merge(df1, df2, how="inner", left_on=key1, right_on=key2)
    return joined

@st.cache_data(show_spinner=False)
def corr_matrix(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    if len(cols) < 2:
        return pd.DataFrame()
    # Ensure numeric
    numeric_df = df[cols].apply(pd.to_numeric, errors="coerce")
    return numeric_df.corr()

# -----------------------
# Main layout
# -----------------------
col1, col2 = st.columns(2)

with col1:
    if file_metrics:
        df_metrics = read_csv(file_metrics)
        st.success(f"Metrics file loaded ✅ – {df_metrics.shape[0]:,} rows, {df_metrics.shape[1]} cols")
        key1 = st.selectbox("🔑 Key column in Metrics file", options=df_metrics.columns, index=df_metrics.columns.get_loc("ID")
                             if "ID" in df_metrics.columns else 0)
        st.dataframe(df_metrics.head())
    else:
        df_metrics = pd.DataFrame()
        key1 = None

with col2:
    if file_instagram:
        df_instagram = read_csv(file_instagram)
        st.success(f"Instagram file loaded ✅ – {df_instagram.shape[0]:,} rows, {df_instagram.shape[1]} cols")
        key2 = st.selectbox("🔑 Key column in Instagram file", options=df_instagram.columns,
                             index=df_instagram.columns.get_loc("shortcode") if "shortcode" in df_instagram.columns else 0)
        st.dataframe(df_instagram.head())
    else:
        df_instagram = pd.DataFrame()
        key2 = None

# -----------------------
# Join operation
# -----------------------

if not df_metrics.empty and not df_instagram.empty:
    st.divider()
    if st.button("🚀 Join Tables & Analyze", type="primary"):
        with st.spinner("Joining tables…"):
            joined_df = join_frames(df_metrics, key1, df_instagram, key2)
        if joined_df.empty:
            st.error("No matching records found. Ensure key columns contain overlapping values.")
        else:
            st.success(f"Joined successfully! {joined_df.shape[0]:,} rows ✨")

            # --------------
            # Correlation UI
            # --------------
            st.subheader("📊 Correlation Analysis")

            # Default dictionary user can edit
            default_dict = {"correlation_cols": ["likes", "comments", "PctWordsTactic", "PctSentencesTactic"]}
            default_json = json.dumps(default_dict, indent=2)
            user_json = st.text_area("Modify the correlation dictionary (JSON)", value=default_json, height=150)
            try:
                user_dict = json.loads(user_json)
                corr_cols = [c for c in user_dict.get("correlation_cols", []) if c in joined_df.columns]
            except json.JSONDecodeError:
                st.warning("Invalid JSON – using default columns")
                corr_cols = [c for c in default_dict["correlation_cols"] if c in joined_df.columns]

            corr_df = corr_matrix(joined_df, corr_cols)
            if not corr_df.empty:
                st.write("### Heatmap")
                # Melt for Altair heatmap
                corr_long = corr_df.reset_index().melt(id_vars="index")
                heat = (
                    alt.Chart(corr_long)
                    .mark_rect()
                    .encode(
                        x="variable:N",
                        y="index:N",
                        color=alt.Color("value:Q", scale=alt.Scale(scheme="redblue", domain=[-1, 1])),
                        tooltip=["index", "variable", alt.Tooltip("value:Q", format=".3f")],
                    )
                    .properties(width=300, height=300)
                )
                st.altair_chart(heat, use_container_width=True)
                st.dataframe(corr_df)
            else:
                st.info("Need at least two numeric columns for correlation.")

            # --------------
            # Scatter plot
            # --------------
            st.subheader("📈 Scatterplot")
            num_cols = joined_df.select_dtypes(include=[np.number]).columns
            if len(num_cols) >= 2:
                x_axis = st.selectbox("Choose X axis", num_cols, index=num_cols.get_loc("PctWordsTactic") if "PctWordsTactic" in num_cols else 0)
                y_axis = st.selectbox("Choose Y axis", num_cols, index=num_cols.get_loc("likes") if "likes" in num_cols else 1)
                scatter = (
                    alt.Chart(joined_df)
                    .mark_circle(opacity=0.7)
                    .encode(
                        x=alt.X(f"{x_axis}:Q", title=x_axis),
                        y=alt.Y(f"{y_axis}:Q", title=y_axis),
                        tooltip=list(joined_df.columns[:6]),
                    )
                    .interactive()
                    .properties(height=350)
                )
                st.altair_chart(scatter, use_container_width=True)
            else:
                st.info("Not enough numeric columns for scatter plot.")

            # --------------
            # Download
            # --------------
            st.subheader("⬇️ Download Joined Data")
            csv_buffer = StringIO()
            joined_df.to_csv(csv_buffer, index=False)
            st.download_button(
                "Download IG_joined_result.csv",
                data=csv_buffer.getvalue().encode("utf-8"),
                file_name="IG_joined_result.csv",
                mime="text/csv",
            )

else:
    st.info("👈 Upload both CSV files to get started.")
