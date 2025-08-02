import streamlit as st
import pandas as pd
import nltk
import re
import io
from typing import List, Optional, Tuple


@st.cache_resource
def download_nltk_data():
    """Download NLTK punkt tokenizer data."""
    try:
        nltk.download('punkt', quiet=True)
        return True
    except:
        return False


def guess_id_columns(df: pd.DataFrame) -> List[str]:
    """Guess likely ID columns based on column names."""
    id_keywords = ['id', 'uid', 'post', 'user', 'record', 'index', 'key']
    candidates = []
    
    for col in df.columns:
        col_lower = col.lower()
        if any(keyword in col_lower for keyword in id_keywords):
            candidates.append(col)
    
    return candidates


def guess_text_columns(df: pd.DataFrame) -> List[str]:
    """Guess likely text columns based on column names."""
    text_keywords = ['text', 'caption', 'message', 'content', 'description', 'comment', 'body']
    candidates = []
    
    for col in df.columns:
        col_lower = col.lower()
        if any(keyword in col_lower for keyword in text_keywords):
            candidates.append(col)
    
    return candidates


def clean_text(text: str) -> str:
    """Clean text by removing URLs, hashtags, mentions, emojis, and odd symbols."""
    if pd.isna(text):
        return ""
    
    text = str(text)
    
    # Remove URLs
    text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
    
    # Remove hashtags
    text = re.sub(r'#\w+', '', text)
    
    # Remove mentions
    text = re.sub(r'@\w+', '', text)
    
    # Remove emojis and special unicode characters, keep basic punctuation
    text = re.sub(r'[^\w\s.,!?;:\'"()-]', '', text)
    
    # Clean up extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text


def extract_hashtags(text: str) -> str:
    """Extract hashtags from text and return as space-separated string."""
    if pd.isna(text):
        return ""
    
    text = str(text)
    hashtags = re.findall(r'#\w+', text)
    return ' '.join(hashtags)


def is_punctuation_only(text: str) -> bool:
    """Check if text contains only punctuation."""
    return bool(re.match(r'^[^\w\s]*$', text.strip()))


def split_into_sentences(text: str, use_nltk: bool = True) -> List[str]:
    """Split text into sentences using NLTK or regex fallback."""
    if pd.isna(text) or not text.strip():
        return []
    
    text = str(text).strip()
    
    # Ensure text ends with punctuation
    if not re.search(r'[.!?]$', text):
        text += '.'
    
    sentences = []
    
    if use_nltk:
        try:
            sentences = nltk.sent_tokenize(text)
        except:
            # Fallback to regex
            sentences = re.split(r'[.!?]+', text)
    else:
        sentences = re.split(r'[.!?]+', text)
    
    # Clean and filter sentences
    clean_sentences = []
    for sentence in sentences:
        sentence = sentence.strip()
        if sentence and not is_punctuation_only(sentence):
            clean_sentences.append(sentence)
    
    return clean_sentences


def transform_data(df: pd.DataFrame, id_col: str, text_col: str, include_hashtags: bool = True) -> pd.DataFrame:
    """Transform data into sentence-level format."""
    results = []
    
    for idx, row in df.iterrows():
        record_id = row[id_col]
        text = row[text_col]
        
        if pd.isna(text):
            continue
        
        # Clean text for sentence splitting
        cleaned_text = clean_text(text)
        
        # Split into sentences
        sentences = split_into_sentences(cleaned_text, use_nltk=True)
        
        # Add hashtags as separate sentence if requested
        if include_hashtags:
            hashtags = extract_hashtags(str(text))
            if hashtags:
                sentences.append(hashtags)
        
        # Create output rows
        for sent_idx, sentence in enumerate(sentences, 1):
            results.append({
                'ID': record_id,
                'Sentence_ID': sent_idx,
                'Context': str(text)[:200] + '...' if len(str(text)) > 200 else str(text),
                'Statement': sentence
            })
    
    return pd.DataFrame(results)


def create_column_info(df: pd.DataFrame) -> pd.DataFrame:
    """Create column information table."""
    info_data = []
    
    for col in df.columns:
        sample_val = df[col].dropna().iloc[0] if not df[col].dropna().empty else "N/A"
        info_data.append({
            'Column': col,
            'dtype': str(df[col].dtype),
            'non_null_count': df[col].notna().sum(),
            'sample_value': str(sample_val)[:50] + '...' if len(str(sample_val)) > 50 else str(sample_val)
        })
    
    return pd.DataFrame(info_data)


def main():
    """Main Streamlit app."""
    st.set_page_config(page_title="Text Transform App", layout="wide")
    
    st.title("📝 Text Transform App")
    st.markdown("Upload a CSV and transform text data into sentence-level format")
    
    # Initialize NLTK
    nltk_available = download_nltk_data()
    if not nltk_available:
        st.warning("NLTK punkt tokenizer unavailable. Using regex fallback.")
    
    # File upload
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    
    if uploaded_file is not None:
        try:
            # Load data
            df = pd.read_csv(uploaded_file)
            
            st.success(f"✅ Loaded {len(df)} rows and {len(df.columns)} columns")
            
            # Data preview
            st.subheader("📊 Data Preview")
            preview_rows = st.number_input("Number of rows to preview", min_value=1, max_value=1000, value=500)
            st.dataframe(df.head(preview_rows), use_container_width=True)
            
            # Column information
            st.subheader("📋 Column Information")
            col_info = create_column_info(df)
            st.dataframe(col_info, use_container_width=True)
            
            # Sidebar controls
            st.sidebar.header("🔧 Configuration")
            
            # Guess columns
            id_candidates = guess_id_columns(df)
            text_candidates = guess_text_columns(df)
            
            # Column selection
            id_col = st.sidebar.selectbox(
                "Select ID Column",
                options=df.columns.tolist(),
                index=df.columns.tolist().index(id_candidates[0]) if id_candidates else 0
            )
            
            text_col = st.sidebar.selectbox(
                "Select Text Column",
                options=df.columns.tolist(),
                index=df.columns.tolist().index(text_candidates[0]) if text_candidates else 0
            )
            
            include_hashtags = st.sidebar.checkbox("Include hashtags as separate sentences", value=True)
            
            # Transform button
            if st.sidebar.button("🚀 Transform Data", type="primary"):
                with st.spinner("Transforming data..."):
                    transformed_df = transform_data(df, id_col, text_col, include_hashtags)
                
                if not transformed_df.empty:
                    st.subheader("✨ Transformed Data")
                    
                    # Display first 20 rows
                    st.dataframe(transformed_df.head(20), use_container_width=True)
                    
                    # KPI metrics
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric("Original Rows", len(df))
                    
                    with col2:
                        st.metric("Generated Sentences", len(transformed_df))
                    
                    with col3:
                        avg_sentences = len(transformed_df) / len(df) if len(df) > 0 else 0
                        st.metric("Avg Sentences/Record", f"{avg_sentences:.2f}")
                    
                    # Download button
                    csv_buffer = io.StringIO()
                    transformed_df.to_csv(csv_buffer, index=False)
                    csv_data = csv_buffer.getvalue()
                    
                    st.download_button(
                        label="📥 Download Transformed CSV",
                        data=csv_data,
                        file_name="transformed_data.csv",
                        mime="text/csv"
                    )
                else:
                    st.warning("No sentences generated. Please check your data and column selections.")
        
        except Exception as e:
            st.error(f"Error processing file: {str(e)}")
    
    else:
        st.info("👆 Please upload a CSV file to get started")


if __name__ == "__main__":
    main()

