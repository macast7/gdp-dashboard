import streamlit as st
import pandas as pd
import json
import re
from typing import List, Dict, Any
import anthropic
from io import StringIO

# Try to import st_tags, fall back to multiselect if not available
try:
    from st_tags import st_tags
    TAGS_AVAILABLE = True
except ImportError:
    TAGS_AVAILABLE = False
    st.warning("st_tags not available. Using multiselect as fallback.")

def setup_page():
    """Configure the Streamlit page"""
    st.set_page_config(
        page_title="Dictionary Refinement",
        page_icon="📚",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    st.title("📚 Dictionary Refinement")
    st.markdown("Craft high-quality keyword lists with AI assistance")

def get_anthropic_client():
    """Initialize Anthropic client"""
    # Check if API key is in secrets or environment
    api_key = None
    
    # Try to get from Streamlit secrets first
    try:
        api_key = st.secrets["ANTHROPIC_API_KEY"]
    except:
        # If not in secrets, ask user to input it
        if "anthropic_api_key" not in st.session_state:
            st.session_state.anthropic_api_key = ""
        
        with st.sidebar:
            api_key = st.text_input(
                "Anthropic API Key",
                type="password",
                value=st.session_state.anthropic_api_key,
                help="Enter your Anthropic API key to generate keyword suggestions"
            )
            st.session_state.anthropic_api_key = api_key
    
    if not api_key:
        return None
    
    return anthropic.Anthropic(api_key=api_key)

def generate_keywords(tactic_definition: str, client: anthropic.Anthropic) -> List[str]:
    """Generate keywords using Anthropic Claude API"""
    prompt = f"""Generate 30 one-word (unigram) keywords that signal the following tactic: {tactic_definition}.

Return as a JSON array of strings. Focus on words that would clearly indicate this tactic when found in text.

Example format: ["word1", "word2", "word3", ...]"""

    try:
        message = client.messages.create(
            model="claude-3-sonnet-20240229",
            max_tokens=1000,
            temperature=0.7,
            messages=[
                {"role": "user", "content": prompt}
            ]
        )
        
        # Extract JSON from response
        response_text = message.content[0].text
        
        # Try to find JSON array in the response
        json_match = re.search(r'\[.*\]', response_text, re.DOTALL)
        if json_match:
            keywords_json = json_match.group()
            keywords = json.loads(keywords_json)
            return [str(word).strip().lower() for word in keywords if word.strip()]
        else:
            st.error("Could not parse keywords from API response")
            return []
    
    except Exception as e:
        st.error(f"Error generating keywords: {str(e)}")
        return []

def display_keywords_editor(keywords: List[str]) -> List[str]:
    """Display editable keyword interface"""
    if TAGS_AVAILABLE:
        # Use st_tags for better UX
        edited_keywords = st_tags(
            label="Edit Keywords",
            text="Press enter to add a keyword",
            value=keywords,
            suggestions=[],
            maxtags=-1,
            key="keyword_tags"
        )
    else:
        # Fallback to multiselect with text input
        st.subheader("Edit Keywords")
        
        # Display current keywords as chips
        if keywords:
            cols = st.columns(min(len(keywords), 5))
            for i, keyword in enumerate(keywords):
                with cols[i % 5]:
                    if st.button(f"❌ {keyword}", key=f"remove_{keyword}_{i}"):
                        keywords.remove(keyword)
                        st.rerun()
        
        # Input for new keywords
        new_keyword = st.text_input("Add new keyword:", key="new_keyword_input")
        if st.button("Add Keyword") and new_keyword.strip():
            if new_keyword.strip().lower() not in [k.lower() for k in keywords]:
                keywords.append(new_keyword.strip().lower())
                st.rerun()
        
        edited_keywords = keywords
    
    return edited_keywords

def preview_sample_hits(keywords: List[str], df: pd.DataFrame) -> None:
    """Preview how keywords match against sample CSV"""
    if df is None or df.empty:
        st.warning("No CSV file uploaded for preview")
        return
    
    if 'Statement' not in df.columns:
        st.error("CSV must contain a 'Statement' column")
        return
    
    # Create regex pattern for keyword matching
    keyword_pattern = r'\b(' + '|'.join(re.escape(k) for k in keywords) + r')\b'
    
    # Find matches
    matches = []
    for idx, row in df.iterrows():
        statement = str(row.get('Statement', ''))
        found_keywords = re.findall(keyword_pattern, statement, re.IGNORECASE)
        
        if found_keywords:
            matches.append({
                'ID': row.get('ID', idx),
                'Statement': statement,
                'Matched_Keywords': list(set([k.lower() for k in found_keywords])),
                'Match_Count': len(set([k.lower() for k in found_keywords]))
            })
    
    if matches:
        st.success(f"Found {len(matches)} statements with keyword matches")
        
        # Show first 10 matches
        preview_df = pd.DataFrame(matches[:10])
        st.dataframe(
            preview_df,
            use_container_width=True,
            column_config={
                "Statement": st.column_config.TextColumn(width="large"),
                "Matched_Keywords": st.column_config.ListColumn(),
                "Match_Count": st.column_config.NumberColumn()
            }
        )
        
        if len(matches) > 10:
            st.info(f"Showing first 10 of {len(matches)} matches")
    else:
        st.warning("No matches found in the sample data")

def download_dictionary(keywords: List[str]) -> None:
    """Create download button for dictionary"""
    if keywords:
        dictionary_data = {
            "keywords": keywords,
            "count": len(keywords),
            "created_at": pd.Timestamp.now().isoformat()
        }
        
        json_str = json.dumps(dictionary_data, indent=2)
        
        st.download_button(
            label="📥 Download Dictionary",
            data=json_str,
            file_name="dictionary.json",
            mime="application/json",
            help="Download your refined keyword dictionary as JSON"
        )
    else:
        st.warning("No keywords to download")

def main():
    """Main application logic"""
    setup_page()
    
    # Initialize session state
    if "keywords" not in st.session_state:
        st.session_state.keywords = []
    if "uploaded_df" not in st.session_state:
        st.session_state.uploaded_df = None
    
    # Sidebar
    with st.sidebar:
        st.header("📋 Configuration")
        
        # Tactic definition input
        tactic_definition = st.text_input(
            "Tactic Definition",
            placeholder="Enter the tactic you want to create keywords for...",
            help="Describe the tactic or concept you want to identify in text"
        )
        
        # File uploader
        uploaded_file = st.file_uploader(
            "Upload Sample CSV",
            type=['csv'],
            help="Upload a CSV file with 'ID' and 'Statement' columns"
        )
        
        # Process uploaded file
        if uploaded_file is not None:
            try:
                df = pd.read_csv(uploaded_file)
                st.session_state.uploaded_df = df
                
                # Validate columns
                required_cols = ['ID', 'Statement']
                missing_cols = [col for col in required_cols if col not in df.columns]
                
                if missing_cols:
                    st.error(f"Missing columns: {missing_cols}")
                    st.info("Required columns: ID, Statement")
                else:
                    st.success(f"✅ CSV loaded: {len(df)} rows")
                    with st.expander("Preview Data"):
                        st.dataframe(df.head(3))
                        
            except Exception as e:
                st.error(f"Error loading CSV: {str(e)}")
        
        # Generate button
        generate_button = st.button(
            "🎯 Generate Keyword Suggestions",
            type="primary",
            disabled=not tactic_definition.strip()
        )
    
    # Main panel
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Generate keywords
        if generate_button:
            if not tactic_definition.strip():
                st.error("Please enter a tactic definition")
                return
            
            client = get_anthropic_client()
            if not client:
                st.error("Please provide your Anthropic API key in the sidebar")
                return
            
            with st.spinner("Generating keyword suggestions..."):
                new_keywords = generate_keywords(tactic_definition, client)
                if new_keywords:
                    st.session_state.keywords = new_keywords
                    st.success(f"Generated {len(new_keywords)} keywords!")
        
        # Display current tactic if keywords exist
        if st.session_state.keywords and tactic_definition:
            st.info(f"**Current Tactic:** {tactic_definition}")
        
        # Keywords editor
        if st.session_state.keywords:
            st.subheader("🏷️ Keyword Editor")
            edited_keywords = display_keywords_editor(st.session_state.keywords)
            st.session_state.keywords = edited_keywords
            
            st.info(f"Current dictionary contains {len(edited_keywords)} keywords")
        else:
            st.info("👆 Enter a tactic definition and click 'Generate Keyword Suggestions' to get started")
    
    with col2:
        st.subheader("📊 Actions")
        
        # Download dictionary
        if st.session_state.keywords:
            download_dictionary(st.session_state.keywords)
        
        # Preview sample hits
        if st.button("🔍 Preview Sample Hits", disabled=not st.session_state.keywords):
            if st.session_state.keywords:
                with st.spinner("Analyzing sample data..."):
                    preview_sample_hits(st.session_state.keywords, st.session_state.uploaded_df)
        
        # Statistics
        if st.session_state.keywords:
            st.subheader("📈 Statistics")
            st.metric("Keywords", len(st.session_state.keywords))
            
            if st.session_state.uploaded_df is not None:
                st.metric("Sample Rows", len(st.session_state.uploaded_df))
    
    # Preview section (if requested)
    if st.session_state.get('show_preview', False):
        st.subheader("🎯 Sample Matches")
        preview_sample_hits(st.session_state.keywords, st.session_state.uploaded_df)

if __name__ == "__main__":
    main()
