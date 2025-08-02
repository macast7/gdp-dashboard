import streamlit as st
import pandas as pd
import re
from datetime import datetime
import nltk
nltk.download('punkt', quiet=True)
from nltk.tokenize import sent_tokenize

def normalize_text(text):
    if not isinstance(text, str): return ''
    return re.sub(r'[\s\u2018\u2019\u201c\u201d]+', ' ', text.strip().replace("\u2019", "'").replace("\u2018", "'").replace('“','\"').replace('”','\"'))

def resolve_speaker(df, speaker_col, turn_col, assume_mode):
    if speaker_col in df.columns and speaker_col:
        df['Speaker'] = df[speaker_col].fillna('unknown')
    else:
        if assume_mode == 'odd_customer':
            df['Speaker'] = df[turn_col].apply(lambda x: 'customer' if int(x) % 2 == 1 else 'salesperson')
        else:
            df['Speaker'] = 'unknown'
    return df

def preprocess(df, id_col, turn_col, statement_col, speaker_col, statement_unit, context_scope,
               selected_speaker, include_both_speakers, window_n, language):

    df = df.copy()
    df = resolve_speaker(df, speaker_col, turn_col, 'odd_customer')
    df[statement_col] = df[statement_col].map(normalize_text)
    df = df.sort_values(by=[id_col, turn_col]).reset_index(drop=True)

    units = []
    for id_val, group in df.groupby(id_col):
        context_rows = group.copy()
        if statement_unit == 'post':
            all_text = ' '.join(context_rows[context_rows['Speaker'] == selected_speaker][statement_col].tolist())
            units.append({"ID": id_val, "Turn": '', "Speaker": selected_speaker, "unit_level": 'post',
                          "unit_index": 1, "statement_text": all_text, "context_rows": context_rows})
        else:
            idx = 0
            for i, row in group.iterrows():
                if row['Speaker'] != selected_speaker: continue
                text = row[statement_col]
                if statement_unit == 'sentence':
                    sents = [normalize_text(s) for s in sent_tokenize(text, language=language)]
                    for s_idx, sent in enumerate(sents):
                        units.append({"ID": row[id_col], "Turn": row[turn_col], "Speaker": row['Speaker'],
                                      "unit_level": 'sentence', "unit_index": s_idx+1, "statement_text": sent,
                                      "context_rows": context_rows})
                else:
                    idx += 1
                    units.append({"ID": row[id_col], "Turn": row[turn_col], "Speaker": row['Speaker'],
                                  "unit_level": 'turn', "unit_index": idx, "statement_text": text,
                                  "context_rows": context_rows})

    rows = []
    for u in units:
        id_val = u['ID']
        unit_level = u['unit_level']
        index = u['unit_index']
        statement_text = u['statement_text']
        rows_same_id = [r for r in units if r['ID'] == id_val and r['unit_level'] == unit_level]

        def get_context():
            context_parts = []
            for r in rows_same_id:
                if r['unit_index'] == index: continue
                if context_scope == 'rolling' and r['unit_index'] >= index: continue
                if not include_both_speakers and r['Speaker'] != u['Speaker']: continue
                context_parts.append(r['statement_text'])
            if context_scope == 'rolling' and window_n > 0:
                context_parts = context_parts[-window_n:]
            return ' '.join(context_parts).strip()

        rows.append({
            'ID': u['ID'],
            'Turn': u['Turn'],
            'Speaker': u['Speaker'],
            'unit_level': unit_level,
            'unit_index': index,
            'context_mode': context_scope,
            'context_includes_both_speakers': include_both_speakers,
            'window_n': window_n,
            'statement_text': statement_text,
            'context_text': get_context()
        })

    df_pairs = pd.DataFrame(rows)
    return df_pairs

def main():
    st.title("Preprocessing App: One-to-One & One-to-Many Communication")

    uploaded_file = st.file_uploader("Upload your CSV file", type="csv")

    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.write("Sample Data:", df.head())

        with st.form("config_form"):
            id_col = st.text_input("ID Column", "ID")
            turn_col = st.text_input("Turn Column", "Turn")
            statement_col = st.text_input("Statement Column", "Statement")
            speaker_col = st.text_input("Speaker Column (optional)", "Speaker")
            statement_unit = st.radio("Statement Unit", ["sentence", "turn", "post"], index=1)
            context_scope = st.radio("Context Scope", ["rolling", "whole"], index=0)
            selected_speaker = st.radio("Speaker (for statements)", ["customer", "salesperson"], index=0)
            include_both = st.checkbox("Include both speakers in context", value=True)
            window_n = st.number_input("Rolling window size N (0 = unlimited)", min_value=0, value=0)
            language = st.text_input("Language for sentence split", "english")
            submit = st.form_submit_button("Run Preprocessing")

        if submit:
            result_df = preprocess(df, id_col, turn_col, statement_col, speaker_col,
                                   statement_unit, context_scope, selected_speaker,
                                   include_both, window_n, language)
            st.write("Processed Data Preview:", result_df.head())
            csv = result_df.to_csv(index=False).encode('utf-8')
            st.download_button("Download Processed CSV", csv, "processed_output.csv", "text/csv")

if __name__ == '__main__':
    main()



