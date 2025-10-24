import streamlit as st
import pandas as pd
from analyzer import analyze_text_record, analyze_file, save_results_df
import matplotlib.pyplot as plt

st.set_page_config(page_title="AI Text Analyzer", layout="centered")

st.title("🧠 AI Text Analyzer")

mode = st.sidebar.radio("Mode", ["Single Text", "Batch File", "View Results"])

if mode == "Single Text":
    text = st.text_area("Enter text to analyze", height=150)
    if st.button("Analyze"):
        if not text.strip():
            st.warning("Please enter text.")
        else:
            rec = analyze_text_record(text)
            st.subheader("Summary")
            st.write(rec['summary'])
            st.subheader("Keywords")
            st.write(rec['keywords'])
            st.subheader("Sentiment & Emotion")
            st.write(f"**Sentiment:** {rec['sentiment']} ({rec['sentiment_score']:.2f})")
            st.write(f"**Emotion:** {rec['emotion']} ({rec['emotion_score']:.2f})")

elif mode == "Batch File":
    uploaded = st.file_uploader("Upload .txt or .csv", type=["txt","csv"])
    if uploaded is not None:
        with open("temp_input_file", "wb") as f:
            f.write(uploaded.getbuffer())
        try:
            df = analyze_file("temp_input_file")
            st.success("Analysis complete")
            st.dataframe(df)
            csv_path = save_results_df(df, out_csv="streamlit_results.csv")
            st.download_button("Download CSV", data=open(csv_path,"rb"), file_name=csv_path)
        except Exception as e:
            st.error(str(e))

elif mode == "View Results":
    try:
        df = pd.read_csv("analysis_results.csv")
        st.dataframe(df)
        counts = df['sentiment'].value_counts()
        fig, ax = plt.subplots()
        ax.pie(counts.values, labels=counts.index, autopct='%1.1f%%')
        ax.set_title("Sentiment distribution")
        st.pyplot(fig)
    except FileNotFoundError:
        st.info("No results CSV (analysis_results.csv) found. Run batch analysis first.")
