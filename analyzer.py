
from transformers import pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd

_sentiment_pipe = None
_emotion_pipe = None
_summarizer = None

def get_sentiment_pipe():
    global _sentiment_pipe
    if _sentiment_pipe is None:
        _sentiment_pipe = pipeline("text-classification", model="distilbert-base-uncased-finetuned-sst-2-english")
    return _sentiment_pipe

def get_emotion_pipe():
    global _emotion_pipe
    if _emotion_pipe is None:
        _emotion_pipe = pipeline("text-classification", model="j-hartmann/emotion-english-distilroberta-base", return_all_scores=False)
    return _emotion_pipe

def get_summarizer():
    global _summarizer
    if _summarizer is None:
        _summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
    return _summarizer

def analyze_sentiment(text):
    pipe = get_sentiment_pipe()
    res = pipe(text[:512])  
    return res[0]['label'], float(res[0]['score'])

def analyze_emotion(text):
    pipe = get_emotion_pipe()
    res = pipe(text[:512]) 
    return res[0]['label'], float(res[0]['score'])

def summarize_text(text, max_length=80, min_length=20):
    summarizer = get_summarizer()
    
    if len(text.split()) < 30:
        return text  
    out = summarizer(text, max_length=max_length, min_length=min_length, do_sample=False)
    return out[0]['summary_text']

def extract_keywords(text, n=5):
    sentences = [s.strip() for s in text.replace('\n',' ').split('.') if s.strip()]
    if len(sentences) == 0:
        return []
    vectorizer = TfidfVectorizer(stop_words='english', ngram_range=(1,2), max_features=1000)
    try:
        X = vectorizer.fit_transform(sentences)
    except ValueError:
        return []
    import numpy as np
    avg = X.mean(axis=0).A1
    terms = vectorizer.get_feature_names_out()
    top_idx = avg.argsort()[::-1][:n]
    keywords = [terms[i] for i in top_idx]
    return keywords

def analyze_text_record(text):
    sentiment_label, sentiment_score = analyze_sentiment(text)
    emotion_label, emotion_score = analyze_emotion(text)
    summary = summarize_text(text)
    keywords = extract_keywords(text, n=5)
    return {
        "text": text,
        "summary": summary,
        "keywords": ", ".join(keywords),
        "sentiment": sentiment_label,
        "sentiment_score": sentiment_score,
        "emotion": emotion_label,
        "emotion_score": emotion_score
    }

def analyze_file(path):
    import os
    ext = os.path.splitext(path)[1].lower()
    records = []
    if ext == '.txt':
        with open(path, 'r', encoding='utf-8') as f:
            content = f.read()
        parts = [p for p in content.splitlines() if p.strip()]
        if not parts:
            parts = [content]
        for part in parts:
            records.append(analyze_text_record(part))
    elif ext == '.csv':
        df = pd.read_csv(path)
        text_col = None
        for c in ['text', 'Text', 'message', 'Message', df.columns[0]]:
            if c in df.columns:
                text_col = c
                break
        if text_col is None:
            text_col = df.columns[0]
        for v in df[text_col].astype(str):
            records.append(analyze_text_record(v))
    else:
        raise ValueError("Unsupported file type. Use .txt or .csv")
    return pd.DataFrame(records)

def save_results_df(df, out_csv="analysis_results.csv"):
    df.to_csv(out_csv, index=False, encoding='utf-8')
    return out_csv
