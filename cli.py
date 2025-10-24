import argparse
from analyzer import analyze_text_record, analyze_file, save_results_df
import pandas as pd
import json

def main():
    parser = argparse.ArgumentParser(description="AI Text Analyzer CLI")
    parser.add_argument("--text", "-t", help="Text to analyze (put in quotes)")
    parser.add_argument("--file", "-f", help="Path to .txt or .csv for batch analysis")
    parser.add_argument("--out", "-o", help="Output CSV file", default="analysis_results.csv")
    args = parser.parse_args()

    if args.text:
        record = analyze_text_record(args.text)
        print("\nAI Analysis Result:")
        print(f"Text: {record['text']}")
        print(f"Summary: {record['summary']}")
        print(f"Keywords: {record['keywords']}")
        print(f"Sentiment: {record['sentiment']} ({record['sentiment_score']:.2f})")
        print(f"Emotion: {record['emotion']} ({record['emotion_score']:.2f})")
    elif args.file:
        df = analyze_file(args.file)
        out = save_results_df(df, args.out)
        print(f"Batch analysis complete. Results saved to: {out}")
    else:
        text = input("Enter text to analyze: ")
        record = analyze_text_record(text)
        print("\nAI Analysis Result:")
        print(f"Summary: {record['summary']}")
        print(f"Keywords: {record['keywords']}")
        print(f"Sentiment: {record['sentiment']} ({record['sentiment_score']:.2f})")
        print(f"Emotion: {record['emotion']} ({record['emotion_score']:.2f})")

if __name__ == "__main__":
    main()
