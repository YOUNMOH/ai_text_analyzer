from transformers import pipeline

analyzer = pipeline("text-classification", model="distilbert-base-uncased-finetuned-sst-2-english")


text = input("Enter text to analyze: ")


result = analyzer(text)


label = result[0]['label']
score = result[0]['score']

print(f"\nAI Analysis Result:")
print(f"Sentiment: {label}")
print(f"Confidence: {score:.2f}")

