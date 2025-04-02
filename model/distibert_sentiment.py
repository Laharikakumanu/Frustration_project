from transformers import pipeline
import pandas as pd
from tqdm import tqdm

# Load sentiment pipeline (DistilBERT fine-tuned on SST-2)
def load_sentiment_pipeline():
    return pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")

# Predict sentiment for a list of texts
def predict_sentiments(texts, classifier):
    results = []
    for text in tqdm(texts, desc="Classifying sentiment"):
        if not isinstance(text, str) or len(text.strip()) < 5:
            results.append("NEUTRAL")
            continue
        try:
            pred = classifier(text[:512])[0]  # truncate long reviews
            label = pred["label"].upper()
            results.append(label)
        except:
            results.append("NEUTRAL")
    return results

# Apply to a DataFrame
def classify_reviews(df, text_col="clean_review"):
    classifier = load_sentiment_pipeline()
    df["sentiment"] = predict_sentiments(df[text_col].tolist(), classifier)
    return df
