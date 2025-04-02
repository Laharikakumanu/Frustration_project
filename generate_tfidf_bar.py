import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer

def plot_tfidf_for_week(app_name, file_path, week_str):
    print(f"🔍 Loading {app_name} reviews for week: {week_str}")

    df = pd.read_csv(file_path, parse_dates=["at"], low_memory=False)
    df["week"] = df["at"].dt.to_period("W").apply(lambda r: r.start_time)

    # Filter to the selected week and only NEGATIVE reviews
    week = pd.to_datetime(week_str)
    df_week = df[(df["week"] == week) & (df["sentiment"] == "NEGATIVE")]

    if df_week.empty:
        print(f"⚠️ No negative reviews found for {app_name} on week {week_str}")
        return

    texts = df_week["clean_review"].dropna().tolist()

    # TF-IDF extraction
    vectorizer = TfidfVectorizer(max_features=10, stop_words="english")
    X = vectorizer.fit_transform(texts)
    keywords = vectorizer.get_feature_names_out()
    scores = X.sum(axis=0).A1

    # Plot
    plt.figure(figsize=(10, 5))
    plt.barh(keywords, scores, color="darkred")
    plt.xlabel("TF-IDF Score")
    plt.title(f"Top Complaint Keywords – {app_name} ({week_str})")
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.show()

# 👉 Choose the app, file, and week here
app_name = "Zoom"
file_path = "outputs/zoom_final.csv"
week_str = "2023-03-06"  # Use exact Monday date from your timeline spike

plot_tfidf_for_week(app_name, file_path, week_str)
