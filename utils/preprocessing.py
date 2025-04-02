import pandas as pd
import re
import string
import langid


def is_english(text):
    try:
        return langid.classify(text)[0] == 'en'
    except:
        return False
   

def normalize_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+|https\S+", '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def clean_reviews(df, app_name, text_col="content", date_col="at", version_col="appVersion"):
    print(f"\n Cleaning: {app_name} ({len(df)} reviews)")

    # Drop null/blank reviews
    df = df.dropna(subset=[text_col])
    df[text_col] = df[text_col].astype(str)
    df = df[df[text_col].str.strip().str.len() > 10]

    # Parse date
    df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
    df = df.dropna(subset=[date_col])

    # Filter English reviews
    df['is_english'] = df[text_col].apply(is_english)
    df = df[df['is_english']]

    # Normalize review text
    df['clean_review'] = df[text_col].apply(normalize_text)

    # Drop duplicates
    df = df.drop_duplicates(subset=['clean_review', date_col])

    # Add app label
    df['app'] = app_name

    print(f" Cleaned: {len(df)} reviews remain after processing.")
    return df
