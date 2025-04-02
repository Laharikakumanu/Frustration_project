import pandas as pd
from model.distibert_sentiment import classify_reviews

# Load the mapped datasets
zoom_df = pd.read_csv("outputs/zoom_mapped.csv")
webex_df = pd.read_csv("outputs/webex_mapped.csv")
firefox_df = pd.read_csv("outputs/firefox_mapped.csv")

# Classify and save each one
for app_name, df in [("Zoom", zoom_df), ("Webex", webex_df), ("Firefox", firefox_df)]:
    print(f"\n🔍 Processing sentiment for {app_name}...")
    df = classify_reviews(df, text_col="clean_review")
    df.to_csv(f"outputs/{app_name.lower()}_final.csv", index=False)
    print(f"✅ Done with {app_name}: saved to outputs/{app_name.lower()}_final.csv")
