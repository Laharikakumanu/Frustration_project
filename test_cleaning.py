import pandas as pd
from utils.preprocessing import clean_reviews

# Load renamed datasets
zoom_raw = pd.read_csv("data/Zoom.csv", low_memory=False)
webex_raw = pd.read_csv("data/Webex.csv")
firefox_raw = pd.read_csv("data/Firefox.csv")

# Clean each one
zoom_clean = clean_reviews(zoom_raw, "Zoom", text_col="content", date_col="at", version_col="appVersion")
webex_clean = clean_reviews(webex_raw, "Webex", text_col="content", date_col="at", version_col="appVersion")
firefox_clean = clean_reviews(firefox_raw, "Firefox", text_col="content", date_col="at", version_col="appVersion")

# Save the cleaned outputs
zoom_clean.to_csv("outputs/zoom_cleaned.csv", index=False)
webex_clean.to_csv("outputs/webex_cleaned.csv", index=False)
firefox_clean.to_csv("outputs/firefox_cleaned.csv", index=False)
