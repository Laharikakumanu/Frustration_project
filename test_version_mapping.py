import pandas as pd
from utils.version_labels import load_version_config, assign_versions

# Load cleaned data
zoom = pd.read_csv("outputs/zoom_cleaned.csv", parse_dates=["at"])
webex = pd.read_csv("outputs/webex_cleaned.csv", parse_dates=["at"])
firefox = pd.read_csv("outputs/firefox_cleaned.csv", parse_dates=["at"])

# Load version release data
version_config = load_version_config("config/app_versions.json")

# Assign mapped version labels
zoom_mapped = assign_versions(zoom, "Zoom", version_config)
webex_mapped = assign_versions(webex, "Webex", version_config)
firefox_mapped = assign_versions(firefox, "Firefox", version_config)

# Save updated files
zoom_mapped.to_csv("outputs/zoom_mapped.csv", index=False)
webex_mapped.to_csv("outputs/webex_mapped.csv", index=False)
firefox_mapped.to_csv("outputs/firefox_mapped.csv", index=False)

print("✅ Version labels added and saved.")
