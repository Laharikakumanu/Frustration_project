import json
import pandas as pd
from datetime import datetime

# Load version timeline from config
def load_version_config(json_path="config/app_versions.json"):
    with open(json_path, 'r') as f:
        return json.load(f)

# Assign closest version based on date
def assign_versions(df, app_name, config, date_col="at"):
    version_map = config.get(app_name, {})
    if not version_map:
        print(f"No version data found for {app_name}")
        return df

    # Convert version dates to datetime
    version_dates = {
        version: pd.to_datetime(release_date)
        for version, release_date in version_map.items()
    }

    # Sort versions by release date
    sorted_versions = sorted(version_dates.items(), key=lambda x: x[1])

    def find_version(review_date):
        for version, rel_date in reversed(sorted_versions):
            if review_date >= rel_date:
                return version
        return sorted_versions[0][0]  # fallback to earliest version

    df['app_version_mapped'] = df[date_col].apply(find_version)
    return df
