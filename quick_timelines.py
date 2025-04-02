import pandas as pd
import matplotlib.pyplot as plt

# App file paths and display names
apps = {
    "Zoom": "outputs/zoom_final.csv",
    "Webex": "outputs/webex_final.csv",
    "Firefox": "outputs/firefox_final.csv"
}

for app_name, path in apps.items():
    print(f"\n🔄 Generating plot for {app_name}...")
    
    df = pd.read_csv(path, parse_dates=["at"], low_memory=False)
    df["week"] = df["at"].dt.to_period("W").apply(lambda r: r.start_time)

    # Weekly sentiment counts
    weekly = df.groupby("week")["sentiment"].value_counts().unstack().fillna(0)
    weekly["neg_percent"] = (weekly.get("NEGATIVE", 0) / weekly.sum(axis=1)) * 100
    weekly = weekly.reset_index()

    # Plot using Matplotlib (lightweight)
    plt.figure(figsize=(10, 4))
    plt.plot(weekly["week"], weekly["neg_percent"], marker='o', label=app_name)
    plt.title(f"{app_name} – Weekly Frustration %")
    plt.xlabel("Week")
    plt.ylabel("% Negative Reviews")
    plt.grid(True)
    plt.tight_layout()
    plt.xticks(rotation=45)
    plt.legend()
    plt.show()
