import pandas as pd
import plotly.graph_objects as go
import json
import os

# Load version data
with open("config/app_versions.json") as f:
    version_data = json.load(f)

# Colors
colors = {
    "Zoom": "#1f77b4",
    "Webex": "#2ca02c",
    "Firefox": "#d62728"
}

def process_app_data(file_path, app_name):
    df = pd.read_csv(file_path, parse_dates=["at"], low_memory=False)
    df["week"] = df["at"].dt.to_period("W").apply(lambda r: r.start_time)
    weekly_stats = df.groupby("week")["sentiment"].value_counts().unstack().fillna(0)
    weekly_stats["negative_percent"] = (weekly_stats.get("NEGATIVE", 0) / weekly_stats.sum(axis=1)) * 100
    weekly_stats.reset_index(inplace=True)
    return df, weekly_stats

def plot_frustration_timeline(app_name, weekly_stats):
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=weekly_stats["week"],
        y=weekly_stats["negative_percent"],
        mode='lines+markers',
        name=f"{app_name} Neg %",
        line=dict(color=colors[app_name])
    ))

    for version, rel_date in version_data.get(app_name, {}).items():
        rel_date = pd.to_datetime(rel_date)

        fig.add_vline(
            x=rel_date,
            line_dash="dot",
            line_color="gray",
            opacity=0.5
        )

        fig.add_annotation(
            x=rel_date,
            y=weekly_stats["negative_percent"].max() * 0.95,
            text=version,
            showarrow=False,
            font=dict(color="gray", size=10),
            bgcolor="rgba(255,255,255,0.6)"
        )

    fig.update_layout(
        title=f"Frustration Timeline – {app_name}",
        xaxis_title="Week",
        yaxis_title="% Negative Reviews",
        template="plotly_white"
    )

    os.makedirs("outputs/plots", exist_ok=True)
    # Try saving with Kaleido, fallback to show
    try:
        import kaleido
        fig.write_image(f"outputs/plots/{app_name.lower()}_timeline.png")
        print(f"✅ Saved: {app_name.lower()}_timeline.png")
    except:
        print(f"⚠️ Could not save {app_name.lower()}_timeline.png — displaying instead.")
        fig.show()

# Run for all apps
apps = {
    "Zoom": "outputs/zoom_final.csv",
    "Webex": "outputs/webex_final.csv",
    "Firefox": "outputs/firefox_final.csv"
}

for app_name, path in apps.items():
    df, stats = process_app_data(path, app_name)
    plot_frustration_timeline(app_name, stats)
