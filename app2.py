import streamlit as st
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from wordcloud import WordCloud
import os 



github_token = st.secrets["github_token"]

st.set_page_config(page_title="Frustration Dashboard", layout="wide")

# ---------- Load Data ----------
@st.cache_data
def load_data():
    files = {
        "Zoom": "outputs/zoom_final.csv",
        "Webex": "outputs/webex_final.csv",
        "Firefox": "outputs/firefox_final.csv"
    }
    data = {}
    for app, path in files.items():
        df = pd.read_csv(path, parse_dates=["at"], low_memory=False)
        df["week"] = df["at"].dt.to_period("W").apply(lambda r: r.start_time)
        data[app] = df
    return data

app_data = load_data()

st.title("📊 App Review  Dashboard")

# ---------- Tabs for Pages ----------
tabs = st.tabs(["Negative Review Timeline", "Complaint Analyzer", "Complaint Radar"])

# ---------- Page 1: Timeline ----------
with tabs[0]:
    st.header("📈 Weekly Negative Review Timeline")
    app_choice = st.selectbox("Choose App", list(app_data.keys()))
    df = app_data[app_choice]
    weekly = df.groupby("week")["sentiment"].value_counts().unstack().fillna(0)
    weekly["neg_percent"] = (weekly.get("NEGATIVE", 0) / weekly.sum(axis=1)) * 100
    weekly = weekly.reset_index()

    selected_week = st.selectbox("Select a week to drill down:", weekly["week"].astype(str))

    fig = px.line(weekly, x="week", y="neg_percent", markers=True,
                  title=f"{app_choice} – % Negative Reviews Per Week",
                  labels={"neg_percent": "% Negative Reviews"})
    fig.update_layout(template="plotly_white")
    st.plotly_chart(fig, use_container_width=True)

    # Show additional details for the selected week
    st.subheader(f"🔹 Drill-down for {selected_week}")
    df["week"] = df["week"].astype(str)
    selected_reviews = df[(df["week"] == selected_week) & (df["sentiment"] == "NEGATIVE")]
    st.write(f"Found {len(selected_reviews)} negative reviews.")

    if not selected_reviews.empty:
        # TF-IDF Keywords
        texts = selected_reviews["clean_review"].dropna().tolist()
        vectorizer = TfidfVectorizer(max_features=10, stop_words="english")
        X = vectorizer.fit_transform(texts)
        keywords = vectorizer.get_feature_names_out()
        scores = X.sum(axis=0).A1

        fig, ax = plt.subplots(figsize=(10, 4))
        ax.barh(keywords, scores, color="darkred")
        ax.set_title("Top Complaint Keywords")
        ax.invert_yaxis()
        st.pyplot(fig)

        # Word Cloud
        wordcloud = WordCloud(width=800, height=300, background_color='white').generate(" ".join(texts))
        st.image(wordcloud.to_array(), caption="Word Cloud of Complaints", use_column_width=True)

        # Representative Reviews
        st.subheader("💬 Representative User Reviews")
        for i, row in selected_reviews.head(3).iterrows():
            st.markdown(f"- _\"{row['content']}\"_")



# ---------- Page 3: Complaint Analyzer ----------
with tabs[1]:
    st.header("🔧 Complaint Analyzer")
    st.markdown("""
This page classifies complaints into common types:

- **UI**: `layout`, `screen`, `design`, `text`, `navigation`  
- **Performance**: `slow`, `lag`, `freeze`, `delay`, `load`  
- **Crashes**: `crash`, `error`, `fail`, `broken`  
- **Features**: `feature`, `function`, `option`, `customize`, `tool`  
- **Privacy/Security**: `privacy`, `security`, `data`, `permission`, `track`
""")

    app_choice = st.selectbox("Choose App for Complaint Categories", list(app_data.keys()), key="cat_app")
    df = app_data[app_choice]
    keyword_map = {
    "UI": ["layout", "screen", "design", "text", "navigation"],
    "Performance": ["slow", "lag", "freeze", "delay", "load"],
    "Crashes": ["crash", "error", "fail", "broken"],
    "Features": ["feature", "function", "option", "customize", "tool"],
    "Privacy/Security": ["privacy", "security", "data", "permission", "track"]
}


    df["complaint_type"] = "Uncategorized"
    for category, keywords in keyword_map.items():
        mask = df["clean_review"].str.contains("|".join(keywords), case=False, na=False)
        df.loc[mask, "complaint_type"] = category

    weekly_cat = df.groupby(["week", "complaint_type"]).size().unstack(fill_value=0).reset_index()
    fig = px.bar(weekly_cat, x="week", y=weekly_cat.columns[1:],
                 title=f"Complaint Types Over Time – {app_choice}",
                 labels={"value": "# of Complaints"})
    fig.update_layout(barmode="stack", template="plotly_white")
    st.plotly_chart(fig, use_container_width=True)

# ---------- Page 4: Complaint Radar Chart ----------
# ---------- Page 4: Complaint Radar Chart ----------
with tabs[2]:
    st.header("📍 Complaint Radar Chart")

    app_choice = st.selectbox("Select App", list(app_data.keys()), key="radar_app")
    df = app_data[app_choice]

    # Date range selection
    min_date = df["at"].min().date()
    max_date = df["at"].max().date()
    from_date, to_date = st.date_input("Select Date Range:", [min_date, max_date], key="radar_dates")

    if from_date > to_date:
        st.warning("⚠️ 'From' date must be before 'To' date.")
    else:
        # Filter and clean
        filtered_df = df[(df["at"].dt.date >= from_date) & (df["at"].dt.date <= to_date)].copy()
        filtered_df["clean_review"] = filtered_df["clean_review"].fillna("").str.lower()

        # Expanded keyword categories
        keyword_map = {
            "🖥️ UI": ["layout", "layouts", "screen", "screens", "design", "text", "navigation", "navigate"],
            "🚀 Performance": ["slow", "slowness", "lag", "laggy", "freeze", "freezing", "delay", "delays", "load", "loading"],
            "💥 Crashes": ["crash", "crashes", "crashing", "error", "errors", "fail", "failed", "failing", "broken"],
            "⚙️ Features": ["feature", "features", "function", "functions", "option", "options", "customize", "tool", "tools"],
            "🔐 Privacy/Security": ["privacy", "secure", "security", "data", "permission", "permissions", "track", "tracking"]
        }

        # Categorize reviews
        filtered_df["complaint_type"] = "📦 Others"
        for category, keywords in keyword_map.items():
            mask = filtered_df["clean_review"].str.contains("|".join(keywords), case=False, na=False)
            filtered_df.loc[mask, "complaint_type"] = category

        # Checkbox to include or exclude "📦 Others"
        show_others = st.checkbox("Include '📦 Others' in Radar Chart", value=True)

        # Prepare full category list
        categories = list(keyword_map.keys())
        if show_others:
            categories.append("📦 Others")

        counts = filtered_df["complaint_type"].value_counts().to_dict()

        # Build radar data
        radar_data = {"Category": [], "Count": []}
        for cat in categories:
            radar_data["Category"].append(cat)
            radar_data["Count"].append(counts.get(cat, 0))

        radar_df = pd.DataFrame(radar_data)

        # Toggle for Count or Percentage
        view_mode = st.radio("View Mode:", ["Count", "Percentage"], horizontal=True)
        if view_mode == "Percentage":
            total = sum(radar_df["Count"])
            radar_df["Count"] = [round((c / total) * 100, 2) if total > 0 else 0 for c in radar_df["Count"]]

        # Radar Chart Title
        title_suffix = " (with Others)" if show_others else ""
        title = f"{app_choice} Complaint Radar – {from_date} to {to_date}{title_suffix}"

        # Plot radar chart
        fig = px.line_polar(
            radar_df,
            r="Count",
            theta="Category",
            line_close=True,
            title=title,
            markers=False
        )
        fig.update_traces(
            fill='toself',
            line_color='royalblue'
        )
        fig.update_layout(
            template="plotly_white",
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    showline=True,
                    showticklabels=True,
                    ticks='outside'
                )
            )
        )
        st.plotly_chart(fig, use_container_width=True)

        with st.expander("🧪 Sample Clean Reviews"):
            st.write(filtered_df["clean_review"].dropna().head(10).tolist())

        with st.expander("🔍 Match Breakdown"):
            for category, keywords in keyword_map.items():
                match_count = filtered_df["clean_review"].str.contains("|".join(keywords), case=False, na=False).sum()
                st.write(f"{category}: {match_count} matches")

        with st.expander("📋 View Data Table"):
            st.dataframe(radar_df)

