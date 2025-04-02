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


st.title("📊 App Review Frustration Dashboard")

# ---------- Tabs for Pages ----------
tabs = st.tabs(["Frustration Timeline", "Drill-Down Explorer", "Complaint Analyzer", "Multi-App Comparison"])

# ---------- Page 1: Timeline ----------
with tabs[0]:
    st.header("📈 Weekly Frustration Timeline")
    app_choice = st.selectbox("Choose App", list(app_data.keys()))
    df = app_data[app_choice]
    weekly = df.groupby("week")["sentiment"].value_counts().unstack().fillna(0)
    weekly["neg_percent"] = (weekly.get("NEGATIVE", 0) / weekly.sum(axis=1)) * 100
    weekly = weekly.reset_index()

    fig = px.line(weekly, x="week", y="neg_percent", markers=True,
                  title=f"{app_choice} – % Negative Reviews Per Week",
                  labels={"neg_percent": "% Negative Reviews"})
    fig.update_layout(template="plotly_white")
    st.plotly_chart(fig, use_container_width=True)

# ---------- Page 2: Drill-Down ----------
with tabs[1]:
    st.header("🔎 Drill-Down Explorer")
    app_choice = st.selectbox("Select App", list(app_data.keys()), key="drill_app")
    df = app_data[app_choice]
    week_list = sorted(df["week"].dropna().unique())
    week_choice = st.selectbox("Select Week", week_list)

    filtered = df[(df["week"] == pd.to_datetime(week_choice)) & (df["sentiment"] == "NEGATIVE")]
    st.write(f"Found {len(filtered)} negative reviews.")

    if not filtered.empty:
        # TF-IDF Bar Chart
        texts = filtered["clean_review"].dropna().tolist()
        vectorizer = TfidfVectorizer(max_features=10, stop_words="english")
        X = vectorizer.fit_transform(texts)
        keywords = vectorizer.get_feature_names_out()
        scores = X.sum(axis=0).A1

        fig, ax = plt.subplots(figsize=(10, 4))
        ax.barh(keywords, scores, color="darkred")
        ax.set_title(f"Top TF-IDF Complaint Keywords – {app_choice}")
        ax.invert_yaxis()
        st.pyplot(fig)

        # Word Cloud
        wordcloud = WordCloud(width=800, height=300, background_color='white').generate(" ".join(texts))
        st.image(wordcloud.to_array(), caption="Word Cloud of Complaints", use_column_width=True)

# ---------- Page 3: Complaint Analyzer ----------
with tabs[2]:
    st.header("🔧 Complaint Analyzer")
    st.markdown("""
    This page classifies complaints into common types:
    - **UI Issues**: layout, screen, design, text, navigation
    - **Performance**: slow, lag, freeze, delay, load
    - **Bugs / Crashes**: crash, error, fail, broken
    - **Other**: pricing, login, notifications, ads
    """)
    app_choice = st.selectbox("Choose App for Complaint Categories", list(app_data.keys()), key="cat_app")
    df = app_data[app_choice]
    keyword_map = {
        "UI": ["layout", "screen", "design", "text", "navigation"],
        "Performance": ["slow", "lag", "freeze", "delay", "load"],
        "Crashes": ["crash", "error", "fail", "broken"],
        "Other": ["pricing", "login", "notification", "ads"]
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

# ---------- Page 4: Multi-App Comparison ----------
with tabs[3]:
    st.header("📊 Multi-App Frustration Comparison")
    df_all = []
    for app, df in app_data.items():
        weekly = df.groupby("week")["sentiment"].value_counts().unstack().fillna(0)
        weekly["neg_percent"] = (weekly.get("NEGATIVE", 0) / weekly.sum(axis=1)) * 100
        weekly = weekly.reset_index()
        weekly["App"] = app
        df_all.append(weekly[["week", "neg_percent", "App"]])
    merged = pd.concat(df_all)

    fig = px.line(merged, x="week", y="neg_percent", color="App", markers=True,
                  title="Weekly Frustration Comparison Across Apps",
                  labels={"neg_percent": "% Negative Reviews"})
    fig.update_layout(template="plotly_white")
    st.plotly_chart(fig, use_container_width=True)
