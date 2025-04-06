import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from wordcloud import WordCloud

st.set_page_config(page_title="Sentiment Analysis Dashboard", layout="wide")

# ---------- Load Data ----------
@st.cache_data

def load_data():
    df = pd.read_csv("outputs/zoom_final.csv", parse_dates=["at"], low_memory=False)
    df = df[df["app_version_mapped"].notna() & df["sentiment"].notna() & df["clean_review"].notna()]
    df["week"] = df["at"].dt.to_period("W").apply(lambda r: r.start_time)
    return df

df = load_data()

st.title("Sentiment Analysis Dashboard for Zoom")

# ---------- Tabs ----------
tabs = st.tabs([
    "1. Overall Sentiment by Version",
    "2. Drill-Down Explorer",
    "3. Weekly Negative Reviews Timeline"
])

# ---------- Tab 1: Sentiment Distribution ----------
with tabs[0]:
    st.header("1. Overall Sentiment Distribution by App Version")
    sentiment_counts = df.groupby("app_version_mapped")["sentiment"].value_counts().unstack()
    sentiment_counts = sentiment_counts.reindex(columns=["POSITIVE","NEGATIVE"], fill_value=0)

    fig = px.bar(sentiment_counts,
                 x=sentiment_counts.index,
                 y=["POSITIVE", "NEGATIVE"],
                 title="Sentiment Breakdown per Version",
                 labels={"value": "Review Count", "app_version_mapped": "Version"},
                 barmode="group")
    fig.update_layout(template="plotly_white")
    st.plotly_chart(fig, use_container_width=True)

# ---------- Tab 2: Drill-Down Explorer ----------
with tabs[1]:
    st.header("2. Drill-Down Explorer for Selected Version")
    version_list = df["app_version_mapped"].dropna().unique().tolist()
    selected_version = st.selectbox("Select a Version to Explore", sorted(version_list))
    df_version = df[df["app_version_mapped"] == selected_version]

    st.subheader("Complaint Category Radar Chart")
    keyword_map = {
        "UI": ["layout", "screen", "design", "text", "navigation"],
        "Performance": ["slow", "lag", "freeze", "delay", "load"],
        "Crashes": ["crash", "error", "fail", "broken"],
        "Other": ["pricing", "login", "notification", "ads"]
    }
    category_counts = {}
    for category, keywords in keyword_map.items():
        mask = df_version["clean_review"].str.contains("|".join(keywords), case=False, na=False)
        category_counts[category] = mask.sum()

    radar_fig = go.Figure()
    radar_fig.add_trace(go.Scatterpolar(
        r=list(category_counts.values()),
        theta=list(category_counts.keys()),
        fill='toself',
        name='Complaint Categories'
    ))
    radar_fig.update_layout(polar=dict(radialaxis=dict(visible=True)),
                            showlegend=False,
                            title="Complaint Categories for Version: " + selected_version)
    st.plotly_chart(radar_fig, use_container_width=True)

    st.subheader("Top Complaint Keywords and Word Cloud")
    texts = df_version[df_version["sentiment"] == "NEGATIVE"]["clean_review"].dropna().tolist()
    if texts:
        vectorizer = TfidfVectorizer(max_features=10, stop_words="english")
        X = vectorizer.fit_transform(texts)
        keywords = vectorizer.get_feature_names_out()
        scores = X.sum(axis=0).A1

        fig, ax = plt.subplots(figsize=(10, 4))
        ax.barh(keywords, scores, color="darkred")
        ax.set_title("Top TF-IDF Complaint Keywords")
        ax.invert_yaxis()
        st.pyplot(fig)

        wordcloud = WordCloud(width=800, height=300, background_color='white').generate(" ".join(texts))
        st.image(wordcloud.to_array(), caption="Word Cloud of Complaint Terms", use_column_width=True)

    st.subheader("Representative Negative Reviews")
    for i, row in df_version[df_version["sentiment"] == "NEGATIVE"].head(3).iterrows():
        st.markdown(f"- \"{row['clean_review']}\"")

# ---------- Tab 3: Weekly Frustration Timeline ----------
with tabs[2]:
    st.header("3. Weekly Negative Timeline for Zoom")
    weekly = df.groupby("week")["sentiment"].value_counts().unstack().fillna(0)
    weekly["neg_percent"] = (weekly.get("NEGATIVE", 0) / weekly.sum(axis=1)) * 100
    weekly = weekly.reset_index()

    fig = px.line(weekly, x="week", y="neg_percent", markers=True,
                  title="% Negative Reviews Over Time (Weekly)",
                  labels={"neg_percent": "% Negative Reviews", "week": "Week"})
    fig.update_layout(template="plotly_white")
    st.plotly_chart(fig, use_container_width=True)
