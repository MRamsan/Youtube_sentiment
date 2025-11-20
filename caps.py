import streamlit as st
import pandas as pd
import numpy as np
from googleapiclient.discovery import build
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import seaborn as sns
import re

sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 100


# ====================================================
#   SENTIMENT ANALYZER CLASS
# ====================================================
class YouTubeSentimentAnalyzer:
    def __init__(self, api_key):
        self.api_key = api_key
        try:
            self.youtube = build('youtube', 'v3', developerKey=api_key)
        except Exception as e:
            st.error(f"❌ Failed to initialize YouTube API: {e}")
        self.vader = SentimentIntensityAnalyzer()
        self.comments_df = None
        self.video_info = None

    def extract_video_id(self, url):
        patterns = [
            r'(?:https?:\/\/)?(?:www\.)?youtube\.com\/watch\?v=([a-zA-Z0-9_-]{11})',
            r'(?:https?:\/\/)?(?:www\.)?youtu\.be\/([a-zA-Z0-9_-]{11})',
            r'(?:https?:\/\/)?(?:www\.)?youtube\.com\/embed\/([a-zA-Z0-9_-]{11})',
        ]
        for pattern in patterns:
            match = re.search(pattern, url)
            if match:
                return match.group(1)
        return url  # assume it's a pure video ID

    def get_video_info(self, video_id):
        try:
            request = self.youtube.videos().list(
                part='snippet,statistics',
                id=video_id
            )
            response = request.execute()
        except Exception as e:
            st.error(f"❌ YouTube API Error (video info): {e}")
            return None

        if not response.get("items"):
            st.error("❌ No video found with this ID.")
            return None

        video = response['items'][0]

        self.video_info = {
            'title': video['snippet']['title'],
            'channel': video['snippet']['channelTitle'],
            'published_at': video['snippet']['publishedAt'],
            'views': int(video['statistics'].get('viewCount', 0)),
            'likes': int(video['statistics'].get('likeCount', 0)),
            'comments_count': int(video['statistics'].get('commentCount', 0))
        }
        return self.video_info

    def get_comments(self, video_id, max_comments=300):
        comments = []
        next_page_token = None

        while len(comments) < max_comments:
            try:
                request = self.youtube.commentThreads().list(
                    part='snippet',
                    videoId=video_id,
                    maxResults=min(100, max_comments - len(comments)),
                    pageToken=next_page_token,
                    textFormat='plainText'
                )
                response = request.execute()

            except Exception as e:
                st.error(f"❌ YouTube API Error (comments): {e}")
                return []

            for item in response.get('items', []):
                c = item['snippet']['topLevelComment']['snippet']
                comments.append({
                    'author': c['authorDisplayName'],
                    'text': c['textDisplay'],
                    'likes': c['likeCount'],
                    'published_at': c['publishedAt']
                })

            next_page_token = response.get('nextPageToken')
            if not next_page_token:
                break

        return comments

    def clean_text(self, text):
        text = re.sub(r'http\S+|www\S+|https\S+', '', text)
        text = re.sub(r'@\w+', '', text)
        text = re.sub(r'[^\w\s.,!?]', '', text)
        return ' '.join(text.split())

    def analyze(self, comments):
        data = []

        for comment in comments:
            cleaned = self.clean_text(comment['text'])
            if len(cleaned.split()) < 2:
                continue

            blob = TextBlob(cleaned)
            vader = self.vader.polarity_scores(cleaned)

            data.append({
                'comment': cleaned,
                'polarity': blob.sentiment.polarity,
                'subjectivity': blob.sentiment.subjectivity,
                'vader_pos': vader['pos'],
                'vader_neg': vader['neg'],
                'vader_neu': vader['neu'],
                'vader_compound': vader['compound']
            })

        df = pd.DataFrame(data)
        self.comments_df = df
        return df


# ====================================================
#   STREAMLIT APP
# ====================================================
def main():
    st.title("🎬 YouTube Comment Sentiment Analyzer")
    st.write("Analyze sentiment, generate visual insights for any YouTube video.")

    # Load Secret API Key
    try:
        api_key = st.secrets["YOUTUBE_API_KEY"]
    except:
        st.error("❌ Missing API key in Streamlit Secrets.\nGo to → Settings → Secrets")
        return

    analyzer = YouTubeSentimentAnalyzer(api_key)
    url = st.text_input("Enter YouTube Video URL")

    if st.button("Analyze"):
        if not url:
            st.error("❌ Please enter a valid YouTube URL.")
            return

        # Extract Video ID
        video_id = analyzer.extract_video_id(url)

        # Fetch Video Info
        with st.spinner("📌 Fetching video details..."):
            video_info = analyzer.get_video_info(video_id)

        if not video_info:
            return

        st.subheader("🎥 Video Information")
        st.json(video_info)

        # Fetch Comments
        with st.spinner("💬 Fetching comments..."):
            comments = analyzer.get_comments(video_id)

        if len(comments) == 0:
            st.error("❌ No comments found or API blocked the request.")
            return

        st.success(f"✔ Retrieved {len(comments)} comments.")

        # Sentiment Analysis
        with st.spinner("🧠 Running sentiment analysis..."):
            df = analyzer.analyze(comments)

        st.subheader("📊 Sentiment Analysis Results")
        st.dataframe(df)

        # ============================
        # Word Cloud
        # ============================
        st.subheader("☁️ Word Cloud of Comments")
        all_text = " ".join(df['comment'])
        wc = WordCloud(width=800, height=400).generate(all_text)

        fig, ax = plt.subplots()
        ax.imshow(wc, interpolation="bilinear")
        ax.axis("off")
        st.pyplot(fig)

        # ============================
        # Sentiment Histogram
        # ============================
        st.subheader("📈 Sentiment Distribution (Compound Score)")
        fig2, ax2 = plt.subplots()
        sns.histplot(df['vader_compound'], bins=20, ax=ax2)
        st.pyplot(fig2)

        # ============================
        # Download Results
        # ============================
        csv = df.to_csv(index=False).encode("utf-8")
        st.download_button("⬇ Download Sentiment Data (CSV)", csv, "sentiment_results.csv", "text/csv")


if __name__ == "__main__":
    main()
