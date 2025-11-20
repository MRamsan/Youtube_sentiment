import streamlit as st
import pandas as pd
import numpy as np
from googleapiclient.discovery import build
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import re
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import seaborn as sns

sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 100

class YouTubeSentimentAnalyzer:
    def __init__(self, api_key):
        self.api_key = api_key
        self.youtube = build('youtube', 'v3', developerKey=api_key)
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
        return url

    def get_video_info(self, video_id):
        request = self.youtube.videos().list(part='snippet,statistics', id=video_id)
        response = request.execute()
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
            request = self.youtube.commentThreads().list(
                part='snippet', videoId=video_id, maxResults=min(100, max_comments - len(comments)),
                pageToken=next_page_token, textFormat='plainText'
            )
            response = request.execute()
            for item in response['items']:
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
        text = ' '.join(text.split())
        return text

    def analyze(self, comments):
        data = []
        for comment in comments:
            cleaned = self.clean_text(comment['text'])
            if len(cleaned.split()) < 2:
                continue
            blob = TextBlob(cleaned)
            vader = self.vader.polarity_scores(cleaned)
            data.append({
                'text': comment['text'],
                'cleaned': cleaned,
                'likes': comment['likes'],
                'textblob_polarity': blob.sentiment.polarity,
                'textblob_sentiment': 'Positive' if blob.sentiment.polarity > 0.1 else 'Negative' if blob.sentiment.polarity < -0.1 else 'Neutral',
                'vader_compound': vader['compound'],
                'vader_sentiment': 'Positive' if vader['compound'] > 0.05 else 'Negative' if vader['compound'] < -0.05 else 'Neutral'
            })
        self.comments_df = pd.DataFrame(data)
        return self.comments_df

# ---------------- STREAMLIT APP ----------------
st.title("📊 YouTube Comment Sentiment Analyzer")
api_key = st.text_input("Enter YouTube API Key", type="password")
url = st.text_input("Enter YouTube Video URL")
max_comments = st.slider("Comments to analyze", 50, 500, 200)

if st.button("Analyze"):
    if not api_key:
        st.error("API Key required!")
    else:
        try:
            analyzer = YouTubeSentimentAnalyzer(api_key)
            vid = analyzer.extract_video_id(url)
            info = analyzer.get_video_info(vid)

            st.subheader("Video Information")
            st.write(info)

            comments = analyzer.get_comments(vid, max_comments)
            df = analyzer.analyze(comments)
            st.success(f"Analyzed {len(df)} comments")
            st.dataframe(df)

            st.subheader("Sentiment Distribution (VADER)")
            fig, ax = plt.subplots()
            df['vader_sentiment'].value_counts().plot(kind='bar', ax=ax)
            st.pyplot(fig)

            st.subheader("Word Cloud")
            wc = WordCloud(width=500, height=300, background_color='white').generate(" ".join(df['cleaned']))
            fig2, ax2 = plt.subplots()
            ax2.imshow(wc)
            ax2.axis('off')
            st.pyplot(fig2)

            st.download_button("Download CSV", df.to_csv(index=False), file_name="sentiment_results.csv")

        except Exception as e:
            st.error(f"Error: {e}")
