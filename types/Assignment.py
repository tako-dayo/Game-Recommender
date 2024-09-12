import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import os
if not os.path.exists(game.csv):
    gdown.download(url, output, quiet=False)
    
import zipfile
with zipfile.ZipFile(game.csv, 'r') as zip_ref:
    zip_ref.extractall()


# Title of the app
st.title("🎮Steam Games Recommender🎮")

# Load the dataset (assuming the dataset is locally available or use st.file_uploader() for user upload)
path = '/kaggle/input/steam-games-complete-dataset/steam_games.csv'
column_names = ['url', 'types', 'name', 'desc_snippet', 'recent_reviews', 'all_reviews', 'release_date', 'developer', 'publisher', 'popular_tags', 'game_details', 'languages', 'achievements', 'genre', 'game_description', 'mature_content', 'minimum_requirements', 'recommended_requirements', 'original_price']
df = pd.read_csv(path, names=column_names, low_memory=True)

# Display basic information
st.write("Data Overview")
st.write(df.head())
st.write(df.info())

# Preprocess the data
df1 = df[['name', 'desc_snippet', 'popular_tags', 'genre', 'original_price']]

# Drop missing values
df2 = pd.DataFrame(df1.dropna())

# TF-IDF Vectorizer
tfidf = TfidfVectorizer(stop_words="english")
tfidf_matrix = tfidf.fit_transform(df2["desc_snippet"])

# Cosine similarity matrix
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

# Create an index for the 'name' column
indices = pd.Series(df2.index, index=df2['name']).drop_duplicates()

# User input via Streamlit
game = st.text_input("Please enter a game you like:")

# Define a function for recommendation
def steam_game_recommender(title, cosine_sim, dataframe):
    if title in indices:
        game_index = indices[title]
        sim_scores = pd.DataFrame(cosine_sim[game_index], columns=["score"])
        game_indices = sim_scores.sort_values("score", ascending=False)[1:11].index
        return dataframe['name'].iloc[game_indices]
    else:
        return f"Sorry, {title} is not in the dataset."

# Display recommendations
if game:
    recommendations = steam_game_recommender(game, cosine_sim, df2)
    st.write("Recommended Games:")
    st.write(recommendations)
