import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

path = 'games.csv'
column_names = ['url', 'types', 'name', 'desc_snippet', 'recent_reviews', 'all_reviews', 'release_date', 'developer', 'publisher', 'popular_tags', 'game_details', 'languages', 'achievements', 'genre', 'game_description', 'mature_content', 'minimum_requirements', 'recommended_requirements', 'original_price']
df = pd.read_csv(path, names=column_names, low_memory=True)
df.head()

df.info()

# variables used
df1 = df[['name', 'desc_snippet', 'popular_tags', 'genre', 'original_price']]

# delete missing values
df2 = pd.DataFrame(df1.dropna())

df2.head(10)

tfidf = TfidfVectorizer(stop_words="english")

tfidf_matrix = tfidf.fit_transform(df2["desc_snippet"])

tfidf_matrix.shape

tfidf_matrix.toarray()

tfidf.get_feature_names_out()

tfidf_matrix.toarray()

cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

cosine_sim.shape
cosine_sim[1]

# Asking for user input
game = input("Please enter a game you like: ")

print(f"Recommended Games:")
# Create an index series for the 'name' column
indices = pd.Series(df2.index, index=df2['name']).drop_duplicates()

# Check if the game is in the dataset
if game in indices:
    game_index = indices[game]
    
    # Calculate similarity scores for the game
    sim_scores = pd.DataFrame(cosine_sim[game_index], columns=["score"])
    
    # Get the top 10 similar games (excluding the input game itself)
    game_indices = sim_scores.sort_values("score", ascending=False)[1:11].index
    
    # Display the recommended games
    print(df2["name"].iloc[game_indices])
else:
    print(f"Sorry, {game} is not in the dataset.")

def steam_game_recommender(title, cosine_sim, dataframe):
    indices = pd.Series(dataframe.index, index=dataframe['name'])
    indices = indices[~indices.index.duplicated(keep='last')]
    game_index = indices[title]
    sim_scores = pd.DataFrame(cosine_sim[game_index], columns=["score"])
    game_indices = sim_scores.sort_values("score", ascending=False)[1:11].index
    return dataframe['name'].iloc[game_indices]

steam_game_recommender(game, cosine_sim, df2)
