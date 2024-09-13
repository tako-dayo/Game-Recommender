import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import difflib
import streamlit as st

# Load data
path = 'games.csv'
column_names = ['url', 'types', 'name', 'desc_snippet', 'recent_reviews', 'all_reviews', 'release_date', 'developer', 'publisher', 'popular_tags', 'game_details', 'languages', 'achievements', 'genre', 'game_description', 'mature_content', 'minimum_requirements', 'recommended_requirements', 'original_price']   
df = pd.read_csv(path, names=column_names, low_memory=True)

# Selecting relevant columns and dropping missing values
df1 = df[['name', 'desc_snippet', 'popular_tags','genre', 'original_price']]
df2 = df1.dropna()

# Vectorize the game descriptions using TF-IDF
tfidf = TfidfVectorizer(stop_words="english")
tfidf_matrix = tfidf.fit_transform(df2["desc_snippet"])

# Calculate the cosine similarity matrix
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

# Function using difflib to find the closest game name
def get_closest_match(game_name, game_list):
    matches = difflib.get_close_matches(game_name, game_list, n=1, cutoff=0.6)
    return matches[0] if matches else None

# Function to recommend similar games based on cosine similarity
def recommend_games(game_index, sim_matrix, df, n_recommendations=10):
    sim_scores = pd.Series(sim_matrix[game_index]).sort_values(ascending=False)
    top_indices = sim_scores.iloc[1:n_recommendations + 1].index
    return df.iloc[top_indices]

def main():
    st.title("Game Recommender System")
    
    # Taking user input in Streamlit
    game = st.text_input("Please enter a game you like:").strip().lower()

    if game:
        df2['name'] = df2['name'].str.lower()  # Ensure all names in the dataset are lowercase
        
        # Create an index series for the 'name' column to map game names to their index in the dataframe
        indices = pd.Series(df2.index, index=df2['name']).drop_duplicates()

        # First, check if the user input matches a game directly
        if game in indices:
            game_index = indices[game]
            st.write(f"Exact match found for '{game}':")
            recommendations = recommend_games(game_index, cosine_sim, df2, n_recommendations=10)
            st.table(recommendations[['name', 'genre', 'original_price']])
        else:
            # If not found directly, use difflib to find the closest game
            closest_game = get_closest_match(game, df2['name'].tolist())

            # Check if a difflib match was found
            if closest_game:
                game_index = indices[closest_game]
                st.write(f"Recommended Games based on closest match: '{closest_game}':")
                recommendations = recommend_games(game_index, cosine_sim, df2, n_recommendations=10)
                st.table(recommendations[['name', 'genre', 'original_price']])
            else:
                st.write(f"Sorry, no match for '{game}' found in the dataset.")

if __name__ == "__main__":
    main()

# Streamlit app
st.title("🎮Steam Games Recommender🎮")
st.write("🔎 Find out which game best suits you now 🔎")
