import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Set the page title
st.title("Steam Game Recommender System")

# Upload the CSV file
uploaded_file = st.file_uploader("Choose a file (games.csv)", type="csv")

if uploaded_file is not None:
    # Column names for the dataset
    column_names = ['url', 'types', 'name', 'desc_snippet', 'recent_reviews', 'all_reviews', 'release_date', 'developer', 
                    'publisher', 'popular_tags', 'game_details', 'languages', 'achievements', 'genre', 
                    'game_description', 'mature_content', 'minimum_requirements', 'recommended_requirements', 
                    'original_price']

    # Load the data
    df = pd.read_csv(uploaded_file, names=column_names, low_memory=True)

    # Select relevant columns
    df1 = df[['name', 'desc_snippet', 'popular_tags', 'genre', 'original_price']]

    # Drop missing values
    df2 = pd.DataFrame(df1.dropna())

    # Display the first 10 rows of the cleaned data
    st.write("Here are the first 10 rows of the dataset:")
    st.write(df2.head(10))

    # Create TF-IDF matrix
    tfidf = TfidfVectorizer(stop_words="english")
    tfidf_matrix = tfidf.fit_transform(df2["desc_snippet"])

    # Calculate cosine similarity
    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

    # Function to recommend games based on input title
    def steam_game_recommender(title, cosine_sim, dataframe):
        indices = pd.Series(dataframe.index, index=dataframe['name']).drop_duplicates()
        if title not in indices:
            return []
        game_index = indices[title]
        sim_scores = pd.DataFrame(cosine_sim[game_index], columns=["score"])
        game_indices = sim_scores.sort_values("score", ascending=False)[1:11].index
        return dataframe['name'].iloc[game_indices]

    # User input: game title
    game = st.text_input("Please enter a game you like:")

    if game:
        # Display recommended games
        st.write("Recommended Games:")
        recommended_games = steam_game_recommender(game, cosine_sim, df2)
        if not recommended_games.empty:
            st.write(recommended_games)
        else:
            st.write(f"Sorry, {game} is not in the dataset.")
