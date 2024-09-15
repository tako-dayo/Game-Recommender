import streamlit as st
import pandas as pd

# Load CSV data (ensure the path is correct and CSV is accessible)
path = 'games.csv'
column_names = ['name', 'languages', 'popular_tags', 'genre', 'original_price', 'all_reviews']  # Include 'all_reviews' column
df = pd.read_csv(path, names=column_names, encoding='ISO-8859-1', low_memory=True)

# Selecting only relevant columns
df1 = df[['name', 'languages', 'popular_tags', 'genre', 'original_price', 'all_reviews']]

# Delete missing values
df2 = pd.DataFrame(df1.dropna())

# Streamlit app
st.title("🎮Game Recommender with Reviews🎮")
st.write("🔎 Find games based on language and user reviews 🔎")

# Asking for user input via Streamlit for language search
language = st.text_input("Please enter a language to find games with that language:").lower()

# Ensure all languages in the dataset are lowercase for comparison
df2['languages'] = df2['languages'].str.lower()

# Add a dropdown for filtering by user reviews
reviews_filter = st.selectbox("Select review category:", ["None", "Mostly Positive", "Very Positive", "Overwhelmingly Positive"])

st.write(df2['all_reviews'].unique())


# If the user has entered a language, proceed with the recommendation
if language:
    # Filter games that contain the specified language
    matching_games = df2[df2['languages'].str.contains(language, na=False)]

    # Check if a review filter was selected and apply it
    if reviews_filter != "None":
        # Make sure the review filter is case-insensitive
        matching_games = matching_games[matching_games['all_reviews'].str.contains(reviews_filter, case=False, na=False)]
    
    # Display the top 20 games that match the language and review filter
    if not matching_games.empty:
        st.write(f"Games that support the language '{language}' with '{reviews_filter}' reviews:")
        st.table(matching_games[['name', 'languages', 'all_reviews']].head(20))
    else:
        st.write(f"No games found that support the language '{language}' with '{reviews_filter}' reviews.")
