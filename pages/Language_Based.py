import streamlit as st
import pandas as pd

# Load CSV data
path = 'games.csv'
column_names = ['name', 'desc_snippet', 'recent_reviews', 'all_reviews', 'release_date',
                'popular_tags', 'game_details', 'languages', 'genre', 'game_description',
                'mature_content', 'original_price', 'discount_price']   
df = pd.read_csv(path, names=column_names, encoding='ISO-8859-1', low_memory=True)

# Selecting only relevant columns
df1 = df[['name', 'languages', 'popular_tags', 'genre', 'original_price']]

# Delete missing values
df2 = pd.DataFrame(df1.dropna())

# Streamlit app
st.title("🎮Language Based Recommender🎮")
st.write("🔎 Find similar games for you based on language 🔎")

# Asking for user input via Streamlit for language search
language = st.text_input("Please enter a language to find games with that language:").lower()

# Ensure all languages in the dataset are lowercase for comparison
df2['languages'] = df2['languages'].str.lower()

# If the user has entered a language, proceed with the recommendation
if language:
    # Filter games that contain the specified language
    matching_games = df2[df2['languages'].str.contains(language, na=False)]

    # Check if any games were found
    if not matching_games.empty:
        st.write(f"Games that support the language '{language}':")
        st.table(matching_games[['name', 'languages']])
    else:
        st.write(f"No games found that support the language '{language}'.")
