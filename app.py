import streamlit as st
import pandas as pd
import joblib

bundle = joblib.load('anime_model_bundle.pkl')
model = bundle['model']
feature_names = bundle['features']

st.title("Anime Score Predictor")
st.write("This model is trained to predict a score based on numerical values like popularity, number of episodes, year, and the binarization of genres, tags, and studios. This model was trained using AniList\'s database.")

popularity = st.number_input("Popularity Rank", min_value=1, max_value=500000, value=1500)
episodes = st.number_input("Number of Episodes", min_value=1, max_value=2000, value=12)
season_year = st.number_input("Season Year", min_value=1950, max_value=2030, value=2023)

non_genre_cols = ['popularity', 'episodes', 'seasonYear']
genre_cols = [col for col in feature_names if col not in non_genre_cols]

selected_genres = st.multiselect(
    "Select Genres / Themes",
    options=genre_cols
)

if st.button("Predict Score"):
    input_dict = {
        'popularity': popularity,
        'episodes': episodes,
        'seasonYear': season_year
    }

    for col in genre_cols:
        input_dict[col] = 1 if col in selected_genres else 0
    input_data = pd.DataFrame([input_dict])
    input_data = input_data.reindex(columns=feature_names, fill_value=0)

    prediction = model.predict(input_data)[0]
    st.success(f"Predicted Community Mean Score: **{prediction:.2f} / 100**")
