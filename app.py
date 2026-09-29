import streamlit as st
import pandas as pd
import joblib

# 1. Load the saved model bundle
bundle = joblib.load('anime_model_bundle.pkl')
struct_model = bundle['struct_model']
text_model = bundle['text_model']
hybrid_model = bundle['hybrid_model']
feature_names = bundle['features']
tfidf = bundle['tfidf']
svd = bundle['svd']

# 2. UI Layout & Headers
st.title("Anime Score Predictor")
st.write("Academic Evaluation: Comparing Structured vs. Unstructured vs. Hybrid Models.")

# --- Model Selection Toggle ---
model_mode = st.selectbox(
    "Select Model Architecture",
    ["Structured Data Only (Tabular)", "Unstructured Data Only (Synopsis SVD)", "Hybrid Model (Combined)"]
)

# --- Inputs ---
st.subheader("Input Features")
popularity = st.number_input("Popularity Rank", min_value=1, max_value=20000, value=1500)
episodes = st.number_input("Number of Episodes", min_value=1, max_value=2000, value=12)
season_year = st.number_input("Season Year", min_value=1950, max_value=2030, value=2023)

synopsis_input = st.text_area(
    "Anime Synopsis (Unstructured Text)",
    value="A thrilling adventure about a hero fighting to save the world."
)

# Extract genre columns dynamically from feature names (excluding numerical and text SVD cols)
non_genre_cols = ['popularity', 'episodes', 'seasonYear']
genre_cols = [col for col in feature_names if col not in non_genre_cols and not col.startswith('text_svd_')]

selected_genres = st.multiselect(
    "Select Genres / Themes / Tags",
    options=genre_cols
)

# 3. Prediction Logic
if st.button("Run Prediction"):
    # Build complete input dictionary
    input_dict = {
        'popularity': popularity,
        'episodes': episodes,
        'seasonYear': season_year
    }

    # Map genres
    for col in genre_cols:
        input_dict[col] = 1 if col in selected_genres else 0

    # Process text features via tfidf and svd
    text_vec = tfidf.transform([synopsis_input])
    text_red = svd.transform(text_vec)
    for i in range(10):
        input_dict[f'text_svd_{i}'] = text_red[0, i]

    # Create full input dataframe aligned with model features
    input_data = pd.DataFrame([input_dict])
    input_data = input_data.reindex(columns=feature_names, fill_value=0)

    # Route prediction based on selected mode
    if model_mode == "Structured Data Only (Tabular)":
        # Keep only structured columns for the structured model
        struct_cols = [c for c in feature_names if not c.startswith('text_svd_')]
        X_input_struct = input_data[struct_cols]
        prediction = struct_model.predict(X_input_struct)[0]
        st.info("📊 Running **Structured-Only** Model (Baseline tabular features)")

    elif model_mode == "Unstructured Data Only (Synopsis SVD)":
        # Keep only text SVD columns for the text model
        text_cols = [f'text_svd_{i}' for i in range(10)]
        X_input_text = input_data[text_cols]
        prediction = text_model.predict(X_input_text)[0]
        st.info("📝 Running **Unstructured-Only** Model (Synopsis text features only)")

    else:
        # Hybrid model uses the full dataset features
        prediction = hybrid_model.predict(input_data)[0]
        st.success("🔥 Running **Hybrid Model** (Combined metadata + text features)")

    # Display the result prominently
    st.metric(label="Predicted Community Mean Score", value=f"{prediction:.2f} / 100")
