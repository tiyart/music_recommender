import streamlit as st
st.set_page_config(
    page_title="Music Genre Recommender",
    page_icon="🎧",
    layout="centered",
)

import pandas as pd
import joblib

artist_model = joblib.load("artist_model.pkl")
mbti_genre_model = joblib.load("mbti_genre_model.pkl")
le_genre = joblib.load("le_genre.pkl")
le_artist = joblib.load("le_artist.pkl")
le_mbti = joblib.load("le_mbti.pkl")

st.markdown("""
<style>
.stSelectbox, .stRadio, .stButton {
    margin-bottom: 0 !important;
}
</style>
""", unsafe_allow_html=True)

st.title("🎧 Music Genre Recommender")
st.markdown("Answer a few personality questions and we’ll recommend a music genre and some artists you might love!")

st.subheader("Tell us about yourself")

def determine_mbti(social, info, decision, planning):
    mbti = ""
    mbti += "E" if "Extraversion" in social else "I"
    mbti += "N" if "Intuition" in info else "S"
    mbti += "T" if "Thinking" in decision else "F"
    mbti += "J" if "Judging" in planning else "P"
    return mbti

genre_groups = [
    "Classical/Jazz", "Indie", "Lofi/Melody", "Pop", "R&B", "Rap", "Rock"
]

artist_groups = [
    "ASAP Rocky, Travis Scott, Drake, Kendrick Lamar",
    "Arctic Monkeys, The Smiths, The Pixies, Florence & The Machine",
    "Chris Brown, Daniel Caesar, Brent Faiyaz, Frank Ocean",
    "Green Day, The Beatles, Nirvana, Guns 'n Roses",
    "Mozart, Miles Davis, Beethoven, Frank Sinatra",
    "Shiloh Dynasty, Sagun, Joji, Yot Club",
    "Taylor Swift, Beyonce, Michael Jackson, Sabrina Carpenter"
]

tempo_map = {'Slow/Calm': 0, 'Medium': 1, 'Fast/Energetic': 2}

# Inputs (live-updating)
social = st.selectbox(
    "When it comes to socialising:",
    [
        "I enjoy large social gatherings and meeting new people (Extraversion)",
        "I prefer spending time alone or with a small group of close friends (Introversion)"
    ]
)
info = st.selectbox(
    "When processing information:",
    [
        "I focus on patterns, ideas, and possibilities (Intuition)",
        "I trust facts, data, and real experiences (Sensing)"
    ]
)
decision = st.selectbox(
    "When making decisions:",
    [
        "I prioritise logic and objectivity (Thinking)",
        "I consider emotions and values (Feeling)"
    ]
)
planning = st.selectbox(
    "When planning my day or tasks:",
    [
        "I like structure, planning, and sticking to schedules (Judging)",
        "I prefer being spontaneous and flexible (Perceiving)"
    ]
)
tempo = st.radio(
    "Preferred music tempo:",
    ["Slow/Calm", "Medium", "Fast/Energetic"],
    index=1
)

submit = st.button("🎶 Recommend")

if submit:
    user_mbti = determine_mbti(social, info, decision, planning)
    mbti_encoded = le_mbti.transform([user_mbti])[0]
    genre_input = pd.DataFrame([[mbti_encoded, tempo_map[tempo]]], columns=["MBTI_encoded", "Tempo_Ordinal"])
    genre_pred_encoded = mbti_genre_model.predict(genre_input)[0]

    predicted_genre = genre_groups[genre_pred_encoded] if genre_pred_encoded < len(genre_groups) else "Unknown Genre"

    genre_encoded = le_genre.transform([predicted_genre])[0] if predicted_genre != "Unknown Genre" else 0
    artist_input = pd.DataFrame([[genre_encoded, tempo_map[tempo]]], columns=["Genre_encoded", "Tempo_Ordinal"])
    artist_pred_encoded = artist_model.predict(artist_input)[0]

    predicted_artist_group = artist_groups[artist_pred_encoded] if artist_pred_encoded < len(artist_groups) else "Unknown Artist Group"

    st.success(f"✨ As an **{user_mbti}**, you're matched with **{predicted_genre}** music!")
    st.warning(f"🎤 We think you'll enjoy artists such as **{predicted_artist_group}**.")
else:
    st.info("Click 'Recommend' to get your personalised music suggestion :)")
