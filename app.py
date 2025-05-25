import streamlit as st
import pandas as pd
import joblib

artist_model = joblib.load("artist_model.pkl")
mbti_genre_model = joblib.load("mbti_genre_model.pkl")
le_genre = joblib.load("le_genre.pkl")
le_artist = joblib.load("le_artist.pkl")
le_mbti = joblib.load("le_mbti.pkl")

st.title("🎧 Music Genre Recommender")
st.markdown("Answer a few personality questions and we’ll recommend a music genre and some artists you might love!")

st.header("Tell us about yourself")

st.markdown("<div style='font-size:15px;'>When it comes to socialising:</div>", unsafe_allow_html=True)
social = st.selectbox(
    "",
    [
        "I enjoy large social gatherings and meeting new people (Extraversion)",
        "I prefer smaller groups or alone time (Introversion)"
    ]
)
st.markdown("<div style='font-size:15px;'>When processing information:</div>", unsafe_allow_html=True)
info = st.selectbox(
    "",
    [
        "I focus on patterns, ideas, and possibilities (Intuition)",
        "I focus on facts, details, and reality (Sensing)"
    ]
)
st.markdown("<div style='font-size:15px;'>When making decisions:</div>", unsafe_allow_html=True)
decision = st.selectbox(
    "",
    [
        "I prioritise logic and objectivity (Thinking)",
        "I consider emotions and values (Feeling)"
    ]
)
st.markdown("<div style='font-size:15px;'>When planning my day or tasks:</div>", unsafe_allow_html=True)
planning = st.selectbox(
    "",
    [
        "I like structure, planning, and sticking to schedules (Judging)",
        "I prefer flexibility and spontaneity (Perceiving)"
    ]
)

st.markdown("<div style='font-size: 15px; margin-bottom: -25px;'>Preferred music tempo:</div>", unsafe_allow_html=True)
tempo = st.radio(
    "",
    ["Slow/Calm", "Medium", "Fast/Energetic"],
    index=1
)

tempo_map = {'Slow/Calm': 0, 'Medium': 1, 'Fast/Energetic': 2}
tempo_ordinal = tempo_map[tempo]

def determine_mbti(social, info, decision, planning):
    mbti = ""
    mbti += "E" if "Extraversion" in social else "I"
    mbti += "N" if "Intuition" in info else "S"
    mbti += "T" if "Thinking" in decision else "F"
    mbti += "J" if "Judging" in planning else "P"
    return mbti

genre_groups = [
    "Classical/Jazz",  
    "Indie",           
    "Lofi/Melody",     
    "Pop",        
    "R&B",     
    "Rap",       
    "Rock"  
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

if st.button("🎶 Recommend"):

    user_mbti = determine_mbti(social, info, decision, planning)
    mbti_encoded = le_mbti.transform([user_mbti])[0]
    genre_input = pd.DataFrame([[mbti_encoded, tempo_ordinal]], columns=["MBTI_encoded", "Tempo_Ordinal"])
    genre_pred_encoded = mbti_genre_model.predict(genre_input)[0]

    if genre_pred_encoded < len(genre_groups):
        predicted_genre = genre_groups[genre_pred_encoded]
    else:
        predicted_genre = "Unknown Genre"

    genre_encoded = le_genre.transform([predicted_genre])[0] if predicted_genre != "Unknown Genre" else 0
    artist_input = pd.DataFrame([[genre_encoded, tempo_ordinal]], columns=["Genre_encoded", "Tempo_Ordinal"])
    artist_pred_encoded = artist_model.predict(artist_input)[0]

    if artist_pred_encoded < len(artist_groups):
        predicted_artist_group = artist_groups[artist_pred_encoded]
    else:
        predicted_artist_group = "Unknown Artist Group"

    st.success(f"✨ As an **{user_mbti}**, you're matched with **{predicted_genre}** music!")
    st.info(f"🎤 We think you'll enjoy artists such as **{predicted_artist_group}**.")
else:
    st.info("Click 'Recommend' to get your personalised music suggestion :)")
