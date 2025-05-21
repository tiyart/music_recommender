import streamlit as st
import pandas as pd
import joblib

# Load models
genre_model = joblib.load("mbti_genre_model.pkl")
artist_model = joblib.load("artist_model.pkl")

# Encoders
mbti_map = {
    ('Extraversion', 'Sensing', 'Thinking', 'Judging'): 'ESTJ',
    ('Extraversion', 'Sensing', 'Thinking', 'Perceiving'): 'ESTP',
    ('Extraversion', 'Sensing', 'Feeling', 'Judging'): 'ESFJ',
    ('Extraversion', 'Sensing', 'Feeling', 'Perceiving'): 'ESFP',
    ('Extraversion', 'Intuition', 'Thinking', 'Judging'): 'ENTJ',
    ('Extraversion', 'Intuition', 'Thinking', 'Perceiving'): 'ENTP',
    ('Extraversion', 'Intuition', 'Feeling', 'Judging'): 'ENFJ',
    ('Extraversion', 'Intuition', 'Feeling', 'Perceiving'): 'ENFP',
    ('Introversion', 'Sensing', 'Thinking', 'Judging'): 'ISTJ',
    ('Introversion', 'Sensing', 'Thinking', 'Perceiving'): 'ISTP',
    ('Introversion', 'Sensing', 'Feeling', 'Judging'): 'ISFJ',
    ('Introversion', 'Sensing', 'Feeling', 'Perceiving'): 'ISFP',
    ('Introversion', 'Intuition', 'Thinking', 'Judging'): 'INTJ',
    ('Introversion', 'Intuition', 'Thinking', 'Perceiving'): 'INTP',
    ('Introversion', 'Intuition', 'Feeling', 'Judging'): 'INFJ',
    ('Introversion', 'Intuition', 'Feeling', 'Perceiving'): 'INFP'
}

mbti_list = list(mbti_map.values())
tempo_map = {'Slow/Calm': 0, 'Medium': 1, 'Fast/Energetic': 2}

st.title("🎧 Music Personality Recommender")

with st.form("personality_form"):
    st.subheader("Tell us about yourself")
    st.text_input("Your personality or description (optional):")

    social = st.radio("When it comes to socialising:",
                      ["Introversion", "Extraversion"])
    info = st.radio("When processing information:",
                    ["Sensing", "Intuition"])
    decision = st.radio("When making decisions:",
                        ["Thinking", "Feeling"])
    planning = st.radio("When planning my day or tasks:",
                        ["Judging", "Perceiving"])
    tempo = st.selectbox("Preferred music tempo:", list(tempo_map.keys()))

    submit = st.form_submit_button("Recommend")

if submit:
    mbti = mbti_map.get((social, info, decision, planning), "Unknown")

    if mbti == "Unknown":
        st.error("Unable to determine MBTI type.")
    else:
        mbti_encoded = mbti_list.index(mbti)
        tempo_encoded = tempo_map[tempo]

        genre_pred = genre_model.predict([[mbti_encoded, tempo_encoded]])[0]
        artist_pred = artist_model.predict([[genre_pred, tempo_encoded]])[0]

        # Dummy maps (replace with your label maps if available)
        genre_labels = {
            0: "Pop", 1: "Rock", 2: "Indie", 3: "Hip-Hop", 4: "Classical",
            5: "Jazz", 6: "Electronic"
        }
        artist_labels = {
            0: "Taylor Swift, Olivia Rodrigo",
            1: "Imagine Dragons, Coldplay",
            2: "Arctic Monkeys, The Smiths",
            3: "Kendrick Lamar, Drake",
            4: "Mozart, Yo-Yo Ma",
            5: "Miles Davis, Norah Jones",
            6: "Daft Punk, Calvin Harris"
        }

        genre_name = genre_labels.get(genre_pred, "Unknown Genre")
        artist_group = artist_labels.get(artist_pred, "Unknown Artists")

        st.markdown(f"### As an **{mbti}**, you're matched with **{genre_name}** music.")
        st.markdown(f"🎤 You may enjoy artists such as: **{artist_group}**")
