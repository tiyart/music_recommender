import streamlit as st
import pandas as pd
import joblib

# Load the trained models and encoders
artist_model = joblib.load("artist_model.pkl")
mbti_genre_model = joblib.load("mbti_genre_model.pkl")
le_genre = joblib.load("le_genre.pkl")
le_artist = joblib.load("le_artist.pkl")
le_mbti = joblib.load("le_mbti.pkl")

# Set page title and description with emoji
st.title("🎧 Music Genre Recommender")
st.markdown("Answer a few personality questions and we’ll recommend a music genre and some artists you might love!")

# Section for personality inputs
st.header("🧠 Tell us about yourself")

# Dropdowns for personality traits to determine MBTI
social = st.selectbox(
    "When it comes to socialising:",
    [
        "I enjoy large social gatherings and meeting new people (Extraversion)",
        "I prefer smaller groups or alone time (Introversion)"
    ]
)
info = st.selectbox(
    "When processing information:",
    [
        "I focus on patterns, ideas, and possibilities (Intuition)",
        "I focus on facts, details, and reality (Sensing)"
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
        "I prefer flexibility and spontaneity (Perceiving)"
    ]
)

# Radio buttons for tempo preference
st.markdown("**Preferred music tempo:**")
tempo = st.radio(
    "",
    ["Slow/Calm", "Medium", "Fast/Energetic"]
)

# Map tempo to ordinal value
tempo_map = {'Slow/Calm': 0, 'Medium': 1, 'Fast/Energetic': 2}
tempo_ordinal = tempo_map[tempo]

# Determine MBTI type from user inputs
def determine_mbti(social, info, decision, planning):
    mbti = ""
    mbti += "E" if "Extraversion" in social else "I"
    mbti += "N" if "Intuition" in info else "S"
    mbti += "T" if "Thinking" in decision else "F"
    mbti += "J" if "Judging" in planning else "P"
    return mbti

# Genre mapping based on provided encoding values
genre_groups = [
    "Classical/Jazz",  # Encoded Value 0
    "Indie",           # Encoded Value 1
    "Lofi/Melody",     # Encoded Value 2
    "Pop",             # Encoded Value 3
    "R&B",             # Encoded Value 4
    "Rap",             # Encoded Value 5
    "Rock"             # Encoded Value 6
]

# Artist groups mapping based on provided encoding values
artist_groups = [
    "ASAP Rocky, Travis Scott, Drake, Kendrick Lamar",  # Encoded Value 0
    "Arctic Monkeys, The Smiths, The Pixies, Florence & The Machine",  # Encoded Value 1
    "Chris Brown, Daniel Caesar, Brent Faiyaz, Frank Ocean",  # Encoded Value 2
    "Green Day, The Beatles, Nirvana, Guns 'n Roses",  # Encoded Value 3
    "Mozart, Miles Davis, Beethoven, Frank Sinatra",  # Encoded Value 4
    "Shiloh Dynasty, Sagun, Joji, Yot Club",  # Encoded Value 5
    "Taylor Swift, Beyonce, Michael Jackson, Sabrina Carpenter"  # Encoded Value 6
]

# Button to trigger recommendation
if st.button("Recommend"):
    # Get MBTI type
    user_mbti = determine_mbti(social, info, decision, planning)
    
    # Encode MBTI for genre prediction
    mbti_encoded = le_mbti.transform([user_mbti])[0]
    
    # Predict genre based on MBTI and tempo
    genre_input = pd.DataFrame([[mbti_encoded, tempo_ordinal]], columns=["MBTI_encoded", "Tempo_Ordinal"])
    genre_pred_encoded = mbti_genre_model.predict(genre_input)[0]
    
    # Map the predicted genre encoding to the corresponding genre name
    if genre_pred_encoded < len(genre_groups):
        predicted_genre = genre_groups[genre_pred_encoded]
    else:
        predicted_genre = "Unknown Genre"  # Fallback in case of mismatch
    
    # Predict artist group based on genre and tempo
    genre_encoded = le_genre.transform([predicted_genre])[0] if predicted_genre != "Unknown Genre" else 0
    artist_input = pd.DataFrame([[genre_encoded, tempo_ordinal]], columns=["Genre_encoded", "Tempo_Ordinal"])
    artist_pred_encoded = artist_model.predict(artist_input)[0]
    
    # Map the predicted artist encoding to the corresponding artist group
    if artist_pred_encoded < len(artist_groups):
        predicted_artist_group = artist_groups[artist_pred_encoded]
    else:
        predicted_artist_group = "Unknown Artist Group"  # Fallback in case of mismatch
    
    # Display personalized recommendation
    st.success(f"As an {user_mbti}, you're matched with **{predicted_genre}** music ✨. You may enjoy artists such as **{predicted_artist_group}**.")
else:
    st.info("Click 'Recommend' to get your personalized music suggestion!")

# Add a footer
st.markdown("---")
st.write("Built with Streamlit | Personalized music insights based on personality")
