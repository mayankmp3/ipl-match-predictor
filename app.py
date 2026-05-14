import streamlit as st
import pickle
import pandas as pd
import time

# Load files
model = pickle.load(open("model.pkl", "rb"))
le = pickle.load(open("encoder.pkl", "rb"))
team_wins = pickle.load(open("team_wins.pkl", "rb"))
venue_wins = pickle.load(open("venue_wins.pkl", "rb"))

st.set_page_config(page_title="IPL Predictor", layout="wide")

# 🔥 PREMIUM CSS
st.markdown("""
<style>

/* Background */
.stApp {
    background-image: url("https://images.unsplash.com/photo-1593349480506-8433634cdcbe");
    background-size: cover;
    background-position: center;
}

/* Animated Gradient */
.stApp::before {
    content: "";
    position: fixed;
    width: 100%;
    height: 100%;
    top: 0;
    left: 0;
    background: linear-gradient(
        270deg,
        rgba(0,0,0,0.7),
        rgba(255,75,43,0.4),
        rgba(255,65,108,0.4),
        rgba(0,0,0,0.7)
    );
    background-size: 600% 600%;
    animation: gradientMove 12s ease infinite;
    z-index: -1;
}

/* Animation */
@keyframes gradientMove {
    0% { background-position: 0% 50%; }
    50% { background-position: 100% 50%; }
    100% { background-position: 0% 50%; }
}

/* Glass Card */
.block-container {
    background: rgba(0, 0, 0, 0.6);
    padding: 30px;
    border-radius: 20px;
    backdrop-filter: blur(10px);
}

/* Text */
h1, h2, h3, label {
    color: white;
}

/* Button */
.stButton>button {
    background: linear-gradient(90deg, #ff4b2b, #ff416c);
    color: white;
    border-radius: 10px;
    height: 3em;
    width: 100%;
    font-size: 18px;
    border: none;
    transition: 0.3s;
}

.stButton>button:hover {
    transform: scale(1.05);
    box-shadow: 0px 0px 15px #ff416c;
}

/* Result Box */
.result-box {
    background: linear-gradient(90deg,#ff4b2b,#ff416c);
    padding:25px;
    border-radius:15px;
    text-align:center;
    box-shadow: 0px 0px 25px rgba(255,65,108,0.8);
}

</style>
""", unsafe_allow_html=True)

st.title("🏏 IPL Match Winner Predictor")
st.write("### Predict match outcomes using Machine Learning")

# 🏏 Logos (LOCAL)
team_logos = {
    'Mumbai Indians': 'images/mi.png',
    'Chennai Super Kings': 'images/csk.png',
    'Royal Challengers Bangalore': 'images/rcb.png',
    'Kolkata Knight Riders': 'images/kkr.png',
    'Delhi Capitals': 'images/dc.png',
    'Punjab Kings': 'images/pbks.png',
    'Kings XI Punjab': 'images/pbks.png',
    'Rajasthan Royals': 'images/rr.png',
    'Sunrisers Hyderabad': 'images/srh.png'
}

teams = list(team_wins.keys())
venues = list(venue_wins.keys())

# Inputs
col1, col2 = st.columns(2)

with col1:
    team1 = st.selectbox("Select Team 1", teams)
    team2 = st.selectbox("Select Team 2", teams)

with col2:
    toss_winner = st.selectbox("Toss Winner", [team1, team2])
    toss_decision = st.selectbox("Toss Decision", ['bat', 'field'])

venue = st.selectbox("Select Venue", venues)

# VS Layout
colA, colB, colC = st.columns([1,1,1])

with colA:
    st.image(team_logos.get(team1, 'images/default.png'), width=120)

with colB:
    st.markdown("<h2 style='text-align:center;'>VS</h2>", unsafe_allow_html=True)

with colC:
    st.image(team_logos.get(team2, 'images/default.png'), width=120)

# Feature Engineering
team1_strength = team_wins.get(team1, 0)
team2_strength = team_wins.get(team2, 0)

toss_win = 1 if toss_winner == team1 else 0
bat_first = 1 if toss_decision == 'bat' else 0

strength_diff = team1_strength - team2_strength
venue_adv = venue_wins.get(venue, 0)

h2h = abs(team1_strength - team2_strength)
team1_encoded = le.transform([team1])[0]
team2_encoded = le.transform([team2])[0]

# Prediction
if st.button("🚀 Predict Winner"):

    with st.spinner("Analyzing match... ⚡"):
        time.sleep(2)

    data = pd.DataFrame([[
    team1_encoded,
    team2_encoded,
    toss_win,
    bat_first,
    team1_strength,
    team2_strength,
    strength_diff,
    h2h
]], columns=[
    'team1',
    'team2',
    'toss_win',
    'bat_first',
    'team1_strength',
    'team2_strength',
    'strength_diff',
    'h2h'
])
    prediction = model.predict(data)
    prob = model.predict_proba(data)

    winner = le.inverse_transform(prediction)[0]
    confidence = max(prob[0]) * 100

    st.markdown(f"""
    <div class="result-box">
        <h1>🏆 {winner} Wins!</h1>
        <h3>Confidence: {confidence:.2f}%</h3>
    </div>
    """, unsafe_allow_html=True)