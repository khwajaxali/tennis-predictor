import streamlit as st
from predictor import TennisPredictor

st.set_page_config(page_title="Tennis Predictor", layout="wide")
st.title("🎾 ATP Match Predictor")

# Load predictor once
@st.cache_resource
def load_predictor():
    return TennisPredictor()

pred = load_predictor()
all_players = pred.get_all_players()

st.header("Predict Match Outcome")

# Search box for Player A
st.subheader("Player A")
search_a = st.text_input("Search Player A", key="search_a")

if search_a:
    matches_a = [p for p in all_players if search_a.lower() in p.lower()][:5]
    player_a = st.selectbox("Select Player A", matches_a, key="select_a")
else:
    player_a = st.selectbox("Select Player A", all_players[:100], key="select_a")

# Search box for Player B
st.subheader("Player B")
search_b = st.text_input("Search Player B", key="search_b")

if search_b:
    matches_b = [p for p in all_players if search_b.lower() in p.lower()][:5]
    player_b = st.selectbox("Select Player B", matches_b, key="select_b")
else:
    player_b = st.selectbox("Select Player B", all_players[:100], key="select_b")

# Other params
col1, col2, col3 = st.columns(3)

with col1:
    surface = st.selectbox("Surface", ["Hard", "Clay", "Grass"])

with col2:
    tournament = st.selectbox("Tournament Level",
                             {"ATP 250/500": "A", 
                              "Masters 1000": "M",
                              "Grand Slam": "G", 
                              "ATP Finals": "F", 
                              "Davis Cup": "D"},
                             index=2)  # Default: Grand Slam

with col3:
    round_num = st.selectbox("Round",
                            {"R128": "R128", "R64": "R64", "R32": "R32",
                             "R16": "R16", "QF": "QF", "SF": "SF",
                             "Final": "F"},
                            index=6)  # Default: Final

# Predict button
if st.button("🔮 Predict", use_container_width=True):
    with st.spinner("Computing..."):
        result = pred.predict_match(player_a, player_b, surface, tournament, round_num)
    
    if result.get('error'):
        st.error(f"Error: {result['error']}")
    else:
        col_a, col_b = st.columns(2)
        
        with col_a:
            st.metric(f"{player_a} Win %",
                     f"{result['player_a_win_prob']:.1%}")
        
        with col_b:
            st.metric(f"{player_b} Win %",
                     f"{result['player_b_win_prob']:.1%}")
        
        st.info(f"**Confidence:** {result['confidence']}")
        
        st.write(f"**Model breakdown:** XGBoost {result['xgb_prob']:.1%} | LightGBM {result['lgb_prob']:.1%}")