import streamlit as st
from predictor import TennisPredictor

st.set_page_config(page_title="Tennis Predictor", layout="wide")
st.title("🎾 ATP Match Predictor")


@st.cache_resource
def load_predictor():
    return TennisPredictor()


pred = load_predictor()
all_players = pred.get_all_players()

st.header("Predict Match Outcome")

# ── Player 1 (was Player A) ──────────────────────────────────────────────
col_left, col_right = st.columns(2)

with col_left:
    st.subheader("Player 1")
    search_1 = st.text_input("Search Player 1", key="search_1",
                             placeholder="e.g. Djokovic")
    filtered_1 = ([p for p in all_players if search_1.lower() in p.lower()][:5]
                  if search_1 else all_players[:100])
    if not filtered_1:
        st.warning("No players found — try a different spelling")
        filtered_1 = all_players[:100]
    player_1 = st.selectbox("Select Player 1", filtered_1, key="select_1")

# ── Player 2 (was Player B) ──────────────────────────────────────────────
with col_right:
    st.subheader("Player 2")
    search_2 = st.text_input("Search Player 2", key="search_2",
                             placeholder="e.g. Nadal")
    filtered_2 = ([p for p in all_players if search_2.lower() in p.lower()][:5]
                  if search_2 else all_players[:100])
    if not filtered_2:
        st.warning("No players found — try a different spelling")
        filtered_2 = all_players[:100]
    player_2 = st.selectbox("Select Player 2", filtered_2, key="select_2")

# Swap button
if st.button("🔄 Swap Players"):
    # Use session state to swap
    st.session_state.player_1 = player_2
    st.session_state.player_2 = player_1
    st.rerun()

# ── Match context ─────────────────────────────────────────────────────────
col1, col2, col3 = st.columns(3)

with col1:
    surface = st.selectbox("Surface", ["Hard", "Clay", "Grass"], index=1)

with col2:
    tourn_labels = ["ATP 250/500", "Masters 1000", "Grand Slam",
                    "ATP Finals", "Davis Cup"]
    tourn_values = ["A", "M", "G", "F", "D"]
    tourn_idx = st.selectbox("Tournament Level", range(len(tourn_labels)),
                             format_func=lambda i: tourn_labels[i],
                             index=2)
    tournament = tourn_values[tourn_idx]

with col3:
    round_labels = ["R128", "R64", "R32", "R16", "QF", "SF", "Final"]
    round_values = ["R128", "R64", "R32", "R16", "QF", "SF", "F"]
    round_idx = st.selectbox("Round", range(len(round_labels)),
                             format_func=lambda i: round_labels[i],
                             index=6)
    round_num = round_values[round_idx]

# ── Predict ───────────────────────────────────────────────────────────────
st.divider()

if st.button("🔮 Predict Match", use_container_width=True, type="primary"):

    if player_1 == player_2:
        st.error("Please select two different players.")
    else:
        with st.spinner("Computing prediction..."):
            result = pred.predict_match(
                player_1, player_2, surface, tournament, round_num
            )

        if result.get('error'):
            st.error(f"Error: {result['error']}")
        else:
            # ── Result cards ─────────────────────────────────────────────
            st.subheader("Prediction Result")

            c1, c2, c3 = st.columns([2, 1, 2])

            with c1:
                st.metric(
                    label=f"🎾 {player_1}",
                    value=f"{result['p1_win_prob']:.1%}",
                    delta=f"{result['p1_win_prob'] - 0.5:+.1%} vs 50%"
                )

            with c2:
                st.markdown(
                    f"<h2 style='text-align:center; padding-top:10px'>VS</h2>",
                    unsafe_allow_html=True
                )

            with c3:
                st.metric(
                    label=f"🎾 {player_2}",
                    value=f"{result['p2_win_prob']:.1%}",
                    delta=f"{result['p2_win_prob'] - 0.5:+.1%} vs 50%"
                )

            # Winner call
            winner = player_1 if result['p1_win_prob'] > 0.5 else player_2
            win_prob = max(result['p1_win_prob'], result['p2_win_prob'])
            st.success(f"**Predicted winner: {winner}** ({win_prob:.1%})")

            # Confidence + model breakdown
            conf_color = {"High": "🟢", "Medium": "🟡", "Low": "🔴"}
            st.info(
                f"{conf_color.get(result['confidence'], '⚪')} "
                f"Confidence: **{result['confidence']}**  |  "
                f"XGBoost: {result['xgb_prob']:.1%}  |  "
                f"LightGBM: {result['lgb_prob']:.1%}  |  "
                f"Ensemble: {result['p1_win_prob']:.1%} for {player_1}"
            )

            # Context reminder
            st.caption(
                f"Context: {tourn_labels[tourn_idx]} · {surface} · "
                f"{round_labels[round_idx]}"
            )

            # Quick H2H summary (still works with old method)
            st.divider()
            h2h = pred.get_h2h(player_1, player_2)
            if h2h['total_matches'] > 0:
                st.markdown(f"**H2H record:** {h2h['h2h_record']} "
                            f"({h2h['total_matches']} matches)")
                for surf, rec in h2h['surface_breakdown'].items():
                    st.markdown(f"&nbsp;&nbsp;&nbsp;• {surf}: {rec}",
                                unsafe_allow_html=True)
            else:
                st.markdown("**H2H:** No prior meetings in database")