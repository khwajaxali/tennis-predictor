import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
from predictor import TennisPredictor

st.set_page_config(page_title="Tennis Predictor", layout="wide")
st.title("🎾 ATP Match Predictor")

# Calibration error from retraining
CALIBRATION_ERROR = 0.0667


@st.cache_resource
def load_predictor():
    return TennisPredictor()


pred = load_predictor()
all_players = pred.get_all_players()

st.header("Predict Match Outcome")

# ── Player 1 ──────────────────────────────────────────────────────────────
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

# ── Player 2 ──────────────────────────────────────────────────────────────
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
                player_1, player_2, surface, tournament, round_num,
                include_explanation=True,
                include_baseline=True
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

            # Calibration warning
            st.caption(
                f"📊 Model calibration error: ~{CALIBRATION_ERROR*100:.1f}% — "
                f"predictions are typically within ±{CALIBRATION_ERROR*100:.1f}% of actual win rates."
            )

            # ── Expanded Context Card ────────────────────────────────────
            st.caption(
                f"Context: {tourn_labels[tourn_idx]} · {surface} · "
                f"{round_labels[round_idx]}"
            )
            context_notes = {
                "Grand Slam": "🏆 Highest prestige – best‑of‑5, more pressure",
                "Masters 1000": "💪 Strong field, best‑of‑3, high intensity",
                "ATP 250/500": "📉 Lower stakes, potential variable motivation",
                "ATP Finals": "⭐ Elite 8, indoor conditions",
                "Davis Cup": "🇺🇳 National pride, team environment",
            }
            st.caption(context_notes.get(tourn_labels[tourn_idx], ""))

            # ── Enhanced Analysis ──────────────────────────────────────
            st.divider()
            st.subheader("📝 Analysis")
            analysis = pred.generate_detailed_analysis(result)
            st.markdown(analysis)

            # ── Player Profile Comparison ───────────────────────────────
            st.divider()
            st.subheader("📊 Player Profiles")

            p1_stats = pred._get_player_stats(player_1, surface)
            p2_stats = pred._get_player_stats(player_2, surface)

            cols = st.columns(4)
            metrics = [
                ("Rank", "rank"),
                ("Elo", "elo"),
                ("Win Rate (surface)", "win_rate_surf"),
                ("Days Since Last", "days_since_last"),
            ]
            for i, (label, key) in enumerate(metrics):
                with cols[i]:
                    st.metric(
                        label=label,
                        value=f"{p1_stats[key]:.1f}",
                        delta=f"vs {p2_stats[key]:.1f}"
                    )

            # ── Recent Form Badges ──────────────────────────────────────
            st.divider()
            st.subheader("📈 Recent Form (last 5 matches)")
            form1 = pred.get_recent_form(player_1, surface, n=5)
            form2 = pred.get_recent_form(player_2, surface, n=5)
            colf1, colf2 = st.columns(2)
            with colf1:
                record1 = form1['record'] if form1['n_matches'] >= 3 else f"{form1['record']} (insufficient data)"
                st.markdown(f"**{player_1}**")
                st.markdown(f"Record: {record1}  Streak: {form1['streak']}")
            with colf2:
                record2 = form2['record'] if form2['n_matches'] >= 3 else f"{form2['record']} (insufficient data)"
                st.markdown(f"**{player_2}**")
                st.markdown(f"Record: {record2}  Streak: {form2['streak']}")

            # ── H2H Summary ──────────────────────────────────────────────
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

            # ── SHAP Explanation ──────────────────────────────────────────
            if 'explanation' in result and result['explanation'] and not result['explanation'].get('error'):
                st.divider()
                st.subheader("🔍 Why this prediction?")

                expl = result['explanation']
                contributions = expl['contributions']
                base_prob = expl['base_prob']
                final_prob = expl['final_prob']

                # Create a clean DataFrame with readable labels
                df_contrib = pd.DataFrame(contributions, columns=['Feature', 'Contribution'])
                
                # Add human-readable labels
                feature_labels = {
                    'p1_rank': 'P1 Rank',
                    'p2_rank': 'P2 Rank',
                    'p1_elo': 'P1 Elo',
                    'p2_elo': 'P2 Elo',
                    'p1_elo_surf': 'P1 Surface Elo',
                    'p2_elo_surf': 'P2 Surface Elo',
                    'rank_diff': 'Rank Difference',
                    'elo_diff': 'Elo Difference',
                    'elo_surf_diff': 'Surface Elo Difference',
                    'win_rate_diff': 'Win Rate Difference',
                    'win_rate_surf_diff': 'Surface Win Rate Difference',
                    'p1_win_rate': 'P1 Win Rate',
                    'p2_win_rate': 'P2 Win Rate',
                    'p1_win_rate_surf': 'P1 Surface Win Rate',
                    'p2_win_rate_surf': 'P2 Surface Win Rate',
                    'p1_age': 'P1 Age',
                    'p2_age': 'P2 Age',
                    'p1_days_since_last': 'P1 Days Since Last',
                    'p2_days_since_last': 'P2 Days Since Last',
                    'rank_pts_diff': 'Rank Points Difference',
                    'h2h_diff': 'H2H Difference',
                    'h2h_total': 'H2H Matches',
                    'h2h_win_rate': 'H2H Win Rate',
                    'surface_code': 'Surface',
                    'best_of': 'Best of',
                    'round': 'Round',
                    'p1_ht': 'P1 Height',
                    'p2_ht': 'P2 Height',
                    'p1_hand': 'P1 Hand',
                    'p2_hand': 'P2 Hand',
                    'p1_rank_pts': 'P1 Rank Points',
                    'p2_rank_pts': 'P2 Rank Points',
                    'p1_1st_serve_pct': 'P1 1st Serve %',
                    'p2_1st_serve_pct': 'P2 1st Serve %',
                    'p1_1st_won_pct': 'P1 1st Won %',
                    'p2_1st_won_pct': 'P2 1st Won %',
                    'p1_2nd_won_pct': 'P1 2nd Won %',
                    'p2_2nd_won_pct': 'P2 2nd Won %',
                    'p1_ace_rate': 'P1 Ace Rate',
                    'p2_ace_rate': 'P2 Ace Rate',
                    'p1_return_pts_won': 'P1 Return Pts Won',
                    'p2_return_pts_won': 'P2 Return Pts Won',
                    'p1_bp_save_pct': 'P1 BP Save %',
                    'p2_bp_save_pct': 'P2 BP Save %',
                    'p1_bp_convert_pct': 'P1 BP Convert %',
                    'p2_bp_convert_pct': 'P2 BP Convert %',
                    'p1_matches_last_30': 'P1 Matches (30d)',
                    'p2_matches_last_30': 'P2 Matches (30d)',
                }
                df_contrib['Feature Label'] = df_contrib['Feature'].map(feature_labels).fillna(df_contrib['Feature'])
                
                # Add direction indicator
                df_contrib['Direction'] = df_contrib['Contribution'].apply(
                    lambda x: '🔵 Favours P1' if x > 0 else '🔴 Favours P2'
                )
                df_contrib['Impact'] = df_contrib['Contribution'].apply(
                    lambda x: f"+{x:.2f}" if x > 0 else f"{x:.2f}"
                )

                # Show as a bar chart with improved styling
                fig, ax = plt.subplots(figsize=(10, 5))
                colors = ['#00d4ff' if c > 0 else '#ff6b35' for c in df_contrib['Contribution']]
                ax.barh(df_contrib['Feature Label'], df_contrib['Contribution'], color=colors, height=0.6)
                ax.axvline(0, color='white', linewidth=1, linestyle='--', alpha=0.5)
                ax.set_xlabel('Log-odds contribution', color='white', fontsize=11)
                ax.set_title('Top Feature Contributions', color='white', fontsize=12, fontweight='bold')
                ax.set_facecolor('#1a1a2e')
                fig.patch.set_facecolor('#0d0d0d')
                ax.tick_params(colors='white', labelsize=9)
                for spine in ax.spines.values():
                    spine.set_edgecolor('#2a2a2a')
                
                # Add a subtle grid
                ax.grid(color='#2a2a2a', linewidth=0.5, alpha=0.5)
                
                st.pyplot(fig)

                # Show table with clear interpretation
                st.dataframe(
                    df_contrib[['Feature Label', 'Impact', 'Direction']],
                    use_container_width=True,
                    hide_index=True
                )

                # Explanation of the base vs final
                st.caption(
                    f"**Base probability:** {base_prob:.1%} (average) → "
                    f"**Final (XGBoost):** {final_prob:.1%} → "
                    f"**Ensemble:** {result['p1_win_prob']:.1%}"
                )

            # ── Baseline Comparison ──────────────────────────────────────
            if 'baseline_prob' in result:
                st.divider()
                st.subheader("📊 Baseline Comparison")

                baseline_prob = result['baseline_prob']
                full_prob = result['p1_win_prob']
                diff = full_prob - baseline_prob

                col1, col2, col3 = st.columns(3)
                col1.metric("Full Model", f"{full_prob:.1%}")
                col2.metric("Elo-Only Baseline", f"{baseline_prob:.1%}")
                col3.metric("Difference", f"{diff:+.1%}",
                            delta_color="normal" if diff > 0 else "inverse")

                if abs(diff) > 0.05:
                    st.info(
                        "💡 The full model differs significantly from Elo-only – "
                        "it's using surface, form, and other factors to adjust the prediction."
                    )
                else:
                    st.info(
                        "ℹ️ The full model's prediction is close to Elo-only – "
                        "the extra features add limited value for this matchup."
                    )

            # ── Historical Context (Fixed) ──────────────────────────────
            st.divider()
            st.subheader("📜 Historical Context")

            elo_diff = result['features']['elo_diff']
            
            if hasattr(pred, 'master') and len(pred.master) > 0:
                # Use absolute Elo difference to compare similar magnitudes
                similar = pred.master[
                    (pred.master['surface'] == surface) &
                    (abs(abs(pred.master['winner_elo_pre'] - pred.master['loser_elo_pre']) - abs(elo_diff)).abs() < 50)
                ]
                
                if len(similar) > 10:
                    fav_wins = (similar['winner_elo_pre'] > similar['loser_elo_pre']).mean()
                    upset_rate = 1 - fav_wins
                    st.caption(
                        f"**{len(similar)}** historical matches on {surface} with similar Elo advantage "
                        f"(diff ≈ {abs(elo_diff):.0f}).\n"
                        f"• Higher‑ELO player won: **{fav_wins:.0%}**\n"
                        f"• Upsets: **{upset_rate:.0%}**"
                    )
                else:
                    st.caption(f"Not enough historical data for a similar‑match comparison on {surface}.")
            else:
                st.caption("Historical data not available for this comparison.")