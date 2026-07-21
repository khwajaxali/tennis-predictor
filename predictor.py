"""
Tennis Match Predictor - Backend Logic (COMPLETE FINAL VERSION)
Uses FIXED neutral-format models (p1/p2) with no data leakage.
Ensemble: 55% XGBoost + 45% LightGBM

FIXES INCLUDED:
1. ✅ Full 59-feature support (not just 13)
2. ✅ Temperature scaling (from temperature.json)
3. ✅ StandardScaler integration
4. ✅ Surface-specific H2H (fixes Nadal/Djokovic clay issue)
5. ✅ Symmetric prediction wrapper (fixes 103%/122% sum issues)
6. ✅ Rolling stats for serve, return, fatigue
7. ✅ Proper feature building matching training
8. ✅ SHAP explainability
9. ✅ Baseline Elo-only comparison
10.✅ Recent form retrieval
11.✅ Enhanced human‑readable analysis
"""

import pandas as pd
import numpy as np
import joblib
import json
import warnings
from datetime import datetime
import shap

warnings.filterwarnings('ignore')


class TennisPredictor:
    """
    Tennis match prediction system with NEUTRAL format (p1/p2).
    Production-ready with full 59 features, scaling, and symmetry.
    """
    
    def __init__(self, models_path='./models', data_path='./data'):
        """
        Initialize predictor with FIXED neutral models and load player database.
        """
        print("Loading FIXED neutral models...")
        try:
            # Load models and artifacts
            self.xgb_model = joblib.load(f'{models_path}/xgb_tuned_FIXED.pkl')
            self.lgb_model = joblib.load(f'{models_path}/lgb_model_FIXED.pkl')
            self.scaler = joblib.load(f'{models_path}/scaler.pkl')
            self.feature_cols = joblib.load(f'{models_path}/feature_cols.pkl')
            
            # Load temperature scaling
            with open(f'{models_path}/temperature.json', 'r') as f:
                self.temperature = json.load(f)['temperature']
            
            # Load tour averages for fallbacks
            with open(f'{models_path}/tour_averages.json', 'r') as f:
                self.tour_avg = json.load(f)
            
            # Load player database
            self.master = pd.read_csv(
                f'{data_path}/atp_matches_master.csv',
                parse_dates=['tourney_date']
            )
            
            self.all_players = sorted(set(
                self.master['winner_name'].tolist() +
                self.master['loser_name'].tolist()
            ))
            
            # Mappings for categorical variables
            self.surface_map = {'Hard': 0, 'Clay': 1, 'Grass': 2}
            self.round_map = {'R128': 1, 'R64': 2, 'R32': 3, 'R16': 4,
                             'QF': 5, 'SF': 6, 'F': 7}
            
            # Load SHAP explainer (if available)
            try:
                self.explainer = joblib.load(f'{models_path}/shap_explainer.pkl')
                print("✅ SHAP explainer loaded")
            except FileNotFoundError:
                self.explainer = None
                print("⚠️ SHAP explainer not found; explanations will be disabled.")
            
            print(f"✅ Loaded {len(self.all_players)} players")
            print(f"   Features: {len(self.feature_cols)}")
            print(f"   Ensemble: 55% XGB + 45% LGB")
            print(f"   Temperature: {self.temperature}")
            
        except FileNotFoundError as e:
            print(f"❌ File not found: {e}")
            print("   Make sure model files exist in the specified directory.")
            raise

    def get_all_players(self):
        """Returns a sorted list of all unique players in the database."""
        return self.all_players
    
    # ---------------------------------------------------------------------
    # H2H Methods (Surface-Specific)
    # ---------------------------------------------------------------------
    
    def get_surface_h2h(self, player_a, player_b, surface):
        """
        Get surface-specific head-to-head record between two players.
        Used both for prediction features and UI display.
        """
        a_won = self.master[
            (self.master['winner_name'] == player_a) &
            (self.master['loser_name'] == player_b) &
            (self.master['surface'] == surface)
        ]
        b_won = self.master[
            (self.master['winner_name'] == player_b) &
            (self.master['loser_name'] == player_a) &
            (self.master['surface'] == surface)
        ]
        
        total = len(a_won) + len(b_won)
        if total == 0:
            return 0, 0, 0.5, f"{player_a} 0-0 {player_b}"
        
        win_rate = len(a_won) / total
        record = f"{player_a} {len(a_won)}-{len(b_won)} {player_b}"
        return len(a_won), len(b_won), win_rate, record
    
    # ---------------------------------------------------------------------
    # Recent Form (New)
    # ---------------------------------------------------------------------
    
    def get_recent_form(self, player_name, surface=None, n=5):
        """
        Get the player's recent win/loss record over last n matches.
        Returns a dict: {'wins': int, 'losses': int, 'record': str, 'streak': str}
        """
        won = self.master[self.master['winner_name'] == player_name]
        lost = self.master[self.master['loser_name'] == player_name]
        
        all_matches = pd.concat([
            won[['tourney_date', 'surface']].assign(result=1),
            lost[['tourney_date', 'surface']].assign(result=0)
        ]).sort_values('tourney_date', ascending=False)
        
        if surface:
            all_matches = all_matches[all_matches['surface'] == surface]
        
        recent = all_matches.head(n)
        wins = recent['result'].sum()
        losses = n - wins
        
        # Streak: last match result
        if len(recent) > 0:
            last_result = recent.iloc[0]['result']
            streak = 'W' if last_result == 1 else 'L'
            streak_count = 1
            for i in range(1, len(recent)):
                if recent.iloc[i]['result'] == last_result:
                    streak_count += 1
                else:
                    break
            streak = f"{streak}{streak_count}" if streak_count > 1 else streak
        else:
            streak = '—'
        
        return {
            'wins': int(wins),
            'losses': int(losses),
            'record': f"{wins}-{losses}",
            'streak': streak,
            'n_matches': min(n, len(recent))
        }
    
    # ---------------------------------------------------------------------
    # Player Stats Extraction (Full 59 Features)
    # ---------------------------------------------------------------------
    
    def _get_player_stats(self, player_name, surface=None, n_matches=20):
        """
        Get comprehensive stats for a player including rolling serve stats.
        Matches the training data feature engineering.
        
        KEY FIX: win_rate_surf uses CAREER stats, not rolling (preserves Nadal's clay dominance)
        """
        won = self.master[self.master['winner_name'] == player_name]
        lost = self.master[self.master['loser_name'] == player_name]
        
        # Get tour averages for fallbacks
        surf_key = surface if surface else 'Hard'
        ta = self.tour_avg.get(surf_key, self.tour_avg['Hard'])
        
        def safe_float(v, default):
            try:
                f = float(v)
                return default if np.isnan(f) else f
            except (TypeError, ValueError):
                return default
        
        # Default for unknown players
        if len(won) == 0 and len(lost) == 0:
            return {
                'rank': 100, 'rank_pts': 0, 'age': 26, 'ht': 185, 'hand': 0,
                'elo': 1500, 'elo_surf': 1500, 
                'win_rate': 0.5, 
                'win_rate_surf': 0.5,  # Career win rate on surface
                'first_serve_pct': ta.get('1st_serve_pct', 0.617),
                'first_won_pct': ta.get('1st_won_pct', 0.767),
                'second_won_pct': ta.get('2nd_won_pct', 0.566),
                'ace_rate': ta.get('ace_rate', 0.091),
                'return_pts_won': ta.get('return_pts_won', 0.423),
                'bp_save_pct': ta.get('bp_save_pct', 0.663),
                'bp_convert_pct': ta.get('bp_convert_pct', 0.489),
                'days_since_last': 7.0, 'matches_last_30': 5.0,
            }
        
        # Get most recent match data (for rank, ELO, bio)
        if len(won) > 0 and len(lost) > 0:
            last_won = won.sort_values('tourney_date').iloc[-1]
            last_lost = lost.sort_values('tourney_date').iloc[-1]
            if last_won['tourney_date'] > last_lost['tourney_date']:
                row = last_won
                rank = safe_float(row.get('winner_rank', 100), 100)
                rank_pts = safe_float(row.get('winner_rank_points', 0), 0)
                age = safe_float(row.get('winner_age', 26), 26)
                ht = safe_float(row.get('winner_ht', 185), 185)
                hand = 0 if str(row.get('winner_hand', 'R')).upper() == 'R' else 1
                elo = safe_float(row.get('winner_elo_pre', 1500), 1500)
                elo_surf = safe_float(row.get('winner_elo_surf_pre', 1500), 1500)
            else:
                row = last_lost
                rank = safe_float(row.get('loser_rank', 100), 100)
                rank_pts = safe_float(row.get('loser_rank_points', 0), 0)
                age = safe_float(row.get('loser_age', 26), 26)
                ht = safe_float(row.get('loser_ht', 185), 185)
                hand = 0 if str(row.get('loser_hand', 'R')).upper() == 'R' else 1
                elo = safe_float(row.get('loser_elo_pre', 1500), 1500)
                elo_surf = safe_float(row.get('loser_elo_surf_pre', 1500), 1500)
        elif len(won) > 0:
            row = won.sort_values('tourney_date').iloc[-1]
            rank = safe_float(row.get('winner_rank', 100), 100)
            rank_pts = safe_float(row.get('winner_rank_points', 0), 0)
            age = safe_float(row.get('winner_age', 26), 26)
            ht = safe_float(row.get('winner_ht', 185), 185)
            hand = 0 if str(row.get('winner_hand', 'R')).upper() == 'R' else 1
            elo = safe_float(row.get('winner_elo_pre', 1500), 1500)
            elo_surf = safe_float(row.get('winner_elo_surf_pre', 1500), 1500)
        else:
            row = lost.sort_values('tourney_date').iloc[-1]
            rank = safe_float(row.get('loser_rank', 100), 100)
            rank_pts = safe_float(row.get('loser_rank_points', 0), 0)
            age = safe_float(row.get('loser_age', 26), 26)
            ht = safe_float(row.get('loser_ht', 185), 185)
            hand = 0 if str(row.get('loser_hand', 'R')).upper() == 'R' else 1
            elo = safe_float(row.get('loser_elo_pre', 1500), 1500)
            elo_surf = safe_float(row.get('loser_elo_surf_pre', 1500), 1500)
        
        # ============================================================
        # KEY FIX: Career win rate on surface (NOT rolling)
        # This preserves Nadal's clay dominance and other surface specialists
        # ============================================================
        total = len(won) + len(lost)
        career_win_rate = len(won) / total if total > 0 else 0.5
        
        if surface:
            won_surf = won[won['surface'] == surface]
            lost_surf = lost[lost['surface'] == surface]
            surf_total = len(won_surf) + len(lost_surf)
            career_win_rate_surf = len(won_surf) / surf_total if surf_total > 0 else career_win_rate
        else:
            career_win_rate_surf = career_win_rate
        
        # ============================================================
        # Rolling stats from last n matches (chronological)
        # Used for recent form (win_rate, not win_rate_surf)
        # ============================================================
        all_matches = pd.concat([
            won[['tourney_date', 'surface']].assign(result=1),
            lost[['tourney_date', 'surface']].assign(result=0)
        ]).sort_values('tourney_date').tail(n_matches)
        
        # Days since last match
        if len(all_matches) > 0:
            last_date = all_matches['tourney_date'].iloc[-1]
            days_since_last = (pd.Timestamp.now() - last_date).days
            days_since_last = max(1.0, min(30.0, float(days_since_last)))
        else:
            days_since_last = 7.0
        
        # Matches in last 30 days
        thirty_days_ago = pd.Timestamp.now() - pd.Timedelta(days=30)
        recent_matches = all_matches[all_matches['tourney_date'] >= thirty_days_ago]
        matches_last_30 = float(len(recent_matches))
        
        # Rolling win rate (last n matches) - for recent form
        if len(all_matches) >= 3:
            rolling_win_rate = all_matches['result'].mean()
        else:
            rolling_win_rate = career_win_rate
        
        # Surface-specific rolling win rate (last n matches on surface)
        if surface:
            surf_matches = all_matches[all_matches['surface'] == surface]
            if len(surf_matches) >= 3:
                rolling_win_rate_surf = surf_matches['result'].mean()
            else:
                rolling_win_rate_surf = career_win_rate_surf
        else:
            rolling_win_rate_surf = rolling_win_rate
        
        # Serve stats from won matches (last n wins with stats)
        ws = won.copy()
        if surface:
            ws = ws[ws['surface'] == surface]
        if 'has_stats' in ws.columns:
            ws = ws[ws['has_stats'] == 1]
        ws = ws.sort_values('tourney_date').tail(n_matches)
        
        if len(ws) >= 3:
            svpt = ws['w_svpt'].replace(0, np.nan)
            first_serve_pct = (ws['w_1stIn'] / svpt).mean()
            first_won_pct = (ws['w_1stWon'] / ws['w_1stIn'].replace(0, np.nan)).mean()
            second_denom = (svpt - ws['w_1stIn']).replace(0, np.nan)
            second_won_pct = (ws['w_2ndWon'] / second_denom).mean()
            ace_rate = (ws['w_ace'] / svpt).mean()
            return_pts_won = (1 - (ws['l_1stWon'] + ws['l_2ndWon']) / 
                            ws['l_svpt'].replace(0, np.nan)).mean()
            bp_save_pct = (ws['w_bpSaved'] / ws['w_bpFaced'].replace(0, np.nan)).mean()
            bp_convert_pct = ((ws['l_bpFaced'] - ws['l_bpSaved']) /
                            ws['l_bpFaced'].replace(0, np.nan)).mean()
            
            # Handle NaNs
            first_serve_pct = ta.get('1st_serve_pct', 0.617) if pd.isna(first_serve_pct) else first_serve_pct
            first_won_pct = ta.get('1st_won_pct', 0.767) if pd.isna(first_won_pct) else first_won_pct
            second_won_pct = ta.get('2nd_won_pct', 0.566) if pd.isna(second_won_pct) else second_won_pct
            ace_rate = ta.get('ace_rate', 0.091) if pd.isna(ace_rate) else ace_rate
            return_pts_won = ta.get('return_pts_won', 0.423) if pd.isna(return_pts_won) else return_pts_won
            bp_save_pct = ta.get('bp_save_pct', 0.663) if pd.isna(bp_save_pct) else bp_save_pct
            bp_convert_pct = ta.get('bp_convert_pct', 0.489) if pd.isna(bp_convert_pct) else bp_convert_pct
        else:
            first_serve_pct = ta.get('1st_serve_pct', 0.617)
            first_won_pct = ta.get('1st_won_pct', 0.767)
            second_won_pct = ta.get('2nd_won_pct', 0.566)
            ace_rate = ta.get('ace_rate', 0.091)
            return_pts_won = ta.get('return_pts_won', 0.423)
            bp_save_pct = ta.get('bp_save_pct', 0.663)
            bp_convert_pct = ta.get('bp_convert_pct', 0.489)
        
        return {
            'rank': rank,
            'rank_pts': rank_pts,
            'age': age,
            'ht': ht,
            'hand': hand,
            'elo': elo,
            'elo_surf': elo_surf,
            # KEY FIX: win_rate_surf uses CAREER stats (preserves surface specialists)
            'win_rate': rolling_win_rate,                    # Recent form (rolling)
            'win_rate_surf': career_win_rate_surf,          # Career on surface (FIXED)
            'first_serve_pct': float(first_serve_pct),
            'first_won_pct': float(first_won_pct),
            'second_won_pct': float(second_won_pct),
            'ace_rate': float(ace_rate),
            'return_pts_won': float(return_pts_won),
            'bp_save_pct': float(bp_save_pct),
            'bp_convert_pct': float(bp_convert_pct),
            'days_since_last': days_since_last,
            'matches_last_30': matches_last_30,
        }
    
    # ---------------------------------------------------------------------
    # Feature Vector Construction (Full 59 Features)
    # ---------------------------------------------------------------------
    
    def _build_feature_vector(self, p1_stats, p2_stats, surface, tournament_level, round_num):
        """
        Build the full 59-feature vector matching training.
        """
        surface_code = self.surface_map.get(surface, 0)
        round_code = self.round_map.get(round_num, 5)
        best_of = 5 if tournament_level in ['G', 'D'] else 3
        
        features = {
            # Context
            'surface_code': surface_code,
            'best_of': best_of,
            'round': round_code,
            
            # Raw player stats
            'p1_rank': p1_stats['rank'],
            'p2_rank': p2_stats['rank'],
            'p1_rank_pts': p1_stats['rank_pts'],
            'p2_rank_pts': p2_stats['rank_pts'],
            'p1_age': p1_stats['age'],
            'p2_age': p2_stats['age'],
            'p1_ht': p1_stats['ht'],
            'p2_ht': p2_stats['ht'],
            'p1_hand': p1_stats['hand'],
            'p2_hand': p2_stats['hand'],
            'p1_elo': p1_stats['elo'],
            'p2_elo': p2_stats['elo'],
            'p1_elo_surf': p1_stats['elo_surf'],
            'p2_elo_surf': p2_stats['elo_surf'],
            'p1_win_rate': p1_stats['win_rate'],
            'p2_win_rate': p2_stats['win_rate'],
            'p1_win_rate_surf': p1_stats['win_rate_surf'],
            'p2_win_rate_surf': p2_stats['win_rate_surf'],
            'p1_1st_serve_pct': p1_stats['first_serve_pct'],
            'p2_1st_serve_pct': p2_stats['first_serve_pct'],
            'p1_1st_won_pct': p1_stats['first_won_pct'],
            'p2_1st_won_pct': p2_stats['first_won_pct'],
            'p1_2nd_won_pct': p1_stats['second_won_pct'],
            'p2_2nd_won_pct': p2_stats['second_won_pct'],
            'p1_ace_rate': p1_stats['ace_rate'],
            'p2_ace_rate': p2_stats['ace_rate'],
            'p1_return_pts_won': p1_stats['return_pts_won'],
            'p2_return_pts_won': p2_stats['return_pts_won'],
            'p1_bp_save_pct': p1_stats['bp_save_pct'],
            'p2_bp_save_pct': p2_stats['bp_save_pct'],
            'p1_bp_convert_pct': p1_stats['bp_convert_pct'],
            'p2_bp_convert_pct': p2_stats['bp_convert_pct'],
            'p1_days_since_last': p1_stats['days_since_last'],
            'p2_days_since_last': p2_stats['days_since_last'],
            'p1_matches_last_30': p1_stats['matches_last_30'],
            'p2_matches_last_30': p2_stats['matches_last_30'],
            
            # Difference features
            'rank_diff': p1_stats['rank'] - p2_stats['rank'],
            'rank_pts_diff': p1_stats['rank_pts'] - p2_stats['rank_pts'],
            'elo_diff': p1_stats['elo'] - p2_stats['elo'],
            'elo_surf_diff': p1_stats['elo_surf'] - p2_stats['elo_surf'],
            'age_diff': p1_stats['age'] - p2_stats['age'],
            'ht_diff': p1_stats['ht'] - p2_stats['ht'],
            'win_rate_diff': p1_stats['win_rate'] - p2_stats['win_rate'],
            'win_rate_surf_diff': p1_stats['win_rate_surf'] - p2_stats['win_rate_surf'],
            '1st_serve_pct_diff': p1_stats['first_serve_pct'] - p2_stats['first_serve_pct'],
            '1st_won_pct_diff': p1_stats['first_won_pct'] - p2_stats['first_won_pct'],
            '2nd_won_pct_diff': p1_stats['second_won_pct'] - p2_stats['second_won_pct'],
            'ace_rate_diff': p1_stats['ace_rate'] - p2_stats['ace_rate'],
            'return_pts_won_diff': p1_stats['return_pts_won'] - p2_stats['return_pts_won'],
            'bp_save_pct_diff': p1_stats['bp_save_pct'] - p2_stats['bp_save_pct'],
            'bp_convert_pct_diff': p1_stats['bp_convert_pct'] - p2_stats['bp_convert_pct'],
            'days_since_last_diff': p1_stats['days_since_last'] - p2_stats['days_since_last'],
            'matches_last_30_diff': p1_stats['matches_last_30'] - p2_stats['matches_last_30'],
            
            # H2H placeholders (filled in predict_match)
            'h2h_total': 0,
            'h2h_diff': 0,
            'h2h_win_rate': 0.5,
        }
        return features
    
    # ---------------------------------------------------------------------
    # Temperature Scaling
    # ---------------------------------------------------------------------
    
    def _apply_temperature(self, probs, T):
        """Apply temperature scaling to probabilities."""
        if T == 1.0:
            return probs
        probs = np.clip(probs, 1e-9, 1 - 1e-9)
        logit = np.log(probs / (1 - probs))
        logit_scaled = logit / T
        return 1 / (1 + np.exp(-logit_scaled))
    
    # ---------------------------------------------------------------------
    # Baseline Elo-Only Prediction
    # ---------------------------------------------------------------------
    
    def baseline_elo_prediction(self, player1, player2, surface):
        """
        Simple Elo-only prediction using standard formula.
        Returns probability player1 wins.
        """
        p1_stats = self._get_player_stats(player1, surface)
        p2_stats = self._get_player_stats(player2, surface)
        elo_diff = p1_stats['elo'] - p2_stats['elo']
        prob = 1.0 / (1.0 + 10 ** (-elo_diff / 400.0))
        return prob
    
    # ---------------------------------------------------------------------
    # SHAP Explanation
    # ---------------------------------------------------------------------
    
    def explain_prediction(self, player1, player2, surface,
                           tournament_level='G', round_num='F', top_k=8):
        """
        Return SHAP feature contributions for the prediction.
        Returns a dict with:
          - 'base_prob': base probability (from expected log-odds)
          - 'final_prob': final probability after temperature scaling
          - 'contributions': list of (feature_name, contribution) sorted by abs(contribution)
        """
        if self.explainer is None:
            return {'error': 'SHAP explainer not loaded'}
        
        # Get stats and build feature vector (same as predict_match)
        p1_stats = self._get_player_stats(player1, surface)
        p2_stats = self._get_player_stats(player2, surface)
        h2h_wins_p1, h2h_wins_p2, h2h_win_rate, _ = self.get_surface_h2h(player1, player2, surface)
        
        features = self._build_feature_vector(p1_stats, p2_stats, surface, tournament_level, round_num)
        features['h2h_total'] = h2h_wins_p1 + h2h_wins_p2
        features['h2h_diff'] = h2h_wins_p1 - h2h_wins_p2
        features['h2h_win_rate'] = h2h_win_rate
        
        X = pd.DataFrame([features])[self.feature_cols].values
        X_scaled = self.scaler.transform(X)
        
        # Get SHAP values
        shap_values = self.explainer.shap_values(X_scaled)[0]
        
        # Create contribution list
        contributions = []
        for name, val in zip(self.feature_cols, shap_values):
            if abs(val) > 1e-6:
                contributions.append((name, val))
        contributions.sort(key=lambda x: abs(x[1]), reverse=True)
        
        # Get expected value (base probability in log-odds space)
        expected_value = self.explainer.expected_value
        if isinstance(expected_value, (list, np.ndarray)):
            expected_value = expected_value[0]  # binary classification
        base_prob = 1 / (1 + np.exp(-expected_value))
        
        # Final probability (log-odds sum) then temperature scaling
        total_logit = expected_value + sum(c[1] for c in contributions)
        raw_prob = 1 / (1 + np.exp(-total_logit))
        final_prob = self._apply_temperature(np.array([raw_prob]), self.temperature)[0]
        
        return {
            'base_prob': base_prob,
            'final_prob': final_prob,
            'contributions': contributions[:top_k],
            'all_contributions': contributions,  # full list if needed
        }
    
    # ---------------------------------------------------------------------
    # Core Prediction (Symmetric + H2H + Scaling)
    # ---------------------------------------------------------------------
    
    def predict_match(self, player1, player2, surface,
                      tournament_level='G', round_num='F',
                      include_explanation=False, include_baseline=False):
        """
        Predict match outcome with:
        - Full 59 features
        - Surface-specific H2H
        - StandardScaler normalization
        - Temperature scaling
        - Symmetric averaging (fixes order bias)
        
        Optional:
          include_explanation: if True, add SHAP explanation to result
          include_baseline: if True, add Elo-only baseline probability
        """
        try:
            p1_stats = self._get_player_stats(player1, surface)
            p2_stats = self._get_player_stats(player2, surface)
        except Exception as e:
            return {
                'error': str(e),
                'p1_win_prob': 0.5,
                'p2_win_prob': 0.5,
                'confidence': 'None',
            }
        
        # Get surface-specific H2H
        h2h_wins_p1, h2h_wins_p2, h2h_win_rate, h2h_record = self.get_surface_h2h(player1, player2, surface)
        
        # Build feature vector
        features = self._build_feature_vector(p1_stats, p2_stats, surface, tournament_level, round_num)
        features['h2h_total'] = h2h_wins_p1 + h2h_wins_p2
        features['h2h_diff'] = h2h_wins_p1 - h2h_wins_p2
        features['h2h_win_rate'] = h2h_win_rate
        
        # Create DataFrame and scale
        X = pd.DataFrame([features])[self.feature_cols].values
        X_scaled = self.scaler.transform(X)
        
        # Get raw probabilities
        xgb_prob = self.xgb_model.predict_proba(X_scaled)[0, 1]
        lgb_prob = self.lgb_model.predict_proba(X_scaled)[0, 1]
        raw_prob = 0.55 * xgb_prob + 0.45 * lgb_prob
        
        # Apply temperature scaling
        scaled_prob = self._apply_temperature(np.array([raw_prob]), self.temperature)[0]
        
        # Now do symmetric prediction (swap players and average)
        p1_stats_swapped = self._get_player_stats(player2, surface)
        p2_stats_swapped = self._get_player_stats(player1, surface)
        
        h2h_wins_p1_swapped, h2h_wins_p2_swapped, h2h_win_rate_swapped, _ = self.get_surface_h2h(player2, player1, surface)
        
        features_swapped = self._build_feature_vector(p1_stats_swapped, p2_stats_swapped, surface, tournament_level, round_num)
        features_swapped['h2h_total'] = h2h_wins_p1_swapped + h2h_wins_p2_swapped
        features_swapped['h2h_diff'] = h2h_wins_p1_swapped - h2h_wins_p2_swapped
        features_swapped['h2h_win_rate'] = h2h_win_rate_swapped
        
        X_swapped = pd.DataFrame([features_swapped])[self.feature_cols].values
        X_swapped_scaled = self.scaler.transform(X_swapped)
        
        xgb_prob_swapped = self.xgb_model.predict_proba(X_swapped_scaled)[0, 1]
        lgb_prob_swapped = self.lgb_model.predict_proba(X_swapped_scaled)[0, 1]
        raw_prob_swapped = 0.55 * xgb_prob_swapped + 0.45 * lgb_prob_swapped
        scaled_prob_swapped = self._apply_temperature(np.array([raw_prob_swapped]), self.temperature)[0]
        
        # Symmetric average
        symmetric_prob_p1 = (scaled_prob + (1 - scaled_prob_swapped)) / 2
        symmetric_prob_p1 = np.clip(symmetric_prob_p1, 0.05, 0.95)
        
        # Confidence
        margin = abs(symmetric_prob_p1 - 0.5)
        if margin > 0.2:
            confidence = 'High'
        elif margin > 0.1:
            confidence = 'Medium'
        else:
            confidence = 'Low'
        
        result = {
            'error': None,
            'player1': player1,
            'player2': player2,
            'surface': surface,
            'p1_win_prob': float(symmetric_prob_p1),
            'p2_win_prob': float(1 - symmetric_prob_p1),
            'confidence': confidence,
            'xgb_prob': float(xgb_prob),
            'lgb_prob': float(lgb_prob),
            'raw_prob': float(raw_prob),
            'scaled_prob': float(scaled_prob),
            'h2h_record': h2h_record,
            'features': features,
        }
        
        # Optional extras
        if include_baseline:
            baseline_prob = self.baseline_elo_prediction(player1, player2, surface)
            result['baseline_prob'] = baseline_prob
        
        if include_explanation and self.explainer is not None:
            expl = self.explain_prediction(player1, player2, surface, tournament_level, round_num)
            result['explanation'] = expl
        
        return result
    
    # ---------------------------------------------------------------------
    # Enhanced Human-Readable Analysis (New)
    # ---------------------------------------------------------------------
    
    def generate_detailed_analysis(self, result):
        """
        Generate a detailed, human‑readable analysis paragraph.
        Uses real stats, SHAP contributions, and baseline comparison.
        """
        p1 = result['player1']
        p2 = result['player2']
        prob = result['p1_win_prob']
        winner = p1 if prob > 0.5 else p2
        loser = p2 if prob > 0.5 else p1
        surface = result['surface']
        
        # Extract key stats from the features dict
        feats = result['features']
        p1_elo = feats['p1_elo']
        p2_elo = feats['p2_elo']
        p1_surf_wr = feats['p1_win_rate_surf']
        p2_surf_wr = feats['p2_win_rate_surf']
        p1_days = feats['p1_days_since_last']
        p2_days = feats['p2_days_since_last']
        h2h_wr = feats['h2h_win_rate']
        h2h_total = feats['h2h_total']
        
        # Recent form
        form1 = self.get_recent_form(p1, surface, n=5)
        form2 = self.get_recent_form(p2, surface, n=5)
        rec1 = f"{form1['wins']}-{form1['losses']}" if form1['n_matches'] >= 3 else "insufficient data"
        rec2 = f"{form2['wins']}-{form2['losses']}" if form2['n_matches'] >= 3 else "insufficient data"
        
        lines = []
        lines.append(f"**Analysis:**")
        
        # 1. Who is favoured and by how much?
        lines.append(f"{winner} is predicted to win with a **{max(prob, 1-prob):.0%}** chance.")
        
        # 2. Baseline comparison
        baseline = result.get('baseline_prob')
        if baseline is not None:
            diff = prob - baseline
            if diff > 0.05:
                lines.append(f"The model is **{diff:.0%} more confident** in {winner} than a simple Elo‑based prediction.")
            elif diff < -0.05:
                lines.append(f"The model is **{abs(diff):.0%} more conservative** than Elo for {winner}.")
            else:
                lines.append("The model closely agrees with the Elo baseline.")
        
        # 3. Key drivers (use concrete stats)
        lines.append("\n**Key factors:**")
        added = False
        
        # Elo
        elo_diff = p1_elo - p2_elo
        if abs(elo_diff) > 30:
            fav = p1 if elo_diff > 0 else p2
            lines.append(f"• **Elo**: {fav} has a {abs(elo_diff):.0f}‑point Elo advantage.")
            added = True
        
        # Surface win rate
        surf_diff = p1_surf_wr - p2_surf_wr
        if abs(surf_diff) > 0.03:
            fav = p1 if surf_diff > 0 else p2
            lines.append(f"• **Surface performance**: {fav} has a {abs(surf_diff):.1%} higher win rate on {surface}.")
            added = True
        
        # Recent form
        if form1['n_matches'] >= 3 and form2['n_matches'] >= 3:
            if form1['wins'] > form2['wins'] + 1:
                lines.append(f"• **Recent form**: {p1} has won {form1['wins']} of their last 5 matches ({rec1}), compared to {p2} ({rec2}).")
                added = True
            elif form2['wins'] > form1['wins'] + 1:
                lines.append(f"• **Recent form**: {p2} has won {form2['wins']} of their last 5 matches ({rec2}), compared to {p1} ({rec1}).")
                added = True
        
        # Rest days
        if abs(p1_days - p2_days) > 5:
            more_rest = p1 if p1_days > p2_days else p2
            lines.append(f"• **Rest**: {more_rest} has had {abs(p1_days - p2_days):.0f} more days of rest.")
            added = True
        
        # H2H (if meaningful)
        if h2h_total >= 3 and h2h_wr != 0.5:
            h2h_fav = p1 if h2h_wr > 0.5 else p2
            lines.append(f"• **Head‑to‑head**: {h2h_fav} has won {h2h_wr:.0%} of previous meetings ({h2h_total} total matches).")
            added = True
        
        if not added:
            lines.append("• The players are closely matched across all key metrics.")
        
        # 4. Summary / conclusion
        lines.append("\n**Bottom line**: The prediction blends current form, Elo rating, surface expertise, and historical H2H to make a balanced forecast.")
        
        return "\n".join(lines)
    
    # ---------------------------------------------------------------------
    # Public Methods for UI (unchanged)
    # ---------------------------------------------------------------------
    
    def get_h2h(self, player_a, player_b, surface=None):
        """Get head-to-head record between two players for UI display."""
        if surface:
            a_wins, b_wins, win_rate, record = self.get_surface_h2h(player_a, player_b, surface)
            total = a_wins + b_wins
            surface_breakdown = {surface: f"{player_a} {a_wins}-{b_wins} {player_b}"}
        else:
            a_wins = 0
            b_wins = 0
            surface_breakdown = {}
            for surf in ['Hard', 'Clay', 'Grass']:
                aw, bw, _, _ = self.get_surface_h2h(player_a, player_b, surf)
                a_wins += aw
                b_wins += bw
                if aw + bw > 0:
                    surface_breakdown[surf] = f"{player_a} {aw}-{bw} {player_b}"
            total = a_wins + b_wins
            win_rate = a_wins / total if total > 0 else 0
            record = f"{player_a} {a_wins}-{b_wins} {player_b}"
        
        if total == 0:
            return {
                'player_a': player_a,
                'player_b': player_b,
                'total_matches': 0,
                'h2h_record': f'{player_a} 0-0 {player_b}',
                'player_a_win_rate': 0,
                'surface_breakdown': {}
            }
        
        return {
            'player_a': player_a,
            'player_b': player_b,
            'player_a_wins': a_wins,
            'player_b_wins': b_wins,
            'total_matches': total,
            'h2h_record': record,
            'player_a_win_rate': win_rate,
            'surface_breakdown': surface_breakdown,
        }


# ---------------------------------------------------------------------
# TEST BLOCK
# ---------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 60)
    print("Testing Tennis Predictor (FINAL VERSION)")
    print("=" * 60)
    
    tp = TennisPredictor(models_path='./models', data_path='./data')
    
    # Test critical matchups
    test_cases = [
        ("Rafael Nadal", "Novak Djokovic", "Clay", "Nadal should be favored"),
        ("Novak Djokovic", "Rafael Nadal", "Hard", "Djokovic should be favored"),
        ("Roger Federer", "Rafael Nadal", "Grass", "Federer should be favored"),
        ("Carlos Alcaraz", "Novak Djokovic", "Grass", "Either could win"),
    ]
    
    print("\n" + "=" * 60)
    print("KNOWLEDGE TEST")
    print("=" * 60)
    
    for p1, p2, surface, note in test_cases:
        result = tp.predict_match(p1, p2, surface, "G", "F")
        print(f"\n{p1} vs {p2} on {surface}")
        print(f"  {p1}: {result['p1_win_prob']:.1%}")
        print(f"  {p2}: {result['p2_win_prob']:.1%}")
        print(f"  Confidence: {result['confidence']}")
        print(f"  H2H: {result['h2h_record']}")
        print(f"  Note: {note}")
    
    # Symmetry test
    print("\n" + "=" * 60)
    print("SYMMETRY TEST")
    print("=" * 60)
    
    r1 = tp.predict_match("Rafael Nadal", "Novak Djokovic", "Clay", "G", "F")
    r2 = tp.predict_match("Novak Djokovic", "Rafael Nadal", "Clay", "G", "F")
    
    print(f"\nNadal vs Djokovic on Clay:")
    print(f"  Nadal as Player 1: {r1['p1_win_prob']:.1%}")
    print(f"  Djokovic as Player 1: {r2['p1_win_prob']:.1%}")
    print(f"  Sum (should be ~100%): {r1['p1_win_prob'] + r2['p1_win_prob']:.1%}")