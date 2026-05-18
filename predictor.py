import pandas as pd
import numpy as np
import joblib
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class TennisPredictor:
    def __init__(self, models_path='./models', data_path='./data'):
        """Load models and data once at startup."""
        print("Loading models...")
        try:
            self.xgb_model = joblib.load(f'{models_path}/xgb_tuned.pkl')
            self.lgb_model = joblib.load(f'{models_path}/lgb_model.pkl')
            self.shap_explainer = joblib.load(f'{models_path}/shap_explainer.pkl')
            self.iso_forest = joblib.load(f'{models_path}/iso_forest.pkl')
            self.scaler = joblib.load(f'{models_path}/scaler.pkl')
            
            with open(f'{models_path}/tour_averages.json') as f:
                self.tour_avg = json.load(f)
            
            self.feature_cols = joblib.load(f'{models_path}/feature_cols.pkl')
            
            print("Loading data...")
            self.master = pd.read_csv(f'{data_path}/atp_matches_master.csv',
                                      parse_dates=['tourney_date'])
            
            # Get all unique players
            self.all_players = sorted(set(
                self.master['winner_name'].tolist() +
                self.master['loser_name'].tolist()
            ))
            
            print(f"✅ Models loaded | {len(self.all_players)} players in database")
        
        except FileNotFoundError as e:
            print(f"❌ Error loading models: {e}")
            print(f"Make sure models/ and data/ folders exist with all required files")
            raise
    
    def get_all_players(self):
        """Return list of all players for dropdown."""
        return self.all_players
    
    def get_player_stats(self, player_name, surface=None):
        """Get comprehensive player stats."""
        won = self.master[self.master['winner_name'] == player_name]
        lost = self.master[self.master['loser_name'] == player_name]
        
        if surface:
            won = won[won['surface'] == surface]
            lost = lost[lost['surface'] == surface]
        
        if len(won) + len(lost) == 0:
            return {'error': f'No matches found for {player_name}'}
        
        total = len(won) + len(lost)
        win_rate = len(won) / total
        
        best_rank = None
        if 'winner_rank' in won.columns and len(won) > 0:
            ranks = pd.concat([won['winner_rank'], lost['loser_rank']]).dropna()
            if len(ranks) > 0:
                best_rank = int(ranks[ranks < 999].min())
        
        # Surface breakdown
        surface_breakdown = {}
        for surf in ['Hard', 'Clay', 'Grass']:
            w = len(won[won['surface'] == surf])
            l = len(lost[lost['surface'] == surf])
            total_surf = w + l
            if total_surf > 0:
                surface_breakdown[surf] = {
                    'wins': w,
                    'losses': l,
                    'win_rate': w / total_surf
                }
        
        # Recent form (last 10 matches)
        all_matches = pd.concat([
            won[['tourney_date', 'surface']].assign(result='W'),
            lost[['tourney_date', 'surface']].assign(result='L')
        ]).sort_values('tourney_date').tail(10)
        
        recent_form = ''.join(all_matches['result'].tolist()) if len(all_matches) > 0 else 'N/A'
        
        return {
            'player_name': player_name,
            'total_matches': total,
            'wins': len(won),
            'losses': len(lost),
            'win_rate': win_rate,
            'best_rank': best_rank,
            'surface_breakdown': surface_breakdown,
            'recent_form': recent_form,
            'matches_count': len(all_matches)
        }
    
    def get_h2h(self, player_a, player_b, surface=None):
        """Get head-to-head record between two players."""
        a_won = self.master[
            (self.master['winner_name'] == player_a) &
            (self.master['loser_name'] == player_b)
        ]
        b_won = self.master[
            (self.master['winner_name'] == player_b) &
            (self.master['loser_name'] == player_a)
        ]
        
        if surface:
            a_won = a_won[a_won['surface'] == surface]
            b_won = b_won[b_won['surface'] == surface]
        
        total = len(a_won) + len(b_won)
        
        if total == 0:
            return {
                'player_a': player_a,
                'player_b': player_b,
                'total_matches': 0,
                'h2h_record': f"{player_a} 0-0 {player_b}",
                'surface_breakdown': {}
            }
        
        # Surface breakdown
        surface_breakdown = {}
        for surf in ['Hard', 'Clay', 'Grass']:
            a_surf = len(a_won[a_won['surface'] == surf])
            b_surf = len(b_won[b_won['surface'] == surf])
            if a_surf + b_surf > 0:
                surface_breakdown[surf] = f"{player_a} {a_surf}-{b_surf} {player_b}"
        
        return {
            'player_a': player_a,
            'player_b': player_b,
            'player_a_wins': len(a_won),
            'player_b_wins': len(b_won),
            'total_matches': total,
            'h2h_record': f"{player_a} {len(a_won)}-{len(b_won)} {player_b}",
            'player_a_win_rate': len(a_won) / total if total > 0 else 0.5,
            'surface_breakdown': surface_breakdown
        }
    
    def get_historical_matches(self, player_a, player_b, limit=10):
        """Get past matches between two players."""
        a_won = self.master[
            (self.master['winner_name'] == player_a) &
            (self.master['loser_name'] == player_b)
        ][['tourney_date', 'tourney_name', 'surface', 'score']].copy()
        a_won['winner'] = player_a
        
        b_won = self.master[
            (self.master['winner_name'] == player_b) &
            (self.master['loser_name'] == player_a)
        ][['tourney_date', 'tourney_name', 'surface', 'score']].copy()
        b_won['winner'] = player_b
        
        all_matches = pd.concat([a_won, b_won]).sort_values(
            'tourney_date', ascending=False
        ).head(limit)
        
        if len(all_matches) == 0:
            return None
        
        return all_matches
    
    def get_anomalies(self, player_name, limit=5):
        """Get anomalous sessions for a player."""
        stats = self.master[
            (self.master['winner_name'] == player_name) &
            (self.master['has_stats'] == 1)
        ].copy()
        
        if len(stats) < 5:
            return None
        
        # Compute stats
        stats['1st_serve_pct'] = stats['w_1stIn']  / stats['w_svpt']
        stats['ace_rate']      = stats['w_ace']     / stats['w_svpt']
        stats['df_rate']       = stats['w_df']      / stats['w_svpt']
        stats['bp_save_pct']   = stats['w_bpSaved'] / stats['w_bpFaced'].replace(0, np.nan)
        
        perf_cols = ['1st_serve_pct', 'ace_rate', 'df_rate', 'bp_save_pct']
        stats = stats.dropna(subset=perf_cols)
        
        if len(stats) < 5:
            return None
        
        # Detect anomalies
        anomaly_scores = self.iso_forest.decision_function(stats[perf_cols])
        predictions = self.iso_forest.predict(stats[perf_cols])
        
        stats['anomaly_score'] = anomaly_scores
        stats['is_anomaly'] = predictions == -1
        
        anomalies = stats[stats['is_anomaly']].sort_values('anomaly_score').head(limit)
        
        if len(anomalies) == 0:
            return None
        
        result = []
        for _, row in anomalies.iterrows():
            result.append({
                'date': row['tourney_date'].strftime('%Y-%m-%d'),
                'opponent': row['loser_name'],
                'tournament': row['tourney_name'],
                'surface': row['surface'],
                '1st_serve_pct': row['1st_serve_pct'],
                'ace_rate': row['ace_rate'],
                'df_rate': row['df_rate'],
                'bp_save_pct': row['bp_save_pct'],
                'anomaly_score': row['anomaly_score']
            })
        
        return result
    
    def build_feature_vector(self, player_a_name, player_b_name, surface, 
                         tournament_level, round_num):
   
    
            # Get player histories
            p_a_won = self.master[self.master['winner_name'] == player_a_name]
            p_a_lost = self.master[self.master['loser_name'] == player_a_name]
            p_a_all = pd.concat([p_a_won, p_a_lost]).sort_values('tourney_date')
            
            p_b_won = self.master[self.master['winner_name'] == player_b_name]
            p_b_lost = self.master[self.master['loser_name'] == player_b_name]
            p_b_all = pd.concat([p_b_won, p_b_lost]).sort_values('tourney_date')
            
            if len(p_a_all) == 0 or len(p_b_all) == 0:
                return None, "Insufficient player history in database"
            
            # Get most recent match for each player
            a_recent = p_a_won.iloc[-1] if len(p_a_won) > 0 else p_a_all.iloc[-1]
            b_recent = p_b_won.iloc[-1] if len(p_b_won) > 0 else p_b_all.iloc[-1]
            
            # Compute rolling stats function
            def compute_rolling_stats(matches_won, surface_filter=None):
                """Compute rolling stats from won matches."""
                df = matches_won.copy()
                if surface_filter:
                    df = df[df['surface'] == surface_filter]
                
                if len(df) == 0:
                    # Return tour averages if no matches
                    return {
                        'win_rate': 0.533,
                        'win_rate_surf': 0.545,
                        '1st_serve_pct': 0.617,
                        '1st_won_pct': 0.767,
                        '2nd_won_pct': 0.566,
                        'ace_rate': 0.091,
                        'return_pts_won': 0.423,
                        'bp_save_pct': 0.663,
                        'bp_convert_pct': 0.489,
                    }
                
                # Last 20 matches
                recent = df.tail(20)
                
                # Overall win rate
                win_rate = len(recent) / max(len(recent), 1) if len(recent) > 0 else 0.533
                
                # Surface specific win rate
                if surface_filter:
                    surf_df = recent[recent['surface'] == surface_filter]
                    surf_win_rate = len(surf_df) / max(len(surf_df), 1) if len(surf_df) > 0 else 0.545
                else:
                    surf_win_rate = win_rate
                
                # Get stats from won matches
                stats_df = recent[recent['has_stats'] == 1]
                
                if len(stats_df) > 0:
                    first_in_pct = (stats_df['w_1stIn'] / stats_df['w_svpt']).mean()
                    first_won_pct = (stats_df['w_1stWon'] / stats_df['w_1stIn']).mean()
                    second_won_pct = (stats_df['w_2ndWon'] / (stats_df['w_svpt'] - stats_df['w_1stIn'])).mean()
                    ace_rate = (stats_df['w_ace'] / stats_df['w_svpt']).mean()
                    
                    # Return points won (against opponent's serve)
                    ret_pts = (1 - (stats_df['l_1stWon'] + stats_df['l_2ndWon']) / stats_df['l_svpt']).mean()
                    
                    # BP stats
                    bp_save = (stats_df['w_bpSaved'] / stats_df['w_bpFaced'].replace(0, np.nan)).mean()
                    bp_conv = ((stats_df['l_bpFaced'] - stats_df['l_bpSaved']) / stats_df['l_bpFaced'].replace(0, np.nan)).mean()
                else:
                    # Use tour averages if no stats
                    first_in_pct = 0.617
                    first_won_pct = 0.767
                    second_won_pct = 0.566
                    ace_rate = 0.091
                    ret_pts = 0.423
                    bp_save = 0.663
                    bp_conv = 0.489
                
                return {
                    'win_rate': win_rate,
                    'win_rate_surf': surf_win_rate,
                    '1st_serve_pct': first_in_pct,
                    '1st_won_pct': first_won_pct,
                    '2nd_won_pct': second_won_pct,
                    'ace_rate': ace_rate,
                    'return_pts_won': ret_pts,
                    'bp_save_pct': bp_save,
                    'bp_convert_pct': bp_conv,
                }
            
            a_stats = compute_rolling_stats(p_a_won, surface)
            b_stats = compute_rolling_stats(p_b_won, surface)
            
            # Get H2H record (A's record vs B)
            a_won_vs_b = len(p_a_won[p_a_won['loser_name'] == player_b_name])
            b_won_vs_a = len(p_b_won[p_b_won['loser_name'] == player_a_name])
            h2h_total = a_won_vs_b + b_won_vs_a
            h2h_wr = a_won_vs_b / max(h2h_total, 1) if h2h_total > 0 else 0.5
            
            # Surface encoding
            surface_map = {'Hard': 0, 'Clay': 1, 'Grass': 2}
            round_map = {
                'R128': 1, 'R64': 2, 'R32': 3, 'R16': 4,
                'QF': 5, 'SF': 6, 'F': 7
            }
            
            # Build feature vector — CRITICAL: differences must be A - B consistently
            features_dict = {
                # Match context
                'surface_code': surface_map.get(surface, 0),
                'best_of': 5 if tournament_level in ['G', 'D'] else 3,
                'round': round_map.get(round_num, 3),
                
                # Rank (A - B)
                'winner_rank': float(a_recent.get('winner_rank', 100)),
                'loser_rank': float(b_recent.get('loser_rank', 100)),
                'rank_diff': float(a_recent.get('winner_rank', 100)) - float(b_recent.get('loser_rank', 100)),
                'winner_rank_points': float(a_recent.get('winner_rank_points', 0)),
                'loser_rank_points': float(b_recent.get('loser_rank_points', 0)),
                'rank_pts_diff': float(a_recent.get('winner_rank_points', 0)) - float(b_recent.get('loser_rank_points', 0)),
                
                # Bio
                'winner_age': float(a_recent.get('winner_age', 26)),
                'loser_age': float(b_recent.get('loser_age', 26)),
                'winner_ht': float(a_recent.get('winner_ht', 185)),
                'loser_ht': float(b_recent.get('loser_ht', 185)),
                'winner_hand_code': 0 if a_recent.get('winner_hand') == 'R' else 1,
                'loser_hand_code': 0 if b_recent.get('loser_hand') == 'R' else 1,
                
                # Elo (A - B)
                'winner_elo_pre': float(a_recent.get('winner_elo_pre', 1600)),
                'loser_elo_pre': float(b_recent.get('loser_elo_pre', 1600)),
                'elo_diff': float(a_recent.get('winner_elo_pre', 1600)) - float(b_recent.get('loser_elo_pre', 1600)),
                'elo_surf_diff': float(a_recent.get('winner_elo_surf_pre', 1600)) - float(b_recent.get('loser_elo_surf_pre', 1600)),
                'winner_elo_surf_pre': float(a_recent.get('winner_elo_surf_pre', 1600)),
                'loser_elo_surf_pre': float(b_recent.get('loser_elo_surf_pre', 1600)),
                
                # Win rates (A - B)
                'winner_roll_win_rate': a_stats['win_rate'],
                'loser_roll_win_rate': b_stats['win_rate'],
                'win_rate_diff': a_stats['win_rate'] - b_stats['win_rate'],
                'winner_roll_win_rate_surf': a_stats['win_rate_surf'],
                'loser_roll_win_rate_surf': b_stats['win_rate_surf'],
                'win_rate_surf_diff': a_stats['win_rate_surf'] - b_stats['win_rate_surf'],
                
                # Serve stats (A - B)
                'winner_roll_1st_serve_pct': a_stats['1st_serve_pct'],
                'loser_roll_1st_serve_pct': b_stats['1st_serve_pct'],
                '1st_serve_pct_diff': a_stats['1st_serve_pct'] - b_stats['1st_serve_pct'],
                'winner_roll_1st_won_pct': a_stats['1st_won_pct'],
                'loser_roll_1st_won_pct': b_stats['1st_won_pct'],
                '1st_won_pct_diff': a_stats['1st_won_pct'] - b_stats['1st_won_pct'],
                'winner_roll_2nd_won_pct': a_stats['2nd_won_pct'],
                'loser_roll_2nd_won_pct': b_stats['2nd_won_pct'],
                '2nd_won_pct_diff': a_stats['2nd_won_pct'] - b_stats['2nd_won_pct'],
                'winner_roll_ace_rate': a_stats['ace_rate'],
                'loser_roll_ace_rate': b_stats['ace_rate'],
                'ace_rate_diff': a_stats['ace_rate'] - b_stats['ace_rate'],
                
                # Return & pressure (A - B)
                'winner_roll_return_pts_won': a_stats['return_pts_won'],
                'loser_roll_return_pts_won': b_stats['return_pts_won'],
                'return_pts_won_diff': a_stats['return_pts_won'] - b_stats['return_pts_won'],
                'winner_roll_bp_save_pct': a_stats['bp_save_pct'],
                'loser_roll_bp_save_pct': b_stats['bp_save_pct'],
                'bp_save_pct_diff': a_stats['bp_save_pct'] - b_stats['bp_save_pct'],
                'winner_roll_bp_convert_pct': a_stats['bp_convert_pct'],
                'loser_roll_bp_convert_pct': b_stats['bp_convert_pct'],
                'bp_convert_pct_diff': a_stats['bp_convert_pct'] - b_stats['bp_convert_pct'],
                
                # H2H (A's record vs B)
                'h2h_winner_wins': a_won_vs_b,
                'h2h_loser_wins': b_won_vs_a,
                'h2h_total': h2h_total,
                'h2h_winner_winrate': h2h_wr,
                'h2h_diff': a_won_vs_b - b_won_vs_a,
            }
            
            # Create feature array in correct order
            feature_array = np.array([
                features_dict.get(col, 0) for col in self.feature_cols
            ]).reshape(1, -1)
            
            return feature_array, None

    
    def predict_match(self, player_a_name, player_b_name, surface,
                  tournament_level='A', round_num='F'):
        
            # Build features (A vs B)
            X, error = self.build_feature_vector(
                player_a_name, player_b_name, surface,
                tournament_level, round_num
            )
            
            if X is None:
                return {'error': error}
            
            # Get predictions from both models
            xgb_proba = self.xgb_model.predict_proba(X)[0, 1]
            lgb_proba = self.lgb_model.predict_proba(X)[0, 1]
            
            # Ensemble (weighted average)
            ensemble_proba = 0.55 * xgb_proba + 0.45 * lgb_proba
            
            # Confidence based on margin from 0.5
            conf_margin = abs(ensemble_proba - 0.5)
            if conf_margin > 0.15:
                confidence = 'High'
            elif conf_margin > 0.08:
                confidence = 'Medium'
            else:
                confidence = 'Low'
            
            # Return prediction where proba is A's win probability
            return {
                'player_a': player_a_name,
                'player_b': player_b_name,
                'surface': surface,
                'player_a_win_prob': float(ensemble_proba),
                'player_b_win_prob': float(1 - ensemble_proba),
                'confidence': confidence,
                'xgb_prob': float(xgb_proba),
                'lgb_prob': float(lgb_proba),
                'ensemble_prob': float(ensemble_proba),
                'error': None,
                'model_accuracy': 0.778,
                'grand_slam_accuracy': 0.821
            }