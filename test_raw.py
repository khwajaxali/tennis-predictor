"""
Advanced Diagnostic Suite for TennisPredictor
Verifies symmetry, surface/format logic, and scenario-based accuracy.
"""

from predictor import TennisPredictor

# Initialize
tp = TennisPredictor(models_path='./models', data_path='./data')

def run_test(name, p1, p2, surf, tourney, rnd, verbose=True):
    res = tp.predict_match(p1, p2, surf, tourney, rnd)
    if verbose:
        print(f"\n--- {name} ---")
        if res.get('error'):
            print(f"Status: ❌ Error: {res['error']}")
        else:
            print(f"Match: {p1} vs {p2} [{surf} | {tourney} | {rnd}]")
            print(f"Outcome: {p1} {res['p1_win_prob']:.1%} | {p2} {res['p2_win_prob']:.1%}")
            print(f"Confidence: {res['confidence']}")
    return res

print("=" * 60)
print("EXTENDED BACKEND DIAGNOSTIC")
print("=" * 60)

# 1. Symmetry & Ordering
res1 = run_test("Symmetry A", "Novak Djokovic", "Roger Federer", "Grass", "G", "F")
res2 = run_test("Symmetry B", "Roger Federer", "Novak Djokovic", "Grass", "G", "F")
if abs((res1['p1_win_prob'] + res2['p1_win_prob']) - 1.0) < 0.01:
    print("✅ Symmetry Check: Passed")
else:
    print("❌ Symmetry Check: FAILED")

# 2. Surface Specialist Testing
# Does the model favor the Clay king on Clay, but not on Hard?
run_test("Nadal (Clay King) on Clay", "Rafael Nadal", "Daniil Medvedev", "Clay", "G", "F")
run_test("Nadal on Hard", "Rafael Nadal", "Daniil Medvedev", "Hard", "G", "F")

# 3. Round Progression Logic
run_test("Early Round (R128)", "Carlos Alcaraz", "Qualifier Player", "Hard", "A", "R128")
run_test("Grand Slam Final", "Carlos Alcaraz", "Novak Djokovic", "Hard", "G", "F")

# 4. Generational/Tiered Testing
run_test("Veteran vs Youth", "Andy Murray", "Jannik Sinner", "Hard", "M", "QF")

# 5. Tournament Level Scaling (ATP 250 vs GS)
run_test("ATP 250 Format", "Stefanos Tsitsipas", "Alexander Zverev", "Clay", "A", "SF")
run_test("Grand Slam Format", "Stefanos Tsitsipas", "Alexander Zverev", "Clay", "G", "SF")

# 6. Extreme Gap Test (The "David vs. Goliath" test)
run_test("Extreme Mismatch", "Novak Djokovic", "Zizou Bergs", "Hard", "G", "R128")

# 7. Tournament Stakes Intelligence (The "Pressure" Test)
run_test("Low Stakes (ATP 250)", "Carlos Alcaraz", "Jannik Sinner", "Hard", "A", "F")
run_test("High Stakes (Grand Slam)", "Carlos Alcaraz", "Jannik Sinner", "Hard", "G", "F")

# 8. Rivalry Logic (H2H-informed check)
run_test("Rivalry Logic", "Novak Djokovic", "Daniil Medvedev", "Hard", "G", "F")

# 9. Baseline Match (Near Peers)
run_test("Baseline Match (Near Peers)", "Taylor Fritz", "Frances Tiafoe", "Hard", "A", "SF")

# 10. Robustness & Edge Cases
print(f"\n--- Edge Case Testing ---")
run_test("Invalid Surface", "Novak Djokovic", "Rafael Nadal", "Carpet", "G", "F")
run_test("Same Player", "Novak Djokovic", "Novak Djokovic", "Hard", "G", "F")
run_test("Missing Player", "Unknown Player X", "Rafael Nadal", "Hard", "G", "F")

print(f"\n--- Edge Case Testing ---")
run_test("Invalid Surface", "Novak Djokovic", "Rafael Nadal", "Carpet", "G", "F")
run_test("Same Player", "Novak Djokovic", "Novak Djokovic", "Hard", "G", "F")
res_missing = run_test("Missing Player", "Unknown Player X", "Rafael Nadal", "Hard", "G", "F")

# 11. Feature Inspection (The "Why" Test)
print(f"\n{'='*20} FEATURE SENSITIVITY CHECK {'='*20}")
# Run a specific test and capture the full result dictionary
res_debug = tp.predict_match("Novak Djokovic", "Zizou Bergs", "Hard", "G", "R128")

if not res_debug.get('error'):
    feats = res_debug.get('features', {})
    # Assuming your feature vector uses these keys
    print(f"Rank Diff: {feats.get('rank_diff', 'N/A')}")
    print(f"ELO Diff: {feats.get('elo_diff', 'N/A')}")
    print(f"Surface Win Rate Diff: {feats.get('win_rate_surf_diff', 'N/A')}")
    print("\nIf these values are near 0, your database lookup or feature calculation is failing.")
else:
    print("Could not inspect features due to error in prediction.")

print("\n" + "=" * 60)
print("✅ DIAGNOSTIC COMPLETE")
print("=" * 60)