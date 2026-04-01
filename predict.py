# predict.py

import pandas as pd
import numpy as np
import joblib
from scipy.stats import poisson

# ==============================
# LOAD MODEL & DATA
# ==============================
model_home = joblib.load("model_home.pkl")
model_away = joblib.load("model_away.pkl")
df = joblib.load("data.pkl")
LEAGUE_AVG = joblib.load("league_avg.pkl")

# unieke teams
teams = pd.unique(df[['Thuisteam', 'Uitteam']].values.ravel())

# ==============================
# FEATURES (ALLEEN VERLEDEN)
# ==============================
def get_features(home, away, df, n_form=5):

    def team_stats(team):
        t_home = df[df['Thuisteam'] == team]
        t_away = df[df['Uitteam'] == team]

        gf = pd.concat([t_home['Goals Thuis'], t_away['Goals Uit']])
        ga = pd.concat([t_home['Goals Uit'], t_away['Goals Thuis']])

        return (
            gf.mean() if len(gf) else LEAGUE_AVG,
            ga.mean() if len(ga) else LEAGUE_AVG
        )

    def recent_form(team):
        matches = df[(df['Thuisteam'] == team) | (df['Uitteam'] == team)].tail(n_form)

        if len(matches) == 0:
            return LEAGUE_AVG, LEAGUE_AVG

        gf, ga = [], []
        for _, r in matches.iterrows():
            if r['Thuisteam'] == team:
                gf.append(r['Goals Thuis'])
                ga.append(r['Goals Uit'])
            else:
                gf.append(r['Goals Uit'])
                ga.append(r['Goals Thuis'])

        return np.mean(gf), np.mean(ga)

    home_attack, home_def = team_stats(home)
    away_attack, away_def = team_stats(away)

    home_form_a, home_form_d = recent_form(home)
    away_form_a, away_form_d = recent_form(away)

    return pd.DataFrame([{
        "home_attack": home_attack,
        "home_defense": home_def,
        "away_attack": away_attack,
        "away_defense": away_def,

        "home_form_attack": home_form_a,
        "home_form_defense": home_form_d,
        "away_form_attack": away_form_a,
        "away_form_defense": away_form_d,

        "home_advantage": 1
    }])

# ==============================
# USER INPUT
# ==============================
print("\nTeams:")
for i, t in enumerate(teams):
    print(f"{i}: {t}")

home = teams[int(input("Selecteer Thuisteam (nummer): "))]
away = teams[int(input("Selecteer Uitteam (nummer): "))]

# ==============================
# PREDICT XG
# ==============================
X_pred = get_features(home, away, df)

pred_home = np.expm1(model_home.predict(X_pred)[0])
pred_away = np.expm1(model_away.predict(X_pred)[0])

# realistische limits
pred_home = np.clip(pred_home, 0.2, 5)
pred_away = np.clip(pred_away, 0.2, 5)

print("\n📊 xG voorspelling:")
print(f"{home} {pred_home:.2f} - {pred_away:.2f} {away}")

# ==============================
# POISSON SCORE PROBABILITIES
# ==============================
def predict_scores(lh, la, max_goals=5):
    results = []
    for h in range(max_goals + 1):
        for a in range(max_goals + 1):
            p = poisson.pmf(h, lh) * poisson.pmf(a, la)
            results.append(((h, a), p))
    return sorted(results, key=lambda x: -x[1])

scores = predict_scores(pred_home, pred_away, max_goals=5)

print("\n🏁 Meest waarschijnlijke scores:")
for (h, a), p in scores[:5]:
    print(f"{h}-{a}: {p:.2%}")

# ==============================
# WIN / DRAW / LOSS KANSEN
# ==============================
home_win_prob = 0
draw_prob = 0
away_win_prob = 0

for (h, a), p in scores:
    if h > a:
        home_win_prob += p
    elif h == a:
        draw_prob += p
    else:
        away_win_prob += p

print("\n📊 Kansverdeling:")
print(f"{home} wint: {home_win_prob:.2%}")
print(f"Gelijkspel: {draw_prob:.2%}")
print(f"{away} wint: {away_win_prob:.2%}")

best_score = scores[0][0]
print(f"\n🎯 Voorspelling: {home} {best_score[0]} - {best_score[1]} {away}")