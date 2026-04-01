# train_model.py
from sklearn.metrics import mean_absolute_error
import pandas as pd
import numpy as np
import joblib
from xgboost import XGBRegressor

# ==============================
# LOAD DATA
# ==============================
df = pd.read_excel("wedstrijden_beloften.xlsx")
df.columns = df.columns.str.strip()
# 🔥 BELANGRIJK
df["Datum"] = pd.to_datetime(df["Datum"])

# sorteren op tijd
df = df.sort_values("Datum").reset_index(drop=True)
print(df.head())
LEAGUE_AVG = df[['Goals Thuis', 'Goals Uit']].stack().mean()

# ==============================
# FEATURE ENGINEERING (NO LEAKAGE)
# ==============================
def create_features(df, n_form=5):
    rows = []
    history = []

    for _, row in df.iterrows():
        past = pd.DataFrame(history, columns=df.columns)

        home = row['Thuisteam']
        away = row['Uitteam']

        def team_stats(team):
            if len(past) < 5:
                return LEAGUE_AVG, LEAGUE_AVG

            t_home = past[past['Thuisteam'] == team]
            t_away = past[past['Uitteam'] == team]

            gf = pd.concat([t_home['Goals Thuis'], t_away['Goals Uit']])
            ga = pd.concat([t_home['Goals Uit'], t_away['Goals Thuis']])

            return (
                gf.mean() if len(gf) else LEAGUE_AVG,
                ga.mean() if len(ga) else LEAGUE_AVG
            )

        def recent_form(team):
            matches = past[(past['Thuisteam'] == team) | (past['Uitteam'] == team)].tail(n_form)

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

        rows.append({
            "home_attack": home_attack,
            "home_defense": home_def,
            "away_attack": away_attack,
            "away_defense": away_def,

            "home_form_attack": home_form_a,
            "home_form_defense": home_form_d,
            "away_form_attack": away_form_a,
            "away_form_defense": away_form_d,

            "home_advantage": 1,

            "home_goals": row['Goals Thuis'],
            "away_goals": row['Goals Uit']
        })

        history.append(row)

    return pd.DataFrame(rows).dropna()

print("Features bouwen...")
feature_df = create_features(df)

X = feature_df.drop(columns=["home_goals", "away_goals"])

# 🔥 LOG TRANSFORM (BELANGRIJK)
y_home = np.log1p(feature_df["home_goals"])
y_away = np.log1p(feature_df["away_goals"])

# ==============================
# MODEL
# ==============================
model_home = XGBRegressor(
    n_estimators=400,
    max_depth=4,
    learning_rate=0.03,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_lambda=1.5,
    random_state=42
)

model_away = XGBRegressor(
    n_estimators=400,
    max_depth=4,
    learning_rate=0.03,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_lambda=1.5,
    random_state=42
)

print("Training...")
model_home.fit(X, y_home)
model_away.fit(X, y_away)

# ==============================
# SAVE
# ==============================
joblib.dump(model_home, "model_home.pkl")
joblib.dump(model_away, "model_away.pkl")
joblib.dump(df, "data.pkl")
joblib.dump(LEAGUE_AVG, "league_avg.pkl")

print("✅ Training klaar en opgeslagen!")

pred_home = np.expm1(model_home.predict(X))

mae = mean_absolute_error(feature_df["home_goals"], pred_home)
print("MAE:", mae)