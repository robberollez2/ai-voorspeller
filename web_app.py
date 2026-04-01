import os
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import TimeSeriesSplit
import streamlit as st
from xgboost import XGBRegressor
from scipy.stats import poisson

# =========================================
# Data loading helpers
# =========================================
MODEL_CONFIGS = {
    "Beloften": {
        "excel": "wedstrijden_beloften.xlsx",
        "data_pkl": "data.pkl",
        "model_home": "model_home.pkl",
        "model_away": "model_away.pkl",
        "league_avg": "league_avg.pkl",
    },
    "Eerste ploeg": {
        "excel": "wedstrijden_eersteploeg.xlsx",  # pas aan naar je bestandsnaam
        "data_pkl": "data_eerste.pkl",
        "model_home": "model_home_eerste.pkl",
        "model_away": "model_away_eerste.pkl",
        "league_avg": "league_avg_eerste.pkl",
    },
    "U21": {
        "excel": "wedstrijden_u21.xlsx",  # pas aan naar je bestandsnaam
        "data_pkl": "data_u21.pkl",
        "model_home": "model_home_u21.pkl",
        "model_away": "model_away_u21.pkl",
        "league_avg": "league_avg_u21.pkl",
    },
}


def load_base_dataframe(config):
    """Load baseline match history for a given team model."""
    if os.path.exists(config["data_pkl"]):
        return joblib.load(config["data_pkl"])
    if os.path.exists(config["excel"]):
        df = pd.read_excel(config["excel"])
        df.columns = df.columns.str.strip()
        df["Datum"] = pd.to_datetime(df["Datum"])
        return df.sort_values("Datum").reset_index(drop=True)
    raise FileNotFoundError(
        f"Geen basisdataset gevonden voor {config['excel']} (verwacht {config['data_pkl']} of {config['excel']})"
    )


def load_default_models(df, config):
    """Load trained models for a team; retrain and save if missing."""
    paths = [config["model_home"], config["model_away"], config["league_avg"]]
    if all(os.path.exists(p) for p in paths):
        model_home = joblib.load(config["model_home"])
        model_away = joblib.load(config["model_away"])
        league_avg = joblib.load(config["league_avg"])
        return model_home, model_away, league_avg
    return train_models(df, config)


# =========================================
# Feature engineering
# =========================================

def create_features(df, n_form=5):
    rows = []
    history = []
    league_avg = df[["Goals Thuis", "Goals Uit"]].stack().mean()

    for _, row in df.iterrows():
        past = pd.DataFrame(history, columns=df.columns)

        home = row["Thuisteam"]
        away = row["Uitteam"]

        def team_stats(team):
            if len(past) < 5:
                return league_avg, league_avg

            t_home = past[past["Thuisteam"] == team]
            t_away = past[past["Uitteam"] == team]

            gf = pd.concat([t_home["Goals Thuis"], t_away["Goals Uit"]])
            ga = pd.concat([t_home["Goals Uit"], t_away["Goals Thuis"]])

            return (
                gf.mean() if len(gf) else league_avg,
                ga.mean() if len(ga) else league_avg,
            )

        def recent_form(team):
            matches = past[(past["Thuisteam"] == team) | (past["Uitteam"] == team)].tail(n_form)

            if len(matches) == 0:
                return league_avg, league_avg

            gf, ga = [], []
            for _, r in matches.iterrows():
                if r["Thuisteam"] == team:
                    gf.append(r["Goals Thuis"])
                    ga.append(r["Goals Uit"])
                else:
                    gf.append(r["Goals Uit"])
                    ga.append(r["Goals Thuis"])

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
            "home_goals": row["Goals Thuis"],
            "away_goals": row["Goals Uit"],
        })

        history.append(row)

    return pd.DataFrame(rows).dropna(), league_avg


def get_features(home, away, df, league_avg, n_form=5):
    def team_stats(team):
        t_home = df[df["Thuisteam"] == team]
        t_away = df[df["Uitteam"] == team]

        gf = pd.concat([t_home["Goals Thuis"], t_away["Goals Uit"]])
        ga = pd.concat([t_home["Goals Uit"], t_away["Goals Thuis"]])

        return (
            gf.mean() if len(gf) else league_avg,
            ga.mean() if len(ga) else league_avg,
        )

    def recent_form(team):
        matches = df[(df["Thuisteam"] == team) | (df["Uitteam"] == team)].tail(n_form)

        if len(matches) == 0:
            return league_avg, league_avg

        gf, ga = [], []
        for _, r in matches.iterrows():
            if r["Thuisteam"] == team:
                gf.append(r["Goals Thuis"])
                ga.append(r["Goals Uit"])
            else:
                gf.append(r["Goals Uit"])
                ga.append(r["Goals Thuis"])

        return np.mean(gf), np.mean(ga)

    home_attack, home_def = team_stats(home)
    away_attack, away_def = team_stats(away)

    home_form_a, home_form_d = recent_form(home)
    away_form_a, away_form_d = recent_form(away)

    return pd.DataFrame([
        {
            "home_attack": home_attack,
            "home_defense": home_def,
            "away_attack": away_attack,
            "away_defense": away_def,
            "home_form_attack": home_form_a,
            "home_form_defense": home_form_d,
            "away_form_attack": away_form_a,
            "away_form_defense": away_form_d,
            "home_advantage": 1,
        }
    ])


# =========================================
# Model training & prediction
# =========================================

def train_models(df, config=None):
    features, league_avg = create_features(df)
    X = features.drop(columns=["home_goals", "away_goals"])
    y_home_log = np.log1p(features["home_goals"])
    y_away_log = np.log1p(features["away_goals"])

    tuned_params = _tune_params(X, y_home_log, features["home_goals"])

    model_home = XGBRegressor(**tuned_params)
    model_away = XGBRegressor(**tuned_params)

    model_home.fit(X, y_home_log)
    model_away.fit(X, y_away_log)

    if config:
        joblib.dump(model_home, config["model_home"])
        joblib.dump(model_away, config["model_away"])
        joblib.dump(league_avg, config["league_avg"])
        joblib.dump(df, config["data_pkl"])

    return model_home, model_away, league_avg


def predict_match(home, away, df, model_home, model_away, league_avg):
    X_pred = get_features(home, away, df, league_avg)

    pred_home = np.expm1(model_home.predict(X_pred)[0])
    pred_away = np.expm1(model_away.predict(X_pred)[0])

    pred_home = float(np.clip(pred_home, 0.2, 5))
    pred_away = float(np.clip(pred_away, 0.2, 5))

    scores = predict_scores(pred_home, pred_away)

    home_win = sum(p for (h, a), p in scores if h > a)
    draw = sum(p for (h, a), p in scores if h == a)
    away_win = sum(p for (h, a), p in scores if h < a)

    return {
        "expected": (pred_home, pred_away),
        "scores": scores,
        "probs": {
            "home": home_win,
            "draw": draw,
            "away": away_win,
        },
    }


def predict_scores(lh, la, max_goals=5):
    results = []
    for h in range(max_goals + 1):
        for a in range(max_goals + 1):
            p = poisson.pmf(h, lh) * poisson.pmf(a, la)
            results.append(((h, a), p))
    return sorted(results, key=lambda x: -x[1])


def _tune_params(X, y_log, y_raw):
    """Lightweight time-series CV to choose XGB params; balances bias/variance."""
    param_grid = [
        {"n_estimators": 300, "max_depth": 3, "learning_rate": 0.05},
        {"n_estimators": 400, "max_depth": 3, "learning_rate": 0.03},
        {"n_estimators": 450, "max_depth": 4, "learning_rate": 0.03},
    ]

    base_params = {
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_lambda": 1.5,
        "random_state": 42,
    }

    def mae_goal_space(model, X_val, y_val_log, y_val_raw):
        preds = np.expm1(model.predict(X_val))
        preds = np.clip(preds, 0, None)
        return float(np.mean(np.abs(preds - y_val_raw)))

    best = (None, float("inf"))
    splitter = TimeSeriesSplit(n_splits=3)

    for params in param_grid:
        fold_mae = []
        for train_idx, val_idx in splitter.split(X):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val_log = y_log.iloc[train_idx], y_log.iloc[val_idx]
            y_val_raw = y_raw.iloc[val_idx]

            model = XGBRegressor(**base_params, **params)
            model.fit(X_train, y_train)
            fold_mae.append(mae_goal_space(model, X_val, y_val_log, y_val_raw))

        avg_mae = float(np.mean(fold_mae))
        if avg_mae < best[1]:
            best = (params, avg_mae)

    return {**base_params, **best[0]}


# =========================================
# Streamlit UI
# =========================================
st.set_page_config(page_title="Voetbal voorspeller", page_icon="⚽", layout="centered")
def get_state_key(model_key, suffix):
    return f"{model_key}_{suffix}"

st.title("Match voorspeller")
st.write("Kies twee teams, voorspel de score en bekijk de winstkansen. Upload eventueel je eigen Excel om opnieuw te trainen.")

for _key in MODEL_CONFIGS:
    model_k = get_state_key(_key, "custom_models")
    if model_k not in st.session_state:
        st.session_state[model_k] = None
    df_k = get_state_key(_key, "custom_df")
    if df_k not in st.session_state:
        st.session_state[df_k] = None
    avg_k = get_state_key(_key, "custom_league_avg")
    if avg_k not in st.session_state:
        st.session_state[avg_k] = None


# Load default data/models lazily per model type
@st.cache_resource(show_spinner=False)
def _load_defaults(model_key: str):
    cfg = MODEL_CONFIGS[model_key]
    base_df = load_base_dataframe(cfg)
    model_home, model_away, league_avg = load_default_models(base_df, cfg)
    return base_df, model_home, model_away, league_avg

st.subheader("Modelkeuze")
model_key = st.radio("Kies team", list(MODEL_CONFIGS.keys()), horizontal=True)
cfg = MODEL_CONFIGS[model_key]

st.caption(
    "Vereiste kolommen: Datum, Thuisteam, Uitteam, Goals Thuis, Goals Uit. Datum kan dd-mm-jjjj of jjjj-mm-dd zijn."
)

uploaded = st.file_uploader(f"Upload Excel ({model_key})", type=["xlsx"], key=f"uploader_{model_key}")

# Defaults
base_available = True
try:
    base_df, base_model_home, base_model_away, base_league_avg = _load_defaults(model_key)
except FileNotFoundError as exc:
    base_available = False
    base_df = None
    base_model_home = base_model_away = base_league_avg = None
    st.warning(f"Geen default dataset voor {model_key}: {exc}. Upload een bestand om te trainen.")

if uploaded:
    try:
        user_df = pd.read_excel(uploaded)
        user_df.columns = user_df.columns.str.strip()
        missing = {"Datum", "Thuisteam", "Uitteam", "Goals Thuis", "Goals Uit"} - set(user_df.columns)
        if missing:
            st.error(f"Ontbrekende kolommen: {', '.join(sorted(missing))}")
            user_df = None
        else:
            user_df["Datum"] = pd.to_datetime(user_df["Datum"])
            user_df = user_df.sort_values("Datum").reset_index(drop=True)
            st.success("Bestand geladen. Train het model om deze data te gebruiken.")
            st.dataframe(user_df.head())
            if st.button(f"Train model ({model_key})", type="primary"):
                with st.spinner("Model trainen..."):
                    m_home, m_away, l_avg = train_models(user_df, cfg)
                st.session_state[get_state_key(model_key, "custom_models")] = (m_home, m_away)
                st.session_state[get_state_key(model_key, "custom_df")] = user_df
                st.session_state[get_state_key(model_key, "custom_league_avg")] = l_avg
                st.success(f"Nieuw {model_key}-model getraind en opgeslagen.")
                st.caption("Let op: training gebruikt een lichte CV-tuning; kan iets langer duren.")
    except Exception as exc:  # noqa: BLE001
        st.error(f"Kon het bestand niet lezen: {exc}")

st.divider()

st.subheader("Voorspel een match")
custom_models = st.session_state[get_state_key(model_key, "custom_models")]
use_custom = custom_models is not None

model_choice = "Eigen getraind model" if use_custom else "Standaard model"

if use_custom:
    model_choice = st.radio(
        "Modelbron",
        ["Standaard model", "Eigen getraind model"],
        index=1,
        help="Gebruik je eigen model zodra je hebt getraind.",
    )

if model_choice == "Eigen getraind model" and use_custom:
    current_df = st.session_state[get_state_key(model_key, "custom_df")]
    model_home, model_away = custom_models
    league_avg = st.session_state[get_state_key(model_key, "custom_league_avg")]
elif base_available:
    current_df = base_df
    model_home, model_away = base_model_home, base_model_away
    league_avg = base_league_avg
else:
    st.error("Geen standaardmodel beschikbaar en nog geen eigen model getraind.")
    st.stop()

teams = pd.unique(current_df[["Thuisteam", "Uitteam"]].values.ravel())

col1, col2 = st.columns(2)
with col1:
    home_team = st.selectbox("Thuisploeg", options=teams)
with col2:
    away_team = st.selectbox("Uitploeg", options=[t for t in teams if t != home_team])

if st.button("Voorspel", type="primary"):
    with st.spinner("Voorspellen..."):
        result = predict_match(home_team, away_team, current_df, model_home, model_away, league_avg)

    exp_home, exp_away = result["expected"]
    probs = result["probs"]

    predicted_home = max(0, int(round(exp_home)))
    predicted_away = max(0, int(round(exp_away)))

    st.markdown(
        f"### Voorspelde uitslag: **{home_team} {predicted_home} - {predicted_away} {away_team}**"
    )
    st.caption(
        f"Verwachte goals (xG): {home_team} {exp_home:.2f} - {exp_away:.2f} {away_team}"
    )

    st.write("Winstkansen:")
    st.progress(min(1.0, probs["home"]), text=f"{home_team} wint: {probs['home']:.1%}")
    st.progress(min(1.0, probs["draw"]), text=f"Gelijkspel: {probs['draw']:.1%}")
    st.progress(min(1.0, probs["away"]), text=f"{away_team} wint: {probs['away']:.1%}")

st.divider()

st.subheader("Excel structuur")
st.text("Kolommen:\n- Datum (dd-mm-jjjj of jjjj-mm-dd)\n- Thuisteam\n- Uitteam\n- Goals Thuis\n- Goals Uit")
st.caption("Elke rij is een gespeelde match. Voorzie minstens 10-15 wedstrijden per team voor betere schattingen.")

st.info(
    "Start met 'streamlit run web_app.py' in deze map. Installeer eerst: pip install streamlit pandas numpy joblib xgboost scipy scikit-learn"
)
