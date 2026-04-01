import os
import pandas as pd
import numpy as np
import joblib
import streamlit as st
from xgboost import XGBRegressor
from scipy.stats import poisson

# =========================================
# Data loading helpers
# =========================================
DEFAULT_EXCEL = "wedstrijden_beloften.xlsx"
DEFAULT_DATA_PKL = "data.pkl"
DEFAULT_MODEL_HOME = "model_home.pkl"
DEFAULT_MODEL_AWAY = "model_away.pkl"
DEFAULT_LEAGUE_AVG = "league_avg.pkl"


def load_base_dataframe():
    """Load baseline match history for default predictions."""
    if os.path.exists(DEFAULT_DATA_PKL):
        return joblib.load(DEFAULT_DATA_PKL)
    if os.path.exists(DEFAULT_EXCEL):
        df = pd.read_excel(DEFAULT_EXCEL)
        df.columns = df.columns.str.strip()
        df["Datum"] = pd.to_datetime(df["Datum"])
        return df.sort_values("Datum").reset_index(drop=True)
    raise FileNotFoundError("Geen basisdataset gevonden (data.pkl of wedstrijden_beloften.xlsx)")


def load_default_models(df):
    """Load trained models; retrain if missing."""
    if all(os.path.exists(p) for p in [DEFAULT_MODEL_HOME, DEFAULT_MODEL_AWAY, DEFAULT_LEAGUE_AVG]):
        model_home = joblib.load(DEFAULT_MODEL_HOME)
        model_away = joblib.load(DEFAULT_MODEL_AWAY)
        league_avg = joblib.load(DEFAULT_LEAGUE_AVG)
        return model_home, model_away, league_avg
    return train_models(df)


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

def train_models(df):
    features, league_avg = create_features(df)
    X = features.drop(columns=["home_goals", "away_goals"])
    y_home = np.log1p(features["home_goals"])
    y_away = np.log1p(features["away_goals"])

    params = {
        "n_estimators": 400,
        "max_depth": 4,
        "learning_rate": 0.03,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_lambda": 1.5,
        "random_state": 42,
    }

    model_home = XGBRegressor(**params)
    model_away = XGBRegressor(**params)

    model_home.fit(X, y_home)
    model_away.fit(X, y_away)

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


# =========================================
# Streamlit UI
# =========================================
st.set_page_config(page_title="Voetbal voorspeller", page_icon="⚽", layout="centered")
st.title("Match voorspeller")
st.write("Kies twee teams, voorspel de score en bekijk de winstkansen. Upload eventueel je eigen Excel om opnieuw te trainen.")

if "custom_models" not in st.session_state:
    st.session_state.custom_models = None
    st.session_state.custom_df = None
    st.session_state.custom_league_avg = None

# Load default data/models lazily
@st.cache_resource(show_spinner=False)
def _load_defaults():
    base_df = load_base_dataframe()
    model_home, model_away, league_avg = load_default_models(base_df)
    return base_df, model_home, model_away, league_avg


base_df, base_model_home, base_model_away, base_league_avg = _load_defaults()

st.subheader("Upload eigen dataset (optioneel)")
st.caption("Vereiste kolommen: Datum, Thuisteam, Uitteam, Goals Thuis, Goals Uit. Datum kan dd-mm-jjjj of jjjj-mm-dd zijn.")

uploaded = st.file_uploader("Upload Excel", type=["xlsx"])

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
            if st.button("Train model op geüploade data", type="primary"):
                with st.spinner("Model trainen..."):
                    m_home, m_away, l_avg = train_models(user_df)
                st.session_state.custom_models = (m_home, m_away)
                st.session_state.custom_df = user_df
                st.session_state.custom_league_avg = l_avg
                st.success("Nieuw model getraind en klaar voor voorspellingen.")
    except Exception as exc:  # noqa: BLE001
        st.error(f"Kon het bestand niet lezen: {exc}")

st.divider()

st.subheader("Voorspel een match")
use_custom = st.session_state.custom_models is not None

model_choice = "Eigen getraind model" if use_custom else "Standaard model"

if use_custom:
    model_choice = st.radio(
        "Modelkeuze",
        ["Standaard model", "Eigen getraind model"],
        index=1,
        help="Gebruik je eigen model zodra je hebt getraind.",
    )

if model_choice == "Eigen getraind model" and use_custom:
    current_df = st.session_state.custom_df
    model_home, model_away = st.session_state.custom_models
    league_avg = st.session_state.custom_league_avg
else:
    current_df = base_df
    model_home, model_away = base_model_home, base_model_away
    league_avg = base_league_avg

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

    st.markdown(f"### Verwachte score: **{home_team} {exp_home:.2f} - {exp_away:.2f} {away_team}**")

    st.write("Winstkansen:")
    st.progress(min(1.0, probs["home"]), text=f"{home_team} wint: {probs['home']:.1%}")
    st.progress(min(1.0, probs["draw"]), text=f"Gelijkspel: {probs['draw']:.1%}")
    st.progress(min(1.0, probs["away"]), text=f"{away_team} wint: {probs['away']:.1%}")

st.divider()

st.subheader("Excel structuur")
st.text("Kolommen:\n- Datum (dd-mm-jjjj of jjjj-mm-dd)\n- Thuisteam\n- Uitteam\n- Goals Thuis\n- Goals Uit")
st.caption("Elke rij is een gespeelde match. Voorzie minstens 10-15 wedstrijden per team voor betere schattingen.")

st.info("Start met 'streamlit run web_app.py' in deze map. Installeer eerst: pip install streamlit pandas numpy joblib xgboost scipy scikit-learn")
