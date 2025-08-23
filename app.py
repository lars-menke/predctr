import os, math, re, unicodedata, datetime as dt, requests
import numpy as np
import pandas as pd
import streamlit as st

# ===================== Config & Secrets =====================
ODDS_API_KEY = st.secrets.get("ODDS_API_KEY", os.getenv("ODDS_API_KEY", "")).strip()
REGIONS = "eu,uk"

# Basis-Niveau separat für Heim/Away (verhindert 1:1-Bias)
BASE_HOME_GOALS = 1.62
BASE_AWAY_GOALS = 1.28
HOME_ADV        = 1.15

# Modell-Feintuning
DEFAULT_N       = 5            # Formfenster je Team
DECAY_LAMBDA    = 0.22         # Zeitgewicht pro Woche (höher => jüngeres zählt mehr)
BETA_SHRINK     = 0.8          # moderates Shrinkage als Pseudo-Spiele
RHO_DC          = 0.04         # Dixon–Coles (Low-Score-Feintuning)
DRAW_DEFLATE    = 0.03         # kleine Reduktion aller Unentschieden
MAX_GOALS       = 8            # klarere Top-Ergebnisse
MU_CLIP         = 4.5          # harte Obergrenze für μ

st.set_page_config(page_title="Bundesliga Predictor 25/26", page_icon="⚽", layout="wide")

# ===================== HTTP (mit Timeouts) =====================
@st.cache_data(show_spinner=False, ttl=300)
def http_get_json(url, params=None, timeout=7):
    r = requests.get(url, params=params, timeout=timeout)
    r.raise_for_status()
    return r.json()

def ol_matchday(season: int, matchday: int):
    return http_get_json(f"https://api.openligadb.de/getmatchdata/bl1/{season}/{matchday}", timeout=7)

# ===================== Utils & Modell =====================
def normalize_name(s: str) -> str:
    s0 = (s or "").lower()
    s0 = unicodedata.normalize('NFKD', s0).encode('ascii','ignore').decode('ascii')
    s0 = re.sub(r'\b(fc|sc|sv|vfb|vfl|tsg|rb|1\.|bayer 04|bayer|borussia|hertha bsc|1 fsv|fsv|union)\b','', s0)
    s0 = re.sub(r'[^a-z0-9]+','', s0)
    return s0

def parse_date(s: str) -> dt.datetime | None:
    if not s: return None
    try: return dt.datetime.fromisoformat(s.replace("Z",""))
    except Exception: return None

def weeks_between(later: dt.datetime, earlier: dt.datetime) -> float:
    return max(0.0, (later - earlier).total_seconds() / (7*24*3600))

def extract_ft(m: dict):
    """Gibt (Tore1, Tore2, Zeit) zurück; nur Endstand (resultTypeID==2)."""
    when = m.get("matchDateTimeUTC") or m.get("matchDateTime") or ""
    for r in m.get("matchResults") or []:
        if r.get("resultTypeID") == 2:
            return r.get("pointsTeam1"), r.get("pointsTeam2"), when
    return None, None, when

# ---- Dixon–Coles + Draw-Deflate auf Score-Matrix ----
def dixon_coles_adjust(M: np.ndarray, rho=RHO_DC, draw_deflate=DRAW_DEFLATE) -> np.ndarray:
    A = M.copy()
    n_i, n_j = A.shape
    if n_i >= 2 and n_j >= 2 and rho != 0.0:
        A[0,0] *= (1.0 - rho); A[1,1] *= (1.0 - rho)
        A[1,0] *= (1.0 + rho); A[0,1] *= (1.0 + rho)
    if draw_deflate > 0:
        for d in range(min(n_i, n_j)):
            A[d,d] *= (1.0 - draw_deflate)
    s = A.sum()
    if s > 0: A /= s
    return A

def poisson_pmf(lmbda, k):
    return (lmbda**k) * math.exp(-lmbda) / math.factorial(k)

def score_matrix(mu_h, mu_a, max_goals=MAX_GOALS, apply_dc=True):
    home = [poisson_pmf(mu_h, i) for i in range(max_goals+1)]
    away = [poisson_pmf(mu_a, j) for j in range(max_goals+1)]
    M = np.outer(home, away)
    return dixon_coles_adjust(M) if apply_dc else M

def top_k_scores(mu_h, mu_a, k=3):
    M = score_matrix(mu_h, mu_a, MAX_GOALS, apply_dc=True)
    scores = [((i,j), float(M[i,j])) for i in range(MAX_GOALS+1) for j in range(MAX_GOALS+1)]
    scores.sort(key=lambda x: x[1], reverse=True)
    return scores[:k], M

# ===================== Form-Historie über Saison-Grenzen =====================
def compute_form_history(teams, season, matchday, n=DEFAULT_N, seasons_back=2):
    """
    Liefert je Team eine Liste von (gf, ga, when_str, is_home).
    Wir gehen rückwärts: erst laufende Saison (bis matchday-1), dann s-1, s-2 ...
    bis jedes Team >= n Einträge hat (oder Quellen ausgeschöpft).
    """
    out = {t: [] for t in teams}

    def feed_from(seas: int, md_list):
        for d in md_list:
            for m in ol_matchday(seas, d):
                t1 = m.get("team1",{}).get("teamName"); t2 = m.get("team2",{}).get("teamName")
                g1,g2,when = extract_ft(m)
                if g1 is None: 
                    continue
                if t1 in out and len(out[t1]) < n:
                    out[t1].append((g1, g2, when, True))
                if t2 in out and len(out[t2]) < n:
                    out[t2].append((g2, g1, when, False))
            if all(len(out[t]) >= n for t in teams):
                return True
        return False

    # 1) laufende Saison rückwärts bis matchday-1
    if matchday > 1:
        done = feed_from(season, range(matchday-1, 0, -1))
        if done: return out

    # 2) Vorjahre rückwärts
    for back in range(1, seasons_back+1):
        done = feed_from(season - back, range(34, 0, -1))
        if done: break
    return out

def _weighted_means(records, ref_dt, decay_lambda, beta, avg_base):
    """Zeitgewichtet + Shrinkage Richtung avg_base."""
    ws, gfs, gas = [], [], []
    for gf, ga, when_str, _ in records:
        when_dt = parse_date(when_str) or ref_dt
        w = math.exp(-decay_lambda * weeks_between(ref_dt, when_dt))
        ws.append(w); gfs.append(float(gf)); gas.append(float(ga))
    sum_w = sum(ws) if ws else 0.0
    if sum_w <= 0:
        return avg_base, avg_base
    gf_w = float(np.dot(ws, gfs) / sum_w)
    ga_w = float(np.dot(ws, gas) / sum_w)
    gf_sm = (gf_w*sum_w + beta*avg_base) / (sum_w + beta)
    ga_sm = (ga_w*sum_w + beta*avg_base) / (sum_w + beta)
    return gf_sm, ga_sm

def strengths_from_history(history, ref_date: dt.datetime | None,
                           base_home=BASE_HOME_GOALS, base_away=BASE_AWAY_GOALS,
                           decay_lambda=DECAY_LAMBDA, beta=BETA_SHRINK):
    """
    Für jedes Team vier Faktoren:
      att_home, def_home, att_away, def_away.
    Fallback nur wenn gar keine Daten: -> 1.0.
    """
    out = {}
    ref = ref_date or dt.datetime.utcnow()
    for team, arr in history.items():
        if not arr:
            out[team] = dict(att_home=1.0, def_home=1.0, att_away=1.0, def_away=1.0)
            continue

        home_recs = [r for r in arr if r[3] is True]
        away_recs = [r for r in arr if r[3] is False]
        both = arr

        # Heim: relativ zu BASE_HOME_GOALS
        gf_h, ga_h = _weighted_means(home_recs or both, ref, decay_lambda, beta, base_home)
        att_home = max(0.2, gf_h / max(0.1, base_home))
        def_home = max(0.2, max(0.1, base_home) / max(0.1, ga_h))

        # Auswärts: relativ zu BASE_AWAY_GOALS
        gf_a, ga_a = _weighted_means(away_recs or both, ref, decay_lambda, beta, base_away)
        att_away = max(0.2, gf_a / max(0.1, base_away))
        def_away = max(0.2, max(0.1, base_away) / max(0.1, ga_a))

        out[team] = dict(att_home=att_home, def_home=def_home,
                         att_away=att_away, def_away=def_away)
    return out

def expected_goals_home_away(h_att_home, a_def_away, a_att_away, h_def_home,
                             home_adv=HOME_ADV,
                             base_home=BASE_HOME_GOALS, base_away=BASE_AWAY_GOALS):
    mu_h = home_adv       * h_att_home * (1.0 / a_def_away) * base_home
    mu_a = (1.0/home_adv) * a_att_away * (1.0 / h_def_home) * base_away
    return float(np.clip(mu_h, 0.1, MU_CLIP)), float(np.clip(mu_a, 0.1, MU_CLIP))

# ===================== Backtesting (1X2 aus Score-Matrix) =====================
def brier_score(probs, outcome):
    return sum((probs.get(k,0)- (1 if k==outcome else 0))**2 for k in ["1","X","2"]) / 3.0

def log_loss(probs, outcome, eps=1e-12):
    p = max(probs.get(outcome, eps), eps)
    return -math.log(p)

def run_backtest(season, start_md, end_md, n_form=DEFAULT_N):
    rows, bs_list, ll_list = [], [], []
    for md in range(int(start_md), int(end_md)+1):
        fixtures = ol_matchday(season, md)
        # Teams einsammeln
        teams = set()
        for m in fixtures:
            teams.add(m.get("team1",{}).get("teamName"))
            teams.add(m.get("team2",{}).get("teamName"))
        if not teams:
            continue
        ref_dt = parse_date(fixtures[0].get("matchDateTimeUTC") or fixtures[0].get("matchDateTime") or "") \
                 or dt.datetime.utcnow()
        hist = compute_form_history(list(teams), season, md, n=n_form, seasons_back=2)
        strengths = strengths_from_history(hist, ref_dt)

        for m in fixtures:
            h = m.get("team1",{}).get("teamName"); a = m.get("team2",{}).get("teamName")
            g1,g2,_ = extract_ft(m)
            if g1 is None:   # Spiel evtl. noch nicht gespielt
                continue
            mu_h, mu_a = expected_goals_home_away(
                strengths.get(h,{"att_home":1,"def_home":1})["att_home"],
                strengths.get(a,{"att_away":1,"def_away":1})["def_away"],
                strengths.get(a,{"att_away":1,"def_away":1})["att_away"],
                strengths.get(h,{"att_home":1,"def_home":1})["def_home"]
            )
            _, M = top_k_scores(mu_h, mu_a)
            probs = {
                "1": float(np.triu(M, 1).sum()),
                "X": float(np.trace(M)),
                "2": float(np.tril(M, -1).sum())
            }
            s = sum(probs.values()); 
            if s > 0: probs = {k: v/s for k,v in probs.items()}
            outcome = "1" if g1>g2 else "2" if g2>g1 else "X"
            bs = brier_score(probs, outcome); ll = log_loss(probs, outcome)
            bs_list.append(bs); ll_list.append(ll)
            rows.append({
                "MD": md, "Heim": h, "Gast": a, "Ergebnis": f"{g1}:{g2}",
                "P(1)": round(probs["1"],3), "P(X)": round(probs["X"],3), "P(2)": round(probs["2"],3),
                "Outcome": outcome, "Brier": round(bs,3), "LogLoss": round(ll,3)
            })
    df = pd.DataFrame(rows)
    return df, (float(np.mean(bs_list)) if bs_list else None), (float(np.mean(ll_list)) if ll_list else None)

# ===================== UI =====================
tab_pred, tab_back = st.tabs(["🔮 Vorhersage", "📊 Backtest"])

# ---------- Vorhersage ----------
with tab_pred:
    st.title("🔮 Bundesliga Predictor 2025/26")
    left, mid, right = st.columns(3)
    with left:
        matchday = st.number_input("Spieltag", 1, 34, 1, 1)
    with mid:
        n = st.slider("N Formspiele", 3, 10, int(DEFAULT_N), 1)
    with right:
        st.caption("Form wird zeitgewichtet & regularisiert. Ergebnisse sind keine Wettberatung.")

    season = 2025

    # iOS/Safari: harter Auto-Run beim ersten Render + Fallback
    if "booted" not in st.session_state:
        st.session_state["booted"] = True
        auto_run = True
    else:
        auto_run = False

    recalc = st.button("Vorhersagen berechnen", type="primary")
    run = auto_run or recalc or True  # failsafe: immer rechnen

    if run:
        with st.spinner("Lade Daten & berechne..."):
            md = []
            try:
                md = ol_matchday(season, int(matchday))
            except Exception:
                md = []

            fixtures = [{
                "home": m.get("team1",{}).get("teamName"),
                "away": m.get("team2",{}).get("teamName"),
                "utc":  m.get("matchDateTimeUTC") or m.get("matchDateTime") or ""
            } for m in md]

            teams = sorted({f["home"] for f in fixtures} | {f["away"] for f in fixtures})
            ref_dt = parse_date(fixtures[0]["utc"]) if fixtures and fixtures[0]["utc"] else dt.datetime.utcnow()

            # >>> WICHTIG: Historie über Saison-Grenzen holen (gegen 1:0/1:1-Einheitsbrei)
            hist = compute_form_history(teams, season, int(matchday), n=n, seasons_back=2)
            strengths = strengths_from_history(hist, ref_dt)

            rows = []
            for fx in fixtures:
                s_h = strengths.get(fx["home"], dict(att_home=1.0, def_home=1.0, att_away=1.0, def_away=1.0))
                s_a = strengths.get(fx["away"], dict(att_home=1.0, def_home=1.0, att_away=1.0, def_away=1.0))
                mu_h, mu_a = expected_goals_home_away(
                    s_h["att_home"], s_a["def_away"], s_a["att_away"], s_h["def_home"]
                )
                top3, M = top_k_scores(mu_h, mu_a)
                rows.append({
                    "Heim": fx["home"], "Gast": fx["away"], "Anstoß": fx["utc"],
                    "μ_home": round(mu_h,2), "μ_away": round(mu_a,2),
                    "Top": f"{top3[0][0][0]}:{top3[0][0][1]}", "P(Top)%": round(top3[0][1]*100,1),
                    "2.": f"{top3[1][0][0]}:{top3[1][0][1]}", "P2%": round(top3[1][1]*100,1),
                    "3.": f"{top3[2][0][0]}:{top3[2][0][1]}", "P3%": round(top3[2][1]*100,1)
                })

            if rows:
                st.dataframe(pd.DataFrame(rows), use_container_width=True)
            else:
                st.warning("Keine Fixtures gefunden. Prüfe Saison/Spieltag.")

# ---------- Backtest ----------
with tab_back:
    st.title("📊 Backtest vergangener Saisons")
    col1,col2,col3 = st.columns(3)
    with col1:
        season_bt = st.number_input("Saison", 2016, 2025, 2024, 1)
    with col2:
        start_md = st.number_input("Start-Spieltag", 1, 34, 1, 1)
    with col3:
        end_md   = st.number_input("End-Spieltag",   1, 34, 5, 1)

    if st.button("Backtest starten"):
        with st.spinner("Lade und berechne..."):
            df, bs, ll = run_backtest(int(season_bt), int(start_md), int(end_md), n_form=DEFAULT_N)
            if bs is not None:
                st.success(f"Ø Brier-Score: {bs:.3f} (0=perfekt, 0.333=Zufall)  |  Ø LogLoss: {ll:.3f} (↓ besser)")
            else:
                st.warning("Keine abgeschlossenen Spiele im gewählten Zeitraum gefunden.")
            if not df.empty:
                st.dataframe(df, use_container_width=True)
