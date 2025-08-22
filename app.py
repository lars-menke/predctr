import os, math, re, unicodedata, datetime as dt, requests
import numpy as np
import pandas as pd
import streamlit as st

# Breites Layout und schneller „erste Pixel“-Render
st.set_page_config(page_title="Bundesliga Predictor 25/26", page_icon="⚽", layout="wide")

# Mini-Connectivity-Check (zeigt sofort was an, statt „leerer“ Seite)
try:
    requests.get("https://api.openligadb.de/getcurrentgroup/bl1", timeout=5)
    st.caption("✅ OpenLigaDB erreichbar")
except Exception:
    st.warning("⚠️ Konnte OpenLigaDB nicht erreichen. Prüfe Netzwerk/Firewall.")

# ----------------------- Config & Secrets -----------------------
ODDS_API_KEY = st.secrets.get("ODDS_API_KEY", os.getenv("ODDS_API_KEY", "")).strip()
REGIONS = "eu,uk"                      # mehrere Buchmacher-Regionen
DEFAULT_ALPHA = 0.6                    # Blend Modell vs. Markt
DEFAULT_N = 5                          # Formfenster
LEAGUE_AVG = 2.9                       # Ligamittel Tore/Spiel
HOME_ADV = 1.1                         # Heimvorteil-Faktor

# ----------------------- Small cache ----------------------------
@st.cache_data(show_spinner=False, ttl=300)
def http_get_json(url, params=None, timeout=30):
    r = requests.get(url, params=params, timeout=timeout)
    r.raise_for_status()
    return r.json()

# ----------------------- Data sources ---------------------------
def ol_matchday(season: int, matchday: int):
    return http_get_json(f"https://api.openligadb.de/getmatchdata/bl1/{season}/{matchday}")

def odds_events():
    if not ODDS_API_KEY:
        return []
    return http_get_json("https://api.the-odds-api.com/v4/sports/soccer_germany_bundesliga/events",
                         params={"apiKey": ODDS_API_KEY})

def odds_single_event(event_id: str):
    if not ODDS_API_KEY:
        return {}
    return http_get_json(
        f"https://api.the-odds-api.com/v4/sports/soccer_germany_bundesliga/events/{event_id}/odds",
        params={"apiKey": ODDS_API_KEY, "regions": REGIONS, "markets": "h2h", "oddsFormat": "decimal"}
    )

# ----------------------- Helpers (model) ------------------------
def normalize_name(s: str) -> str:
    s0 = s.lower()
    s0 = unicodedata.normalize('NFKD', s0).encode('ascii','ignore').decode('ascii')
    s0 = re.sub(r'\b(fc|sc|sv|vfb|vfl|tsg|rb|1\.|bayer 04|bayer|borussia|hertha bsc|1 fsv|fsv|union)\b','', s0)
    s0 = re.sub(r'[^a-z0-9]+','', s0)
    return s0

def same_or_adjacent_day(utc1: str, utc2: str) -> bool:
    try:
        d1 = dt.datetime.fromisoformat((utc1 or "").replace("Z","")).date()
        d2 = dt.datetime.fromisoformat((utc2 or "").replace("Z","")).date()
        return abs((d1 - d2).days) <= 1
    except Exception:
        return (utc1 or "")[:10] == (utc2 or "")[:10]

def extract_ft(m: dict):
    for r in m.get("matchResults") or []:
        if r.get("resultTypeID")==2:
            return r.get("pointsTeam1"), r.get("pointsTeam2")
    return None, None

def compute_form_history(teams, season, matchday, n=DEFAULT_N):
    out = {t: [] for t in teams}
    # rückwärts in derselben Saison
    d, steps = matchday-1, 0
    while d>=1 and steps<10 and any(len(out[t])<n for t in teams):
        for m in ol_matchday(season, d):
            t1 = m.get("team1",{}).get("teamName")
            t2 = m.get("team2",{}).get("teamName")
            g1,g2 = extract_ft(m)
            if g1 is None: continue
            if t1 in out and len(out[t1])<n: out[t1].append((g1,g2))
            if t2 in out and len(out[t2])<n: out[t2].append((g2,g1))
        d -= 1; steps += 1
    # falls zu wenig: Vorjahressaison
    if any(len(out[t])<n for t in teams):
        prev, d, steps = season-1, 34, 0
        while d>=1 and steps<12 and any(len(out[t])<n for t in teams):
            for m in ol_matchday(prev, d):
                t1 = m.get("team1",{}).get("teamName")
                t2 = m.get("team2",{}).get("teamName")
                g1,g2 = extract_ft(m)
                if g1 is None: continue
                if t1 in out and len(out[t1])<n: out[t1].append((g1,g2))
                if t2 in out and len(out[t2])<n: out[t2].append((g2,g1))
            d -= 1; steps += 1
    return out

def strengths_from_history(history, league_avg=LEAGUE_AVG):
    avg_side = league_avg/2.0
    out={}
    for team, arr in history.items():
        if not arr: out[team]=(1.0,1.0); continue
        gf = np.array([x[0] for x in arr], dtype=float)
        ga = np.array([x[1] for x in arr], dtype=float)
        att = max(0.2, float(gf.mean())/max(0.1,avg_side))
        deff= max(0.2, max(0.1,avg_side)/max(0.1,ga.mean()))
        out[team]=(att,deff)
    return out

def expected_goals(att_h, def_h, att_a, def_a, home_adv=HOME_ADV, base=LEAGUE_AVG):
    mu_h = home_adv * att_h * (1/def_a) * (base/2.0)
    mu_a = (1/home_adv) * att_a * (1/def_h) * (base/2.0)
    return float(np.clip(mu_h, 0.1, 3.5)), float(np.clip(mu_a, 0.1, 3.5))

def poisson_pmf(lmbda, k): return (lmbda**k)*math.exp(-lmbda)/math.factorial(k)

def score_matrix(mu_h, mu_a, max_goals=6):
    home = [poisson_pmf(mu_h,i) for i in range(max_goals+1)]
    away = [poisson_pmf(mu_a,j) for j in range(max_goals+1)]
    return np.outer(home, away)

def top_k_scores(mu_h, mu_a, k=3, max_goals=6):
    mat = score_matrix(mu_h, mu_a, max_goals)
    scores=[((i,j), float(mat[i,j])) for i in range(max_goals+1) for j in range(max_goals+1)]
    scores.sort(key=lambda x: x[1], reverse=True)
    return scores[:k], mat

def market_probs(odds_payload):
    by_outcome={"1":[], "X":[], "2":[]}
    if not isinstance(odds_payload, dict): return {}
    for bm in odds_payload.get("bookmakers",[]):
        for mkt in bm.get("markets",[]):
            if mkt.get("key")!="h2h": continue
            for sel in mkt.get("outcomes",[]):
                name=str(sel.get("name","")); price=sel.get("price")
                if price is None: continue
                if name==odds_payload.get("home_team"): by_outcome["1"].append(price)
                elif name==odds_payload.get("away_team"): by_outcome["2"].append(price)
                elif name.lower() in ("draw","unentschieden"): by_outcome["X"].append(price)
    probs={}
    for k,arr in by_outcome.items():
        if arr:
            arr=sorted(arr); med=arr[len(arr)//2]
            probs[k]=1.0/med
    s=sum(probs.values())
    return {k:v/s for k,v in probs.items()} if s>0 else {}

def match_events(fixtures, events):
    mp={}
    for fx in fixtures:
        h = normalize_name(fx["home"])
        a = normalize_name(fx["away"])
        best=None
        for ev in events:
            eh = normalize_name(ev.get("home_team",""))
            ea = normalize_name(ev.get("away_team",""))
            if eh==h and ea==a and same_or_adjacent_day(fx["utc"], ev.get("commence_time","")):
                best=ev.get("id"); break
        if best:
            mp[(fx["home"], fx["away"], (fx["utc"] or "")[:10])] = best
    return mp

# ----------------------- UI ------------------------
st.set_page_config(page_title="Bundesliga Predictor 25/26", page_icon="⚽")
st.title("⚽ Bundesliga Predictor 2025/26 — Web-App (Serverless)")

left, mid, right = st.columns(3)
with left:
    matchday = st.number_input("Spieltag", 1, 34, 1, 1)
with mid:
    alpha = st.slider("α Markt-Blend", 0.0, 1.0, DEFAULT_ALPHA, 0.05)
with right:
    n = st.slider("N Formspiele", 3, 10, DEFAULT_N, 1)

season = 2025

# Auto-Run auf iOS/Safari: einmal direkt rechnen
run = True
st.button("Vorhersagen neu berechnen", type="primary", on_click=lambda: None)
if run:
    # ... rechnen ...

    with st.spinner("Lade Daten & berechne..."):
        # Fixtures
        md = ol_matchday(season, int(matchday))
        fixtures=[]
        for m in md:
            fixtures.append({
                "home": m.get("team1",{}).get("teamName"),
                "away": m.get("team2",{}).get("teamName"),
                "utc":  m.get("matchDateTimeUTC") or m.get("matchDateTime") or ""
            })
        teams = sorted({f["home"] for f in fixtures} | {f["away"] for f in fixtures})

        # Form
        hist = compute_form_history(teams, season, int(matchday), n=n)
        strengths = strengths_from_history(hist)

        # Odds
        events = odds_events() if ODDS_API_KEY else []
        ev_map = match_events(fixtures, events) if events else {}

        rows=[]
        for fx in fixtures:
            att_h, def_h = strengths[fx["home"]]
            att_a, def_a = strengths[fx["away"]]
            mu_h, mu_a = expected_goals(att_h, def_h, att_a, def_a)

            top3, mat = top_k_scores(mu_h, mu_a, k=3, max_goals=6)
            # 1X2 (nur für Info/Blend)
            p_model = {
                "1": float(np.triu(mat,1).sum()),
                "X": float(np.trace(mat)),
                "2": float(np.tril(mat,-1).sum())
            }
            s=sum(p_model.values()); p_model={k:v/s for k,v in p_model.items()} if s>0 else p_model

            # Markt
            p_market=None
            ev_id = ev_map.get((fx["home"], fx["away"], (fx["utc"] or "")[:10]))
            if ev_id:
                payload = odds_single_event(ev_id)
                if payload:
                    payload["home_team"]=payload.get("home_team") or fx["home"]
                    payload["away_team"]=payload.get("away_team") or fx["away"]
                    p_market = market_probs(payload)

            # Blend (Info)
            if p_market:
                p_blend={k: alpha*p_model.get(k,0.0)+(1-alpha)*p_market.get(k,0.0) for k in ["1","X","2"]}
                ss=sum(p_blend.values()); p_blend={k:v/ss for k,v in p_blend.items()} if ss>0 else p_blend
                alpha_used=alpha
            else:
                p_blend=p_model; alpha_used=1.0

            rows.append({
                "Heim": fx["home"], "Gast": fx["away"], "Anstoß (UTC)": fx["utc"],
                "μ_home": round(mu_h,2), "μ_away": round(mu_a,2),
                "Top": f"{top3[0][0][0]}:{top3[0][0][1]}", "P(Top)%": round(top3[0][1]*100,1),
                "2.": f"{top3[1][0][0]}:{top3[1][0][1]}",   "P2%": round(top3[1][1]*100,1),
                "3.": f"{top3[2][0][0]}:{top3[2][0][1]}",   "P3%": round(top3[2][1]*100,1),
                "α genutzt": alpha_used
            })

        st.dataframe(pd.DataFrame(rows), use_container_width=True)
        st.caption("Hinweis: Wenn keine Quoten für eine Partie vorliegen, wird automatisch nur das Modell (α=1.0) genutzt. Keine Wettberatung – verantwortungsbewusst spielen.")
else:
    st.info("Spieltag wählen und auf „Vorhersagen berechnen“ klicken.")
