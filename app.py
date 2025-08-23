import os, math, re, unicodedata, datetime as dt, requests
import numpy as np
import pandas as pd
import streamlit as st

# ===================== Konfiguration & Secrets =====================
ODDS_API_KEY = st.secrets.get("ODDS_API_KEY", os.getenv("ODDS_API_KEY", "")).strip()
REGIONS = "eu,uk"

# Modell-Defaults (einfach oben zentral anpassen)
LEAGUE_AVG = 2.9          # durchschnittliche Tore / Spiel in der Liga
HOME_ADV   = 1.1          # Heimvorteil-Faktor
DEFAULT_N  = 5            # wie viele Formspiele
BASE_ALPHA = 0.6          # Baseline Blend (Modell vs. Markt)
DECAY_LAMBDA = 0.25       # Zeitgewichtung (pro Woche): höher = jüngere Spiele zählen stärker
BETA_SHRINK  = 3.0        # Shrinkage in "Pseudo-Spielen" Richtung Ligamittel
RHO_DC       = 0.06       # Dixon-Coles-Rho (>0 reduziert 0:0/1:1 leicht, erhöht 1:0/0:1)

st.set_page_config(page_title="Bundesliga Predictor 25/26", page_icon="⚽", layout="wide")

# ===================== Caching-Wrapper =====================
@st.cache_data(show_spinner=False, ttl=300)
def http_get_json(url, params=None, timeout=30):
    r = requests.get(url, params=params, timeout=timeout)
    r.raise_for_status()
    return r.json()

# ===================== Datenquellen =====================
def ol_matchday(season: int, matchday: int):
    return http_get_json(f"https://api.openligadb.de/getmatchdata/bl1/{season}/{matchday}")

def odds_events():
    if not ODDS_API_KEY:
        return []
    return http_get_json(
        "https://api.the-odds-api.com/v4/sports/soccer_germany_bundesliga/events",
        params={"apiKey": ODDS_API_KEY}
    )

def odds_single_event(event_id: str):
    if not ODDS_API_KEY:
        return {}
    return http_get_json(
        f"https://api.the-odds-api.com/v4/sports/soccer_germany_bundesliga/events/{event_id}/odds",
        params={"apiKey": ODDS_API_KEY, "regions": REGIONS, "markets": "h2h", "oddsFormat": "decimal"}
    )

# ===================== Helper / Modell =====================
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

def parse_date(s: str) -> dt.datetime | None:
    if not s:
        return None
    try:
        return dt.datetime.fromisoformat(s.replace("Z",""))
    except Exception:
        return None

def extract_ft(m: dict):
    # Fulltime-Result (resultTypeID==2); gebe auch Match-Datum zurück
    when = m.get("matchDateTimeUTC") or m.get("matchDateTime") or ""
    for r in m.get("matchResults") or []:
        if r.get("resultTypeID")==2:
            return r.get("pointsTeam1"), r.get("pointsTeam2"), when
    return None, None, when

def weeks_between(later: dt.datetime, earlier: dt.datetime) -> float:
    return max(0.0, (later - earlier).total_seconds() / (7*24*3600))

# ---- Dixon-Coles-Korrektur für Low-Score-Zellen ----
def dixon_coles_adjust(mat: np.ndarray, rho: float = RHO_DC) -> np.ndarray:
    # Kopie, nur 4 Zellen modifizieren: (0,0), (1,0), (0,1), (1,1)
    M = mat.copy()
    max_i, max_j = M.shape
    # Sicherheits-Guards
    if max_i < 2 or max_j < 2 or rho == 0.0:
        return M
    # Faktoren (vereinfachte DC-Variante): weniger 0:0 & 1:1, etwas mehr 1:0 & 0:1
    M[0,0] *= (1.0 - rho)
    M[1,1] *= (1.0 - rho)
    M[1,0] *= (1.0 + rho)
    M[0,1] *= (1.0 + rho)
    # Renormieren
    s = M.sum()
    if s > 0:
        M /= s
    return M

def poisson_pmf(lmbda, k): 
    return (lmbda**k)*math.exp(-lmbda)/math.factorial(k)

def score_matrix(mu_h, mu_a, max_goals=6, apply_dc=True):
    home = [poisson_pmf(mu_h,i) for i in range(max_goals+1)]
    away = [poisson_pmf(mu_a,j) for j in range(max_goals+1)]
    M = np.outer(home, away)
    if apply_dc:
        M = dixon_coles_adjust(M, RHO_DC)
    return M

def top_k_scores(mu_h, mu_a, k=3, max_goals=6):
    mat = score_matrix(mu_h, mu_a, max_goals=max_goals, apply_dc=True)
    scores=[((i,j), float(mat[i,j])) for i in range(max_goals+1) for j in range(max_goals+1)]
    scores.sort(key=lambda x: x[1], reverse=True)
    return scores[:k], mat

# ---- Markt-Probs & Confidence (für dynamisches α) ----
def market_probs_and_confidence(odds_payload):
    by_outcome={"1":[], "X":[], "2":[]}
    if not isinstance(odds_payload, dict):
        return {}, 0.0
    n_books = 0
    for bm in odds_payload.get("bookmakers",[]):
        had_h2h = False
        for mkt in bm.get("markets",[]):
            if mkt.get("key")!="h2h": 
                continue
            had_h2h = True
            for sel in mkt.get("outcomes",[]):
                name=str(sel.get("name","")); price=sel.get("price")
                if price is None: continue
                if name==odds_payload.get("home_team"): by_outcome["1"].append(price)
                elif name==odds_payload.get("away_team"): by_outcome["2"].append(price)
                elif name.lower() in ("draw","unentschieden"): by_outcome["X"].append(price)
        if had_h2h:
            n_books += 1

    probs={}
    spreads=[]
    for k,arr in by_outcome.items():
        if arr:
            arr=sorted(arr)
            med = arr[len(arr)//2]
            probs[k]=1.0/med
            spreads.append((max(arr)-min(arr))/max(arr))  # relative Spanne
    s=sum(probs.values())
    if s>0:
        probs={k:v/s for k,v in probs.items()}

    # Confidence in [0,1]: viele Bookies & kleine Spreads => hohe Confidence
    c_books = min(1.0, n_books/6.0)           # 6+ Bookies => 1.0
    c_spread = 1.0 - min(0.5, (np.mean(spreads) if spreads else 0.5))  # mittlere Spanne klein -> nah 1
    confidence = 0.6*c_books + 0.4*c_spread   # simple Mischung
    return probs, float(max(0.0, min(1.0, confidence)))

def dynamic_alpha(base_alpha: float, confidence: float) -> float:
    # Hohe Markt-Confidence => niedrigere α (mehr Markt), bei niedriger Confidence => höheres α (mehr Modell)
    # Skala:  α = 0.3 + 0.7*(1 - confidence), danach leicht Richtung base_alpha blenden
    a = 0.3 + 0.7*(1.0 - confidence)
    return float(0.5*a + 0.5*base_alpha)

# ===================== Form-Historie (zeitgewichtet + Shrinkage) =====================
def compute_form_history(teams, season, matchday, n=DEFAULT_N):
    """
    Liefert pro Team eine Liste von Tupeln:
      (goals_for, goals_against, match_datetime)
    aus der laufenden Saison rückwärts und ggf. Vorjahres-Fallback.
    """
    out = {t: [] for t in teams}
    # 1) aktuelle Saison rückwärts
    d, steps = matchday-1, 0
    while d>=1 and steps<10 and any(len(out[t])<n for t in teams):
        for m in ol_matchday(season, d):
            t1 = m.get("team1",{}).get("teamName")
            t2 = m.get("team2",{}).get("teamName")
            g1,g2,when = extract_ft(m)
            if g1 is None: continue
            if t1 in out and len(out[t1])<n: out[t1].append((g1,g2, when))
            if t2 in out and len(out[t2])<n: out[t2].append((g2,g1, when))
        d -= 1; steps += 1
    # 2) Vorjahres-Fallback
    if any(len(out[t])<n for t in teams):
        prev, d, steps = season-1, 34, 0
        while d>=1 and steps<12 and any(len(out[t])<n for t in teams):
            for m in ol_matchday(prev, d):
                t1 = m.get("team1",{}).get("teamName")
                t2 = m.get("team2",{}).get("teamName")
                g1,g2,when = extract_ft(m)
                if g1 is None: continue
                if t1 in out and len(out[t1])<n: out[t1].append((g1,g2, when))
                if t2 in out and len(out[t2])<n: out[t2].append((g2,g1, when))
            d -= 1; steps += 1
    return out

def strengths_from_history(history: dict, league_avg=LEAGUE_AVG, decay_lambda=DECAY_LAMBDA, beta=BETA_SHRINK,
                           ref_date: dt.datetime | None = None):
    """
    Zeitgewichtete Attack/Defense mit Shrinkage.
    w_i = exp(-lambda * Wochen_seit_Spiel)
    Shrinkage: Pseudo-Spiele (beta) Richtung Ligamittel (avg_side).
    """
    avg_side = league_avg/2.0
    out={}
    # Referenzdatum: heute, falls nicht vom Fixture gesetzt
    ref = ref_date or dt.datetime.utcnow()

    for team, arr in history.items():
        if not arr:
            out[team]=(1.0,1.0); continue

        # Gewichte berechnen
        ws = []
        gf_list, ga_list = [], []
        for gf, ga, when_str in arr:
            when_dt = parse_date(when_str) or ref
            weeks = weeks_between(ref, when_dt)
            w = math.exp(-decay_lambda * weeks)
            ws.append(w); gf_list.append(float(gf)); ga_list.append(float(ga))

        sum_w = sum(ws) if ws else 0.0
        if sum_w <= 0:
            out[team]=(1.0,1.0); continue

        # gewichtete Mittelwerte
        gf_w = float(np.dot(ws, gf_list) / sum_w)
        ga_w = float(np.dot(ws, ga_list) / sum_w)

        # Shrinkage Richtung Ligamittel (Pseudo-Spiele beta)
        gf_sm = (gf_w*sum_w + beta*avg_side) / (sum_w + beta)
        ga_sm = (ga_w*sum_w + beta*avg_side) / (sum_w + beta)

        # in Faktoren übersetzen
        att  = max(0.2, gf_sm / max(0.1, avg_side))
        deff = max(0.2, max(0.1, avg_side) / max(0.1, ga_sm))

        out[team]=(att, deff)
    return out

def expected_goals(att_h, def_h, att_a, def_a, home_adv=HOME_ADV, base=LEAGUE_AVG):
    mu_h = home_adv * att_h * (1/def_a) * (base/2.0)
    mu_a = (1/home_adv) * att_a * (1/def_h) * (base/2.0)
    return float(np.clip(mu_h, 0.1, 3.5)), float(np.clip(mu_a, 0.1, 3.5))

# ===================== UI =====================
st.title("⚽ Bundesliga Predictor 2025/26 — Web-App (Serverless)")

left, mid, right = st.columns(3)
with left:
    matchday = st.number_input("Spieltag", 1, 34, 1, 1)
with mid:
    base_alpha = st.slider("α-Basis (Modell vs. Markt)", 0.0, 1.0, float(BASE_ALPHA), 0.05)
with right:
    n = st.slider("N Formspiele", 3, 10, int(DEFAULT_N), 1)

season = 2025

# kleines Reachability-Signal (zeigt sofort „etwas“ an)
try:
    requests.get("https://api.openligadb.de/getcurrentgroup/bl1", timeout=5)
    st.caption("✅ OpenLigaDB erreichbar")
except Exception:
    st.warning("⚠️ Konnte OpenLigaDB nicht erreichen. Prüfe Netzwerk/Firewall.")

# ---- iOS-Safari Fix: Auto-Run ohne Gate ----
run = True
st.button("Vorhersagen neu berechnen", type="primary", help="Berechnet mit aktuellen Parametern neu.")

if run:
    with st.spinner("Lade Daten & berechne..."):
        # Fixtures + Referenzdatum (für Zeitgewichtung)
        md = ol_matchday(season, int(matchday))
        fixtures=[]
        for m in md:
            fixtures.append({
                "home": m.get("team1",{}).get("teamName"),
                "away": m.get("team2",{}).get("teamName"),
                "utc":  m.get("matchDateTimeUTC") or m.get("matchDateTime") or ""
            })
        teams = sorted({f["home"] for f in fixtures} | {f["away"] for f in fixtures})
        # Referenzdatum: erster Fixture-Zeitpunkt (falls vorhanden), sonst jetzt
        ref_dt = parse_date(fixtures[0]["utc"]) if fixtures and fixtures[0]["utc"] else dt.datetime.utcnow()

        # Form (zeitgewichtet + Shrinkage)
        hist = compute_form_history(teams, season, int(matchday), n=n)
        strengths = strengths_from_history(hist, ref_date=ref_dt)

        # Markt (Events → je Spiel payload)
        events = odds_events() if ODDS_API_KEY else []
        # schnelle Map: (home_norm, away_norm, date_key) -> event_id
        ev_map={}
        for ev in events:
            k = (normalize_name(ev.get("home_team","")), normalize_name(ev.get("away_team","")), (ev.get("commence_time","") or "")[:10])
            ev_map[k] = ev.get("id")

        rows=[]
        for fx in fixtures:
            att_h, def_h = strengths[fx["home"]]
            att_a, def_a = strengths[fx["away"]]
            mu_h, mu_a = expected_goals(att_h, def_h, att_a, def_a)

            # Score-Matrix inkl. Dixon-Coles
            top3, mat = top_k_scores(mu_h, mu_a, k=3, max_goals=6)
            # 1X2 aus Modell
            p_model = {"1": float(np.triu(mat,1).sum()),
                       "X": float(np.trace(mat)),
                       "2": float(np.tril(mat,-1).sum())}
            s=sum(p_model.values())
            if s>0:
                p_model={k:v/s for k,v in p_model.items()}

            # Markt + dynamisches α
            p_market=None
            confidence=0.0
            key = (normalize_name(fx["home"]), normalize_name(fx["away"]), (fx["utc"] or "")[:10])
            ev_id = ev_map.get(key)
            if not ev_id:
                # toleranter Abgleich (±1 Tag)
                for ev in events:
                    if normalize_name(ev.get("home_team",""))==key[0] and normalize_name(ev.get("away_team",""))==key[1] \
                       and same_or_adjacent_day(fx["utc"], ev.get("commence_time","")):
                        ev_id = ev.get("id"); break
            if ev_id:
                payload = odds_single_event(ev_id)
                if payload:
                    payload["home_team"]=payload.get("home_team") or fx["home"]
                    payload["away_team"]=payload.get("away_team") or fx["away"]
                    p_market, confidence = market_probs_and_confidence(payload)

            if p_market:
                alpha = dynamic_alpha(base_alpha, confidence)
                p_blend={k: alpha*p_model.get(k,0.0)+(1-alpha)*p_market.get(k,0.0) for k in ["1","X","2"]}
                ss=sum(p_blend.values())
                if ss>0:
                    p_blend={k:v/ss for k,v in p_blend.items()}
                alpha_used=alpha
            else:
                p_blend=p_model
                alpha_used=1.0

            rows.append({
                "Heim": fx["home"], "Gast": fx["away"], "Anstoß (UTC)": fx["utc"],
                "μ_home": round(mu_h,2), "μ_away": round(mu_a,2),
                "Top": f"{top3[0][0][0]}:{top3[0][0][1]}", "P(Top)%": round(top3[0][1]*100,1),
                "2.": f"{top3[1][0][0]}:{top3[1][0][1]}",   "P2%": round(top3[1][1]*100,1),
                "3.": f"{top3[2][0][0]}:{top3[2][0][1]}",   "P3%": round(top3[2][1]*100,1),
                "α genutzt": round(alpha_used,2)
            })

        st.dataframe(pd.DataFrame(rows), use_container_width=True)
        st.caption("Hinweis: Wenn keine Quoten für eine Partie vorliegen, wird automatisch nur das Modell (α=1.0) genutzt. Keine Wettberatung – verantwortungsbewusst spielen.")
