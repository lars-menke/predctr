import os, math, re, unicodedata, datetime as dt, requests
import numpy as np
import pandas as pd
import streamlit as st

# ===================== Config & Secrets =====================
ODDS_API_KEY = st.secrets.get("ODDS_API_KEY", os.getenv("ODDS_API_KEY", "")).strip()
REGIONS = "eu,uk"

BASE_HOME_GOALS = 1.62
BASE_AWAY_GOALS = 1.28
HOME_ADV        = 1.15
DEFAULT_N       = 5
BASE_ALPHA      = 0.6
DECAY_LAMBDA    = 0.22
BETA_SHRINK     = 0.6
RHO_DC          = 0.04
DRAW_DEFLATE    = 0.03   # etwas stärker
MAX_GOALS       = 8      # klarere Verteilungen

st.set_page_config(page_title="Bundesliga Predictor 25/26", page_icon="⚽", layout="wide")

# ===================== HTTP =====================
@st.cache_data(show_spinner=False, ttl=300)
def http_get_json(url, params=None, timeout=7):
    r = requests.get(url, params=params, timeout=timeout)
    r.raise_for_status()
    return r.json()

def ol_matchday(season: int, matchday: int):
    return http_get_json(f"https://api.openligadb.de/getmatchdata/bl1/{season}/{matchday}", timeout=7)

def odds_events():
    if not ODDS_API_KEY: return []
    try:
        return http_get_json(
            "https://api.the-odds-api.com/v4/sports/soccer_germany_bundesliga/events",
            params={"apiKey": ODDS_API_KEY}, timeout=7
        )
    except Exception:
        return []

def odds_single_event(event_id: str):
    if not ODDS_API_KEY: return {}
    try:
        return http_get_json(
            f"https://api.the-odds-api.com/v4/sports/soccer_germany_bundesliga/events/{event_id}/odds",
            params={"apiKey": ODDS_API_KEY, "regions": REGIONS, "markets": "h2h", "oddsFormat": "decimal"},
            timeout=7
        )
    except Exception:
        return {}

# ===================== Helpers / Modell =====================
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

def extract_ft(m: dict):
    when = m.get("matchDateTimeUTC") or m.get("matchDateTime") or ""
    for r in m.get("matchResults") or []:
        if r.get("resultTypeID")==2:
            return r.get("pointsTeam1"), r.get("pointsTeam2"), when
    return None, None, when

def weeks_between(later: dt.datetime, earlier: dt.datetime) -> float:
    return max(0.0, (later - earlier).total_seconds() / (7*24*3600))

# ---- DC + Draw-Deflate ----
def dixon_coles_adjust(mat: np.ndarray, rho=RHO_DC, draw_deflate=DRAW_DEFLATE) -> np.ndarray:
    M = mat.copy()
    n_i, n_j = M.shape
    if n_i >= 2 and n_j >= 2 and rho != 0.0:
        M[0,0] *= (1.0 - rho); M[1,1] *= (1.0 - rho)
        M[1,0] *= (1.0 + rho); M[0,1] *= (1.0 + rho)
    if draw_deflate > 0.0:
        for d in range(min(n_i,n_j)): M[d,d] *= (1.0 - draw_deflate)
    s = M.sum()
    if s>0: M/=s
    return M

def poisson_pmf(lmbda, k): 
    return (lmbda**k)*math.exp(-lmbda)/math.factorial(k)

def score_matrix(mu_h, mu_a, max_goals=MAX_GOALS, apply_dc=True):
    home=[poisson_pmf(mu_h,i) for i in range(max_goals+1)]
    away=[poisson_pmf(mu_a,j) for j in range(max_goals+1)]
    M=np.outer(home,away)
    if apply_dc: M=dixon_coles_adjust(M)
    return M

def top_k_scores(mu_h, mu_a, k=3):
    mat = score_matrix(mu_h, mu_a, max_goals=MAX_GOALS, apply_dc=True)
    scores=[((i,j), float(mat[i,j])) for i in range(MAX_GOALS+1) for j in range(MAX_GOALS+1)]
    scores.sort(key=lambda x:x[1], reverse=True)
    return scores[:k], mat

# ===================== Form & Stärken =====================
def compute_form_history(teams, season, matchday, n=DEFAULT_N):
    out={t:[] for t in teams}
    d=matchday-1
    while d>=1 and any(len(out[t])<n for t in teams):
        for m in ol_matchday(season,d):
            t1=m.get("team1",{}).get("teamName"); t2=m.get("team2",{}).get("teamName")
            g1,g2,when=extract_ft(m)
            if g1 is None: continue
            if t1 in out and len(out[t1])<n: out[t1].append((g1,g2,when,True))
            if t2 in out and len(out[t2])<n: out[t2].append((g2,g1,when,False))
        d-=1
    return out

def _weighted_means(records, ref_dt, decay_lambda, beta, avg_base):
    ws,gfs,gas=[],[],[]
    for gf,ga,when_str,_ in records:
        when_dt=parse_date(when_str) or ref_dt
        w=math.exp(-decay_lambda*weeks_between(ref_dt,when_dt))
        ws.append(w); gfs.append(float(gf)); gas.append(float(ga))
    sum_w=sum(ws) if ws else 0.0
    if sum_w<=0: return avg_base,avg_base
    gf_w=np.dot(ws,gfs)/sum_w; ga_w=np.dot(ws,gas)/sum_w
    gf_sm=(gf_w*sum_w+beta*avg_base)/(sum_w+beta)
    ga_sm=(ga_w*sum_w+beta*avg_base)/(sum_w+beta)
    return gf_sm,ga_sm

def strengths_from_history(history, ref_date:dt.datetime|None,
                           base_home=BASE_HOME_GOALS, base_away=BASE_AWAY_GOALS,
                           decay_lambda=DECAY_LAMBDA, beta=BETA_SHRINK):
    out={}; ref=ref_date or dt.datetime.utcnow()
    for team,arr in history.items():
        if not arr:
            out[team]=dict(att_home=1,def_home=1,att_away=1,def_away=1); continue
        home_recs=[r for r in arr if r[3]]; away_recs=[r for r in arr if not r[3]]
        both=arr
        gf_h,ga_h=_weighted_means(home_recs or both,ref,decay_lambda,beta,base_home)
        att_home=max(0.2,gf_h/max(0.1,base_home)); def_home=max(0.2,max(0.1,base_home)/max(0.1,ga_h))
        gf_a,ga_a=_weighted_means(away_recs or both,ref,decay_lambda,beta,base_away)
        att_away=max(0.2,gf_a/max(0.1,base_away)); def_away=max(0.2,max(0.1,base_away)/max(0.1,ga_a))
        out[team]=dict(att_home=att_home,def_home=def_home,att_away=att_away,def_away=def_away)
    return out

def expected_goals_home_away(h_att_home,a_def_away,a_att_away,h_def_home,
                             home_adv=HOME_ADV,
                             base_home=BASE_HOME_GOALS, base_away=BASE_AWAY_GOALS):
    mu_h=home_adv*h_att_home*(1/a_def_away)*base_home
    mu_a=(1/home_adv)*a_att_away*(1/h_def_home)*base_away
    return float(np.clip(mu_h,0.1,4.5)), float(np.clip(mu_a,0.1,4.5))

# ===================== Backtesting =====================
def brier_score(probs, outcome):
    return sum((probs.get(k,0)- (1 if k==outcome else 0))**2 for k in ["1","X","2"])/3

def log_loss(probs, outcome, eps=1e-12):
    p=probs.get(outcome,eps)
    return -math.log(p+eps)

def run_backtest(season,start,end):
    rows=[]; bs_list=[]; ll_list=[]
    for md in range(start,end+1):
        fixtures=ol_matchday(season,md)
        teams=set()
        for m in fixtures:
            teams.add(m.get("team1",{}).get("teamName"))
            teams.add(m.get("team2",{}).get("teamName"))
        ref_dt=parse_date(fixtures[0].get("matchDateTimeUTC") or "") if fixtures else dt.datetime.utcnow()
        hist=compute_form_history(list(teams),season,md)
        strengths=strengths_from_history(hist,ref_dt)
        for m in fixtures:
            h=m.get("team1",{}).get("teamName"); a=m.get("team2",{}).get("teamName")
            g1,g2,_=extract_ft(m)
            if g1 is None: continue
            mu_h,mu_a=expected_goals_home_away(
                strengths[h]["att_home"],strengths[a]["def_away"],
                strengths[a]["att_away"],strengths[h]["def_home"]
            )
            _,mat=top_k_scores(mu_h,mu_a)
            probs={"1":float(np.triu(mat,1).sum()),"X":float(np.trace(mat)),"2":float(np.tril(mat,-1).sum())}
            s=sum(probs.values())
            if s>0: probs={k:v/s for k,v in probs.items()}
            outcome="1" if g1>g2 else "2" if g2>g1 else "X"
            bs=brier_score(probs,outcome); ll=log_loss(probs,outcome)
            bs_list.append(bs); ll_list.append(ll)
            rows.append({"MD":md,"Heim":h,"Gast":a,"Ergebnis":f"{g1}:{g2}",
                         "P(1)":round(probs["1"],2),"P(X)":round(probs["X"],2),"P(2)":round(probs["2"],2),
                         "Outcome":outcome,"Brier":round(bs,3),"LogLoss":round(ll,3)})
    df=pd.DataFrame(rows)
    return df, np.mean(bs_list), np.mean(ll_list)

# ===================== UI =====================
tab_pred, tab_back = st.tabs(["🔮 Vorhersage","📊 Backtest"])

with tab_pred:
    st.title("🔮 Bundesliga Predictor 2025/26")
    left,mid,right=st.columns(3)
    with left: matchday=st.number_input("Spieltag",1,34,1,1)
    with mid: base_alpha=st.slider("α-Basis",0.0,1.0,float(BASE_ALPHA),0.05)
    with right: n=st.slider("N Formspiele",3,10,int(DEFAULT_N),1)
    season=2025
    st.button("Vorhersagen neu berechnen",type="primary")
    with st.spinner("Berechne..."):
        md=ol_matchday(season,int(matchday))
        fixtures=[{"home":m.get("team1",{}).get("teamName"),
                   "away":m.get("team2",{}).get("teamName"),
                   "utc":m.get("matchDateTimeUTC") or m.get("matchDateTime") or ""} for m in md]
        teams=sorted({f["home"] for f in fixtures}|{f["away"] for f in fixtures})
        ref_dt=parse_date(fixtures[0]["utc"]) if fixtures and fixtures[0]["utc"] else dt.datetime.utcnow()
        hist=compute_form_history(teams,season,int(matchday),n=n)
        strengths=strengths_from_history(hist,ref_dt)
        rows=[]
        for fx in fixtures:
            s_h=strengths[fx["home"]]; s_a=strengths[fx["away"]]
            mu_h,mu_a=expected_goals_home_away(s_h["att_home"],s_a["def_away"],s_a["att_away"],s_h["def_home"])
            top3,mat=top_k_scores(mu_h,mu_a)
            p_model={"1":float(np.triu(mat,1).sum()),"X":float(np.trace(mat)),"2":float(np.tril(mat,-1).sum())}
            s=sum(p_model.values()); 
            if s>0: p_model={k:v/s for k,v in p_model.items()}
            rows.append({"Heim":fx["home"],"Gast":fx["away"],"Anstoß":fx["utc"],
                         "μ_home":round(mu_h,2),"μ_away":round(mu_a,2),
                         "Top":f"{top3[0][0][0]}:{top3[0][0][1]}","P(Top)%":round(top3[0][1]*100,1)})
        st.dataframe(pd.DataFrame(rows),use_container_width=True)

with tab_back:
    st.title("📊 Backtest vergangener Saisons")
    season=st.number_input("Saison",2020,2025,2024,1)
    start=st.number_input("Start-Spieltag",1,34,1,1)
    end=st.number_input("End-Spieltag",1,34,5,1)
    if st.button("Backtest starten"):
        with st.spinner("Lade und berechne..."):
            df,bs,ll=run_backtest(season,int(start),int(end))
            st.write(f"🔹 Ø Brier-Score: {bs:.3f} (0 perfekt, 0.333 Zufall)")
            st.write(f"🔹 Ø LogLoss: {ll:.3f} (niedriger = besser)")
            st.dataframe(df,use_container_width=True)
