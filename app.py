"""
FakeBuster — V2: DEEP OCEAN  |  single-page scroll, no tabs, spotlight checker
streamlit run v2_ocean.py
"""
import ssl; ssl._create_default_https_context = ssl.create_default_context
import nltk, os
_nd = os.path.join(os.path.expanduser("~"), "nltk_data")
os.makedirs(_nd, exist_ok=True)
for _p in ["punkt","punkt_tab","stopwords","wordnet","omw-1.4","averaged_perceptron_tagger","averaged_perceptron_tagger_eng"]:
    try: nltk.download(_p, download_dir=_nd, quiet=True, force=False)
    except: pass

import streamlit as st, pandas as pd, numpy as np, matplotlib, warnings
matplotlib.use("Agg"); import matplotlib.pyplot as plt; import seaborn as sns
from scipy import stats; warnings.filterwarnings("ignore")
from nltk.tokenize import word_tokenize; from nltk.corpus import stopwords as nltk_sw
from nltk.stem import WordNetLemmatizer; from textblob import TextBlob
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from scipy.sparse import hstack, csr_matrix

st.set_page_config(page_title="FakeBuster · Ocean", page_icon="🌊", layout="wide", initial_sidebar_state="collapsed")
if "dark" not in st.session_state: st.session_state.dark = True
dark = st.session_state.dark

if dark:
    BG="#030c18"; CARD="rgba(4,18,34,0.95)"; CB="#00d9c018"
    TX="#c0f0ec"; TX2="#3d8c86"; AC="#00d9c0"; AC2="#ff6b9d"
    FAKE_C="#ff6b9d"; REAL_C="#00d9c0"
    PBG="#040e1e"; PTX="#3d8c86"
    SPOT_BG="rgba(0,20,40,0.97)"
else:
    BG="#f0fffe"; CARD="rgba(255,255,255,0.97)"; CB="#00a99530"
    TX="#003a36"; TX2="#2d7a74"; AC="#00a995"; AC2="#d04070"
    FAKE_C="#c03060"; REAL_C="#00a995"
    PBG="#f5fffd"; PTX="#2d7a74"
    SPOT_BG="rgba(240,255,254,0.98)"

st.markdown(f"""<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Mono:wght@400;500&family=Outfit:wght@300;400;500;600&display=swap');
#MainMenu,[data-testid="stToolbar"],[data-testid="stDecoration"],[data-testid="stSidebar"],[data-testid="stSidebarCollapseButton"],[data-testid="collapsedControl"]{{display:none!important}}
[data-testid="stHeader"]{{background:{BG}!important;border-bottom:none!important;box-shadow:none!important}}
.stApp{{background:{BG}!important;font-family:'Outfit',sans-serif!important;color:{TX}!important;min-height:100vh;overflow-x:hidden}}
.stApp::before{{content:'';position:fixed;inset:0;background:radial-gradient(ellipse 70% 50% at 20% 15%,rgba(0,217,192,0.05),transparent),radial-gradient(ellipse 50% 60% at 80% 85%,rgba(255,107,157,0.04),transparent);z-index:0;pointer-events:none}}
#bc{{position:fixed;inset:0;z-index:0;pointer-events:none}}
.main .block-container{{position:relative;z-index:1;padding:.5rem 0 2rem!important;max-width:100%!important}}
/* TOPBAR */
.topbar{{display:flex;align-items:center;justify-content:space-between;padding:12px 2rem;border-bottom:1px solid {AC}20;background:rgba(3,12,24,0.95);backdrop-filter:blur(20px);position:sticky;top:0;z-index:100}}
.tb-brand{{font-family:'Syne',sans-serif;font-size:1.1rem;font-weight:800;background:linear-gradient(120deg,{AC},{AC2});-webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text;letter-spacing:.06em}}
.tb-sub{{font-family:'DM Mono',monospace;font-size:.56rem;color:{AC}60;letter-spacing:.15em;margin-top:2px}}
/* HERO — large centered */
.hero{{text-align:center;padding:56px 2rem 40px;position:relative}}
.hero-title{{font-family:'Syne',sans-serif!important;font-size:5rem;font-weight:800;background:linear-gradient(120deg,{AC},{AC2},{AC});background-size:200% auto;-webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text;letter-spacing:.06em;margin:0;animation:shimmer 5s linear infinite}}
@keyframes shimmer{{to{{background-position:200% center}}}}
.hero-sub{{font-family:'DM Mono',monospace;font-size:.7rem;color:{TX2};letter-spacing:.2em;text-transform:uppercase;margin:14px auto 0;max-width:600px}}
/* SPOTLIGHT — centered large input area */
.spotlight{{background:{SPOT_BG};border:1px solid {AC}25;border-radius:28px;padding:36px 40px;max-width:900px;margin:0 auto 32px;backdrop-filter:blur(24px);box-shadow:0 20px 80px rgba(0,217,192,0.08)}}
.spot-title{{font-family:'Syne',sans-serif;font-size:1.1rem;font-weight:700;color:{AC};text-transform:uppercase;letter-spacing:.12em;margin:0 0 16px;text-align:center}}
/* RESULT CARD — appears inline below input */
.rcard{{background:{CARD};border-radius:20px;padding:28px 32px;max-width:900px;margin:0 auto 24px;backdrop-filter:blur(24px);box-shadow:0 8px 48px {AC}08;border:1px solid {CB};position:relative;overflow:hidden}}
.rcard::before{{content:'';position:absolute;top:0;left:30%;right:30%;height:2px;background:linear-gradient(90deg,transparent,{AC}80,transparent)}}
.vfake{{border-left:5px solid {FAKE_C}}}
.vreal{{border-left:5px solid {REAL_C}}}
.vfake h2{{font-family:'Syne',sans-serif;color:{FAKE_C}!important;margin:0 0 8px;font-size:1.8rem;font-weight:800}}
.vreal h2{{font-family:'Syne',sans-serif;color:{REAL_C}!important;margin:0 0 8px;font-size:1.8rem;font-weight:800}}
.vfake p,.vreal p{{color:{TX}!important;font-size:.92rem;margin:0}}
/* METRIC PILL ROW */
.mprow{{display:flex;gap:10px;flex-wrap:wrap;margin:16px 0}}
.mp{{background:rgba(0,217,192,.07);border:1px solid {AC}25;border-radius:50px;padding:8px 16px;text-align:center;min-width:110px}}
.mp-l{{font-family:'DM Mono',monospace;font-size:.55rem;color:{TX2};text-transform:uppercase;letter-spacing:.1em;display:block}}
.mp-v{{font-family:'Syne',sans-serif;font-size:1.1rem;font-weight:700;color:{AC};display:block}}
/* SECTION DIVIDER */
.sdiv{{display:flex;align-items:center;gap:16px;padding:0 2rem;margin:40px 0 20px}}
.sdiv-line{{flex:1;height:1px;background:linear-gradient(90deg,transparent,{AC}30,transparent)}}
.sdiv-title{{font-family:'Syne',sans-serif;font-size:.9rem;font-weight:700;color:{TX2};text-transform:uppercase;letter-spacing:.15em;white-space:nowrap}}
/* DASHBOARD GRID */
.dash-grid{{display:grid;grid-template-columns:1fr 1fr;gap:16px;padding:0 2rem;margin-bottom:16px}}
.dash-card{{background:{CARD};border:1px solid {CB};border-radius:22px;padding:22px;backdrop-filter:blur(20px);box-shadow:0 4px 24px {AC}05}}
.dash-card-full{{background:{CARD};border:1px solid {CB};border-radius:22px;padding:22px;backdrop-filter:blur(20px);box-shadow:0 4px 24px {AC}05;margin:0 2rem 16px}}
/* KPI ROW */
.krow{{display:grid;grid-template-columns:repeat(4,1fr);gap:0;margin:0 2rem 20px;border:1px solid {CB};border-radius:16px;overflow:hidden}}
.kc{{background:{CARD};padding:18px 16px;text-align:center;border-right:1px solid {CB}}}
.kc:last-child{{border-right:none}}
.kc-l{{font-family:'DM Mono',monospace;font-size:.58rem;color:{TX2};text-transform:uppercase;letter-spacing:.1em;display:block;margin-bottom:4px}}
.kc-v{{font-family:'Syne',sans-serif;font-size:1.6rem;font-weight:700;color:{AC};display:block}}
/* CLOUD ROW */
.cw{{display:flex;gap:12px;flex-wrap:wrap;padding:0 2rem;margin-bottom:24px}}
.cloud{{background:{CARD};border:1px solid {CB};border-radius:20px 20px 20px 4px;padding:14px 18px;flex:1;min-width:180px;max-width:250px;backdrop-filter:blur(16px);animation:fc ease-in-out infinite;box-shadow:0 4px 16px {AC}05;position:relative}}
.cloud::after{{content:'';position:absolute;bottom:-9px;left:16px;width:14px;height:9px;background:{CARD};border-right:1px solid {CB};border-bottom:1px solid {CB};border-radius:0 0 6px 0;clip-path:polygon(0 0,100% 0,100% 100%)}}
@keyframes fc{{0%,100%{{transform:translateY(0)}}50%{{transform:translateY(-8px)}}}}
.cl-l{{font-family:'DM Mono',monospace;font-size:.58rem;text-transform:uppercase;letter-spacing:.12em;margin-bottom:5px;font-weight:500}}
.cl-f{{color:{FAKE_C}}} .cl-r{{color:{REAL_C}}}
.cl-tx{{font-family:'Outfit',sans-serif;font-size:.8rem;color:{TX2};line-height:1.5;font-style:italic}}
.cl-sc{{font-family:'DM Mono',monospace;font-size:.7rem;margin-top:6px;font-weight:500}}
/* PILL */
.pill{{display:inline-block;padding:6px 14px;border-radius:50px;font-family:'Outfit',sans-serif;font-size:.8rem;margin:4px 0;font-weight:500}}
.pr{{background:rgba(255,107,157,.12);border:1px solid {FAKE_C}50;color:#ff8fb8}}
.pg{{background:rgba(0,217,192,.1);border:1px solid {REAL_C}50;color:{REAL_C}}}
.pa{{background:rgba(255,190,0,.12);border:1px solid #ffbe0050;color:#ffcc55}}
/* WIDGETS */
.stTextArea textarea{{background:{SPOT_BG}!important;border:2px solid {AC}35!important;border-radius:16px!important;color:{TX}!important;font-family:'Outfit',sans-serif!important;font-size:.9rem!important}}
.stTextArea textarea:focus{{border-color:{AC}!important;box-shadow:0 0 0 3px {AC}15!important}}
.stButton>button{{background:linear-gradient(135deg,{AC},{AC2})!important;color:white!important;border:none!important;border-radius:50px!important;font-family:'Outfit',sans-serif!important;font-weight:600!important;font-size:.92rem!important;padding:12px 28px!important;transition:all .25s!important;box-shadow:0 4px 20px {AC}30}}
.stButton>button:hover{{transform:translateY(-2px)!important;box-shadow:0 8px 32px {AC}50!important}}
[data-testid="stExpander"]{{background:{CARD}!important;border:1px solid {CB}!important;border-radius:16px!important}}
[data-testid="stExpander"] summary{{font-family:'DM Mono',monospace!important;color:{AC}!important;font-size:.78rem!important;letter-spacing:.08em}}
[data-testid="stMultiSelect"]>div{{background:{CARD}!important;border-color:{CB}!important;border-radius:12px!important}}
[data-testid="stAlert"]{{background:{CARD}!important;border-color:{CB}!important;border-radius:16px!important;color:{TX}!important}}
[data-testid="stDataFrame"]{{background:{CARD}!important;border:1px solid {CB}!important;border-radius:16px!important}}
.stCaption{{color:{TX2}!important;font-family:'DM Mono',monospace!important;font-size:.64rem!important}}
::-webkit-scrollbar{{width:6px}}::-webkit-scrollbar-thumb{{background:{AC}30;border-radius:3px}}
</style>
<canvas id="bc"></canvas>
<script>(function(){{const c=document.getElementById('bc');if(!c)return;const x=c.getContext('2d');let W,H;const bs=[];function rs(){{W=c.width=window.innerWidth;H=c.height=window.innerHeight;}}rs();window.addEventListener('resize',rs);for(let i=0;i<50;i++)bs.push({{x:Math.random()*3000,y:Math.random()*1200+1200,r:Math.random()*3.5+1,sp:Math.random()*.3+.07,a:Math.random()*.2+.04,wo:Math.random()*Math.PI*2,ws:Math.random()*.012+.004}});function draw(){{c.width=W;const now=Date.now()/1000;bs.forEach(b=>{{b.y-=b.sp;if(b.y<-10){{b.y=H+10;b.x=Math.random()*W;}}const wx=b.x+Math.sin(now*b.ws+b.wo)*16;x.beginPath();x.arc(wx,b.y,b.r,0,Math.PI*2);x.strokeStyle=`rgba(0,217,192,${{b.a}})`;x.lineWidth=1;x.stroke();x.beginPath();x.arc(wx,b.y,b.r*.35,0,Math.PI*2);x.fillStyle=`rgba(0,217,192,${{b.a*.4}})`;x.fill();}});requestAnimationFrame(draw);}}draw();}})();</script>""", unsafe_allow_html=True)

lem=WordNetLemmatizer(); sw=set(nltk_sw.words("english"))
def preprocess(t):
    return " ".join(lem.lemmatize(w) for w in word_tokenize(str(t).lower()) if w.isalpha() and w not in sw)
def get_ling(t):
    wds=t.split() or ["..."]; wc=max(len(wds),1); b=TextBlob(t)
    return dict(wc=wc,sent=b.sentiment.polarity,subj=b.sentiment.subjectivity,ttr=len(set(wds))/wc,excl=(t.count("!")/wc)*100,superl=sum(1 for w in wds if w.lower().endswith("est"))/wc*100)
@st.cache_resource(show_spinner=False)
def train_model():
    p=os.path.join(os.path.dirname(__file__),"deceptive-opinion.csv"); df=pd.read_csv(p)
    df["label"]=(df["deceptive"]=="deceptive").astype(int); df["tc"]=df["text"].apply(preprocess)
    tf=TfidfVectorizer(ngram_range=(1,2),max_features=10000,sublinear_tf=True); Xt=tf.fit_transform(df["tc"])
    lg=np.array([[get_ling(t)[k] for k in ["sent","ttr","wc","excl","superl"]] for t in df["text"]],dtype=float)
    clf=LogisticRegression(max_iter=1000,random_state=42); clf.fit(hstack([Xt,csr_matrix(lg)]),df["label"].values)
    return clf,tf
def predict(t,clf,tf):
    l=get_ling(t); Xt=tf.transform([preprocess(t)]); lg=np.array([[l["sent"],l["ttr"],l["wc"],l["excl"],l["superl"]]],dtype=float)
    X=hstack([Xt,csr_matrix(lg)]); prob=clf.predict_proba(X)[0]; pred=clf.predict(X)[0]
    return int(pred),float(prob[1]),float(prob[0])
@st.cache_data
def load_results():
    p=os.path.join(os.path.dirname(__file__),"results_summary.csv")
    if os.path.exists(p): return pd.read_csv(p)
    return pd.DataFrame({"model":["LR","SVM","RF","GB"]*4,"config":["A_tfidf"]*4+["B_linguistic"]*4+["A_tfidf"]*4+["B_linguistic"]*4,"dataset":["ott"]*8+["salminen"]*8,"F1":[.891,.887,.850,.832,.890,.607,.864,.838,.847,.853,.826,.786,.851,.848,.853,.849],"F1_sd":[.012,.015,.025,.021,.018,.260,.023,.023,.017,.013,.019,.015,.023,.021,.018,.022],"Precision":[.891,.885,.847,.833,.888,.675,.868,.842,.861,.855,.824,.812,.870,.839,.863,.859],"Recall":[.893,.889,.855,.834,.892,.792,.862,.836,.834,.851,.828,.763,.833,.860,.844,.840],"ROC_AUC":[.961,np.nan,.931,.916,.954,np.nan,.941,.917,.929,np.nan,.903,.883,.932,np.nan,.934,.932]})
df_res=load_results()
with st.spinner("🌊 Initialising…"): clf,tfidf=train_model()

# STICKY TOPBAR
t1,t2,t3=st.columns([3,4,1])
with t1: st.markdown(f'<div class="tb-brand">FAKEBUSTER</div><div class="tb-sub">◉ DEEP OCEAN · DS7010 DISSERTATION</div>',unsafe_allow_html=True)
with t3:
    if st.button("☀" if dark else "🌙",use_container_width=True): st.session_state.dark=not dark; st.rerun()

with st.expander("⚙ Dashboard Filters",expanded=False):
    fc1,fc2,fc3=st.columns(3)
    with fc1: f_ds=st.multiselect("Dataset",df_res["dataset"].unique().tolist(),default=df_res["dataset"].unique().tolist())
    with fc2: f_cfg=st.multiselect("Config",df_res["config"].unique().tolist(),default=df_res["config"].unique().tolist())
    with fc3: f_mdl=st.multiselect("Classifier",df_res["model"].unique().tolist(),default=df_res["model"].unique().tolist())

# ═══════════════════════════════════════════════════════════════════
# SECTION 1 — HERO
# ═══════════════════════════════════════════════════════════════════
st.markdown(f'<div class="hero"><div class="hero-title">FakeBuster</div><div class="hero-sub">◉ AI-Powered Fake Review Detection · DS7010 Dissertation · Logistic Regression + TF-IDF + Linguistic Features</div></div>',unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════
# SECTION 2 — EXAMPLE CLOUDS
# ═══════════════════════════════════════════════════════════════════
st.markdown(f'<div class="sdiv"><div class="sdiv-line"></div><div class="sdiv-title">◉ Example Detections</div><div class="sdiv-line"></div></div>',unsafe_allow_html=True)
CLOUDS=[("FAKE","Absolutely the BEST hotel ever!!! Perfect in every way!!!","94% Fake"),("GENUINE","Decent stay. Room small but clean. Breakfast average.","91% Genuine"),("FAKE","Most AMAZING experience!!! Absolutely perfect and flawless!","89% Fake"),("GENUINE","Check-in slow. Nice view. AC worked. Would stay again.","87% Genuine"),("FAKE","Greatest hotel in the universe!!! Recommend to everyone!!!","96% Fake")]
html='<div class="cw">'
for i,(lb,tx,sc) in enumerate(CLOUDS):
    f=lb=="FAKE"; cl="cl-f" if f else "cl-r"; sc_c=FAKE_C if f else REAL_C
    html+=f'<div class="cloud" style="animation-duration:{5+i}s;animation-delay:{i*1.1}s;"><div class="cl-l {cl}">{"⚠" if f else "✓"} {lb}</div><div class="cl-tx">"{tx}"</div><div class="cl-sc" style="color:{sc_c}">◉ {sc}</div></div>'
st.markdown(html+'</div>',unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════
# SECTION 3 — SPOTLIGHT CHECKER (centred, large)
# ═══════════════════════════════════════════════════════════════════
st.markdown(f'<div class="sdiv"><div class="sdiv-line"></div><div class="sdiv-title">◉ Analyse a Review</div><div class="sdiv-line"></div></div>',unsafe_allow_html=True)

FAKE_EX="This is the most amazing hotel I have ever stayed at in my entire life!!! Staff were absolutely incredible, rooms spotless. Best experience EVER!!! Recommending to everyone!!!"
GENUINE_EX="Stayed two nights in October. Room clean, decent size. Breakfast average. Location near station convenient. Staff polite but not very helpful. Would probably stay again."

spot=st.container()
with spot:
    _, sc, _ = st.columns([1,4,1])
    with sc:
        st.markdown(f'<div class="spot-title">◉ Paste or type a hotel review below</div>',unsafe_allow_html=True)
        e1,e2,_=st.columns([1,1,2])
        if e1.button("🎲 Fake Example",use_container_width=True): st.session_state["rv"]=FAKE_EX
        if e2.button("🎲 Genuine Example",use_container_width=True): st.session_state["rv"]=GENUINE_EX
        rv=st.text_area("R",value=st.session_state.get("rv",""),height=150,placeholder="Paste or type a hotel review here for deep analysis…",label_visibility="collapsed")
        run=st.button("🌊 ANALYSE WITH FAKEBUSTER",use_container_width=True)

# RESULT appears inline below, also centered
if run and rv and rv.strip():
    pred,pf,pg=predict(rv,clf,tfidf); l=get_ling(rv)
    _,rc,_=st.columns([1,4,1])
    with rc:
        cls="vfake" if pred==1 else "vreal"
        icon="⚠" if pred==1 else "✓"; label="DECEPTIVE FAKE" if pred==1 else "AUTHENTIC GENUINE"
        conf=pf if pred==1 else pg; conf_c=FAKE_C if pred==1 else REAL_C
        st.markdown(f'<div class="rcard {cls}"><h2>{icon} {label}</h2><p>FakeBuster classifies this review as <strong>{"FAKE" if pred==1 else "GENUINE"}</strong> with <strong style="color:{conf_c}">{conf*100:.1f}% confidence</strong>. {"Linguistic patterns match known deceptive writing signatures." if pred==1 else "Linguistic patterns are consistent with authentic user-generated content."}</p></div>',unsafe_allow_html=True)
        st.markdown(f'<div class="mprow"><div class="mp"><span class="mp-l">Words</span><span class="mp-v">{l["wc"]}</span></div><div class="mp"><span class="mp-l">Sentiment</span><span class="mp-v">{l["sent"]:+.2f}</span></div><div class="mp"><span class="mp-l">TTR</span><span class="mp-v">{l["ttr"]:.2f}</span></div><div class="mp"><span class="mp-l">Subjectivity</span><span class="mp-v">{l["subj"]:.2f}</span></div><div class="mp"><span class="mp-l">Excl/100w</span><span class="mp-v">{l["excl"]:.1f}</span></div><div class="mp"><span class="mp-l">Superl/100w</span><span class="mp-v">{l["superl"]:.1f}</span></div><div class="mp" style="background:rgba(255,107,157,.1)"><span class="mp-l">Fake %</span><span class="mp-v" style="color:{FAKE_C}">{pf*100:.0f}%</span></div><div class="mp" style="background:rgba(0,217,192,.1)"><span class="mp-l">Real %</span><span class="mp-v" style="color:{REAL_C}">{pg*100:.0f}%</span></div></div>',unsafe_allow_html=True)
        plt.rcParams.update({"figure.facecolor":PBG,"axes.facecolor":PBG,"axes.edgecolor":CB,"text.color":PTX,"axes.labelcolor":PTX,"xtick.color":PTX,"ytick.color":PTX})
        cv1,cv2=st.columns(2)
        with cv1:
            fig,ax=plt.subplots(figsize=(5,2.5)); fig.patch.set_facecolor(PBG)
            ax.barh(["Genuine","Fake"],[pg*100,pf*100],color=[REAL_C,FAKE_C],height=.45,alpha=.88)
            for v,y in zip([pg*100,pf*100],[0,1]): ax.text(v+1,y,f"{v:.1f}%",va="center",fontsize=10,color=PTX,fontweight="bold")
            ax.set_xlim(0,115); ax.set_title("Confidence",fontsize=10,color=AC,fontfamily="monospace",fontweight="bold")
            for s in ["top","right"]: ax.spines[s].set_visible(False)
            ax.spines["left"].set_color(CB); ax.spines["bottom"].set_color(CB)
            plt.tight_layout(); st.pyplot(fig,use_container_width=True); plt.close()
        with cv2:
            lbs=["Words÷200","Sentiment","TTR","Subjectivity","Excl÷5","Superl÷2"]
            vr=[min(l["wc"]/200,1),(l["sent"]+1)/2,l["ttr"],l["subj"],min(l["excl"]/5,1),min(l["superl"]/2,1)]
            N=len(lbs); angs=[n/N*2*np.pi for n in range(N)]+[0]; vp=vr+vr[:1]
            fig,ax=plt.subplots(figsize=(4.5,3.5),subplot_kw=dict(polar=True))
            fig.patch.set_facecolor(PBG); ax.set_facecolor(PBG)
            cr_=FAKE_C if pred==1 else REAL_C
            ax.plot(angs,vp,color=cr_,linewidth=2.5,zorder=3); ax.fill(angs,vp,color=cr_,alpha=.18)
            ax.set_xticks(angs[:-1]); ax.set_xticklabels(lbs,fontsize=7.5,color=PTX)
            ax.set_ylim(0,1); ax.set_yticks([]); ax.grid(color=CB,alpha=.5); ax.spines["polar"].set_color(CB)
            ax.set_title("Feature Radar",fontsize=10,color=AC,fontfamily="monospace",fontweight="bold",pad=14)
            plt.tight_layout(); st.pyplot(fig,use_container_width=True); plt.close()
        sigs=[]
        if l["sent"]>0.5: sigs.append(("pr","⚠ Very high sentiment — fake signal"))
        if l["excl"]>2: sigs.append(("pr","⚠ Excessive exclamations"))
        if l["ttr"]<0.5: sigs.append(("pr","⚠ Low vocabulary richness"))
        elif l["ttr"]>.75: sigs.append(("pg","✓ High vocabulary richness"))
        if l["superl"]>1: sigs.append(("pr","⚠ Many superlatives"))
        if l["subj"]>.7: sigs.append(("pr","⚠ High subjectivity"))
        if l["sent"]<0: sigs.append(("pg","✓ Mentions negatives — genuine signal"))
        if not sigs: sigs.append(("pa","◈ TF-IDF pattern used for decision"))
        st.markdown("".join(f'<div class="pill {k}">{m}</div> ' for k,m in sigs),unsafe_allow_html=True)
elif run: st.warning("◉ Please enter a review before analysing.")

# ═══════════════════════════════════════════════════════════════════
# SECTION 4 — RESULTS DASHBOARD (no tabs — all inline, scrollable)
# ═══════════════════════════════════════════════════════════════════
dff=df_res[df_res["dataset"].isin(f_ds)&df_res["config"].isin(f_cfg)&df_res["model"].isin(f_mdl)]
best=df_res.loc[df_res["F1"].idxmax()]
st.markdown(f'<div class="sdiv"><div class="sdiv-line"></div><div class="sdiv-title">◉ Model Performance Dashboard</div><div class="sdiv-line"></div></div>',unsafe_allow_html=True)
st.markdown(f'<div class="krow"><div class="kc"><span class="kc-l">Best F1</span><span class="kc-v">{df_res["F1"].max():.3f}</span></div><div class="kc"><span class="kc-l">Avg F1</span><span class="kc-v">{df_res["F1"].mean():.3f}</span></div><div class="kc"><span class="kc-l">Best AUC</span><span class="kc-v">{df_res["ROC_AUC"].max():.3f}</span></div><div class="kc"><span class="kc-l">Best Model</span><span class="kc-v" style="font-size:1.1rem">{best["model"]}</span></div></div>',unsafe_allow_html=True)

def pd_():
    plt.rcParams.update({"figure.facecolor":PBG,"axes.facecolor":PBG,"axes.edgecolor":CB,"text.color":PTX,"axes.labelcolor":PTX,"xtick.color":PTX,"ytick.color":PTX,"grid.color":CB,"grid.alpha":.3})

pd_()
dc1,dc2=st.columns(2)
with dc1:
    with st.container():
        st.markdown(f'<p style="font-family:Syne,sans-serif;font-weight:700;font-size:.85rem;color:{AC};text-transform:uppercase;letter-spacing:.1em;margin:0 0 8px">F1-Score — Ott Dataset</p>',unsafe_allow_html=True)
        sub=df_res[df_res["dataset"]=="ott"]; mdls=sub["model"].unique(); x,w=np.arange(len(mdls)),.35
        fig,ax=plt.subplots(figsize=(6,4)); fig.patch.set_facecolor(PBG)
        for cfg,off,col,lbl in [("A_tfidf",-w/2,AC,"Config A"),("B_linguistic",w/2,AC2,"Config B")]:
            s=sub[sub["config"]==cfg].set_index("model").reindex(mdls); f1=s["F1"].fillna(0); sd=s["F1_sd"].fillna(0)
            ax.bar(x+off,f1,w,label=lbl,color=col,alpha=.85); ax.errorbar(x+off,f1,yerr=sd,fmt="none",color=PTX,capsize=3,lw=.8,alpha=.5)
        ax.set_xticks(x); ax.set_xticklabels(mdls,fontsize=11); ax.set_ylim(.4,1.05)
        ax.legend(fontsize=9,facecolor=PBG,edgecolor=CB,labelcolor=PTX)
        for s in ["top","right"]: ax.spines[s].set_visible(False)
        ax.spines["left"].set_color(CB); ax.spines["bottom"].set_color(CB)
        plt.tight_layout(); st.pyplot(fig,use_container_width=True); plt.close()
with dc2:
    with st.container():
        st.markdown(f'<p style="font-family:Syne,sans-serif;font-weight:700;font-size:.85rem;color:{AC};text-transform:uppercase;letter-spacing:.1em;margin:0 0 8px">F1-Score — Salminen Dataset</p>',unsafe_allow_html=True)
        sub=df_res[df_res["dataset"]=="salminen"]; mdls=sub["model"].unique()
        fig,ax=plt.subplots(figsize=(6,4)); fig.patch.set_facecolor(PBG)
        for cfg,off,col,lbl in [("A_tfidf",-w/2,AC,"Config A"),("B_linguistic",w/2,AC2,"Config B")]:
            s=sub[sub["config"]==cfg].set_index("model").reindex(mdls); f1=s["F1"].fillna(0); sd=s["F1_sd"].fillna(0)
            ax.bar(x+off,f1,w,label=lbl,color=col,alpha=.85); ax.errorbar(x+off,f1,yerr=sd,fmt="none",color=PTX,capsize=3,lw=.8,alpha=.5)
        ax.set_xticks(x); ax.set_xticklabels(mdls,fontsize=11); ax.set_ylim(.4,1.05)
        ax.legend(fontsize=9,facecolor=PBG,edgecolor=CB,labelcolor=PTX)
        for s in ["top","right"]: ax.spines[s].set_visible(False)
        ax.spines["left"].set_color(CB); ax.spines["bottom"].set_color(CB)
        plt.tight_layout(); st.pyplot(fig,use_container_width=True); plt.close()

dc3,dc4=st.columns(2)
with dc3:
    st.markdown(f'<p style="font-family:Syne,sans-serif;font-weight:700;font-size:.85rem;color:{AC};text-transform:uppercase;letter-spacing:.1em;margin:0 0 8px">F1 Heatmap</p>',unsafe_allow_html=True)
    heat=df_res.pivot_table(index=["config","model"],columns="dataset",values="F1").round(3)
    fig2,ax2=plt.subplots(figsize=(5,4)); fig2.patch.set_facecolor(PBG); ax2.set_facecolor(PBG)
    sns.heatmap(heat,annot=True,fmt=".3f",cmap="YlGnBu",linewidths=.5,ax=ax2,vmin=.5,vmax=1,annot_kws={"size":10,"color":PTX},linecolor=CB)
    ax2.set_ylabel(""); ax2.tick_params(colors=PTX)
    plt.tight_layout(); st.pyplot(fig2,use_container_width=True); plt.close()
with dc4:
    st.markdown(f'<p style="font-family:Syne,sans-serif;font-weight:700;font-size:.85rem;color:{AC};text-transform:uppercase;letter-spacing:.1em;margin:0 0 8px">Generalisation: Ott → Salminen</p>',unsafe_allow_html=True)
    bo=df_res[(df_res["config"]=="B_linguistic")&(df_res["dataset"]=="ott")]; bs=df_res[(df_res["config"]=="B_linguistic")&(df_res["dataset"]=="salminen")]
    mg=pd.merge(bo[["model","F1"]].rename(columns={"F1":"F1_ott"}),bs[["model","F1"]].rename(columns={"F1":"F1_sal"}),on="model")
    fig4,ax4=plt.subplots(figsize=(5,4)); fig4.patch.set_facecolor(PBG); ax4.set_facecolor(PBG)
    ax4.scatter(mg["F1_ott"],mg["F1_sal"],color=AC2,s=100,zorder=3)
    for _,r in mg.iterrows(): ax4.annotate(r["model"],(r["F1_ott"],r["F1_sal"]),textcoords="offset points",xytext=(6,4),fontsize=10,color=PTX)
    lm=[min(mg["F1_ott"].min(),mg["F1_sal"].min())-.02,max(mg["F1_ott"].max(),mg["F1_sal"].max())+.02]
    ax4.plot(lm,lm,color=AC,ls="--",lw=1,alpha=.6,label="y=x"); ax4.set_xlim(lm); ax4.set_ylim(lm)
    ax4.set_xlabel("F1 on Ott"); ax4.set_ylabel("F1 on Salminen")
    ax4.legend(fontsize=9,facecolor=PBG,edgecolor=CB,labelcolor=PTX)
    for s in ["top","right"]: ax4.spines[s].set_visible(False)
    ax4.spines["left"].set_color(CB); ax4.spines["bottom"].set_color(CB)
    plt.tight_layout(); st.pyplot(fig4,use_container_width=True); plt.close()

# ═══════════════════════════════════════════════════════════════════
# SECTION 5 — ABLATION + TABLE
# ═══════════════════════════════════════════════════════════════════
st.markdown(f'<div class="sdiv"><div class="sdiv-line"></div><div class="sdiv-title">◉ Ablation Study</div><div class="sdiv-line"></div></div>',unsafe_allow_html=True)
for ds in df_res["dataset"].unique():
    st.markdown(f'<p style="font-family:Syne,sans-serif;font-weight:700;font-size:.82rem;color:{TX2};margin:4px 0 8px;padding:0 2rem">Dataset: {ds.upper()}</p>',unsafe_allow_html=True)
    sub=df_res[df_res["dataset"]==ds]
    ab=pd.merge(sub[sub["config"]=="A_tfidf"][["model","F1"]].rename(columns={"F1":"F1_A"}),sub[sub["config"]=="B_linguistic"][["model","F1"]].rename(columns={"F1":"F1_B"}),on="model")
    ab["ΔF1"]=(ab["F1_B"]-ab["F1_A"]).round(3); ab["Dir"]=ab["ΔF1"].apply(lambda x:"▲ Better" if x>0 else("▼ Worse" if x<0 else"= Equal"))
    st.dataframe(ab.style.map(lambda v:("color:#00d9c0;font-weight:bold" if "▲" in str(v) else "color:#ff6b9d;font-weight:bold" if "▼" in str(v) else ""),subset=["Dir"]).format({"F1_A":"{:.3f}","F1_B":"{:.3f}","ΔF1":"{:+.3f}"}),use_container_width=True,hide_index=True)
    t,p=stats.ttest_rel(ab["F1_B"],ab["F1_A"]); st.info(f"{'🟢' if p<.05 else '🟡'} t={t:.3f} · p={p:.4f} · "+("Significant at α=0.05" if p<.05 else "Not significant at α=0.05"))

st.markdown(f'<div class="sdiv"><div class="sdiv-line"></div><div class="sdiv-title">◉ Full Results Table</div><div class="sdiv-line"></div></div>',unsafe_allow_html=True)
st.dataframe(dff.sort_values(["dataset","config","F1"],ascending=[True,True,False]).style.format({"F1":"{:.3f}","F1_sd":"{:.3f}","Precision":"{:.3f}","Recall":"{:.3f}","ROC_AUC":lambda v:f"{v:.3f}" if pd.notna(v) else "—"}).background_gradient(subset=["F1"],cmap="YlGnBu"),use_container_width=True,hide_index=True)
st.download_button("⬇ Export CSV",dff.to_csv(index=False).encode(),"results.csv","text/csv")
st.markdown(f'<div style="text-align:center;padding:32px 0 16px;font-family:DM Mono,monospace;font-size:.58rem;color:{AC}25;letter-spacing:.15em">◉ FAKEBUSTER · DS7010 · LR+TF-IDF+LINGUISTIC · OTT ET AL. 2011 ◉</div>',unsafe_allow_html=True)
