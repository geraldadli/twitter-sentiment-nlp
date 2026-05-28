"""
app.py — Twitter Financial Sentiment Analyser
Model: geraldadli/twitter-sentiment-nlp (Hugging Face Hub)
"""

import re
import time
import numpy as np
import pandas as pd
import streamlit as st

# ─────────────────────────────────────────────────────────────────────────────
#  CONFIG
# ─────────────────────────────────────────────────────────────────────────────
HF_MODEL_ID = "geraldadli/twitter-sentiment-nlp"
MAX_LENGTH  = 64

# LABEL_0/1/2 → human label  (matches training: 0=Bearish, 1=Bullish, 2=Neutral)
ID2LABEL = {"0": "Bearish", "1": "Bullish", "2": "Neutral"}

LABEL_COLOR = {"Bullish": "#A6E3A1", "Bearish": "#F38BA8", "Neutral": "#F9E2AF"}
LABEL_BG    = {"Bullish": "rgba(166,227,161,0.12)", "Bearish": "rgba(243,139,168,0.12)", "Neutral": "rgba(249,226,175,0.12)"}
LABEL_ICON  = {"Bullish": "▲", "Bearish": "▼", "Neutral": "◆"}

# ─────────────────────────────────────────────────────────────────────────────
#  PAGE SETUP
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Market Sentiment Analyser",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;600;700&family=JetBrains+Mono:wght@400;600&display=swap');

  html, body, [data-testid="stAppViewContainer"] {
    background: #07090F; color: #CDD6F4;
    font-family: 'Space Grotesk', sans-serif;
  }
  [data-testid="stAppViewContainer"] {
    background:
      radial-gradient(ellipse 70% 40% at 15% 0%,  rgba(137,180,250,.07) 0%, transparent 65%),
      radial-gradient(ellipse 50% 35% at 85% 100%, rgba(166,227,161,.05) 0%, transparent 65%),
      #07090F;
  }
  [data-testid="stHeader"] { background: transparent; }

  /* ── Typography ── */
  .eyebrow {
    font-family: 'JetBrains Mono', monospace; font-size: .68rem;
    letter-spacing: .22em; color: #89B4FA; text-transform: uppercase;
    text-align: center; margin-bottom: .5rem;
  }
  .hero-title {
    font-family: 'Space Grotesk', sans-serif; font-weight: 700;
    font-size: clamp(2rem,5vw,3.2rem); line-height: 1.08;
    letter-spacing: -.03em; color: #CDD6F4; text-align: center; margin: 0;
  }
  .hero-title .b { color: #89B4FA; }
  .hero-title .g { color: #A6E3A1; }
  .hero-sub {
    font-family: 'JetBrains Mono', monospace; font-size: .72rem;
    color: #313244; text-align: center; margin-top: .8rem;
  }
  .section-lbl {
    font-family: 'JetBrains Mono', monospace; font-size: .64rem;
    letter-spacing: .16em; color: #313244; text-transform: uppercase;
    margin-bottom: .5rem;
  }

  /* ── Model pill ── */
  .pill {
    display: inline-flex; align-items: center; gap: .45rem;
    font-family: 'JetBrains Mono', monospace; font-size: .7rem;
    background: rgba(137,180,250,.08); border: 1px solid rgba(137,180,250,.2);
    border-radius: 100px; padding: .28rem .9rem; color: #89B4FA;
    margin: .7rem auto 0; text-align: center;
  }
  .dot {
    width: 6px; height: 6px; border-radius: 50%; background: #A6E3A1;
    box-shadow: 0 0 6px #A6E3A1; animation: blink 2s ease-in-out infinite;
  }
  @keyframes blink { 0%,100%{opacity:1} 50%{opacity:.35} }

  /* ── Card ── */
  .card {
    background: rgba(255,255,255,.028); border: 1px solid rgba(255,255,255,.07);
    border-radius: 16px; padding: 1.6rem 1.8rem; margin-bottom: 1.1rem;
    position: relative; overflow: hidden;
  }
  .card::before {
    content:''; position:absolute; inset:0;
    background: linear-gradient(135deg,rgba(137,180,250,.03) 0%,transparent 60%);
    pointer-events:none;
  }

  /* ── Sentiment result ── */
  .result-label {
    font-family: 'Space Grotesk', sans-serif; font-weight: 700;
    font-size: 2.8rem; letter-spacing: -.04em; line-height: 1; margin: .3rem 0;
  }
  .badge {
    display: inline-flex; align-items: center; gap: .3rem;
    font-family: 'JetBrains Mono', monospace; font-size: .7rem; font-weight: 600;
    padding: .26rem .85rem; border-radius: 100px; text-transform: uppercase;
    letter-spacing: .05em; margin-top: .5rem;
  }

  /* ── Prob bars ── */
  .prob-row  { margin-bottom: .8rem; }
  .prob-head { display:flex; justify-content:space-between; margin-bottom:.25rem; }
  .prob-name { font-family:'JetBrains Mono',monospace; font-size:.7rem; color:#585B70;
               letter-spacing:.1em; text-transform:uppercase; }
  .prob-pct  { font-family:'JetBrains Mono',monospace; font-size:.82rem; font-weight:600; }
  .prob-track { background:rgba(255,255,255,.05); border-radius:100px; height:7px; overflow:hidden; }
  .prob-fill  { height:7px; border-radius:100px; transition:width .6s cubic-bezier(.4,0,.2,1); }

  /* ── Token bar ── */
  .token-row {
    display:flex; align-items:center; gap:.7rem; margin-top:.65rem;
    font-family:'JetBrains Mono',monospace; font-size:.68rem; color:#45475A;
    background:rgba(255,255,255,.03); border:1px solid rgba(255,255,255,.06);
    border-radius:8px; padding:.38rem .8rem;
  }
  .token-track { flex:1; height:4px; background:rgba(255,255,255,.07); border-radius:2px; }
  .token-fill  { height:4px; border-radius:2px; }
  .infer-ms    { font-family:'JetBrains Mono',monospace; font-size:.66rem;
                 color:#313244; text-align:right; margin-top:.4rem; }

  /* ── Distribution bar ── */
  .dist-bar { display:flex; height:9px; border-radius:100px; overflow:hidden; margin:.5rem 0 .6rem; }
  .dist-lbl { font-family:'JetBrains Mono',monospace; font-size:.7rem; }

  /* ── Inputs ── */
  textarea, .stTextInput input {
    background:rgba(255,255,255,.03) !important; border:1px solid rgba(255,255,255,.08) !important;
    color:#CDD6F4 !important; border-radius:12px !important;
    font-family:'JetBrains Mono',monospace !important; font-size:.87rem !important;
    caret-color:#89B4FA;
  }
  textarea:focus, .stTextInput input:focus {
    border-color:rgba(137,180,250,.45) !important;
    box-shadow:0 0 0 3px rgba(137,180,250,.07) !important;
  }

  /* ── Buttons ── */
  [data-testid="baseButton-primary"] {
    background:linear-gradient(135deg,#89B4FA,#74C7EC) !important;
    color:#07090F !important; font-family:'JetBrains Mono',monospace !important;
    font-weight:600 !important; font-size:.82rem !important;
    letter-spacing:.05em !important; border:none !important;
    border-radius:10px !important; transition:opacity .15s,transform .1s !important;
  }
  [data-testid="baseButton-primary"]:hover { opacity:.88 !important; transform:translateY(-1px) !important; }
  [data-testid="baseButton-secondary"] {
    background:transparent !important; color:#45475A !important;
    font-family:'JetBrains Mono',monospace !important; font-size:.76rem !important;
    border:1px solid rgba(255,255,255,.08) !important; border-radius:10px !important;
  }
  [data-testid="baseButton-secondary"]:hover { color:#89B4FA !important; border-color:rgba(137,180,250,.3) !important; }

  /* ── Tabs ── */
  [data-testid="stTabs"] button {
    font-family:'JetBrains Mono',monospace !important; font-size:.76rem !important;
    color:#45475A !important; border-radius:8px 8px 0 0 !important;
  }
  [data-testid="stTabs"] button[aria-selected="true"] {
    color:#89B4FA !important; border-bottom-color:#89B4FA !important;
  }

  /* ── Metrics ── */
  [data-testid="stMetric"] {
    background:rgba(255,255,255,.02); border:1px solid rgba(255,255,255,.06);
    border-radius:12px; padding:.75rem 1rem;
  }
  [data-testid="stMetricLabel"] {
    font-family:'JetBrains Mono',monospace !important; font-size:.64rem !important;
    color:#45475A !important; text-transform:uppercase; letter-spacing:.1em;
  }
  [data-testid="stMetricValue"] {
    font-family:'Space Grotesk',sans-serif !important; font-weight:700 !important;
    font-size:1.5rem !important; color:#CDD6F4 !important;
  }

  /* ── Table ── */
  [data-testid="stDataFrame"] { font-family:'JetBrains Mono',monospace !important; font-size:.8rem !important; }

  /* ── Chip ── */
  .chip {
    display:inline-block; font-family:'JetBrains Mono',monospace; font-size:.69rem;
    background:rgba(255,255,255,.04); border:1px solid rgba(255,255,255,.08);
    border-radius:6px; padding:.12rem .48rem; color:#585B70;
    margin-right:.3rem; margin-bottom:.3rem;
  }

  hr { border-color:rgba(255,255,255,.05) !important; margin:1.1rem 0 !important; }

  .footer {
    text-align:center; font-family:'JetBrains Mono',monospace;
    font-size:.64rem; color:#1E2535; padding:2rem 0 1rem; letter-spacing:.08em;
  }

  ::-webkit-scrollbar { width:5px; }
  ::-webkit-scrollbar-track { background:#07090F; }
  ::-webkit-scrollbar-thumb { background:#1E2535; border-radius:3px; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
#  PREPROCESSING  (must match training — clean_basic from notebook)
# ─────────────────────────────────────────────────────────────────────────────
_URL     = re.compile(r"https?://\S+|www\.\S+")
_MENTION = re.compile(r"@\w+")
_HASHTAG = re.compile(r"#(\w+)")
_SPACE   = re.compile(r"\s+")
_RPUNCT  = re.compile(r"([!?.,]){2,}")
_RCHAR   = re.compile(r"(.)\1{2,}")
_NUMBER  = re.compile(r"\b\d+([.,]\d+)?\b")

_EMOJI = {
    "📈":"emoji_bullish","🚀":"emoji_bullish","🔥":"emoji_bullish","💹":"emoji_bullish",
    "📉":"emoji_bearish","💥":"emoji_bearish","😡":"emoji_bearish",
    "😐":"emoji_neutral","🤔":"emoji_hedge","😂":"emoji_sarcasm","🙃":"emoji_sarcasm",
}

def clean(text: str) -> str:
    text = str(text)
    text = _URL.sub(" ", text)
    text = _MENTION.sub(" ", text)
    text = _HASHTAG.sub(r"\1", text)
    for emoji, token in _EMOJI.items():
        text = text.replace(emoji, f" {token} ")
    text = text.lower()
    text = _NUMBER.sub(" <NUM> ", text)
    text = _RPUNCT.sub(r" \1 ", text)
    text = _RCHAR.sub(r"\1\1", text)
    text = re.sub(r"[^a-z<>()!?.,\s_]", " ", text)
    return _SPACE.sub(" ", text).strip()


# ─────────────────────────────────────────────────────────────────────────────
#  MODEL LOADING
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_model():
    from transformers import pipeline as hf_pipeline
    return hf_pipeline(
        "text-classification",
        model=HF_MODEL_ID,
        top_k=None,          # return scores for ALL 3 classes
        truncation=True,
        max_length=MAX_LENGTH,
        device=-1,           # CPU
    )

@st.cache_resource(show_spinner=False)
def load_tokenizer():
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(HF_MODEL_ID)


# ─────────────────────────────────────────────────────────────────────────────
#  INFERENCE HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def label_from_raw(raw_label: str) -> str:
    """Map 'LABEL_0' / 'Bearish' / '0' → canonical label string."""
    if raw_label in ("Bearish", "Bullish", "Neutral"):
        return raw_label
    # LABEL_0, LABEL_1, LABEL_2
    if raw_label.startswith("LABEL_"):
        return ID2LABEL[raw_label.split("_")[1]]
    # bare index "0", "1", "2"
    return ID2LABEL.get(raw_label, raw_label)

def predict_one(clf, tokenizer, text: str):
    cleaned  = clean(text)
    toks     = tokenizer(cleaned, truncation=True, max_length=MAX_LENGTH)
    n_tokens = len(toks["input_ids"])

    t0  = time.perf_counter()
    raw = clf(cleaned)
    ms  = (time.perf_counter() - t0) * 1000

    # raw is list-of-list or list-of-dict depending on transformers version
    items = raw[0] if isinstance(raw[0], list) else raw
    conf  = {label_from_raw(d["label"]): d["score"] for d in items}
    pred  = max(conf, key=conf.get)
    return pred, conf, n_tokens, ms

def predict_batch(clf, texts: list) -> list:
    cleaned = [clean(t) for t in texts]
    raw     = clf(cleaned)
    labels  = []
    for item in raw:
        items = item if isinstance(item, list) else [item]
        conf  = {label_from_raw(d["label"]): d["score"] for d in items}
        labels.append(max(conf, key=conf.get))
    return labels


# ─────────────────────────────────────────────────────────────────────────────
#  UI COMPONENTS
# ─────────────────────────────────────────────────────────────────────────────
def render_bars(conf: dict):
    for label, score in sorted(conf.items(), key=lambda x: -x[1]):
        color = LABEL_COLOR[label]
        pct   = score * 100
        st.markdown(f"""
          <div class="prob-row">
            <div class="prob-head">
              <span class="prob-name">{LABEL_ICON[label]} {label}</span>
              <span class="prob-pct" style="color:{color};">{pct:.2f}%</span>
            </div>
            <div class="prob-track">
              <div class="prob-fill" style="width:{pct:.2f}%;background:{color};"></div>
            </div>
          </div>
        """, unsafe_allow_html=True)

def render_token_bar(n: int, mx: int):
    pct   = min(n / mx * 100, 100)
    color = "#F38BA8" if pct > 85 else "#F9E2AF" if pct > 65 else "#89B4FA"
    st.markdown(f"""
      <div class="token-row">
        <span>tokens</span>
        <div class="token-track">
          <div class="token-fill" style="width:{pct:.1f}%;background:{color};"></div>
        </div>
        <span style="color:{color};">{n} / {mx}</span>
      </div>
    """, unsafe_allow_html=True)

def render_result(pred: str, conf: dict, n_tokens: int, ms: float):
    color = LABEL_COLOR[pred]
    bg    = LABEL_BG[pred]
    icon  = LABEL_ICON[pred]
    top   = conf[pred] * 100
    st.markdown(f"""
      <div class="card">
        <div class="section-lbl">Prediction</div>
        <div class="result-label" style="color:{color};">{icon} {pred.upper()}</div>
        <span class="badge" style="background:{bg};color:{color};
              border:1px solid {color}33;">{icon} {top:.2f}% confidence</span>
        <hr>
        <div class="section-lbl" style="margin-bottom:.85rem;">Score breakdown</div>
      </div>
    """, unsafe_allow_html=True)
    render_bars(conf)
    render_token_bar(n_tokens, MAX_LENGTH)
    st.markdown(f'<div class="infer-ms">⚡ {ms:.1f} ms</div>', unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
#  HERO
# ─────────────────────────────────────────────────────────────────────────────
st.markdown(f"""
  <div style="padding:2.2rem 0 1.2rem;">
    <div class="eyebrow">◈ NLP · Financial Sentiment · Transformer</div>
    <h1 class="hero-title">
      <span class="b">Market</span> Sentiment<br><span class="g">Analyser</span>
    </h1>
    <div style="display:flex;justify-content:center;">
      <span class="pill"><span class="dot"></span>{HF_MODEL_ID}</span>
    </div>
    <div class="hero-sub">Bearish · Bullish · Neutral · max_length={MAX_LENGTH} · CPU</div>
  </div>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
#  LOAD MODEL
# ─────────────────────────────────────────────────────────────────────────────
with st.spinner(f"Loading model from Hugging Face Hub… (first run ~30 s)"):
    try:
        clf       = load_model()
        tokenizer = load_tokenizer()
    except Exception as e:
        st.error(f"Could not load `{HF_MODEL_ID}`: {e}")
        st.stop()


# ─────────────────────────────────────────────────────────────────────────────
#  TABS
# ─────────────────────────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["  Single Tweet  ", "  Batch Analysis  ", "  Model Info  "])


# ════════ TAB 1 — Single tweet ════════════════════════════════════════════════
with tab1:
    st.markdown("<div style='height:.7rem'></div>", unsafe_allow_html=True)
    col_in, col_out = st.columns([1.1, 0.9], gap="large")

    with col_in:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="section-lbl">Enter financial tweet</div>', unsafe_allow_html=True)

        # Example buttons BEFORE text_area (Streamlit session state rule)
        st.markdown('<div class="section-lbl" style="margin-bottom:.4rem;">Quick examples</div>',
                    unsafe_allow_html=True)
        ec1, ec2, ec3 = st.columns(3)
        _examples = {
            "▲ Bullish": "$NVDA smashes earnings — revenue up 122%, beats all estimates 📈",
            "▼ Bearish": "Fed signals further rate hikes; markets brace for downturn 📉",
            "◆ Neutral": "Goldman Sachs maintains Q4 outlook with no revision to estimates",
        }
        for _col, (_lbl, _txt) in zip([ec1, ec2, ec3], _examples.items()):
            with _col:
                if st.button(_lbl, key=f"ex_{_lbl}", use_container_width=True):
                    st.session_state["_val"] = _txt

        tweet = st.text_area(
            label="tweet", label_visibility="collapsed",
            placeholder="Paste any financial tweet, headline, or market comment…",
            height=135, key="single_ta",
            value=st.session_state.get("_val", ""),
        )
        st.markdown("</div>", unsafe_allow_html=True)
        go = st.button("Analyse  ›", type="primary", use_container_width=True)

    with col_out:
        if go and tweet.strip():
            with st.spinner("Running inference…"):
                pred, conf, n_tok, ms = predict_one(clf, tokenizer, tweet)
            render_result(pred, conf, n_tok, ms)
            with st.expander("Preprocessed text"):
                st.code(clean(tweet), language=None)
        elif go:
            st.warning("Please enter a tweet first.")
        else:
            st.markdown("""
              <div class="card" style="text-align:center;padding:3rem 1.5rem;opacity:.35;">
                <div style="font-size:2.5rem;margin-bottom:.6rem;">🧠</div>
                <div style="font-family:'JetBrains Mono',monospace;font-size:.76rem;color:#45475A;">
                  Result will appear here
                </div>
              </div>
            """, unsafe_allow_html=True)


# ════════ TAB 2 — Batch ═══════════════════════════════════════════════════════
with tab2:
    st.markdown("<div style='height:.7rem'></div>", unsafe_allow_html=True)

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="section-lbl">One tweet per line</div>', unsafe_allow_html=True)
    batch_txt = st.text_area(
        label="batch", label_visibility="collapsed",
        placeholder=(
            "$TSLA reports record deliveries for Q2\n"
            "Inflation data worse than expected; recession fears mount\n"
            "Apple remains focused on long-term growth, Cook says\n"
            "Oil prices drop sharply on OPEC output increase\n"
            "Microsoft Azure revenues grow 28% in latest quarter"
        ),
        height=190, key="batch_ta",
    )
    st.markdown("</div>", unsafe_allow_html=True)
    go_batch = st.button("Run Batch  ›", type="primary", key="batch_btn")

    if go_batch:
        lines = [l.strip() for l in batch_txt.splitlines() if l.strip()]
        if not lines:
            st.warning("Paste at least one tweet.")
        else:
            with st.spinner(f"Analysing {len(lines)} tweet(s)…"):
                t0     = time.perf_counter()
                labels = predict_batch(clf, lines)
                total  = (time.perf_counter() - t0) * 1000

            df     = pd.DataFrame({"Tweet": lines, "Sentiment": labels})
            counts = df["Sentiment"].value_counts()
            bull   = counts.get("Bullish", 0)
            bear   = counts.get("Bearish", 0)
            neut   = counts.get("Neutral", 0)
            n      = len(df)

            # Metrics
            m1, m2, m3, m4, m5 = st.columns(5)
            m1.metric("Total",      n)
            m2.metric("▲ Bullish",  bull)
            m3.metric("▼ Bearish",  bear)
            m4.metric("◆ Neutral",  neut)
            m5.metric("⚡ ms total", f"{total:.0f}")

            # Stacked distribution bar
            bp = bull/n*100; rp = bear/n*100; np_ = neut/n*100
            st.markdown(f"""
              <div class="section-lbl" style="margin-top:.8rem;">Distribution</div>
              <div class="dist-bar">
                <div style="width:{bp:.1f}%;background:#A6E3A1;" title="Bullish {bp:.1f}%"></div>
                <div style="width:{rp:.1f}%;background:#F38BA8;" title="Bearish {rp:.1f}%"></div>
                <div style="width:{np_:.1f}%;background:#F9E2AF;" title="Neutral {np_:.1f}%"></div>
              </div>
              <div style="display:flex;gap:1.4rem;margin-bottom:.9rem;">
                <span class="dist-lbl" style="color:#A6E3A1;">▲ {bp:.1f}%</span>
                <span class="dist-lbl" style="color:#F38BA8;">▼ {rp:.1f}%</span>
                <span class="dist-lbl" style="color:#F9E2AF;">◆ {np_:.1f}%</span>
              </div>
            """, unsafe_allow_html=True)

            def _color(v):
                return {"Bullish":"color:#A6E3A1;font-weight:700",
                        "Bearish":"color:#F38BA8;font-weight:700",
                        "Neutral":"color:#F9E2AF;font-weight:700"}.get(v,"")

            st.dataframe(
                df.style.map(_color, subset=["Sentiment"]),
                use_container_width=True,
                height=min(60 + n*38, 520),
            )
            st.download_button(
                "Download CSV",
                data=df.to_csv(index=False).encode(),
                file_name="sentiment_results.csv",
                mime="text/csv",
            )


# ════════ TAB 3 — Model info ══════════════════════════════════════════════════
with tab3:
    st.markdown("<div style='height:.7rem'></div>", unsafe_allow_html=True)
    ca, cb = st.columns(2, gap="large")

    with ca:
        st.markdown(f"""
          <div class="card">
            <div class="section-lbl">Model</div>
            <div style="font-family:'Space Grotesk',sans-serif;font-weight:700;
                        font-size:1.35rem;color:#CDD6F4;margin-bottom:.5rem;">
              DistilBERT-base-uncased
            </div>
            <div style="font-family:'JetBrains Mono',monospace;font-size:.78rem;
                        color:#585B70;line-height:1.7;margin-bottom:.9rem;">
              Fully fine-tuned on Twitter Financial News Sentiment<br>
              (Hugging Face: <span style="color:#89B4FA;">{HF_MODEL_ID}</span>)
            </div>
            <hr>
            <div class="section-lbl">Label mapping</div>
            <div style="font-family:'JetBrains Mono',monospace;font-size:.8rem;line-height:2;">
              <span style="color:#45475A;">LABEL_0</span>
              <span style="color:#313244;margin:0 .4rem;">→</span>
              <span style="color:#F38BA8;font-weight:600;">Bearish ▼</span><br>
              <span style="color:#45475A;">LABEL_1</span>
              <span style="color:#313244;margin:0 .4rem;">→</span>
              <span style="color:#A6E3A1;font-weight:600;">Bullish ▲</span><br>
              <span style="color:#45475A;">LABEL_2</span>
              <span style="color:#313244;margin:0 .4rem;">→</span>
              <span style="color:#F9E2AF;font-weight:600;">Neutral ◆</span>
            </div>
            <hr>
            <div class="section-lbl">Inference config</div>
            <div style="margin-top:.4rem;">
              <span class="chip">max_length=64</span>
              <span class="chip">top_k=None</span>
              <span class="chip">truncation=True</span>
              <span class="chip">device=CPU</span>
            </div>
          </div>
        """, unsafe_allow_html=True)

    with cb:
        st.markdown("""
          <div class="card">
            <div class="section-lbl">Preprocessing — clean_basic()</div>
        """, unsafe_allow_html=True)
        for step in [
            "Strip URLs & @mentions",
            "Expand #hashtags → bare word",
            "Map emojis → semantic tokens",
            "Lowercase everything",
            "Normalise numbers → &lt;NUM&gt;",
            "Collapse repeated punctuation",
            "Collapse repeated characters",
            "Remove non-alphanumeric chars",
        ]:
            st.markdown(
                f'<div style="font-family:\'JetBrains Mono\',monospace;font-size:.76rem;'
                f'color:#45475A;padding:.22rem 0;border-bottom:1px solid rgba(255,255,255,.04);">'
                f'<span style="color:#89B4FA;margin-right:.5rem;">›</span>{step}</div>',
                unsafe_allow_html=True
            )
        st.markdown("""
            <hr>
            <div class="section-lbl">GitHub repo structure</div>
        """, unsafe_allow_html=True)
        for fname, note in [
            ("app.py",            "This file"),
            ("requirements.txt",  "Dependencies"),
        ]:
            st.markdown(
                f'<div style="font-family:\'JetBrains Mono\',monospace;font-size:.74rem;'
                f'color:#45475A;padding:.26rem 0;border-bottom:1px solid rgba(255,255,255,.04);">'
                f'<span style="color:#A6E3A1;margin-right:.5rem;">✓</span>'
                f'<span style="color:#CDD6F4;">{fname}</span>'
                f'<span style="color:#313244;margin-left:.5rem;">— {note}</span></div>',
                unsafe_allow_html=True
            )
        st.markdown("</div>", unsafe_allow_html=True)

st.markdown(
    f'<div class="footer">{HF_MODEL_ID} · Twitter Financial News Sentiment · Bearish / Bullish / Neutral</div>',
    unsafe_allow_html=True
)