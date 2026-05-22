"""
app.py  —  Sentiment-Driven Market Analyser
Financial Tweet Sentiment  |  TF-IDF + LinearSVC
"""

import re
import json
import joblib
import numpy as np
import pandas as pd
import streamlit as st

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Market Sentiment Analyser",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Custom CSS — Bloomberg terminal meets modern SaaS ─────────────────────────
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=Syne:wght@400;600;800&display=swap');

  /* ── Base ── */
  html, body, [data-testid="stAppViewContainer"] {
    background-color: #080C14;
    color: #E2E8F0;
    font-family: 'Syne', sans-serif;
  }

  [data-testid="stAppViewContainer"] {
    background:
      radial-gradient(ellipse 80% 50% at 50% -10%, rgba(0, 200, 100, 0.07) 0%, transparent 70%),
      #080C14;
  }

  [data-testid="stHeader"] { background: transparent; }
  [data-testid="stSidebar"] { background: #0D1420; border-right: 1px solid #1E2D40; }

  /* ── Hero header ── */
  .hero {
    text-align: center;
    padding: 2.5rem 0 1.5rem;
    position: relative;
  }
  .hero-eyebrow {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.72rem;
    letter-spacing: 0.22em;
    color: #00C96B;
    text-transform: uppercase;
    margin-bottom: 0.6rem;
  }
  .hero-title {
    font-family: 'Syne', sans-serif;
    font-weight: 800;
    font-size: clamp(2rem, 5vw, 3.4rem);
    line-height: 1.08;
    letter-spacing: -0.02em;
    color: #F0F4FF;
    margin: 0;
  }
  .hero-title span { color: #00C96B; }
  .hero-sub {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.8rem;
    color: #4A6080;
    margin-top: 0.9rem;
  }

  /* ── Cards ── */
  .card {
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 14px;
    padding: 1.6rem 1.8rem;
    margin-bottom: 1.2rem;
    position: relative;
    overflow: hidden;
    transition: border-color .2s;
  }
  .card::before {
    content: '';
    position: absolute;
    inset: 0;
    background: linear-gradient(135deg, rgba(0,201,107,0.04) 0%, transparent 60%);
    pointer-events: none;
  }
  .card:hover { border-color: rgba(0,201,107,0.25); }

  /* ── Sentiment badges ── */
  .badge {
    display: inline-flex;
    align-items: center;
    gap: 0.4rem;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.78rem;
    font-weight: 600;
    padding: 0.3rem 0.9rem;
    border-radius: 100px;
    letter-spacing: 0.06em;
    text-transform: uppercase;
  }
  .badge-bullish  { background: rgba(0,201,107,0.15); color: #00C96B; border: 1px solid rgba(0,201,107,0.35); }
  .badge-bearish  { background: rgba(239,68,68,0.15);  color: #F87171; border: 1px solid rgba(239,68,68,0.35); }
  .badge-neutral  { background: rgba(245,158,11,0.15); color: #FCD34D; border: 1px solid rgba(245,158,11,0.35); }

  /* ── Big result display ── */
  .result-sentiment {
    font-family: 'Syne', sans-serif;
    font-weight: 800;
    font-size: 3.2rem;
    letter-spacing: -0.03em;
    line-height: 1;
    margin: 0.4rem 0;
  }
  .result-bullish { color: #00C96B; }
  .result-bearish { color: #F87171; }
  .result-neutral { color: #FCD34D; }

  /* ── Confidence bar ── */
  .conf-label {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.7rem;
    color: #4A6080;
    text-transform: uppercase;
    letter-spacing: 0.12em;
    margin-bottom: 0.25rem;
  }
  .conf-track {
    background: rgba(255,255,255,0.06);
    border-radius: 100px;
    height: 6px;
    overflow: hidden;
  }
  .conf-fill-bullish { background: linear-gradient(90deg, #00C96B, #00E87A); border-radius: 100px; height: 6px; transition: width .6s ease; }
  .conf-fill-bearish { background: linear-gradient(90deg, #EF4444, #F87171); border-radius: 100px; height: 6px; transition: width .6s ease; }
  .conf-fill-neutral { background: linear-gradient(90deg, #F59E0B, #FCD34D); border-radius: 100px; height: 6px; transition: width .6s ease; }

  /* ── Tabs ── */
  [data-testid="stTabs"] button {
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 0.8rem !important;
    letter-spacing: 0.08em;
    color: #4A6080 !important;
    padding: 0.5rem 1.2rem !important;
    border-radius: 8px 8px 0 0 !important;
    transition: color .2s;
  }
  [data-testid="stTabs"] button[aria-selected="true"] {
    color: #00C96B !important;
    border-bottom-color: #00C96B !important;
  }

  /* ── Input ── */
  textarea, .stTextInput input {
    background: rgba(255,255,255,0.04) !important;
    border: 1px solid rgba(255,255,255,0.1) !important;
    color: #E2E8F0 !important;
    border-radius: 10px !important;
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 0.9rem !important;
    caret-color: #00C96B;
    transition: border-color .2s !important;
  }
  textarea:focus, .stTextInput input:focus {
    border-color: rgba(0,201,107,0.5) !important;
    box-shadow: 0 0 0 3px rgba(0,201,107,0.08) !important;
  }

  /* ── Buttons ── */
  [data-testid="baseButton-primary"] {
    background: #00C96B !important;
    color: #080C14 !important;
    font-family: 'IBM Plex Mono', monospace !important;
    font-weight: 600 !important;
    font-size: 0.85rem !important;
    letter-spacing: 0.06em !important;
    border: none !important;
    border-radius: 10px !important;
    padding: 0.55rem 1.6rem !important;
    transition: opacity .15s, transform .1s !important;
  }
  [data-testid="baseButton-primary"]:hover { opacity: 0.88 !important; transform: translateY(-1px) !important; }
  [data-testid="baseButton-secondary"] {
    background: transparent !important;
    color: #4A6080 !important;
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 0.8rem !important;
    border: 1px solid rgba(255,255,255,0.1) !important;
    border-radius: 10px !important;
  }

  /* ── Data table ── */
  [data-testid="stDataFrame"] { font-family: 'IBM Plex Mono', monospace !important; font-size: 0.82rem !important; }

  /* ── Metric ── */
  [data-testid="stMetric"] { background: rgba(255,255,255,0.03); border: 1px solid rgba(255,255,255,0.07); border-radius: 10px; padding: 0.8rem 1rem; }
  [data-testid="stMetricLabel"] { font-family: 'IBM Plex Mono', monospace !important; font-size: 0.7rem !important; color: #4A6080 !important; text-transform: uppercase; letter-spacing: 0.1em; }
  [data-testid="stMetricValue"] { font-family: 'Syne', sans-serif !important; font-weight: 800 !important; font-size: 1.6rem !important; color: #F0F4FF !important; }

  /* ── Divider ── */
  hr { border-color: rgba(255,255,255,0.06) !important; margin: 1.5rem 0 !important; }

  /* ── Section label ── */
  .section-label {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.68rem;
    color: #2A4060;
    letter-spacing: 0.18em;
    text-transform: uppercase;
    margin-bottom: 0.6rem;
  }

  /* ── Preprocessing chip ── */
  .chip {
    display: inline-block;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.72rem;
    background: rgba(255,255,255,0.05);
    border: 1px solid rgba(255,255,255,0.09);
    border-radius: 6px;
    padding: 0.15rem 0.55rem;
    color: #607090;
    margin-right: 0.3rem;
    margin-bottom: 0.3rem;
  }

  /* ── Footer ── */
  .footer {
    text-align: center;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.68rem;
    color: #1E2D40;
    padding: 2rem 0 1rem;
    letter-spacing: 0.08em;
  }

  /* ── Scrollbar ── */
  ::-webkit-scrollbar { width: 6px; }
  ::-webkit-scrollbar-track { background: #080C14; }
  ::-webkit-scrollbar-thumb { background: #1E2D40; border-radius: 3px; }
</style>
""", unsafe_allow_html=True)

# ── Preprocessing (identical to notebook) ─────────────────────────────────────
URL_RE          = re.compile(r"https?://\S+|www\.\S+")
MENTION_RE      = re.compile(r"@\w+")
HASHTAG_RE      = re.compile(r"#(\w+)")
MULTI_SPACE_RE  = re.compile(r"\s+")
REPEAT_PUNCT_RE = re.compile(r"([!?.,]){2,}")
REPEAT_CHAR_RE  = re.compile(r"(.)\1{2,}")
NUMBER_RE       = re.compile(r"\b\d+([.,]\d+)?\b")

EMOJI_MAP = {
    "📈": "emoji_bullish", "🚀": "emoji_bullish", "🔥": "emoji_bullish", "💹": "emoji_bullish",
    "📉": "emoji_bearish", "💥": "emoji_bearish", "😡": "emoji_bearish",
    "😐": "emoji_neutral",  "🤔": "emoji_hedge",
    "😂": "emoji_sarcasm",  "🙃": "emoji_sarcasm",
}

def replace_emojis(text: str) -> str:
    for e, token in EMOJI_MAP.items():
        text = text.replace(e, f" {token} ")
    return text

def clean_basic(text: str) -> str:
    text = str(text)
    text = URL_RE.sub(" ", text)
    text = MENTION_RE.sub(" ", text)
    text = HASHTAG_RE.sub(r"\1", text)
    text = replace_emojis(text)
    text = text.lower()
    text = NUMBER_RE.sub(" <NUM> ", text)
    text = REPEAT_PUNCT_RE.sub(r" \1 ", text)
    text = REPEAT_CHAR_RE.sub(r"\1\1", text)
    text = re.sub(r"[^a-z<>()!?.,\s_]", " ", text)
    text = MULTI_SPACE_RE.sub(" ", text).strip()
    return text

# ── Model loading ─────────────────────────────────────────────────────────────
LABEL_MAP  = {0: "Bearish", 1: "Bullish", 2: "Neutral"}
LABEL_CLR  = {"Bullish": "bullish", "Bearish": "bearish", "Neutral": "neutral"}
LABEL_ICON = {"Bullish": "▲", "Bearish": "▼", "Neutral": "◆"}

@st.cache_resource(show_spinner=False)
def load_model():
    model     = joblib.load("best_model_tfidf_svm.joblib")
    with open("label_map.json") as f:
        lmap  = {int(k): v for k, v in json.load(f).items()}
    return model, lmap

# ── Prediction helpers ────────────────────────────────────────────────────────
def predict_single(model, text: str):
    """Return (label_str, confidence_dict)."""
    cleaned = clean_basic(text)
    pred_id = model.predict([cleaned])[0]
    label   = LABEL_MAP[pred_id]

    # LinearSVC → decision function scores → softmax-like proxy
    scores = model.decision_function([cleaned])[0]      # shape (3,)
    exp    = np.exp(scores - scores.max())
    probs  = exp / exp.sum()
    conf   = {LABEL_MAP[i]: float(probs[i]) for i in range(3)}
    return label, conf

def predict_batch(model, texts: list[str]):
    cleaned = [clean_basic(t) for t in texts]
    pred_ids = model.predict(cleaned)
    labels   = [LABEL_MAP[p] for p in pred_ids]
    return labels

# ── Render helpers ────────────────────────────────────────────────────────────
def render_confidence_bar(label: str, conf: dict):
    cls = LABEL_CLR[label].lower()
    for sentiment, pct in sorted(conf.items(), key=lambda x: -x[1]):
      s_cls = LABEL_CLR[sentiment].lower()
      st.markdown(f"""
        <div style="margin-bottom:0.7rem;">
          <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:0.2rem;">
            <span class="conf-label">{LABEL_ICON[sentiment]} {sentiment}</span>
            <span style="font-family:'IBM Plex Mono',monospace;font-size:0.78rem;color:#607090;">{pct*100:.1f}%</span>
          </div>
          <div class="conf-track">
            <div class="conf-fill-{s_cls}" style="width:{pct*100:.1f}%"></div>
          </div>
        </div>
      """, unsafe_allow_html=True)

def render_result(label: str, conf: dict):
    cls  = LABEL_CLR[label]
    icon = LABEL_ICON[label]
    top_pct = conf[label] * 100
    st.markdown(f"""
      <div class="card">
        <div class="section-label">Prediction result</div>
        <div class="result-sentiment result-{cls}">{icon} {label.upper()}</div>
        <div style="margin:1rem 0 0.5rem;">
          <span class="badge badge-{cls}">{icon} {top_pct:.1f}% confidence</span>
        </div>
        <hr style="margin:1rem 0!important;">
        <div class="section-label" style="margin-bottom:0.9rem;">Score breakdown</div>
      </div>
    """, unsafe_allow_html=True)
    render_confidence_bar(label, conf)

# ─────────────────────────────────────────────────────────────────────────────
#  LAYOUT
# ─────────────────────────────────────────────────────────────────────────────

# Hero
st.markdown("""
<div class="hero">
  <div class="hero-eyebrow">◈ NLP · Financial Sentiment · TF-IDF + LinearSVC</div>
  <h1 class="hero-title">Market <span>Sentiment</span><br>Analyser</h1>
  <div class="hero-sub">Twitter Financial News  ·  Bearish / Bullish / Neutral</div>
</div>
""", unsafe_allow_html=True)

# Load model
with st.spinner("Loading model…"):
    try:
        model, lmap = load_model()
        model_ok = True
    except FileNotFoundError:
        model_ok = False

if not model_ok:
    st.error(
        "**Model file not found.**  "
        "Run `save_best_model.py` (or the notebook Section 11 cell) to generate "
        "`best_model_tfidf_svm.joblib` and `label_map.json`, then place them "
        "in the same directory as this app."
    )
    st.stop()

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab_single, tab_batch, tab_about = st.tabs(["  Single Tweet  ", "  Batch Analysis  ", "  Model Info  "])

# ════════════════════════════════════════════════════════════
#  TAB 1 — Single tweet analyser
# ════════════════════════════════════════════════════════════
with tab_single:
    st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)

    col_input, col_result = st.columns([1.1, 0.9], gap="large")

    with col_input:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="section-label">Enter financial tweet</div>', unsafe_allow_html=True)
        tweet_input = st.text_area(
            label="tweet_text",
            label_visibility="collapsed",
            placeholder="e.g.  $AAPL beats Q3 earnings estimates, revenue surges 12% YoY 📈",
            height=140,
            key="single_input",
        )

        # Quick-fill examples
        st.markdown('<div class="section-label" style="margin-top:1rem;">Try an example</div>', unsafe_allow_html=True)
        ex_col1, ex_col2, ex_col3 = st.columns(3)
        examples = {
            "▲ Bullish": "$NVDA smashes earnings — revenue up 122%, beats on all metrics",
            "▼ Bearish": "Fed signals further rate hikes; markets brace for prolonged downturn",
            "◆ Neutral": "Goldman Sachs maintains its Q4 outlook with no revision to estimates",
        }
        for col, (lbl, ex_text) in zip([ex_col1, ex_col2, ex_col3], examples.items()):
            with col:
                if st.button(lbl, key=f"ex_{lbl}", use_container_width=True):
                    st.session_state["single_input"] = ex_text
                    st.rerun()

        st.markdown("</div>", unsafe_allow_html=True)

        analyse_btn = st.button("Analyse Tweet ›", type="primary", use_container_width=True, key="analyse_btn")

    with col_result:
        text_to_analyse = st.session_state.get("single_input", "") or tweet_input
        if analyse_btn and text_to_analyse.strip():
            label, conf = predict_single(model, text_to_analyse)
            render_result(label, conf)

            # Show preprocessed text
            with st.expander("Preprocessed text"):
                st.markdown(
                    f'<span class="chip">clean_basic()</span>',
                    unsafe_allow_html=True
                )
                st.code(clean_basic(text_to_analyse), language=None)
        elif analyse_btn:
            st.warning("Please enter a tweet first.")
        else:
            st.markdown("""
              <div class="card" style="text-align:center;padding:3rem 1.5rem;opacity:0.45;">
                <div style="font-size:2.8rem;margin-bottom:0.8rem;">📊</div>
                <div style="font-family:'IBM Plex Mono',monospace;font-size:0.8rem;color:#4A6080;">
                  Sentiment result will appear here
                </div>
              </div>
            """, unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════
#  TAB 2 — Batch analyser
# ════════════════════════════════════════════════════════════
with tab_batch:
    st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="section-label">Paste tweets — one per line</div>', unsafe_allow_html=True)
    batch_input = st.text_area(
        label="batch_tweets",
        label_visibility="collapsed",
        placeholder=(
            "$TSLA reports record deliveries for Q2 2024\n"
            "Inflation data worse than expected; recession fears mount\n"
            "Apple's Tim Cook says company remains focused on long-term growth\n"
            "Oil prices drop sharply on OPEC output increase\n"
            "Microsoft Azure revenues grow 28% in latest quarter"
        ),
        height=200,
        key="batch_input",
    )
    st.markdown("</div>", unsafe_allow_html=True)

    batch_btn = st.button("Run Batch Analysis ›", type="primary", key="batch_btn")

    if batch_btn:
        raw_lines = [l.strip() for l in batch_input.splitlines() if l.strip()]
        if not raw_lines:
            st.warning("Please paste at least one tweet.")
        else:
            with st.spinner(f"Analysing {len(raw_lines)} tweet(s)…"):
                labels = predict_batch(model, raw_lines)

            df = pd.DataFrame({"Tweet": raw_lines, "Sentiment": labels})

            # Summary metrics
            counts = df["Sentiment"].value_counts()
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Total Tweets",  len(df))
            m2.metric("▲ Bullish",  counts.get("Bullish", 0))
            m3.metric("▼ Bearish",  counts.get("Bearish", 0))
            m4.metric("◆ Neutral",  counts.get("Neutral", 0))

            st.markdown("<div style='height:0.8rem'></div>", unsafe_allow_html=True)

            # Colour-coded table
            def colour_sentiment(val):
                colours = {
                    "Bullish": "color:#00C96B;font-weight:700;",
                    "Bearish": "color:#F87171;font-weight:700;",
                    "Neutral": "color:#FCD34D;font-weight:700;",
                }
                return colours.get(val, "")

            st.dataframe(
                df.style.map(colour_sentiment, subset=["Sentiment"]),
                use_container_width=True,
                height=min(60 + len(df) * 38, 480),
            )

            # Download
            csv = df.to_csv(index=False).encode("utf-8")
            st.download_button(
                "Download results as CSV",
                data=csv,
                file_name="sentiment_results.csv",
                mime="text/csv",
            )

# ════════════════════════════════════════════════════════════
#  TAB 3 — Model info
# ════════════════════════════════════════════════════════════
with tab_about:
    st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)

    col_a, col_b = st.columns(2, gap="large")

    with col_a:
        st.markdown("""
          <div class="card">
            <div class="section-label">Best Model</div>
            <div style="font-family:'Syne',sans-serif;font-weight:800;font-size:1.5rem;color:#F0F4FF;margin-bottom:0.8rem;">
              TF-IDF + Linear SVM
            </div>
            <div style="font-family:'IBM Plex Mono',monospace;font-size:0.82rem;color:#607090;line-height:1.7;">
              Selected as best performer across five models trained on the
              <em>Twitter Financial News Sentiment</em> dataset (Hugging Face).
              Outperforms frozen FinBERT and DistilBERT in every metric.
            </div>
            <hr>
            <div class="section-label">Validation Metrics</div>
            <table style="width:100%;font-family:'IBM Plex Mono',monospace;font-size:0.8rem;border-collapse:collapse;">
              <tr style="color:#4A6080;"><td>Accuracy</td><td style="text-align:right;color:#00C96B;font-weight:600;">84.09%</td></tr>
              <tr style="color:#4A6080;"><td>Macro F1</td><td style="text-align:right;color:#00C96B;font-weight:600;">0.7821</td></tr>
              <tr style="color:#4A6080;"><td>Macro Precision</td><td style="text-align:right;color:#00C96B;font-weight:600;">0.7895</td></tr>
              <tr style="color:#4A6080;"><td>Macro Recall</td><td style="text-align:right;color:#607090;">0.7753</td></tr>
              <tr style="color:#4A6080;"><td>Weighted F1</td><td style="text-align:right;color:#00C96B;font-weight:600;">0.8396</td></tr>
            </table>
          </div>
        """, unsafe_allow_html=True)

    with col_b:
        st.markdown("""
          <div class="card">
            <div class="section-label">Preprocessing Pipeline</div>
            <div style="font-family:'IBM Plex Mono',monospace;font-size:0.82rem;color:#607090;line-height:1.8;margin-bottom:0.8rem;">
              Applied via <code style="color:#00C96B;background:rgba(0,201,107,0.1);padding:0.1rem 0.35rem;border-radius:4px;">clean_basic()</code>
            </div>
        """, unsafe_allow_html=True)

        steps = [
            "Remove URLs & @mentions",
            "Expand #hashtags → bare word",
            "Map emojis → semantic tokens",
            "Lowercase all text",
            "Normalise numbers → &lt;NUM&gt;",
            "Collapse repeated punctuation",
            "Collapse repeated characters",
            "Strip non-alphanumeric chars",
        ]
        for s in steps:
            st.markdown(
                f'<div style="font-family:\'IBM Plex Mono\',monospace;font-size:0.78rem;color:#4A6080;'
                f'padding:0.25rem 0;border-bottom:1px solid rgba(255,255,255,0.04);">'
                f'<span style="color:#00C96B;margin-right:0.5rem;">›</span>{s}</div>',
                unsafe_allow_html=True
            )

        st.markdown("""
            <div style="margin-top:1rem;">
              <div class="section-label">Vectoriser</div>
              <div style="display:flex;flex-wrap:wrap;gap:0.4rem;margin-top:0.4rem;">
                <span class="chip">ngram (1,2)</span>
                <span class="chip">max_features=25k</span>
                <span class="chip">min_df=2</span>
                <span class="chip">sublinear_tf=True</span>
              </div>
              <div class="section-label" style="margin-top:1rem;">Classifier</div>
              <div style="display:flex;flex-wrap:wrap;gap:0.4rem;margin-top:0.4rem;">
                <span class="chip">LinearSVC</span>
                <span class="chip">class_weight=balanced</span>
              </div>
            </div>
          </div>
        """, unsafe_allow_html=True)

    # Model comparison table
    st.markdown("<div style='height:0.5rem'></div>", unsafe_allow_html=True)
    st.markdown('<div class="section-label">Full model comparison</div>', unsafe_allow_html=True)
    comparison_df = pd.DataFrame([
        {"Model": "TF-IDF + Linear SVM ✅",           "Accuracy": 0.8409, "Precision (M)": 0.7895, "Recall (M)": 0.7753, "F1 (Macro)": 0.7821, "F1 (Weighted)": 0.8396},
        {"Model": "TF-IDF + Logistic Regression",      "Accuracy": 0.8166, "Precision (M)": 0.7470, "Recall (M)": 0.7813, "F1 (Macro)": 0.7620, "F1 (Weighted)": 0.8202},
        {"Model": "FinBERT (frozen)",                   "Accuracy": 0.7567, "Precision (M)": 0.6829, "Recall (M)": 0.7163, "F1 (Macro)": 0.6960, "F1 (Weighted)": 0.7603},
        {"Model": "FinBERT + Pragmatic Gating",         "Accuracy": 0.7688, "Precision (M)": 0.7108, "Recall (M)": 0.6729, "F1 (Macro)": 0.6893, "F1 (Weighted)": 0.7634},
        {"Model": "DistilBERT-base-uncased (frozen)",   "Accuracy": 0.7680, "Precision (M)": 0.7114, "Recall (M)": 0.6641, "F1 (Macro)": 0.6649, "F1 (Weighted)": 0.7607},
    ])

    num_cols = [c for c in comparison_df.columns if c != "Model"]
    st.dataframe(
        comparison_df.style
            .format({c: "{:.4f}" for c in num_cols})
            .highlight_max(subset=num_cols, color="#0d3320")
            .set_properties(**{"font-family": "IBM Plex Mono, monospace", "font-size": "13px"}),
        use_container_width=True,
        hide_index=True,
    )

# Footer
st.markdown("""
<div class="footer">
  Sentiment-Driven Market Analysis · TF-IDF + LinearSVC · Twitter Financial News Sentiment (Hugging Face)
</div>
""", unsafe_allow_html=True)
