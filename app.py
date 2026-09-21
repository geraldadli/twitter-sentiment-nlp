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

# Display order follows the poster; each class keeps its color everywhere.
CLASSES     = ["Bullish", "Bearish", "Neutral"]
LABEL_COLOR = {"Bullish": "#1DAB8F", "Bearish": "#D23A3F", "Neutral": "#C78511"}
LABEL_TEXT  = {"Bullish": "#5FD0B5", "Bearish": "#F07A7E", "Neutral": "#E6AA4A"}  # lighter tints for text
LABEL_GLYPH = {"Bullish": "▲", "Bearish": "▼", "Neutral": "◆"}
LABEL_ICON  = {"Bullish": "trend_up", "Bearish": "trend_down", "Neutral": "equal"}
LABEL_MATERIAL = {"Bullish": ":material/trending_up:", "Bearish": ":material/trending_down:",
                  "Neutral": ":material/equal:"}
LABEL_DESC  = {"Bullish": "Rising market expectations", "Bearish": "Falling market expectations",
               "Neutral": "Factual, no directional signal"}

# Headline metrics of the deployed model (validation split)
MACRO_F1 = 0.848
ACCURACY = 0.884

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
  @import url('https://fonts.googleapis.com/css2?family=Montserrat:wght@600;700;800&family=Inter:wght@400;500;600;700&display=swap');

  :root {
    --bg: #0F1D36; --bg-deep: #0B172C; --surface: #152845; --surface-2: #1B3254;
    --line: #243F66; --line-2: #2E4E7A;
    --ink: #E8F3FA; --ink-2: #A8BBD2; --ink-3: #8095B1;
    --title: #BDE9F4; --teal: #22B294; --cyan: #45D2E6; --tile: #0E3549;
    --display: 'Montserrat', sans-serif; --body: 'Inter', sans-serif;
  }

  /* ── App shell ── */
  html, body, [data-testid="stAppViewContainer"] {
    background: var(--bg); color: var(--ink); font-family: var(--body);
  }
  [data-testid="stAppViewContainer"] {
    background: radial-gradient(ellipse 55% 45% at 85% 0%, rgba(69,210,230,.07), transparent 70%), var(--bg);
  }
  [data-testid="stHeader"] { background: transparent; }
  [data-testid="stMainBlockContainer"], .block-container {
    max-width: 1180px; padding-top: 2.2rem; padding-bottom: 4rem;
  }

  /* ── Hero ── */
  .hero { display: grid; grid-template-columns: 1fr auto; align-items: end; gap: 32px; padding-bottom: 32px; }
  .eyebrow {
    font: 700 .9rem/1.2 var(--display); letter-spacing: .2em;
    text-transform: uppercase; color: var(--teal);
  }
  .hero-title {
    margin-top: 12px; font: 800 clamp(2.3rem, 4.6vw, 4.1rem)/1.04 var(--display);
    letter-spacing: -.02em; color: var(--title);
  }
  .hero-title span { display: block; }
  .candles { width: min(320px, 28vw); height: auto; }
  .candle { transform-box: fill-box; transform-origin: 50% 50%;
            animation: rise .7s cubic-bezier(.2,.7,.2,1) both; }
  @keyframes rise { from { transform: scaleY(0); opacity: 0; } }

  /* ── Stats row ── */
  .stats {
    display: grid; grid-template-columns: repeat(3, 1fr);
    border-top: 1px solid var(--line-2); border-bottom: 1px solid var(--line-2); margin-bottom: 32px;
  }
  .stat { padding: 24px 24px 22px; }
  .stat + .stat { border-left: 1px solid var(--line-2); }
  .stat-value { font: 800 clamp(2rem, 4.2vw, 3.2rem)/1.05 var(--display); color: var(--title); }
  .stat-label { margin-top: 10px; font: 700 .95rem/1.2 var(--display); letter-spacing: .14em; color: var(--teal); }
  .stat-sub   { margin-top: 4px; font-size: .92rem; color: var(--ink-3); }

  /* ── Tabs as poster "key feature" cards ── */
  [data-testid="stTabs"] [role="tablist"] { gap: 16px; border: 0; box-shadow: none; }
  [data-testid="stTab"] {
    flex: 1; height: auto; justify-content: flex-start;
    padding: 14px 18px; margin: 0; background: var(--surface);
    border-top: 3px solid var(--line-2); border-radius: 2px 2px 8px 8px;
    transition: background .15s, border-color .15s;
  }
  [data-testid="stTab"]:hover { background: var(--surface-2); }
  [data-testid="stTab"][aria-selected="true"] { background: var(--surface-2); border-top-color: var(--title); }
  [data-testid="stTab"] p {
    display: flex; align-items: center; gap: 14px;
    font: 700 1.1rem/1.25 var(--display) !important; color: var(--ink-2);
  }
  [data-testid="stTab"][aria-selected="true"] p { color: var(--ink); }
  [data-testid="stTab"] span[role="img"] {
    display: grid !important; place-items: center; width: 48px; height: 48px; flex: none;
    background: var(--tile); border-radius: 4px; color: var(--cyan); font-size: 26px; line-height: 1;
  }
  [data-testid="stTabs"] .react-aria-SelectionIndicator { display: none; }
  [data-testid="stTabs"] [role="tabpanel"] { padding-top: 20px; }

  /* ── Cards ── */
  .card, .st-key-card_input, .st-key-card_batch {
    background: var(--surface); border: 1px solid var(--line); border-radius: 8px; padding: 22px;
  }
  .card { margin-bottom: 16px; }
  .card-title {
    font: 700 .8rem/1.2 var(--display); letter-spacing: .16em;
    text-transform: uppercase; color: var(--teal); margin-bottom: 14px;
  }
  .card-title.sub { margin-top: 22px; }
  .muted { color: var(--ink-3); }

  /* ── Inputs & buttons ── */
  [data-testid="stTextAreaRootElement"] {
    background: var(--bg-deep) !important; border: 1px solid var(--line) !important; border-radius: 8px;
  }
  [data-testid="stTextAreaRootElement"]:focus-within {
    border-color: var(--cyan) !important; box-shadow: 0 0 0 3px rgba(69,210,230,.15);
  }
  [data-testid="stTextArea"] textarea {
    background: transparent; color: var(--ink); font: 400 1rem/1.55 var(--body); caret-color: var(--cyan);
  }
  [data-testid="stTextArea"] textarea::placeholder { color: var(--ink-3); }

  [data-testid="stBaseButton-primary"] {
    background: var(--title); border: 1px solid var(--title); color: var(--bg-deep);
    min-height: 44px; border-radius: 8px;
  }
  [data-testid="stBaseButton-primary"]:hover { background: #D6F4FB; border-color: #D6F4FB; color: var(--bg-deep); }
  [data-testid="stBaseButton-primary"] p { font: 700 .95rem var(--display); }
  [data-testid="stBaseButton-secondary"] {
    background: transparent; border: 1px solid var(--line-2); color: var(--ink-2); border-radius: 8px;
  }
  [data-testid="stBaseButton-secondary"]:hover { background: var(--surface-2); border-color: var(--line-2); color: var(--ink); }
  [class*="st-key-ex_"] [data-testid="stBaseButton-secondary"] { border-radius: 999px; min-height: 36px; }
  [class*="st-key-ex_"] p { font-size: .88rem; }
  .st-key-ex_Bullish [data-testid="stIconMaterial"] { color: #1DAB8F; }
  .st-key-ex_Bearish [data-testid="stIconMaterial"] { color: #D23A3F; }
  .st-key-ex_Neutral [data-testid="stIconMaterial"] { color: #C78511; }
  .examples-label { font-size: .82rem; color: var(--ink-3); margin-top: 2px; }

  [data-testid="stExpander"] details { background: var(--surface); border: 1px solid var(--line); border-radius: 8px; }
  [data-testid="stExpander"] summary p { font-size: .88rem; color: var(--ink-2); }
  [data-testid="stAlert"] { border-radius: 8px; }

  /* ── Prediction ── */
  .result-head { display: flex; align-items: center; gap: 18px; margin-bottom: 22px; }
  .result-icon {
    flex: none; display: grid; place-items: center; width: 68px; height: 68px; border-radius: 8px;
    color: var(--c); background: color-mix(in srgb, var(--c) 16%, transparent);
    border: 1px solid color-mix(in srgb, var(--c) 45%, transparent);
  }
  .result-label {
    font: 800 2.6rem/1 var(--display); letter-spacing: -.02em;
    color: color-mix(in oklab, var(--c) 72%, white);
  }
  .result-conf { margin-top: 6px; color: var(--ink-2); }
  .result-conf b { color: var(--ink); }
  .result-head.is-empty { --c: #8095B1; }
  .result-head.is-empty .result-label { color: var(--ink-3); }

  .probs { display: grid; gap: 14px; }
  .prob-head { display: flex; justify-content: space-between; margin-bottom: 6px; font-size: .92rem; }
  .prob-name { display: inline-flex; align-items: center; gap: 8px; color: var(--ink-2); }
  .prob.is-top .prob-name { color: var(--ink); font-weight: 600; }
  .prob-val  { color: var(--ink); font-weight: 600; font-variant-numeric: tabular-nums; }
  .prob-track { height: 10px; background: var(--bg-deep); border-radius: 0 4px 4px 0; overflow: hidden; }
  .prob-fill  { height: 100%; background: var(--c); border-radius: 0 4px 4px 0; }

  .meta { display: flex; gap: 32px; margin-top: 22px; padding-top: 16px; border-top: 1px solid var(--line); }
  .meta span { display: block; font-size: .78rem; color: var(--ink-3); }
  .meta b { font-weight: 600; font-variant-numeric: tabular-nums; }
  .token-track { width: 110px; height: 5px; margin-top: 6px; background: var(--bg-deep); border-radius: 3px; overflow: hidden; }
  .token-fill  { height: 100%; background: var(--cyan); border-radius: 0 3px 3px 0; }

  /* ── Batch summary ── */
  .summary { display: grid; grid-template-columns: repeat(4, 1fr); gap: 12px; }
  .tile {
    display: flex; flex-direction: column; gap: 2px; padding: 14px 16px;
    background: var(--bg-deep); border: 1px solid var(--line); border-top: 3px solid var(--c);
    border-radius: 2px 2px 8px 8px;
  }
  .tile-name  { display: inline-flex; align-items: center; gap: 6px; font-size: .88rem; color: var(--ink-2); }
  .tile-value { font: 800 2rem/1.15 var(--display); color: var(--ink); }
  .tile-sub   { font-size: .85rem; color: var(--ink-3); }
  .dist { display: flex; gap: 2px; height: 14px; margin-top: 18px; }
  .dist-seg { flex-basis: 0; min-width: 6px; }
  .dist-seg:first-child { border-radius: 4px 0 0 4px; }
  .dist-seg:last-child  { border-radius: 0 4px 4px 0; }
  .dist-seg:only-child  { border-radius: 4px; }
  [data-testid="stDataFrame"] { border: 1px solid var(--line); border-radius: 8px; overflow: hidden; }

  /* ── Model info ── */
  .model-name { font: 800 1.5rem/1.2 var(--display); color: var(--title); }
  .hub-link { display: inline-block; margin-top: 6px; font-size: .9rem; color: var(--cyan) !important; }
  .kv { display: grid; grid-template-columns: 1fr 1fr; gap: 14px 20px; margin-top: 18px; }
  .kv span { display: block; font-size: .8rem; color: var(--ink-3); }
  .kv b { font-weight: 600; }
  .labels { list-style: none; margin: 0; padding: 0; display: grid; gap: 10px; }
  .labels li {
    display: flex; align-items: center; gap: 12px; flex-wrap: wrap; margin: 0; padding: 11px 14px;
    background: var(--bg-deep); border-left: 3px solid var(--c); border-radius: 2px 8px 8px 2px;
  }
  .label-id {
    display: grid; place-items: center; width: 28px; height: 28px; border-radius: 4px;
    background: var(--surface-2); font: 700 .85rem/1 monospace; color: var(--ink-2);
  }
  .label-name { display: inline-flex; align-items: center; gap: 6px; font-weight: 700; }
  .labels .muted { font-size: .88rem; }
  .steps { list-style: none; margin: 0; padding: 0; display: grid; gap: 8px; }
  .steps li { display: flex; align-items: center; gap: 12px; margin: 0; color: var(--ink-2); font-size: .93rem; }
  .step-n {
    flex: none; display: grid; place-items: center; width: 30px; height: 30px; border-radius: 50%;
    background: var(--tile); border: 1px solid var(--line-2);
    font: 800 .85rem/1 var(--display); color: var(--title);
  }
  .stack { display: flex; flex-wrap: wrap; gap: 12px; }
  .stack span { padding: 10px 18px; border: 2px solid var(--title); font: 700 .95rem/1 var(--display); color: var(--ink); }

  /* ── Small screens ── */
  @media (max-width: 760px) {
    .hero { grid-template-columns: 1fr; }
    .candles { display: none; }
    .stats { grid-template-columns: 1fr 1fr; }
    .stat { padding: 18px 12px 18px 0; }
    .stat + .stat { padding-left: 14px; }
    .stat:last-child { grid-column: span 2; padding-left: 0; border-left: 0; border-top: 1px solid var(--line-2); }
    [data-testid="stTabs"] [role="tablist"] { gap: 8px; }
    [data-testid="stTab"] { padding: 12px; }
    [data-testid="stTab"] p { flex-direction: column; align-items: flex-start; gap: 8px; font-size: .9rem !important; white-space: normal; }
    [data-testid="stTab"] span[role="img"] { width: 38px; height: 38px; font-size: 22px; }
    .summary { grid-template-columns: repeat(2, 1fr); }
    .result-label { font-size: 2.1rem; }
  }
  @media (prefers-reduced-motion: reduce) { .candle { animation: none; } }
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
_ICON_PATHS = {
    "trend_up":   '<polyline points="22 7 13.5 15.5 8.5 10.5 2 17"/><polyline points="16 7 22 7 22 13"/>',
    "trend_down": '<polyline points="22 17 13.5 8.5 8.5 13.5 2 7"/><polyline points="16 17 22 17 22 11"/>',
    "equal":      '<line x1="5" x2="19" y1="9" y2="9"/><line x1="5" x2="19" y1="15" y2="15"/>',
    "target":     '<circle cx="12" cy="12" r="10"/><circle cx="12" cy="12" r="6"/><circle cx="12" cy="12" r="2"/>',
}

def icon(name: str, size: int = 16, color: str = "currentColor", stroke: float = 2) -> str:
    return (f'<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="{color}" '
            f'stroke-width="{stroke}" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">'
            f'{_ICON_PATHS[name]}</svg>')

def html(markup: str):
    """Render raw HTML. Lines are stripped so Markdown never reads indentation as a code block."""
    st.markdown("\n".join(l.strip() for l in markup.splitlines() if l.strip()), unsafe_allow_html=True)

def candles_svg(n: int = 17, w: int = 380, h: int = 230) -> str:
    """Decorative candlestick chart from the poster (deterministic upward random walk)."""
    seed, price, data = 11, 0.0, []
    for _ in range(n):
        seed = seed * 16807 % 2147483647; r1 = seed / 2147483647
        seed = seed * 16807 % 2147483647; r2 = seed / 2147483647
        seed = seed * 16807 % 2147483647; r3 = seed / 2147483647
        o, c = price, price + (r1 - 0.32) * 10
        data.append((o, c, max(o, c) + r2 * 4, min(o, c) - r3 * 4))
        price = c
    hi, lo = max(d[2] for d in data), min(d[3] for d in data)
    y = lambda v: h - 6 - (v - lo) / (hi - lo) * (h - 12)
    step = w / n
    parts = []
    for i, (o, c, top, bottom) in enumerate(data):
        x = step * i + step / 2
        color = "#1E8FA3" if c >= o else "#2D5E9E"
        y0, y1 = y(max(o, c)), y(min(o, c))
        parts.append(
            f'<g class="candle" style="animation-delay:{i * 45}ms">'
            f'<line x1="{x:.1f}" x2="{x:.1f}" y1="{y(top):.1f}" y2="{y(bottom):.1f}" stroke="{color}" stroke-width="1.5"/>'
            f'<rect x="{x - step * .27:.1f}" y="{y0:.1f}" width="{step * .54:.1f}" height="{max(y1 - y0, 2):.1f}" rx="1.5" fill="{color}"/></g>'
        )
    return f'<svg class="candles" viewBox="0 0 {w} {h}" aria-hidden="true">{"".join(parts)}</svg>'

def prob_rows(conf=None) -> str:
    top = max(conf, key=conf.get) if conf else None
    rows = []
    for c in CLASSES:
        p = conf.get(c, 0.0) if conf else 0.0
        rows.append(
            f'<div class="prob{" is-top" if c == top else ""}" style="--c:{LABEL_COLOR[c]}">'
            f'<div class="prob-head"><span class="prob-name">{icon(LABEL_ICON[c], color=LABEL_COLOR[c])}{c}</span>'
            f'<span class="prob-val">{f"{p * 100:.1f}%" if conf else "—"}</span></div>'
            f'<div class="prob-track"><div class="prob-fill" style="width:{p * 100:.2f}%"></div></div></div>'
        )
    return "".join(rows)

def render_result(pred=None, conf=None, n_tokens=None, ms=None):
    if pred is None:
        head = (f'<div class="result-head is-empty"><span class="result-icon">{icon("target", 30)}</span>'
                f'<div><div class="result-label">—</div><div class="result-conf">Awaiting tweet</div></div></div>')
        meta = ""
    else:
        head = (f'<div class="result-head" style="--c:{LABEL_COLOR[pred]}">'
                f'<span class="result-icon">{icon(LABEL_ICON[pred], 34, stroke=2.2)}</span>'
                f'<div><div class="result-label">{pred}</div>'
                f'<div class="result-conf"><b>{conf[pred] * 100:.1f}%</b> confidence</div></div></div>')
        fill = min(n_tokens / MAX_LENGTH * 100, 100)
        meta = (f'<div class="meta"><div><span>Tokens</span><b>{n_tokens} / {MAX_LENGTH}</b>'
                f'<div class="token-track"><div class="token-fill" style="width:{fill:.1f}%"></div></div></div>'
                f'<div><span>Latency</span><b>{ms:.0f} ms</b></div></div>')
    html(f'<div class="card"><div class="card-title">Prediction</div>{head}'
         f'<div class="probs">{prob_rows(conf)}</div>{meta}</div>')


# ─────────────────────────────────────────────────────────────────────────────
#  HERO
# ─────────────────────────────────────────────────────────────────────────────
html(f"""
<div class="hero">
  <div>
    <div class="eyebrow">Financial tweet sentiment classifier</div>
    <div class="hero-title" role="heading" aria-level="1"><span>Sentiment-Driven</span><span>Market Analysis</span></div>
  </div>
  {candles_svg()}
</div>
<div class="stats">
  <div class="stat"><div class="stat-value">{MACRO_F1:.3f}</div><div class="stat-label">Macro F1</div><div class="stat-sub">Validation set</div></div>
  <div class="stat"><div class="stat-value">{ACCURACY * 100:.1f}%</div><div class="stat-label">Accuracy</div><div class="stat-sub">Validation set</div></div>
  <div class="stat"><div class="stat-value">DistilBERT</div><div class="stat-label">Deployed model</div><div class="stat-sub">Fine-tuned transformer</div></div>
</div>
""")


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
tab1, tab2, tab3 = st.tabs([
    ":material/tag: Single Tweet",
    ":material/database: Batch Processing",
    ":material/psychology: Model Info",
])

EXAMPLES = {
    "Bullish": "$NVDA smashes earnings — revenue up 122%, beats all estimates 📈",
    "Bearish": "Stocks tumble as Fed signals more rate hikes",
    "Neutral": "Goldman Sachs maintains Q4 outlook with no revision to estimates",
}

def use_example(text: str):
    st.session_state["single_ta"] = text


# ════════ TAB 1 — Single tweet ════════════════════════════════════════════════
with tab1:
    col_in, col_out = st.columns([1.1, 0.9], gap="large")

    with col_in:
        with st.container(key="card_input"):
            html('<div class="card-title">Tweet</div>')
            tweet = st.text_area(
                label="tweet", label_visibility="collapsed",
                placeholder="Paste a financial tweet…",
                height=150, key="single_ta",
            )
            html('<div class="examples-label">Examples</div>')
            for _col, _lbl in zip(st.columns(3), CLASSES):
                with _col:
                    st.button(_lbl, key=f"ex_{_lbl}", icon=LABEL_MATERIAL[_lbl], help=EXAMPLES[_lbl],
                              on_click=use_example, args=(EXAMPLES[_lbl],), width="stretch")
            go = st.button("Analyse", type="primary", width="stretch")

    with col_out:
        if go and tweet.strip():
            with st.spinner("Running inference…"):
                pred, conf, n_tok, ms = predict_one(clf, tokenizer, tweet)
            render_result(pred, conf, n_tok, ms)
            with st.expander("Preprocessed text"):
                st.code(clean(tweet), language=None)
        elif go:
            st.warning("Please enter a tweet first.")
            render_result()
        else:
            render_result()


# ════════ TAB 2 — Batch ═══════════════════════════════════════════════════════
with tab2:
    with st.container(key="card_batch"):
        html('<div class="card-title">Tweets · one per line</div>')
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
        go_batch = st.button("Run Batch", type="primary", key="batch_btn")

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
            n      = len(df)

            # Summary tiles + stacked distribution bar
            tiles = [f'<div class="tile" style="--c:#BDE9F4"><span class="tile-name">Total</span>'
                     f'<span class="tile-value">{n}</span><span class="tile-sub">{total:.0f} ms</span></div>']
            segs = []
            for c in CLASSES:
                k = int(counts.get(c, 0))
                tiles.append(
                    f'<div class="tile" style="--c:{LABEL_COLOR[c]}">'
                    f'<span class="tile-name">{icon(LABEL_ICON[c], color=LABEL_COLOR[c])}{c}</span>'
                    f'<span class="tile-value">{k}</span><span class="tile-sub">{k / n * 100:.1f}%</span></div>'
                )
                if k:
                    segs.append(f'<div class="dist-seg" style="flex-grow:{k};background:{LABEL_COLOR[c]}" '
                                f'title="{c}: {k} tweets ({k / n * 100:.1f}%)"></div>')
            html(f'<div class="card"><div class="summary">{"".join(tiles)}</div>'
                 f'<div class="dist">{"".join(segs)}</div></div>')

            view = df.assign(Sentiment=df["Sentiment"].map(lambda v: f"{LABEL_GLYPH[v]}  {v}"))
            view.index = range(1, n + 1)
            st.dataframe(
                view.style.map(lambda v: f"color:{LABEL_TEXT[v.split()[-1]]};font-weight:600", subset=["Sentiment"]),
                width="stretch",
                height=min((n + 1) * 35 + 3, 520),
            )
            st.download_button(
                "Download CSV",
                data=df.to_csv(index=False).encode(),
                file_name="sentiment_results.csv",
                mime="text/csv",
                icon=":material/download:",
            )


# ════════ TAB 3 — Model info ══════════════════════════════════════════════════
with tab3:
    ca, cb = st.columns(2, gap="large")

    with ca:
        label_items = "".join(
            f'<li style="--c:{LABEL_COLOR[name]}"><span class="label-id">{idx}</span>'
            f'<span class="label-name">{icon(LABEL_ICON[name], color=LABEL_COLOR[name])}{name}</span>'
            f'<span class="muted">{LABEL_DESC[name]}</span></li>'
            for idx, name in ID2LABEL.items()
        )
        html(f"""
        <div class="card">
          <div class="card-title">Model</div>
          <div class="model-name">DistilBERT-base-uncased</div>
          <a class="hub-link" href="https://huggingface.co/{HF_MODEL_ID}" target="_blank">{HF_MODEL_ID} ↗</a>
          <div class="kv">
            <div><span>Max tokens</span><b>{MAX_LENGTH}</b></div>
            <div><span>Device</span><b>CPU</b></div>
            <div><span>Scores</span><b>All 3 classes</b></div>
            <div><span>Truncation</span><b>On</b></div>
          </div>
          <div class="card-title sub">Labels</div>
          <ul class="labels">{label_items}</ul>
        </div>
        """)

    with cb:
        steps = "".join(
            f'<li><span class="step-n">{i}</span>{step}</li>'
            for i, step in enumerate([
                "Strip URLs &amp; @mentions",
                "Expand #hashtags → bare word",
                "Map emojis → semantic tokens",
                "Lowercase everything",
                "Normalise numbers → &lt;NUM&gt;",
                "Collapse repeated punctuation",
                "Collapse repeated characters",
                "Remove non-alphanumeric chars",
            ], start=1)
        )
        tech = "".join(f"<span>{t}</span>" for t in
                       ["Python", "Streamlit", "DistilBERT", "FinBERT", "TF-IDF", "Scikit-Learn"])
        html(f"""
        <div class="card">
          <div class="card-title">Preprocessing</div>
          <ul class="steps">{steps}</ul>
        </div>
        <div class="card">
          <div class="card-title">Technology</div>
          <div class="stack">{tech}</div>
        </div>
        """)
