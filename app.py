import streamlit as st
import pandas as pd
import re
import unicodedata
import joblib
import urllib.request
import urllib.parse
import xml.etree.ElementTree as ET
from datetime import datetime

try:
    from duckduckgo_search import DDGS
    HAS_DDG = True
except ImportError:
    HAS_DDG = False

try:
    from sentence_transformers import CrossEncoder
    HAS_NLI = True
except ImportError:
    HAS_NLI = False

st.set_page_config(
    page_title="VeriFact AI — Misinformation Command Center",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    :root {
        --vf-bg: #ffffff;
        --vf-panel: #ffffff;
        --vf-blue: #2563eb;
        --vf-purple: #7c3aed;
        --vf-teal: #0f9f95;
        --vf-text: #172033;
        --vf-muted: #64748b;
        --vf-border: #e2e8f0;
    }
    .stApp { background: #ffffff !important; color: #172033 !important; }
    [data-testid="stAppViewContainer"], [data-testid="stMain"],
    [data-testid="stMainBlockContainer"] { background: #ffffff !important; }
    [data-testid="stMainBlockContainer"] { padding-top: 1.8rem; padding-bottom: 2.5rem; }
    [data-testid="stHeader"] { background: rgba(255,255,255,.94) !important; }

    /* A crisp white page with a colorful indigo-to-teal sidebar. */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #eef2ff 0%, #f5f3ff 52%, #ecfeff 100%) !important;
        border-right: 1px solid #dbe4ff;
    }
    [data-testid="stSidebar"] * { color: #253153 !important; }
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p { color: #53617d !important; }
    [data-testid="stSidebar"] [data-testid="stButton"] button {
        background: linear-gradient(110deg, #4f46e5, #7c3aed) !important;
        color: #ffffff !important; border: 0 !important; border-radius: 10px !important;
    }

    .stApp h1, .stApp h2, .stApp h3, .stApp h4 {
        color: #172554 !important; letter-spacing: -0.025em;
    }
    .stApp h1 { border-bottom: 3px solid #c4b5fd; padding-bottom: .35rem; }
    .stApp p, .stApp li, .stApp label,
    [data-testid="stMarkdownContainer"], [data-testid="stWidgetLabel"] { color: #334155; }
    .stApp [data-testid="stCaptionContainer"], .stApp small { color: #64748b !important; }

    .command-card {
        background: linear-gradient(135deg, #ffffff 0%, #f5f7ff 100%);
        border: 1px solid #dbe4ff; border-top: 3px solid #6366f1;
        border-radius: 16px; padding: 24px; margin-bottom: 20px;
        box-shadow: 0 8px 24px rgba(51,65,125,.08);
    }
    .metric-box {
        background: linear-gradient(135deg, #eef2ff 0%, #f5f3ff 60%, #ecfeff 100%);
        border: 1px solid #d7ddff; border-radius: 14px; padding: 17px 14px;
        text-align: center; box-shadow: 0 5px 16px rgba(79,70,229,.08);
    }
    .metric-value { font-size: 2rem; font-weight: 800; color: #4f46e5 !important; }
    .metric-label { font-size: .75rem; color: #596780 !important; text-transform: uppercase; letter-spacing: .055em; }
    .badge-real { background: #dcfce7; color: #166534 !important; border: 1px solid #86efac; padding: 6px 14px; border-radius: 20px; font-weight: 700; font-size: .85rem; }
    .badge-fake { background: #ffe4e6; color: #9f1239 !important; border: 1px solid #fda4af; padding: 6px 14px; border-radius: 20px; font-weight: 700; font-size: .85rem; }
    .badge-warning { background: #fef3c7; color: #92400e !important; border: 1px solid #fcd34d; padding: 6px 14px; border-radius: 20px; font-weight: 700; font-size: .85rem; }

    .stButton > button, [data-testid="stDownloadButton"] > button,
    [data-testid="stFormSubmitButton"] > button {
        background: linear-gradient(105deg, #4f46e5 0%, #7c3aed 58%, #0891b2 100%) !important;
        color: #ffffff !important; border: 0 !important; border-radius: 10px !important;
        font-weight: 700 !important; box-shadow: 0 4px 12px rgba(79,70,229,.18);
        transition: transform 120ms ease, box-shadow 120ms ease;
    }
    .stButton > button:hover, [data-testid="stDownloadButton"] > button:hover,
    [data-testid="stFormSubmitButton"] > button:hover {
        box-shadow: 0 7px 18px rgba(79,70,229,.24); transform: translateY(-1px);
        filter: saturate(1.12);
    }

    /* White inputs with subtle borders and a clear focus state. */
    input, textarea, [data-baseweb="select"] > div,
    [data-baseweb="input"] > div, [data-baseweb="textarea"] {
        background: #ffffff !important; color: #172033 !important;
        border-color: #cbd5e1 !important; border-radius: 9px !important;
    }
    input::placeholder, textarea::placeholder { color: #94a3b8 !important; }
    input:focus, textarea:focus { border-color: #818cf8 !important; }

    /* Colorful tab pills; the active tab gets a saturated gradient underline. */
    [data-testid="stTabs"] [role="tablist"] {
        gap: 8px; border-bottom: 1px solid #e2e8f0; padding-bottom: 7px;
    }
    [data-testid="stTabs"] button[role="tab"] {
        background: #f1f5f9 !important; color: #475569 !important;
        border: 1px solid #e2e8f0 !important; border-radius: 10px 10px 7px 7px !important;
        padding: 8px 15px !important; font-weight: 650 !important;
    }
    /* Each tab gets its own colour theme: tinted pill + coloured text when idle,
       solid gradient + white text when selected. */
    [data-testid="stTabs"] button[role="tab"] p { font-weight: 650 !important; }
    [data-testid="stTabs"] button[role="tab"]:nth-child(1) { background: #e0e7ff !important; border-color: #a5b4fc !important; }
    [data-testid="stTabs"] button[role="tab"]:nth-child(1) p { color: #3730a3 !important; }
    [data-testid="stTabs"] button[role="tab"]:nth-child(2) { background: #dcfce7 !important; border-color: #86efac !important; }
    [data-testid="stTabs"] button[role="tab"]:nth-child(2) p { color: #166534 !important; }
    [data-testid="stTabs"] button[role="tab"]:nth-child(3) { background: #ffedd5 !important; border-color: #fdba74 !important; }
    [data-testid="stTabs"] button[role="tab"]:nth-child(3) p { color: #9a3412 !important; }
    [data-testid="stTabs"] button[role="tab"]:nth-child(4) { background: #fce7f3 !important; border-color: #f9a8d4 !important; }
    [data-testid="stTabs"] button[role="tab"]:nth-child(4) p { color: #9d174d !important; }
    [data-testid="stTabs"] button[role="tab"]:nth-child(5) { background: #cffafe !important; border-color: #67e8f9 !important; }
    [data-testid="stTabs"] button[role="tab"]:nth-child(5) p { color: #155e75 !important; }
    [data-testid="stTabs"] button[role="tab"][aria-selected="true"] { box-shadow: 0 5px 12px rgba(0,0,0,.16); }
    [data-testid="stTabs"] button[role="tab"][aria-selected="true"] p { color: #ffffff !important; }
    [data-testid="stTabs"] button[role="tab"]:nth-child(1)[aria-selected="true"] { background: linear-gradient(105deg, #4f46e5, #7c3aed) !important; border-color: #4f46e5 !important; }
    [data-testid="stTabs"] button[role="tab"]:nth-child(2)[aria-selected="true"] { background: linear-gradient(105deg, #16a34a, #0d9488) !important; border-color: #16a34a !important; }
    [data-testid="stTabs"] button[role="tab"]:nth-child(3)[aria-selected="true"] { background: linear-gradient(105deg, #ea580c, #f59e0b) !important; border-color: #ea580c !important; }
    [data-testid="stTabs"] button[role="tab"]:nth-child(4)[aria-selected="true"] { background: linear-gradient(105deg, #db2777, #e11d48) !important; border-color: #db2777 !important; }
    [data-testid="stTabs"] button[role="tab"]:nth-child(5)[aria-selected="true"] { background: linear-gradient(105deg, #0891b2, #2563eb) !important; border-color: #0891b2 !important; }
    [data-testid="stTabs"] [data-baseweb="tab-highlight"] { background: transparent !important; }

    /* Keep text on buttons readable (inner <p> otherwise inherits dark body text). */
    .stButton > button *, [data-testid="stDownloadButton"] > button *,
    [data-testid="stFormSubmitButton"] > button *, [data-testid="stLinkButton"] a *,
    [data-testid="stSidebar"] [data-testid="stButton"] button * { color: #ffffff !important; }
    [data-testid="stLinkButton"] a {
        background: linear-gradient(105deg, #16a34a, #0d9488) !important; color: #ffffff !important;
        border: 0 !important; border-radius: 10px !important; font-weight: 700 !important;
    }

    /* Text inside the verdict card must be dark on the light card. */
    .command-card p { color: #334155 !important; }
    .command-card h2, .command-card h3 { color: #172554 !important; }

    /* Code / copy blocks: light background, dark text. */
    [data-testid="stCode"], [data-testid="stCode"] pre, [data-testid="stCode"] code {
        background: #f8fafc !important; color: #1e293b !important;
    }
    [data-testid="stCode"] { border: 1px solid #e2e8f0; border-radius: 10px; }
    [data-testid="stTabs"] [role="tabpanel"] {
        background: #ffffff !important; border: 1px solid #e2e8f0 !important;
        border-radius: 0 12px 12px 12px; padding: 18px 12px 10px;
    }
    [data-testid="stMetric"], [data-testid="stAlert"], [data-testid="stExpander"],
    [data-testid="stDataFrame"], [data-testid="stTable"], [data-testid="stForm"],
    [data-testid="stFileUploader"], [data-testid="stTextInput"], [data-testid="stTextArea"],
    [data-testid="stSelectbox"], [data-testid="stRadio"], [data-testid="stCheckbox"],
    [data-testid="stNumberInput"], [data-testid="stDateInput"] {
        border-color: #e2e8f0; border-radius: 12px;
    }
    [data-testid="stExpander"], [data-testid="stForm"], [data-testid="stFileUploader"] {
        background: #ffffff !important;
    }
    [data-testid="stFileUploader"] section { background: #f8fafc !important; border-color: #cbd5e1 !important; }
    [data-testid="stAlert"] { background: #f8fafc !important; color: #334155 !important; }
    [data-testid="stProgressBar"] > div > div { background: linear-gradient(90deg, #4f46e5, #a855f7, #0d9488) !important; }
    hr, [data-testid="stDivider"] { border-color: #e2e8f0 !important; }
    a { color: #4f46e5 !important; }

    /* Restyle existing inline result panels to fit the white theme. */
    [style*="background:rgba(15,23,42"], [style*="background: rgba(15, 23, 42"],
    [style*="background:rgba(167,139,250"] {
        background: linear-gradient(135deg, #ffffff, #f5f7ff) !important;
        border-color: #dbe4ff !important; color: #334155 !important;
        box-shadow: 0 5px 18px rgba(51,65,125,.07);
    }
    [style*="background:rgba(15,23,42"] *, [style*="background: rgba(15, 23, 42"] *,
    [style*="background:rgba(167,139,250"] * { color: #334155 !important; }
    [data-testid="stDataFrame"] th, [data-testid="stTable"] th {
        background: #eef2ff !important; color: #312e81 !important;
    }
    [data-testid="stDataFrame"] td, [data-testid="stTable"] td {
        background: #ffffff !important; color: #334155 !important;
    }
    [data-testid="stDataFrame"] [role="grid"] { background: #ffffff !important; }
</style>
""", unsafe_allow_html=True)

if 'verification_history' not in st.session_state:
    st.session_state.verification_history = []
QUERY_STOP_WORDS = {
    'the', 'is', 'at', 'which', 'on', 'a', 'an', 'and', 'or', 'in', 'to', 'for', 'of', 'with',
    'that', 'this', 'it', 'from', 'by', 'as', 'are', 'was', 'were', 'been', 'be', 'have', 'has',
    'had', 'will', 'would', 'says', 'said', 'according', 'announced', 'new', 'news', 'breaking'
}

def unicode_words(text):
    words, current = [], []
    for ch in text:
        if unicodedata.category(ch)[0] in ('L', 'M', 'N'):
            current.append(ch)
        else:
            if current:
                words.append(''.join(current))
                current = []
    if current:
        words.append(''.join(current))
    return words

def extract_search_queries(text):
    sentences = [s.strip() for s in re.split(r'[.!?]\s+', text) if len(s.strip()) > 10]
    lead_sentence = sentences[0] if sentences else text

    raw_tokens = unicode_words(text)
    entities = []
    general_words = []
    for w in raw_tokens:
        if not w:
            continue
        lw = w.lower()
        if lw in QUERY_STOP_WORDS:
            continue
        if (w.isupper() and len(w) >= 2) or re.search(r'\d', w) or (w[0].isupper() and len(w) >= 3):
            entities.append(w)
        elif len(lw) > 3:
            general_words.append(lw)

    entities = list(dict.fromkeys(entities))
    general_words = list(dict.fromkeys(general_words))

    remaining_slots = max(0, 5 - len(entities[:4]))
    q1_terms = entities[:4] + general_words[:remaining_slots]
    q1 = " ".join(q1_terms) if q1_terms else text[:60]

    q2 = " ".join(lead_sentence.split()[:8])

    return [q1, q2]

LANGUAGE_OPTIONS = {
    'English': {'hl': 'en-IN', 'gl': 'IN', 'ceid': 'IN:en'},
    'Hindi': {'hl': 'hi-IN', 'gl': 'IN', 'ceid': 'IN:hi'},
}

def fetch_google_news_rss(query, lang='English'):
    try:
        lang_params = LANGUAGE_OPTIONS.get(lang, LANGUAGE_OPTIONS['English'])
        encoded_q = urllib.parse.quote(query)
        rss_url = f"https://news.google.com/rss/search?q={encoded_q}&hl={lang_params['hl']}&gl={lang_params['gl']}&ceid={lang_params['ceid']}"
        req = urllib.request.Request(rss_url, headers={'User-Agent': 'Mozilla/5.0'})
        
        with urllib.request.urlopen(req, timeout=5) as response:
            xml_data = response.read()
            
        root = ET.fromstring(xml_data)
        items = []
        for item in root.findall('.//item')[:8]:
            title = item.find('title').text if item.find('title') is not None else ''
            link = item.find('link').text if item.find('link') is not None else ''
            pub_date = item.find('pubDate').text if item.find('pubDate') is not None else ''
            source_el = item.find('source')
            source = source_el.text if source_el is not None else 'Google News'
            
            items.append({
                'title': title,
                'snippet': title,
                'link': link,
                'source': source,
                'date': pub_date
            })
        return items
    except Exception:
        return []

def fetch_duckduckgo_news(query):
    if not HAS_DDG:
        return []
    try:
        with DDGS() as ddgs:
            results = list(ddgs.news(query, max_results=8))
            items = []
            for r in results:
                items.append({
                    'title': r.get('title', ''),
                    'snippet': r.get('body', r.get('title', '')),
                    'link': r.get('url', ''),
                    'source': r.get('source', 'DuckDuckGo News'),
                    'date': r.get('date', '')
                })
            return items
    except Exception:
        return []

def fetch_live_news_with_fallback(query, lang='English'):
    articles = fetch_google_news_rss(query, lang=lang)
    if not articles:
        articles = fetch_duckduckgo_news(query)
    return articles

FACTCHECK_RSS_FEEDS = [
    ("PIB Fact Check", "https://pib.gov.in/PressReleseDetail.aspx?rss=1"),
    ("AltNews", "https://www.altnews.in/feed/"),
    ("BOOM Live", "https://www.boomlive.in/feed"),
    ("Factly", "https://factly.in/feed/"),
]

def fetch_factcheck_rss(query):
    query_terms = [t.lower() for t in re.findall(r'[a-zA-Z0-9]+', query) if len(t) > 2]
    if not query_terms:
        return []

    results = []
    for source_name, feed_url in FACTCHECK_RSS_FEEDS:
        try:
            req = urllib.request.Request(feed_url, headers={'User-Agent': 'Mozilla/5.0'})
            with urllib.request.urlopen(req, timeout=5) as response:
                xml_data = response.read()
            root = ET.fromstring(xml_data)
            for item in root.findall('.//item')[:20]:
                title_el = item.find('title')
                desc_el = item.find('description')
                link_el = item.find('link')
                date_el = item.find('pubDate')
                title = title_el.text if title_el is not None else ''
                desc = desc_el.text if desc_el is not None else ''
                combined_lower = (title + ' ' + (desc or '')).lower()
                hits = sum(1 for t in query_terms if t in combined_lower)
                min_hits_needed = 1 if len(query_terms) <= 2 else 2
                if hits >= min_hits_needed:
                    results.append({
                        'title': title,
                        'snippet': re.sub('<[^<]+?>', '', desc or title),
                        'link': link_el.text if link_el is not None else '',
                        'source': source_name,
                        'date': date_el.text if date_el is not None else '',
                        'source_type': 'factcheck'
                    })
        except Exception:
            continue
    return results

def fetch_all_sources(query, lang='English'):
    general = fetch_live_news_with_fallback(query, lang=lang)
    factcheck = fetch_factcheck_rss(query)
    return general + factcheck

@st.cache_resource
def load_trained_model():
    """Load either supported model/vectorizer filename pair."""
    candidates = [
        ("tfidf_vectorizer.pkl", "model (1).pkl"),
        ("vectorizer.pkl", "model (2).pkl"),
    ]
    errors = []
    for vectorizer_file, model_file in candidates:
        try:
            vectorizer = joblib.load(vectorizer_file)
            model = joblib.load(model_file)
            return model, vectorizer, None
        except Exception as exc:
            errors.append(f"{vectorizer_file} + {model_file}: {exc}")
    return None, None, "\n".join(errors)

trained_model, trained_vectorizer, model_load_error = load_trained_model()

def model_clean_text(text):
    text = str(text)
    text = re.sub(r'^.*?\(Reuters\)\s*-\s*', '', text, flags=re.IGNORECASE)
    text = re.sub(r'^.*?[A-Z]{2,}\s*\(AP\)\s*-\s*', '', text, flags=re.IGNORECASE)
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    return text.lower().strip()

# -----------------------------------------------------------------------------
# VeriFact AI — refreshed interface (visual layout adapted from app (3).py)
# Main application based on final code.py, with the refreshed interface.
# Live web feeds are used only for Sources/Timeline context, never for verdicts.
# -----------------------------------------------------------------------------

if 'verification_history' not in st.session_state:
    st.session_state.verification_history = []
if 'claim_input' not in st.session_state:
    st.session_state.claim_input = ''

def _load_benchmark_claim(claim):
    # Streamlit reruns the script after a button click; update the keyed widget
    # state so the selected benchmark appears in the claim text area.
    st.session_state.claim_input = claim
    st.session_state.last_result = None
if 'last_result' not in st.session_state:
    st.session_state.last_result = None

with st.sidebar:
    st.markdown("<h2 style='color:#4f46e5; margin-bottom:0;'>🛡️ VeriFact AI</h2>", unsafe_allow_html=True)
    st.markdown("<p style='font-size:0.82rem;'>Misinformation Command Center</p>", unsafe_allow_html=True)
    st.divider()
    analysis_mode = st.radio(
        "Workspace",
        ["📰 Text / Article Fact-Checker", "📋 Verification History"],
        index=0,
        key="analysis_mode_radio"
    )
    st.divider()
    st.markdown("### ⚡ Test Claim Benchmarks")
    if st.button("🔴 Fake: Plastic Currency Rumor", use_container_width=True):
        _load_benchmark_claim("The Reserve Bank of India has announced that all paper currency notes will be replaced with plastic bank notes next month.")
    if st.button("🟢 Real: ISRO Gaganyaan Engine", use_container_width=True):
        _load_benchmark_claim("ISRO successfully completed core stage engine testing for the Gaganyaan human spaceflight mission.")
    if st.button("🚨 Clickbait: 5G Scalar Waves", use_container_width=True):
        _load_benchmark_claim("BREAKING URGENT: Secret government plot leaked as 5G cell towers emit scalar frequencies!")

st.markdown("<h1 style='margin-bottom:4px;'>VeriFact AI Command Center</h1>", unsafe_allow_html=True)

def _predict_with_trained_model(text):
    """Return a model-only verdict. No live search or source data is consulted."""
    if trained_model is None or trained_vectorizer is None:
        raise RuntimeError("The trained model could not be loaded. Place the trained model and vectorizer files beside this app.")
    cleaned = model_clean_text(text)
    if not cleaned:
        raise ValueError("Please enter a longer claim or article text.")
    matrix = trained_vectorizer.transform([cleaned])
    prediction = trained_model.predict(matrix)[0]
    probabilities = trained_model.predict_proba(matrix)[0]
    classes = list(trained_model.classes_)
    by_class = {str(label).upper(): float(prob) for label, prob in zip(classes, probabilities)}
    real_prob = by_class.get("0", by_class.get("REAL", 0.0))
    fake_prob = by_class.get("1", by_class.get("FAKE", 0.0))
    confidence = max(real_prob, fake_prob)
    predicted_label = str(prediction).upper()
    # The supplied training pipeline maps 0 -> REAL and 1 -> FAKE.
    if predicted_label in ("1", "FAKE", "FALSE"):
        model_label = "FALSE"
    elif predicted_label in ("0", "REAL", "TRUE"):
        model_label = "TRUE"
    else:
        model_label = "TRUE" if real_prob >= fake_prob else "FALSE"
    if confidence < 0.60:
        verdict, css_class = "⚠️ UNCERTAIN", "badge-warning"
    elif model_label == "FALSE":
        verdict, css_class = "🚨 FALSE", "badge-fake"
    else:
        verdict, css_class = "🟢 TRUE", "badge-real"
    return {
        "verdict": verdict,
        "css_class": css_class,
        "truth_index": int(round(real_prob * 100)),
        "confidence": confidence,
        "real_probability": real_prob,
        "fake_probability": fake_prob,
        "summary": (
            f"The trained TF-IDF + Logistic Regression model predicts {model_label} "
            f"with {confidence * 100:.1f}% confidence."
            if confidence >= 0.60 else
            f"The model is not sufficiently confident; its highest class probability is {confidence * 100:.1f}%."
        )
    }

def _timeline_from_articles(articles):
    parsed=[]
    from email.utils import parsedate_to_datetime
    for article in articles:
        raw=article.get('date') or ''
        if not raw:
            continue
        try:
            dt=parsedate_to_datetime(raw)
            if dt.tzinfo:
                dt=dt.replace(tzinfo=None)
            parsed.append((dt, article))
        except Exception:
            continue
    if not parsed:
        return None, []
    parsed.sort(key=lambda item:item[0])
    return {
        'earliest_date': parsed[0][0],
        'latest_date': parsed[-1][0],
        'span_days': (parsed[-1][0]-parsed[0][0]).days,
        'distinct_dates': len({item[0].date() for item in parsed}),
        'earliest_source': parsed[0][1].get('source','Unknown'),
        'latest_source': parsed[-1][1].get('source','Unknown')
    }, parsed

def _make_reply(verdict, language='English'):
    if language == 'Hindi':
        return f"इस दावे की जाँच VeriFact AI के प्रशिक्षित मॉडल से की गई। मॉडल का परिणाम: {verdict}. कृपया इसे आगे साझा करने से पहले विश्वसनीय स्रोतों से भी जाँच लें।"
    return f"I checked this claim using VeriFact AI's trained model. Model result: {verdict}. Please verify it with reliable sources before forwarding it further."

if analysis_mode == "📰 Text / Article Fact-Checker":
    st.markdown("### 📰 Check a claim")
    user_input = st.text_area(
        "Enter a news claim, article paragraph, or viral post:",
        value=st.session_state.get('claim_input',''),
        height=150,
        placeholder="Paste a headline, claim, or article paragraph here...",
        key="claim_input"
    )
    search_lang = st.selectbox(
        "Language for live source search:",
        options=list(LANGUAGE_OPTIONS.keys()), index=0,
        help="This setting affects only the Sources and Claim Timeline panels. It never changes the trained-model verdict."
    )
    if st.button("🔍 Analyze with Trained Model", type="primary", use_container_width=True):
        if not user_input.strip():
            st.warning("Please enter a claim before analyzing.")
        else:
            try:
                with st.spinner("Running the trained model..."):
                    result = _predict_with_trained_model(user_input)
                # Live feeds are fetched separately, strictly for the source and timeline panels.
                all_articles=[]
                with st.spinner("Collecting live sources for the separate evidence panels..."):
                    for query in extract_search_queries(user_input):
                        all_articles.extend(fetch_all_sources(query, lang=search_lang))
                seen=set(); unique_articles=[]
                for article in all_articles:
                    key=(article.get('title',''), article.get('link',''))
                    if key not in seen:
                        seen.add(key); unique_articles.append(article)
                timeline, dated_articles = _timeline_from_articles(unique_articles)
                result.update({"claim":user_input, "articles":unique_articles, "timeline":timeline, "dated_articles":dated_articles})
                st.session_state.last_result=result
                st.session_state.verification_history.append({
                    'timestamp':datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    'claim':user_input[:60] + ("..." if len(user_input)>60 else ""),
                    'verdict':result['verdict'],
                    'truth_index':f"{result['truth_index']}%",
                    'corroboration':"Sources shown separately"
                })
            except Exception as exc:
                st.error(str(exc))

    result=st.session_state.get('last_result')
    if result:
        st.markdown(f"""
        <div class="command-card">
            <span class="{result['css_class']}">{result['verdict']}</span>
            <h2 style="margin-top:16px;">Truth Index: {result['truth_index']}%</h2>
            <p>{result['summary']}</p>
            <p style="font-size:.82rem;color:#64748b;">Verdict source: trained TF-IDF + Logistic Regression model. Live search is not used to calculate this result.</p>
        </div>
        """, unsafe_allow_html=True)
        c1,c2,c3=st.columns(3)
        with c1:
            st.markdown(f"<div class='metric-box'><div class='metric-value'>{result['truth_index']}%</div><div class='metric-label'>Truth Index</div></div>",unsafe_allow_html=True)
        with c2:
            st.markdown(f"<div class='metric-box'><div class='metric-value'>{result['confidence']*100:.1f}%</div><div class='metric-label'>Model Confidence</div></div>",unsafe_allow_html=True)
        with c3:
            st.markdown(f"<div class='metric-box'><div class='metric-value'>{len(result['articles'])}</div><div class='metric-label'>Sources Found (separate)</div></div>",unsafe_allow_html=True)

        tab2,tab3,tab4=st.tabs(["🟢 Live News & Sources","🕰️ Claim Timeline","💬 Forward-Back Reply"])
        with tab2:
            st.markdown("#### Live News & Source Authority")
            st.caption("These links are contextual references only. They do not change the trained-model result.")
            if result['articles']:
                for article in result['articles'][:10]:
                    source=article.get('source','Unknown source')
                    st.markdown(f"**[{article.get('title','Untitled source')}]({article.get('link','#')})**  \n{source} · {article.get('date','Date unavailable')}")
                    if article.get('snippet'):
                        st.caption(str(article['snippet'])[:350])
                    st.divider()
            else:
                st.info("No matching live sources were returned for this claim.")
        with tab3:
            st.markdown("#### Claim Coverage Timeline")
            st.caption("Shows dates found in the live source results; it does not establish when a claim first originated.")
            timeline=result['timeline']
            if timeline:
                a,b=st.columns(2)
                with a:
                    st.markdown(f"<div class='metric-box'><div class='metric-value' style='font-size:1.2rem'>{timeline['earliest_date'].strftime('%d %b %Y')}</div><div class='metric-label'>Earliest matched coverage</div></div>",unsafe_allow_html=True)
                    st.caption(f"Source: {timeline['earliest_source']}")
                with b:
                    st.markdown(f"<div class='metric-box'><div class='metric-value' style='font-size:1.2rem'>{timeline['latest_date'].strftime('%d %b %Y')}</div><div class='metric-label'>Latest matched coverage</div></div>",unsafe_allow_html=True)
                    st.caption(f"Source: {timeline['latest_source']}")
                st.info(f"Coverage spans {timeline['span_days']} days across {timeline['distinct_dates']} distinct dates.")
                for dt,article in result['dated_articles']:
                    st.markdown(f"**{dt.strftime('%d %b %Y')}** — [{article.get('title','Untitled')}]({article.get('link','#')}) · {article.get('source','Unknown')}")
            else:
                st.info("No parseable publication dates were found in the live source results.")
        with tab4:
            st.markdown("#### A short reply to forward")
            reply_lang=st.selectbox("Reply language:", ["English","Hindi"], key="reply_language")
            reply=_make_reply(result['verdict'],reply_lang)
            st.text_area("Message:",value=reply,height=110,disabled=True)
            st.code(reply)

elif analysis_mode == "📋 Verification History":
    st.markdown("### 📋 Verification History")
    history=st.session_state.verification_history
    if history:
        st.dataframe(pd.DataFrame(history),use_container_width=True,hide_index=True)
        if st.button("Clear session history"):
            st.session_state.verification_history=[]
            st.rerun()
    else:
        st.info("No checks have been recorded in this session yet.")
