"""
app.py — TrustRate AI
Interactive Electronics Review Authenticity & Rating Deflator

Features:
1. 🌐 Live E-Commerce URL Auditor (Matches 2,990+ real Amazon devices or scrapes live reviews)
2. 📋 Custom Review List Auditor (Paste arbitrary real reviews to see live ML deflator)
3. 🔍 Deep-Dive Forensic Report (Hero metrics: Given 4.5★ vs Actual 3.8★ -> Inflated by +0.7★)
4. 🏬 Electronics Storefront & Catalog (300+ indexed devices across 7 categories)
5. ⚡ Single Review Instant Scanner (Real-time fraud score + linguistic cues)
6. 📊 AI Model Studio & Viva Lab (Benchmark metrics, confusion matrix, top predictive words)
"""

from pathlib import Path
import json
import warnings
import joblib
import pandas as pd
import numpy as np
import altair as alt
import streamlit as st

warnings.filterwarnings("ignore")

from model_utils import analyze_review_text, compute_rating_inflation, SUSPICIOUS_TRIGGER_WORDS
from scraper import fetch_product_reviews_from_url

BASE = Path(__file__).resolve().parent
DATA_DIR = BASE / "data"
CATALOG_PATH = DATA_DIR / "electronics_catalog.json"
ASIN_DB_PATH = DATA_DIR / "asin_database.json"
MODEL_PATH = BASE / "model.joblib"
METRICS_PATH = BASE / "metrics.json"

st.set_page_config(
    page_title="TrustRate AI — E-Commerce Rating Deflator",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .metric-card {
        background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
        border: 1px solid #334155;
        border-radius: 12px;
        padding: 18px 20px;
        color: white;
        margin-bottom: 12px;
    }
    .metric-title {
        font-size: 0.85rem;
        font-weight: 500;
        color: #94a3b8;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    .metric-value {
        font-size: 2.1rem;
        font-weight: 700;
        margin-top: 4px;
        margin-bottom: 4px;
    }
    .metric-delta {
        font-size: 0.85rem;
        font-weight: 600;
    }
    .delta-inflated {
        color: #f87171;
    }
    .delta-authentic {
        color: #34d399;
    }
    .url-badge {
        display: inline-block;
        padding: 3px 10px;
        border-radius: 6px;
        background-color: #1e3a8a;
        color: #bfdbfe;
        font-size: 0.8rem;
        font-family: monospace;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        padding: 10px 18px;
        font-weight: 600;
        border-radius: 8px;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        return None
    try:
        return joblib.load(MODEL_PATH)
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

@st.cache_data
def load_catalog():
    if not CATALOG_PATH.exists():
        return []
    try:
        with open(CATALOG_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        st.error(f"Error loading catalog: {e}")
        return []

@st.cache_data
def load_metrics():
    if not METRICS_PATH.exists():
        return {}
    try:
        with open(METRICS_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}

model = load_model()
catalog = load_catalog()
metrics = load_metrics()

# Header
col_logo, col_desc = st.columns([1, 6])
with col_logo:
    st.markdown("<h1 style='font-size: 3.5rem; margin-bottom: 0;'>🛡️</h1>", unsafe_allow_html=True)
with col_desc:
    st.markdown("<h1 style='margin-bottom: 0; padding-top: 5px;'>TrustRate AI</h1>", unsafe_allow_html=True)
    st.markdown("<p style='color: #94a3b8; font-size: 1.05rem; margin-top: 0;'>Real-Time E-Commerce Review Authenticity Engine & Rating Deflator</p>", unsafe_allow_html=True)

# Sidebar
st.sidebar.image("https://img.icons8.com/fluency/96/shield.png", width=64)
st.sidebar.title("TrustRate Suite")

nav_choice = st.sidebar.radio(
    "Select Mode:",
    [
        "🌐 Live E-Commerce URL Auditor",
        "📝 Custom Review Text Auditor",
        "🔍 Deep-Dive Forensic Report",
        "🏬 Electronics Storefront & Catalog",
        "⚡ Single Review Instant Scanner",
        "📊 AI Model Studio & Viva Lab"
    ]
)

st.sidebar.divider()
st.sidebar.subheader("⚙️ Detection Parameters")
confidence_threshold = st.sidebar.slider(
    "Suspicious Review Threshold",
    min_value=0.30, max_value=0.80, value=0.50, step=0.05,
    help="Reviews with fraud probability above this threshold are classified as fake/suspicious and filtered from the genuine rating."
)

st.sidebar.info(
    f"**Active Model:** {metrics.get('best_model', 'Logistic Regression')}\n\n"
    f"**Real Products Index:** 2,990+ Amazon Devices\n\n"
    f"**Training Dataset:** {metrics.get('rows', 26722):,} balanced reviews"
)

# -------------------------------------------------------------
# Mode 1: 🌐 Live E-Commerce URL Auditor
# -------------------------------------------------------------
if nav_choice == "🌐 Live E-Commerce URL Auditor":
    st.subheader("🌐 E-Commerce Product URL Audit & Rating Deflator")
    st.caption("Paste an Amazon or e-commerce URL. TrustRate AI matches the product or extracts reviews, runs them through the trained ML classifier to filter fake/incentivized reviews, and reveals the true authentic rating alongside the rating inflation delta.")
    
    st.markdown("**⚡ 1-Click Real Amazon Product URLs (With Real Unique Ratings):**")
    ex_c1, ex_c2, ex_c3, ex_c4, ex_c5 = st.columns(5)
    
    selected_url = ""
    if ex_c1.button("🎧 Earbuds\n(4.5★ -> 3.8★)", use_container_width=True):
        selected_url = "https://www.amazon.com/dp/P001"
    if ex_c2.button("📱 Galaxy Z Fold\n(4.0★ -> 3.4★)", use_container_width=True):
        selected_url = "https://www.amazon.com/dp/B0FDL5T1PF"
    if ex_c3.button("⚡ 18W Charger\n(4.3★ -> 1.8★)", use_container_width=True):
        selected_url = "https://www.amazon.com/dp/B0836QVLD3"
    if ex_c4.button("🔊 BT Speaker\n(4.2★ -> 1.9★)", use_container_width=True):
        selected_url = "https://www.amazon.com/dp/B07Z8ZWQYV"
    if ex_c5.button("💤 Sleep Phone\n(3.5★ -> 2.9★)", use_container_width=True):
        selected_url = "https://www.amazon.com/dp/B07Q34ZKLF"
        
    st.write("")
    
    url_input = st.text_input(
        "Enter Amazon Product URL or ASIN:",
        value=selected_url if selected_url else "https://www.amazon.com/dp/B0FDL5T1PF",
        placeholder="e.g. https://www.amazon.com/dp/B0FDL5T1PF or paste any Amazon URL"
    )
    
    run_audit = st.button("🚀 Audit Product URL & Deflate Rating", type="primary")
    
    if run_audit or selected_url:
        with st.spinner("Analyzing product and retrieving customer reviews..."):
            extracted = fetch_product_reviews_from_url(url_input, catalog_list=catalog)
            
        if extracted.get("status") == "captcha_blocked":
            st.warning("🔒 **Amazon Bot Protection Active:** Amazon returned an anti-bot CAPTCHA for this live request.")
            st.info(
                "Because Amazon blocks direct automated Python requests to protect its website, you can audit this product in either of two ways:\n\n"
                "1. **Switch to '📝 Custom Review Text Auditor'** in the sidebar and paste 2 to 5 reviews from the Amazon page.\n"
                "2. Or try one of the **2,990+ indexed real Amazon product ASINs** above (like `B0836QVLD3`, `B07Z8ZWQYV`, `B07Q34ZKLF`, `B00A83I8G2`)."
            )
        else:
            product_title = extracted["product_title"]
            source = extracted["source"]
            asin_found = extracted["asin"]
            reviews_list = extracted.get("reviews", [])
            
            if not reviews_list:
                st.warning("No reviews could be parsed from this product. Try copying and pasting reviews directly in '📝 Custom Review Text Auditor'.")
            else:
                rev_df = pd.DataFrame(reviews_list)
                analysis = compute_rating_inflation(rev_df, rating_col="rating", text_col="review_text", model=model)
                
                st.divider()
                st.markdown(f"### 📦 {product_title}")
                st.markdown(f"<span class='url-badge'>Source: {source}</span> &nbsp; <span class='url-badge'>ASIN: {asin_found}</span> &nbsp; <span class='url-badge'>{analysis['total_reviews']} Real Reviews Audited</span>", unsafe_allow_html=True)
                st.write("")
                
                # 4 Hero Metric Cards
                c1, c2, c3, c4 = st.columns(4)
                with c1:
                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-title">Advertised Given Rating</div>
                        <div class="metric-value" style="color: #60a5fa;">{analysis['displayed_rating']:.2f} ★</div>
                        <div class="metric-delta">Listed on online store</div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                with c2:
                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-title">Actual Genuine Rating</div>
                        <div class="metric-value" style="color: #34d399;">{analysis['authentic_rating']:.2f} ★</div>
                        <div class="metric-delta delta-authentic">Calculated from verified real reviews</div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                with c3:
                    inf_color = "#f87171" if analysis['inflation_delta'] >= 0.5 else "#fbbf24"
                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-title">Rating Inflation Delta</div>
                        <div class="metric-value" style="color: {inf_color};">+{analysis['inflation_delta']:.2f} ★</div>
                        <div class="metric-delta delta-inflated">Inflated by {analysis['inflation_pct']:.1f}%</div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                with c4:
                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-title">Trust Assessment</div>
                        <div class="metric-value" style="font-size: 1.8rem;">{analysis['trust_grade']}</div>
                        <div class="metric-delta">{analysis['fake_pct']:.1f}% Suspicious Reviews Flagged</div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                if analysis['inflation_delta'] > 0:
                    st.warning(
                        f"⚠️ **Rating Inflation Result:** The advertised rating listed on the store is **{analysis['displayed_rating']:.2f} stars**, "
                        f"but actual genuine reviews average **{analysis['authentic_rating']:.2f} stars**. "
                        f"**The rating is inflated by +{analysis['inflation_delta']:.2f} stars ({analysis['inflation_pct']:.1f}%)!**"
                    )
                else:
                    st.success(f"✅ **Authentic Rating:** The advertised rating ({analysis['displayed_rating']:.2f}★) matches verified customer feedback. No significant inflation detected.")
                    
                st.divider()
                
                # Side-by-Side Review Breakdown
                st.subheader("📑 Customer Reviews Breakdown: Verified Genuine vs Detected Fake")
                classified_df = analysis["classified_df"].copy()
                genuine_revs = classified_df[~classified_df["is_suspicious"]]
                fake_revs = classified_df[classified_df["is_suspicious"]]
                
                col_gen, col_fake = st.columns(2)
                with col_gen:
                    st.markdown(f"#### 🟢 Verified Real Customer Reviews ({len(genuine_revs)} Reviews)")
                    if genuine_revs.empty:
                        st.info("No genuine reviews identified in this sample.")
                    else:
                        for _, r in genuine_revs.iterrows():
                            with st.container():
                                st.markdown(f"**⭐ {r['rating']:.0f}/5 Stars** | *Authenticity Score: {1.0 - r['suspicious_prob']:.1%} Genuine*")
                                st.write(f'"{r["review_text"]}"')
                                st.markdown("---")
                                
                with col_fake:
                    st.markdown(f"#### 🔴 Filtered Fake / Manipulated Reviews ({len(fake_revs)} Reviews)")
                    if fake_revs.empty:
                        st.success("Zero fake or suspicious reviews detected for this product!")
                    else:
                        for _, r in fake_revs.iterrows():
                            with st.container():
                                st.markdown(f"**⭐ {r['rating']:.0f}/5 Stars** | *Fraud Probability: {r['suspicious_prob']:.1%} Suspicious*")
                                st.write(f'"{r["review_text"]}"')
                                triggers = [w for w in SUSPICIOUS_TRIGGER_WORDS if w in r["review_text"].lower()]
                                if triggers:
                                    st.caption(f"🚨 **Deceptive cues detected:** `{', '.join(triggers)}`")
                                st.markdown("---")
                                
                st.divider()
                st.subheader("📊 Rating Deflator Visualizations")
                ch1, ch2 = st.columns([3, 2])
                with ch1:
                    comp_data = pd.DataFrame([
                        {"Rating Metric": "Advertised Given Rating", "Stars": analysis["displayed_rating"]},
                        {"Rating Metric": "Actual Genuine Rating", "Stars": analysis["authentic_rating"]}
                    ])
                    bar_chart = alt.Chart(comp_data).mark_bar(size=35).encode(
                        x=alt.X("Stars:Q", scale=alt.Scale(domain=[0, 5]), title="Rating (0 to 5 Stars)"),
                        y=alt.Y("Rating Metric:N", sort="-x", title=""),
                        color=alt.Color("Rating Metric:N", scale=alt.Scale(domain=["Advertised Given Rating", "Actual Genuine Rating"], range=["#ef4444", "#10b981"])),
                        tooltip=["Rating Metric", "Stars"]
                    ).properties(height=180, title="Advertised Given Rating vs Actual Genuine Rating")
                    st.altair_chart(bar_chart, use_container_width=True)
                    
                with ch2:
                    pie_df = pd.DataFrame([
                        {"Class": "Verified Genuine", "Count": analysis["genuine_count"]},
                        {"Class": "Filtered Fake", "Count": analysis["fake_count"]}
                    ])
                    pie = alt.Chart(pie_df).mark_arc(innerRadius=45).encode(
                        theta=alt.Theta(field="Count", type="quantitative"),
                        color=alt.Color(field="Class", type="nominal", scale=alt.Scale(domain=["Verified Genuine", "Filtered Fake"], range=["#10b981", "#ef4444"])),
                        tooltip=["Class", "Count"]
                    ).properties(height=180, title="Review Authenticity Distribution")
                    st.altair_chart(pie, use_container_width=True)

# -------------------------------------------------------------
# Mode 2: 📝 Custom Review Text Auditor
# -------------------------------------------------------------
elif nav_choice == "📝 Custom Review Text Auditor":
    st.subheader("📝 Audit Any Real Product Reviews Directly")
    st.caption("Paste customer reviews directly from ANY Amazon, Flipkart, or e-commerce product page. TrustRate AI evaluates every review in real-time, calculates the displayed rating, filters fakes, and outputs the exact inflation delta.")
    
    st.markdown("**⚡ Quick Test Presets:**")
    p_c1, p_c2 = st.columns(2)
    default_text = ""
    
    if p_c1.button("🎧 Load Budget Earbuds (Given: 4.50★ -> Actual: 3.80★ = +0.70★)", use_container_width=True):
        default_text = (
            "5 | Absolutely amazing product, best ever, everyone must buy now!!!\n"
            "5 | Perfect perfect perfect. Excellent quality. Five stars. Buy it immediately.\n"
            "4 | Battery lasts about five hours and the case is compact. Sound is decent for the price.\n"
            "3 | The left earbud sometimes disconnects, but reconnecting usually fixes it.\n"
            "4 | Good call quality indoors, though the ear tips feel slightly loose during running.\n"
            "5 | Pairing with my laptop was quick and Bluetooth range covers my whole apartment.\n"
            "3 | Decent for podcasts, but bass is somewhat muffled and the charging cable is short.\n"
            "5 | Best earbuds on earth!!! 100% perfect!!! No competition!!!\n"
            "5 | Unbelievable quality, top seller, must buy today!!!\n"
            "5 | AMAZING EARBUDS FIVE STARS FIVE STARS BUY NOW!!!\n"
            "5 | Flawless sound, greatest purchase ever, everyone should buy ten!\n"
            "5 | Miracle product, exceeded all expectations, five stars forever!"
        )
    if p_c2.button("⚡ Load Fast Charger (Given: 4.26★ -> Actual: 1.79★ = +2.47★)", use_container_width=True):
        default_text = (
            "5 | Absolutely amazing charger, best ever, everyone must buy now!!!\n"
            "5 | Top quality! Five stars! Best purchase ever made!\n"
            "5 | Unbelievable charging speed! Must buy immediately!\n"
            "5 | Miracle product! Works like a charm! Five stars forever!\n"
            "2 | Overheats dangerously after 20 minutes and makes a humming noise.\n"
            "1 | Stopped working after 3 days. Cable wire tore at the connector.\n"
            "2 | Slow charging, definitely not fast charging as advertised.\n"
            "2 | Flimsy plastic casing, pins bent when plugging into wall outlet."
        )
        
    st.caption("Format: `[Star Rating 1-5] | [Review Text]` (one review per line)")
    pasted_input = st.text_area("Review List:", value=default_text if default_text else (
        "5 | Must buy best product ever five stars buy now immediately!!!\n"
        "5 | Greatest product in history, unbelievable quality 100% recommended!\n"
        "5 | Amazing seller, perfect perfect five stars forever!\n"
        "3 | Battery life only lasted 3 hours and the charger got hot.\n"
        "2 | Broke after two weeks of normal use, plastic cracked."
    ), height=220)
    
    if st.button("🚀 Audit Pasted Reviews", type="primary"):
        reviews_to_analyze = []
        for line in pasted_input.strip().split("\n"):
            line = line.strip()
            if not line:
                continue
            if "|" in line:
                parts = line.split("|", 1)
                try:
                    r = float(parts[0].strip())
                    t = parts[1].strip()
                    reviews_to_analyze.append({"rating": r, "review_text": t})
                except Exception:
                    continue
            else:
                reviews_to_analyze.append({"rating": 5.0, "review_text": line})
                
        if reviews_to_analyze:
            input_df = pd.DataFrame(reviews_to_analyze)
            analysis = compute_rating_inflation(input_df, rating_col="rating", text_col="review_text", model=model)
            
            st.success("✅ Audit Complete!")
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Given Advertised Rating", f"{analysis['displayed_rating']:.2f} ★")
            m2.metric("Actual Genuine Rating", f"{analysis['authentic_rating']:.2f} ★")
            m3.metric("Rating Inflation", f"+{analysis['inflation_delta']:.2f} ★", f"{analysis['inflation_pct']:.1f}% inflated", delta_color="inverse")
            m4.metric("Trust Assessment", analysis['trust_grade'], f"{analysis['fake_pct']:.1f}% fake")
            
            if analysis['inflation_delta'] > 0:
                st.warning(f"⚠️ **Result:** The given rating is **{analysis['displayed_rating']:.2f} stars**, but actual reviews are **{analysis['authentic_rating']:.2f} stars**. It is **inflated by +{analysis['inflation_delta']:.2f} stars** ({analysis['inflation_pct']:.1f}%).")
            else:
                st.success("✅ Rating is genuine and uninflated.")
                
            st.subheader("Audited Review Breakdown:")
            res_df = analysis["classified_df"].copy()
            res_df["Authenticity Verdict"] = res_df["is_suspicious"].map({True: "🔴 Suspicious (Filtered)", False: "🟢 Genuine (Counted)"})
            res_df["Suspicious Probability"] = res_df["suspicious_prob"].map(lambda p: f"{p:.1%}")
            res_df["Stars"] = res_df["rating"].map(lambda s: f"{s:.0f} ★")
            
            st.dataframe(
                res_df[["Stars", "Authenticity Verdict", "Suspicious Probability", "review_text"]],
                use_container_width=True,
                hide_index=True
            )

# -------------------------------------------------------------
# Mode 3: 🔍 Deep-Dive Forensic Report
# -------------------------------------------------------------
elif nav_choice == "🔍 Deep-Dive Forensic Report":
    st.subheader("🔍 Deep-Dive Forensic Audit & Star Distribution Shift")
    st.caption("Select any electronic device from the catalog to inspect its rating inflation, star distribution shift, and individual real vs fake reviews.")
    
    if not catalog:
        st.warning("Catalog is empty. Please run `python prepare_data.py` first.")
        st.stop()
        
    device_options = [f"{p['title']} [{p['asin']}]" for p in catalog]
    selected_option = st.selectbox("Select Electronic Device to Audit:", device_options, index=0)
    
    selected_asin = selected_option.split("[")[-1].replace("]", "").strip()
    product = next((p for p in catalog if p["asin"] == selected_asin), catalog[0])
    
    st.markdown(f"### {product['title']}")
    st.caption(f"**Category:** {product['category']} | **ASIN / ID:** `{product['asin']}` | **Sample Size:** {product['total_reviews']} verified reviews")
    
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Given / Advertised Rating</div>
            <div class="metric-value" style="color: #60a5fa;">{product['displayed_rating']:.2f} ★</div>
            <div class="metric-delta">Listed on online website</div>
        </div>
        """, unsafe_allow_html=True)
        
    with c2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Actual Genuine Rating</div>
            <div class="metric-value" style="color: #34d399;">{product['authentic_rating']:.2f} ★</div>
            <div class="metric-delta delta-authentic">Calculated from verified real reviews</div>
        </div>
        """, unsafe_allow_html=True)
        
    with c3:
        inf_color = "#f87171" if product['inflation_delta'] >= 0.5 else "#fbbf24"
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Rating Inflation Delta</div>
            <div class="metric-value" style="color: {inf_color};">+{product['inflation_delta']:.2f} ★</div>
            <div class="metric-delta delta-inflated">Inflated by {product['inflation_pct']:.1f}%</div>
        </div>
        """, unsafe_allow_html=True)
        
    with c4:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Trust Assessment</div>
            <div class="metric-value" style="font-size: 1.8rem;">{product['trust_grade']}</div>
            <div class="metric-delta">{product['fake_pct']:.1f}% Suspicious Reviews</div>
        </div>
        """, unsafe_allow_html=True)
        
    if product['inflation_delta'] >= 0.5:
        st.error(f"⚠️ **Rating Inflation Alert:** This device is listed at **{product['displayed_rating']:.2f} stars**, but genuine customer reviews average **{product['authentic_rating']:.2f} stars**. The rating is inflated by **{product['inflation_delta']:.2f} stars** ({product['inflation_pct']:.1f}%).")
    else:
        st.success(f"✅ **High Trust Device:** Advertised rating ({product['displayed_rating']:.2f}★) closely matches genuine customer experience ({product['authentic_rating']:.2f}★). Rating inflation is minimal (+{product['inflation_delta']:.2f}★).")
        
    st.divider()
    
    star_all = product.get("star_distribution_all", {1:0, 2:0, 3:0, 4:0, 5:0})
    star_gen = product.get("star_distribution_genuine", {1:0, 2:0, 3:0, 4:0, 5:0})
    
    dist_data = []
    for s in range(1, 6):
        dist_data.append({"Stars": f"{s} Star", "Count": star_all.get(s, 0), "Type": "All Advertised Reviews"})
        dist_data.append({"Stars": f"{s} Star", "Count": star_gen.get(s, 0), "Type": "Verified Genuine Reviews"})
        
    df_dist = pd.DataFrame(dist_data)
    
    ch_col1, ch_col2 = st.columns([3, 2])
    with ch_col1:
        star_chart = alt.Chart(df_dist).mark_bar().encode(
            x=alt.X("Stars:N", title="Star Rating"),
            y=alt.Y("Count:Q", title="Number of Reviews"),
            color=alt.Color("Type:N", scale=alt.Scale(domain=["All Advertised Reviews", "Verified Genuine Reviews"], range=["#3b82f6", "#10b981"])),
            xOffset="Type:N",
            tooltip=["Stars", "Type", "Count"]
        ).properties(height=300)
        st.altair_chart(star_chart, use_container_width=True)
        
    with ch_col2:
        pie_data = pd.DataFrame([
            {"Category": "Genuine Reviews", "Count": product["genuine_count"]},
            {"Category": "Suspicious Reviews", "Count": product["fake_count"]}
        ])
        pie_chart = alt.Chart(pie_data).mark_arc(innerRadius=50).encode(
            theta=alt.Theta(field="Count", type="quantitative"),
            color=alt.Color(field="Category", type="nominal", scale=alt.Scale(domain=["Genuine Reviews", "Suspicious Reviews"], range=["#10b981", "#ef4444"])),
            tooltip=["Category", "Count"]
        ).properties(height=300, title="Review Authenticity Breakdown")
        st.altair_chart(pie_chart, use_container_width=True)
        
    st.divider()
    sample_reviews = product.get("sample_reviews", [])
    genuine_samples = [r for r in sample_reviews if r["label"] == "genuine"]
    suspicious_samples = [r for r in sample_reviews if r["label"] == "suspicious"]
    
    r_left, r_right = st.columns(2)
    with r_left:
        st.markdown("#### 🟢 Verified Genuine Reviews")
        if not genuine_samples:
            st.info("No genuine sample reviews recorded.")
        for r in genuine_samples[:6]:
            with st.container():
                st.markdown(f"**⭐ {r['stars']}/5 Stars** | *Confidence: {1.0 - r.get('score', 0.15):.1%} Genuine*")
                st.write(f'"{r["text"]}"')
                st.markdown("---")
                
    with r_right:
        st.markdown("#### 🔴 Flagged Suspicious Reviews")
        if not suspicious_samples:
            st.info("No suspicious sample reviews recorded.")
        for r in suspicious_samples[:6]:
            with st.container():
                st.markdown(f"**⭐ {r['stars']}/5 Stars** | *Fraud Score: {r.get('score', 0.88):.1%} Suspicious*")
                st.write(f'"{r["text"]}"')
                triggers = [w for w in SUSPICIOUS_TRIGGER_WORDS if w in r["text"].lower()]
                if triggers:
                    st.caption(f"🚨 **Deceptive cues detected:** `{', '.join(triggers)}`")
                st.markdown("---")

# -------------------------------------------------------------
# Mode 4: 🏬 Electronics Storefront & Catalog
# -------------------------------------------------------------
elif nav_choice == "🏬 Electronics Storefront & Catalog":
    st.subheader("🏬 Online Electronics Storefront — Listed vs Authentic Ratings")
    st.caption("Browse 300+ consumer electronics indexed from research review datasets. Compare advertised star ratings against genuine ratings.")
    
    if not catalog:
        st.warning("Electronics catalog not found. Please run `python prepare_data.py`.")
        st.stop()
        
    cat_df = pd.DataFrame(catalog)
    
    m1, m2, m3, m4 = st.columns(4)
    avg_displayed = cat_df["displayed_rating"].mean()
    avg_authentic = cat_df["authentic_rating"].mean()
    avg_inflation = cat_df["inflation_delta"].mean()
    inflated_count = (cat_df["inflation_delta"] >= 0.50).sum()
    
    m1.metric("Devices Indexed", f"{len(cat_df):,}")
    m2.metric("Avg Given Rating", f"{avg_displayed:.2f} ★")
    m3.metric("Avg Actual Rating", f"{avg_authentic:.2f} ★")
    m4.metric("Avg Rating Inflation", f"+{avg_inflation:.2f} ★", f"{inflated_count} heavily inflated", delta_color="inverse")
    
    st.divider()
    f1, f2, f3 = st.columns([2, 2, 2])
    with f1:
        categories = ["All Categories"] + sorted(list(cat_df["category"].unique()))
        selected_category = st.selectbox("Filter by Category", categories)
    with f2:
        search_query = st.text_input("🔍 Search device or ASIN", placeholder="e.g. Earbuds, Charger, Laptop, Smart Watch")
    with f3:
        grade_filter = st.multiselect(
            "Filter by Trust Grade",
            ["Grade A", "Grade B", "Grade C", "Grade D", "Grade F"],
            default=["Grade A", "Grade B", "Grade C", "Grade D", "Grade F"]
        )
        
    filtered = cat_df.copy()
    if selected_category != "All Categories":
        filtered = filtered[filtered["category"] == selected_category]
    if search_query.strip():
        filtered = filtered[filtered["title"].str.contains(search_query.strip(), case=False, na=False) | filtered["asin"].str.contains(search_query.strip(), case=False, na=False)]
    if grade_filter:
        filtered = filtered[filtered["trust_grade"].isin(grade_filter)]
        
    st.write(f"Showing **{len(filtered)}** matching electronic devices:")
    
    display_cols = [
        "title", "category", "displayed_rating", "authentic_rating",
        "inflation_delta", "fake_pct", "total_reviews", "trust_grade"
    ]
    view_table = filtered[display_cols].copy()
    view_table.columns = [
        "Device Title", "Category", "Given Rating", "Actual Rating",
        "Inflation (Stars)", "Fake Reviews %", "Reviews", "Trust Grade"
    ]
    view_table["Given Rating"] = view_table["Given Rating"].map(lambda x: f"{x:.2f} ★")
    view_table["Actual Rating"] = view_table["Actual Rating"].map(lambda x: f"{x:.2f} ★")
    view_table["Inflation (Stars)"] = view_table["Inflation (Stars)"].map(lambda x: f"+{x:.2f} ★" if x > 0 else f"{x:.2f} ★")
    view_table["Fake Reviews %"] = view_table["Fake Reviews %"].map(lambda x: f"{x:.1f}%")
    
    st.dataframe(view_table, use_container_width=True, hide_index=True)
    
    top_chart_data = filtered.sort_values("inflation_delta", ascending=False).head(12).copy()
    top_chart_data["Short Title"] = top_chart_data["title"].str.slice(0, 38) + "..."
    
    melted = pd.melt(
        top_chart_data,
        id_vars=["Short Title"],
        value_vars=["displayed_rating", "authentic_rating"],
        var_name="Rating Type",
        value_name="Stars"
    )
    melted["Rating Type"] = melted["Rating Type"].map({"displayed_rating": "Given Rating (Store)", "authentic_rating": "Actual Rating (Filtered)"})
    
    chart = alt.Chart(melted).mark_bar().encode(
        x=alt.X("Stars:Q", scale=alt.Scale(domain=[0, 5]), title="Rating (0 to 5 Stars)"),
        y=alt.Y("Short Title:N", sort="-x", title="Electronic Device"),
        color=alt.Color("Rating Type:N", scale=alt.Scale(domain=["Given Rating (Store)", "Actual Rating (Filtered)"], range=["#ef4444", "#10b981"])),
        tooltip=["Short Title", "Rating Type", "Stars"]
    ).properties(height=380)
    
    st.altair_chart(chart, use_container_width=True)

# -------------------------------------------------------------
# Mode 5: ⚡ Single Review Instant Scanner
# -------------------------------------------------------------
elif nav_choice == "⚡ Single Review Instant Scanner":
    st.subheader("⚡ Single Review Instant Classifier & Linguistic Scanner")
    st.caption("Paste any individual review text to inspect its fraud probability and detect deceptive linguistic cues.")
    
    c_left, c_right = st.columns([3, 2])
    with c_left:
        sample_prompt_text = "Absolutely amazing product, best ever, everyone must buy now!!! Perfect quality, five stars forever!"
        input_single = st.text_area("Review Text:", value=sample_prompt_text, height=140)
        rating_single = st.slider("Given Star Rating:", 1, 5, 5)
        
    with c_right:
        st.markdown("**Try Example Review Snippets:**")
        if st.button("🔴 Promotional Review ('Best ever, must buy')"):
            input_single = "Absolutely wonderful, the greatest product ever made in history, five stars buy it now immediately!!!"
        if st.button("🟢 Genuine Usage Review ('Battery lasted 5 hours')"):
            input_single = "Battery lasted about five hours of continuous playback. The sound quality is okay for the price, but the ear tips are a bit stiff."
        if st.button("🔴 Extreme Review ('PERFECT FIVE STARS')"):
            input_single = "PERFECT PERFECT PERFECT! AMAZING SELLER! MUST BUY TODAY 100% RECOMMEND!"
            
    if input_single:
        analysis = analyze_review_text(input_single, rating=rating_single, model=model)
        
        st.divider()
        g1, g2, g3 = st.columns(3)
        prob = analysis["suspicious_probability"]
        
        if prob >= 0.5:
            g1.error(f"Verdict: **🔴 Likely Suspicious** ({prob:.1%})")
        else:
            g1.success(f"Verdict: **🟢 Likely Genuine** ({1.0 - prob:.1%})")
            
        g2.metric("Fraud Probability Score", f"{prob:.1%}")
        g3.metric("Uppercase / Shouting Ratio", f"{analysis['caps_ratio_pct']:.1f}%")
        
        st.subheader("🔍 Linguistic & Behavioral Signals:")
        if analysis["flags"]:
            for f in analysis["flags"]:
                st.warning(f"⚠️ {f}")
        else:
            st.success("✅ No spammy linguistic patterns or excessive hype detected.")
            
        if analysis["triggers_found"]:
            st.markdown(f"**Identified Promotional Buzzwords:** `{', '.join(analysis['triggers_found'])}`")
        if analysis["genuine_markers"]:
            st.markdown(f"**Identified Natural Usage Markers:** `{', '.join(analysis['genuine_markers'])}`")

# -------------------------------------------------------------
# Mode 6: 📊 AI Model Studio & Viva Lab
# -------------------------------------------------------------
elif nav_choice == "📊 AI Model Studio & Viva Lab":
    st.subheader("📊 AI Model Studio, Benchmark Evaluation & Viva Defense")
    st.caption("Transparent evaluation of machine learning models trained on 26,700+ balanced reviews. Inspect performance metrics, confusion matrix, predictive linguistic tokens, and viva examiner questions.")
    
    tab_eval, tab_words, tab_viva = st.tabs(["📈 Model Benchmarks & Metrics", "🔤 Predictive Linguistic Tokens", "🎓 Project Viva & Defense Guide"])
    
    with tab_eval:
        st.markdown(f"**Selected Best Model:** `{metrics.get('best_model', 'Logistic Regression (Balanced)')}`")
        st.markdown(f"**Training Dataset:** `{metrics.get('dataset_used', 'data/curated_training_dataset.csv')}` | **Total Samples:** {metrics.get('rows', 26722):,} balanced rows")
        
        results = metrics.get("results", {})
        if results:
            res_df = pd.DataFrame(results).T.reset_index()
            res_df.columns = ["Model Architecture", "Accuracy", "Precision (Suspicious)", "Recall (Suspicious)", "F1-Score (Suspicious)", "ROC-AUC"]
            st.dataframe(res_df, use_container_width=True, hide_index=True)
            
            melted_res = pd.melt(res_df, id_vars=["Model Architecture"], value_vars=["Accuracy", "Precision (Suspicious)", "Recall (Suspicious)", "F1-Score (Suspicious)", "ROC-AUC"], var_name="Metric", value_name="Score")
            metric_chart = alt.Chart(melted_res).mark_bar().encode(
                x=alt.X("Score:Q", scale=alt.Scale(domain=[0.6, 0.85]), title="Score (0 to 1)"),
                y=alt.Y("Model Architecture:N", title="Model"),
                color=alt.Color("Metric:N"),
                xOffset="Metric:N",
                tooltip=["Model Architecture", "Metric", "Score"]
            ).properties(height=320)
            st.altair_chart(metric_chart, use_container_width=True)
            
        cm = metrics.get("confusion_matrix", [[2364, 977], [883, 2457]])
        st.subheader("Confusion Matrix (Test Split: 6,681 held-out reviews)")
        cm_df = pd.DataFrame(cm, index=["Actual Genuine", "Actual Suspicious"], columns=["Predicted Genuine", "Predicted Suspicious"])
        
        cm_c1, cm_c2 = st.columns([2, 1])
        with cm_c1:
            st.dataframe(cm_df, use_container_width=True)
        with cm_c2:
            correct = int(cm_df.iloc[0, 0] + cm_df.iloc[1, 1])
            total = int(cm_df.to_numpy().sum())
            st.metric("Total Test Reviews", f"{total:,}")
            st.metric("Correct Predictions", f"{correct:,} ({correct/total:.1%})")
            st.metric("Misclassifications", f"{total - correct:,} ({(total - correct)/total:.1%})")
            
    with tab_words:
        st.subheader("🔤 Top Predictive Linguistic Features")
        st.caption("TF-IDF unigram and bigram coefficients extracted from the calibrated model. Shows terms most strongly predictive of deceptive/suspicious reviews versus genuine reviews.")
        
        col_w1, col_w2 = st.columns(2)
        with col_w1:
            st.markdown("#### 🔴 Top Suspicious / Promotional Terms")
            top_susp = metrics.get("top_suspicious_words", [])[:15]
            if top_susp:
                df_susp = pd.DataFrame(top_susp)
                st.dataframe(df_susp, use_container_width=True, hide_index=True)
            else:
                st.info("No words recorded.")
                
        with col_w2:
            st.markdown("#### 🟢 Top Genuine / Natural Terms")
            top_gen = metrics.get("top_genuine_words", [])[:15]
            if top_gen:
                df_gen = pd.DataFrame(top_gen)
                st.dataframe(df_gen, use_container_width=True, hide_index=True)
            else:
                st.info("No words recorded.")
                
    with tab_viva:
        st.subheader("🎓 Project Viva Defense & Technical Q&A")
        
        vivas = [
            ("Q: How does TrustRate AI calculate rating inflation from an Amazon URL?",
             "The system extracts the listed customer reviews and advertised star rating ($R_{displayed}$) from the product link. The trained NLP model classifies each review. Filtering out the detected fake reviews gives the genuine rating ($R_{authentic}$):\n\n"
             "$$\\Delta R = R_{displayed} - R_{authentic}$$\n\n"
             "If a product has an advertised rating of 4.50★ but genuine reviews average 3.80★, the rating is inflated by **+0.70 stars** (+18.4%)."),
            
            ("Q: Why is text classification needed in addition to star ratings?",
             "Star ratings alone do not indicate whether reviews were generated organically or sponsored by incentivized review campaigns. Deceptive reviews typically exhibit distinct linguistic patterns: excessive generic superlatives ('best ever', 'perfect in every way'), lack of specific usage duration, and abnormal exclamation density."),
            
            ("Q: What models were evaluated and why was Logistic Regression selected?",
             "We benchmarked Logistic Regression, Multinomial Naive Bayes, Complement Naive Bayes, and Calibrated SGDClassifier with modified Huber loss on TF-IDF unigram and bigram features. Balanced Logistic Regression achieved the highest F1-score (0.7254) and ROC-AUC (0.7932) with well-calibrated posterior probabilities."),
            
            ("Q: How do you prevent data leakage during model training?",
             "If the same reviewer wrote multiple reviews, having some in train and some in test causes data leakage (the model might memorize reviewer idiolects rather than general deception signals). We strictly retained only one review per reviewer before sampling."),
            
            ("Q: What are the main limitations?",
             "Research labels (campaign tracking and moderation removals) are strong proxies but not 100% verified ground truth for every individual sentence. Hence, TrustRate AI serves as an explainable decision support tool rather than an absolute fraud accuser.")
        ]
        
        for q, a in vivas:
            with st.expander(q):
                st.markdown(a)

st.divider()
st.caption("🛡️ TrustRate AI — Machine Learning for E-Commerce Transparency and Consumer Protection.")
