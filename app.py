import streamlit as st
import requests
from bs4 import BeautifulSoup
from model import load_model_and_predict
import re

# --- 1. PAGE SETUP & GLOBAL CSS THEME ---
st.set_page_config(page_title="Ultimate Fake News Analyzer", page_icon="🕵️‍♂️", layout="wide")

# Injecting the background color and smooth styling (WITH DARK MODE FIX)
st.markdown("""
<style>
    .stApp {
        background-color: #f0f2f6;
    }
    /* Force text, tabs, and input labels to dark grey so they don't vanish in dark mode */
    p, span, label, div[data-baseweb="tab"] {
        color: #31333F !important;
    }
    .metric-card {
        background-color: #ffffff;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
        text-align: center;
        border: 1px solid #e6e6e6;
    }
</style>
""", unsafe_allow_html=True)

if 'history' not in st.session_state:
    st.session_state.history = []

# --- HELPERS: SCRAPER & HIGHLIGHTER ---
def scrape_article(url):
    try:
        headers = {'User-Agent': 'Mozilla/5.0'}
        response = requests.get(url, headers=headers, timeout=10)
        soup = BeautifulSoup(response.content, 'html.parser')
        paragraphs = soup.find_all('p')
        return " ".join([p.get_text() for p in paragraphs])
    except Exception as e:
        return None

def highlight_smart_words(text, real_words, fake_words):
    highlighted_text = text
    for word in fake_words:
        highlighted_text = re.sub(f'(?i)\\b({word})\\b', r'<mark style="background-color: #ffcccc; color: #cc0000; font-weight: bold; border-radius: 3px; padding: 0 2px;">\1</mark>', highlighted_text)
    for word in real_words:
        highlighted_text = re.sub(f'(?i)\\b({word})\\b', r'<mark style="background-color: #ccffcc; color: #006600; font-weight: bold; border-radius: 3px; padding: 0 2px;">\1</mark>', highlighted_text)
    return highlighted_text

# --- MAIN UI DASHBOARD ---
st.title("🕵️‍♂️ Ultimate Fake News Analyzer")
st.markdown("*Advanced Machine Learning Classification & Linguistic Analysis*")
st.markdown("<br>", unsafe_allow_html=True) # Smooth Spacing

tab1, tab2 = st.tabs(["📝 Paste Text", "🔗 Analyze Live URL"])
news_text = ""

with tab1:
    text_input = st.text_area("Paste the full article text here:", height=200)
    if st.button("🚀 Analyze Text"):
        news_text = text_input

with tab2:
    url_input = st.text_input("Paste a news website URL:")
    if st.button("🕸️ Scrape & Analyze URL"):
        with st.spinner("Extracting text from website..."):
            scraped_text = scrape_article(url_input)
            if scraped_text and len(scraped_text) > 50:
                news_text = scraped_text
                st.success("✅ Article successfully extracted!")
            else:
                st.error("❌ Could not extract enough text. The site might have bot-protection.")

# --- ANALYSIS EXECUTION ---
if news_text:
    results = load_model_and_predict(news_text)
    
    if results:
        st.markdown("<br><hr><br>", unsafe_allow_html=True)
        
        # Save to history
        st.session_state.history.insert(0, {"snippet": news_text[:60] + "...", "verdict": results["prediction"]})
        st.session_state.history = st.session_state.history[:5]
        
        # --- 2. BIG CENTERED VERDICT ---
        verdict_color = "#28a745" if results["prediction"] == "REAL" else "#dc3545"
        verdict_icon = "✅" if results["prediction"] == "REAL" else "❌"
        
        st.markdown(f"""
        <div style="text-align: center; padding: 20px; background: white; border-radius: 15px; box-shadow: 0 4px 15px rgba(0,0,0,0.1); margin-bottom: 30px; border-top: 8px solid {verdict_color};">
            <h2 style="color: #555; margin: 0;">FINAL VERDICT</h2>
            <h1 style="font-size: 4.5rem; color: {verdict_color}; margin: 10px 0;">{results['prediction']} {verdict_icon}</h1>
        </div>
        """, unsafe_allow_html=True)
        
        # --- 3. CUSTOM CSS METRIC CARDS ---
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown(f"""
            <div class="metric-card" style="border-top: 4px solid {verdict_color};">
                <h4 style="color:#666; margin-bottom: 5px;">🎯 Prediction</h4>
                <h2 style="color:{verdict_color}; margin: 0;">{results['prediction']}</h2>
            </div>
            """, unsafe_allow_html=True)
            
        with col2:
            st.markdown(f"""
            <div class="metric-card" style="border-top: 4px solid #17a2b8;">
                <h4 style="color:#666; margin-bottom: 5px;">🎭 Article Tone</h4>
                <h3 style="color:#333; margin: 0; font-size: 1.2rem;">{results['sentiment'].split(" (")[0]}</h3>
            </div>
            """, unsafe_allow_html=True)
            
        with col3:
            st.markdown(f"""
            <div class="metric-card" style="border-top: 4px solid #6c757d;">
                <h4 style="color:#666; margin-bottom: 5px;">📝 Word Count</h4>
                <h2 style="color:#333; margin: 0;">{results['length']}</h2>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<br><br>", unsafe_allow_html=True)

        # --- 4. CONFIDENCE BREAKDOWN ---
        st.markdown("### 📊 Confidence Breakdown")
        col_prob1, col_prob2 = st.columns(2)
        
        real_p = results['real_prob']
        fake_p = results['fake_prob']
        margin = abs(real_p - fake_p)
        
        col_prob1.write(f"**🟢 Real Probability:** {real_p}%")
        col_prob1.progress(int(real_p))
        col_prob2.write(f"**🔴 Fake Probability:** {fake_p}%")
        col_prob2.progress(int(fake_p))

        st.markdown("<br>", unsafe_allow_html=True)

        # --- 5. DYNAMIC REASONING ---
        st.markdown("#### 🤖 Why did the model make this decision?")
        if fake_p > 70:
            st.error(f"(Margin: {round(margin, 1)}%) **Strong Fake Signal:** The model detected a strong presence of sensational keywords and unverified narrative structures common in fabricated news.")
        elif real_p > 70:
            st.success(f"(Margin: {round(margin, 1)}%) **Strong Real Signal:** The text exhibits a clear, factual reporting structure with vocabulary highly consistent with trusted, verified news sources.")
        elif margin < 10:
            st.warning(f"(Margin: {round(margin, 1)}%) **Borderline Case:** The model detected heavily conflicting signals. This usually happens when an article mixes factual data with extreme opinionated vocabulary.")
        else:
             st.info(f"(Margin: {round(margin, 1)}%) **Moderate Lean:** The model leans towards {results['prediction']}, but the signals are somewhat mixed. It contains elements of factual reporting blended with unusual phrasing.")

        # --- THE KILLER FEATURE: FACT CHECK SUGGESTION ---
        if results["prediction"] == "FAKE" or margin < 20:
            st.markdown("""
            <div style="padding: 15px; background-color: #fff3cd; border-left: 6px solid #ffecb5; border-radius: 5px; margin-top: 15px;">
                <h4 style="color: #856404; margin: 0;">⚠️ Fact Check Suggestion</h4>
                <p style="color: #856404; margin: 5px 0 0 0;">Because this article shows signs of low reliability or mixed signals, we strongly recommend verifying these claims with trusted sources like <a href="https://www.reuters.com" target="_blank">Reuters</a>, <a href="https://apnews.com" target="_blank">AP News</a>, or <a href="https://www.bbc.com" target="_blank">BBC News</a>.</p>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # --- 6. EXPLAINABILITY & HIGHLIGHTER ---
        st.markdown("### 🧠 Model Explainability (Word Impact)")
        col_expl1, col_expl2 = st.columns(2)
        with col_expl1:
            st.success("**🟢 Real Indicators:** " + ", ".join([f"`{w}`" for w in results["top_real_words"][:5]]))
        with col_expl2:
            st.error("**🔴 Fake Indicators:** " + ", ".join([f"`{w}`" for w in results["top_fake_words"][:5]]))
        
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("#### 📄 Document Scan (Model-Detected Keywords Highlighted)")
        highlighted = highlight_smart_words(news_text, results["top_real_words"], results["top_fake_words"])
        
        # Fixed text color in the document scan box for dark mode users
        st.markdown(f"""
        <div style="padding: 20px; border-radius: 10px; background-color: #ffffff; border: 1px solid #ddd; max-height: 400px; overflow-y: auto; line-height: 1.8; font-size: 1.1rem; box-shadow: inset 0 2px 4px rgba(0,0,0,0.05); color: #31333F;">
            {highlighted}
        </div>
        """, unsafe_allow_html=True)

# --- SIDEBAR HISTORY ---
with st.sidebar:
    st.header("🕰️ Recent Scans")
    if not st.session_state.history:
        st.info("No history yet.")
    for item in st.session_state.history:
        color = "green" if item["verdict"] == "REAL" else "red"
        st.markdown(f"""
        <div style="padding: 10px; border-left: 4px solid {color}; background-color: white; margin-bottom: 10px; border-radius: 0 5px 5px 0; box-shadow: 0 2px 4px rgba(0,0,0,0.05);">
            <strong style="color: {color};">{item['verdict']}</strong><br>
            <span style="font-size: 0.85rem; color: #555;">{item['snippet']}</span>
        </div>
        """, unsafe_allow_html=True)