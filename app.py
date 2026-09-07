import streamlit as st
import joblib
import re

# Set page layout
st.set_page_config(page_title="Fake News Detector", page_icon="📰", layout="centered")

# Cache model loading so it only loads once into memory
@st.cache_resource
def load_assets():
    vectorizer = joblib.load('tfidf_vectorizer.pkl')
    model = joblib.load('model (1).pkl')
    return vectorizer, model

vectorizer, model = load_assets()

# Cleaning function matching the training phase
def clean_text(text):
    text = re.sub(r'^.*?\(Reuters\)\s*-\s*', '', text, flags=re.IGNORECASE)
    text = re.sub(r'^.*?[A-Z]{2,}\s*\(AP\)\s*-\s*', '', text, flags=re.IGNORECASE)
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    return text.lower().strip()

# App UI
st.title("📰 Fake News Detector")
st.write("Enter an article or headline below to assess its credibility.")

user_input = st.text_area("News Text", height=200, placeholder="Paste news article content here...")

if st.button("Analyze News"):
    if not user_input.strip():
        st.warning("Please enter some text before analyzing.")
    else:
        cleaned = clean_text(user_input)
        vec_input = vectorizer.transform([cleaned])
        
        prediction = model.predict(vec_input)[0]
        proba = model.predict_proba(vec_input)[0]

        st.divider()
        if prediction == 1:
            st.error(f"🚨 **Verdict: Likely FAKE News** ({proba[1]*100:.1f}% confidence)")
        else:
            st.success(f"✅ **Verdict: Likely REAL News** ({proba[0]*100:.1f}% confidence)")
