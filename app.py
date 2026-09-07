import streamlit as st
import joblib
import os

# Set up page configuration
st.set_page_config(page_title="Fake News Detector", page_icon="🛡️", layout="centered")

# 1. Load the trained model safely
@st.cache_resource  # Caches the model so it doesn't reload on every click
def load_model():
    model_path = "model(1).pkl"
    if os.path.exists(model_path):
        return joblib.load(model_path)
    else:
        st.error(f"Error: '{model_path}' not found in the repository. Please check your filenames!")
        return None

model = load_model()

# 2. User Interface
st.title("🛡️ Fake News Detection System")
st.write("Paste a news headline or article body below to verify its authenticity using machine learning.")

# Text area for user input
user_input = st.text_area("News Content", placeholder="Type or paste your news text here...", height=200)

# 3. Prediction Logic
if st.button("Analyze News"):
    if not user_input.strip():
        st.warning("Please enter some text before analyzing.")
    elif model is None:
        st.error("Model is not loaded properly.")
    else:
        with st.spinner("Analyzing text..."):
            try:
                # Pass the text to your model
                # (Wraps input in a list because ML models expect a batch/array of inputs)
                prediction = model.predict([user_input])[0]
                
                st.write("---")
                st.subheader("Analysis Result:")
                
                # Customizing display based on common model outputs (0 or 1 / Real or Fake)
                # Adjust these labels if your model uses 0 for Fake and 1 for Real!
                if str(prediction).lower() in ['fake', '1', 1]:
                    st.error("🚨 **Warning:** This news text is highly likely to be **FAKE**.")
                else:
                    st.success("✅ **Verified:** This news text appears to be **REAL**.")
                    
            except Exception as e:
                st.error(f"An error occurred during prediction: {e}")
                st.info("Tip: If your model requires a separate TF-IDF Vectorizer/Tokenizer, make sure it's loaded and applied in the script.")
