import streamlit as st
import joblib
import re

# Load Model and Vectorizer
vectorizer = joblib.load("datasets/vectorizer.pkl")
model = joblib.load("datasets/fake_news_model.pkl")

# Function to clean text
def clean_text(text):
    text = text.lower()  # Convert to lowercase
    text = re.sub(r'\W', ' ', text)  # Remove special characters
    text = re.sub(r'\s+', ' ', text)  # Remove extra spaces
    return text

# Streamlit App
st.title("📰 Fake News Detection System")
st.markdown("### Enter a news article below to check if it's **Fake** or **Real**.")

# Text Input
news_input = st.text_area("Enter News Text", height=200)

if st.button("Analyze"):
    if news_input.strip():
        # Preprocess and Predict
        cleaned_text = clean_text(news_input)
        vectorized_text = vectorizer.transform([cleaned_text]).toarray()
        prediction = model.predict(vectorized_text)[0]

        # Display Result
        if prediction == 0:
            st.error("❌ Fake News Detected!")
        else:
            st.success("✅ This News is True!")
    else:
        st.warning("⚠️ Please enter some text to analyze.")


