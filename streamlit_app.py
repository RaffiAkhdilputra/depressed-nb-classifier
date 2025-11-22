import streamlit as st
import joblib
import pandas as pd

# Page Configuration
st.set_page_config(
    page_title="Depression Sentiment Classifier",
    page_icon="🧠",
    layout="centered"
)

# Custom CSS for styling
st.markdown("""
    <style>
    .stTextArea textarea {
        font-size: 16px;
    }
    .stButton button {
        background-color: #4CAF50; 
        color: white; 
        font-size: 18px;
    }
    </style>
    """, unsafe_allow_html=True)

# Function to load the model
@st.cache_resource
def load_model():
    try:
        # Load the pipeline we saved in train_model.py
        model = joblib.load('model/model.pkl')
        vectorizer = joblib.load('model/tfidf_vectorizer.pkl')
        return model, vectorizer
    except FileNotFoundError:
        st.error("Model file not found. Please send an email to raffiakdilputra123@gmail.com about the problem.")
        return None, None

# Load model
model, vectorizer = load_model()

# Header
st.title("🧠 Mental Health Text Classifier")
st.markdown("This app uses a **Naive Bayes** machine learning model to detect potential signs of depression in text.")
st.markdown("---")

# Input Section
user_input = st.text_area("Enter your text here:", height=150, placeholder="Type something... e.g., 'I feel empty inside' or 'Today is a great day!'")

if st.button("Analyze Text"):
    if user_input.strip() == "":
        st.warning("Please enter some text first.")
    else:
        if model and vectorizer:
            # 1. Transform text to numbers (Vectorize)
            text_vectorized = vectorizer.transform([user_input])
            
            # 2. Predict (Model receives the vectorized 2D array)
            prediction = model.predict(text_vectorized)[0]
            probability = model.predict_proba(text_vectorized).max()
            
            # Display Results
            st.markdown("### Result:")
            
            if prediction == 1:
                prediction = "Depressed"
            else:
                prediction = "Not Depressed"

            if prediction == "Depressed":
                st.error(f"**Classification: {prediction}**")
                st.write(f"Confidence: {probability:.2%}")
                st.info("Note: This AI tool is not a substitute for professional diagnosis. If you need help, please contact a mental health professional.")
            else:
                st.success(f"**Classification: {prediction}**")
                st.write(f"Confidence: {probability:.2%}")

# Footer
st.markdown("---")
st.caption("Built with Streamlit & Scikit-Learn | Naive Bayes Classifier")