"""
Alexa Sentiment Analyzer - Streamlit App
========================================
This app uses a trained Logistic Regression model with TfidfVectorizer
to predict sentiment from Amazon Alexa product reviews.

The model was trained using:
- TfidfVectorizer for text vectorization
- Logistic Regression classifier
- 80-20 train-test split
- Full preprocessing pipeline (lowercase, remove punctuation, stopwords, stemming)
"""

import streamlit as st
import pandas as pd
import numpy as np
import re
import pickle
import os
import nltk
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords

# --- NLTK Setup ---
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

# Define global objects
STOPWORDS = set(stopwords.words('english'))
stemmer = PorterStemmer()

# --- Load Saved Model ---
@st.cache_resource
def load_model():
    """Load the trained model and vectorizer from pickle files."""
    # Get the script directory (frontend folder)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Go up from frontend/ to project root, then to backend/model
    project_root = os.path.dirname(script_dir)
    backend_dir = os.path.join(project_root, 'backend')
    model_dir = os.path.join(backend_dir, 'model')
    
    model_path = os.path.join(model_dir, 'sentiment_model.pkl')
    vectorizer_path = os.path.join(model_dir, 'tfidf_vectorizer.pkl')

    if not os.path.exists(model_path) or not os.path.exists(vectorizer_path):
        st.error(f"Model files not found at: {model_dir}")
        st.info("Please ensure you have trained the model first.")
        st.info("Run: python backend/train_model.py from the project root")
        st.stop()
    
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    
    with open(vectorizer_path, 'rb') as f:
        tfidf_vectorizer = pickle.load(f)
    
    return model, tfidf_vectorizer


# --- Preprocessing Function ---
def preprocess_text(text):
    """
    Apply the same preprocessing as used in training:
    1. Lowercase
    2. Remove punctuation and numbers
    3. Remove stopwords
    4. Apply stemming
    """
    # Convert to lowercase
    text = text.lower()
    
    # Remove punctuation and numbers
    text = re.sub(r'[^a-zA-Z]', ' ', text)
    
    # Split into words
    words = text.split()
    
    # Remove stopwords and apply stemming
    words = [stemmer.stem(word) for word in words if word not in STOPWORDS and len(word) > 1]
    
    # Join back into a single string
    processed_text = ' '.join(words)
    
    return processed_text


# --- Main Streamlit App ---
def main():
    # Load model and vectorizer
    MODEL, TFIDF_VECTORIZER = load_model()
    
    # Page configuration
    st.set_page_config(
        page_title="Alexa Sentiment Analyzer",
        page_icon="🎙️",
        layout="centered"
    )
    
    # Header
    st.title("🎙️ Alexa Sentiment Analyzer")
    st.markdown("Analyze the sentiment of Amazon Alexa product reviews")
    st.divider()
    
    # Input section
    st.subheader("Enter a Review")
    review_text = st.text_area(
        "Type or paste your Amazon Alexa review here:",
        height=150,
        placeholder="e.g., 'I love my Alexa! The sound quality is amazing...'"
    )
    
    # Analyze button
    if st.button("🔍 Analyze Sentiment", type="primary"):
        if review_text.strip():
            try:
                # Preprocess the review
                processed_review = preprocess_text(review_text)
                
                # Transform using the vectorizer
                review_tfidf = TFIDF_VECTORIZER.transform([processed_review])
                
                # Make prediction
                prediction = MODEL.predict(review_tfidf)[0]
                prediction_proba = MODEL.predict_proba(review_tfidf)[0]
                
                # Get probabilities
                negative_prob = prediction_proba[0]
                positive_prob = prediction_proba[1]
                
                # Display results
                st.divider()
                st.subheader("📊 Analysis Results")
                
                # Prediction result
                if prediction == 1:
                    st.success("✅ **Positive Review**")
                else:
                    st.error("❌ **Negative Review**")
                
                # Confidence visualization
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("**Negative**")
                    st.progress(1 - positive_prob)
                    st.markdown(f"**{negative_prob*100:.1f}%**")
                        
                with col2:
                    st.markdown("**Positive**")
                    st.progress(positive_prob)
                    st.markdown(f"**{positive_prob*100:.1f}%**")
                
                # Detailed metrics
                st.divider()
                st.subheader("📈 Detailed Metrics")
                
                col_neg, col_pos = st.columns(2)
                
                with col_neg:
                    st.metric(
                        label="Negative Probability",
                        value=f"{negative_prob*100:.2f}%"
                    )
                
                with col_pos:
                    st.metric(
                        label="Positive Probability",
                        value=f"{positive_prob*100:.2f}%"
                    )
                
                # Show processed text
                st.caption(f"**Processed Text (Stems):** `{processed_review}`")
                    
            except Exception as e:
                st.error(f"Error during prediction: {str(e)}")
        else:
            st.warning("⚠️ Please enter a review to analyze.")
    
    # Sample reviews section
    st.divider()
    with st.expander("📋 Sample Reviews to Try"):
        st.markdown("""
        **Positive Examples:**
        - "I love my Alexa! The sound quality is amazing and it's so easy to use."
        - "Great product, works perfectly. Highly recommend to everyone!"
        - "Perfect for my smart home setup. Love all the features!"
        
        **Negative Examples:**
        - "Terrible product, stopped working after one week. Very disappointed."
        - "Poor quality and bad customer service. Waste of money."
        - "The device is always offline and doesn't respond to commands."
        """)
    
    # Footer
    st.divider()
    st.markdown("""
    <div style='text-align: center; color: gray;'>
        <small>Model: Logistic Regression | Vectorizer: TfidfVectorizer | Split: 80-20</small>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
