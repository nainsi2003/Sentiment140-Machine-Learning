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
    # Get the directory where the script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(script_dir, 'model', 'sentiment_model.pkl')
    vectorizer_path = os.path.join(script_dir, 'model', 'tfidf_vectorizer.pkl')

    if not os.path.exists(model_path) or not os.path.exists(vectorizer_path):
        st.error(f"Model files not found at: {model_path}")
        st.info("Run: python train_model.py")
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
    
    # Title and description
    st.title("🎙️ Alexa Review Sentiment Analyzer")
    st.markdown("""
    This application uses a **Logistic Regression** model with **TfidfVectorizer** 
    to analyze the sentiment of Amazon Alexa product reviews.
    
    The model was trained on **5,000+ reviews** with an 80-20 train-test split.
    """)
    
    st.divider()
    
    # Input area
    st.subheader("📝 Enter a Review")
    review_input = st.text_area(
        "Type or paste your Amazon Alexa product review:",
        placeholder="e.g., 'I love this product! The sound quality is amazing and setup was so easy.'\n\nOr try: 'Terrible product, stopped working after a week. Very disappointed.'",
        height=150
    )
    
    # Analyze button
    if st.button("🔍 Analyze Sentiment", type="primary", use_container_width=True):
        if review_input.strip():
            with st.spinner('Analyzing sentiment...'):
                try:
                    # 1. Preprocess the input text
                    processed_review = preprocess_text(review_input)
                    
                    # Guard against empty processed text
                    if not processed_review.strip():
                        st.warning("⚠️ The review contains only stopwords or non-alphabetical characters. Please enter a more substantial review.")
                        return
                    
                    # 2. Vectorize the processed text
                    review_vector = TFIDF_VECTORIZER.transform([processed_review])
                    
                    # 3. Predict the sentiment
                    prediction = MODEL.predict(review_vector)[0]
                    probabilities = MODEL.predict_proba(review_vector)[0]
                    
                    # Extract probabilities
                    negative_prob = probabilities[0]
                    positive_prob = probabilities[1]
                    
                    # 4. Display results
                    st.divider()
                    st.subheader("📊 Analysis Result")
                    
                    # Create columns for result display
                    col_result, col_gauge = st.columns([1, 1])
                    
                    # Left column: Main prediction
                    with col_result:
                        if prediction == 1:
                            st.success("✅ **POSITIVE Review**")
                            st.markdown(f"**Confidence:** {positive_prob*100:.2f}%")
                        else:
                            st.error("❌ **NEGATIVE Review**")
                            st.markdown(f"**Confidence:** {negative_prob*100:.2f}%")
                    
                    # Right column: Visual gauge
                    with col_gauge:
                        st.markdown("##### Sentiment Gauge")
                        
                        # Labels
                        neg_col, pos_col = st.columns(2)
                        with neg_col:
                            st.markdown(f"<div style='text-align: left; font-size: 0.8rem;'>Negative<br>**{negative_prob*100:.1f}%**</div>", unsafe_allow_html=True)
                        with pos_col:
                            st.markdown(f"<div style='text-align: right; font-size: 0.8rem;'>Positive<br>**{positive_prob*100:.1f}%**</div>", unsafe_allow_html=True)
                        
                        # Progress bar
                        st.progress(positive_prob)
                    
                    # Detailed metrics
                    st.divider()
                    st.subheader("📈 Detailed Metrics")
                    
                    col_neg, col_pos = st.columns(2)
                    
                    with col_neg:
                        st.metric(
                            label="Negative Probability",
                            value=f"{negative_prob*100:.2f}%",
                            delta=f"{negative_prob*100 - 50:.2f}%" if negative_prob > 0.5 else None
                        )
                    
                    with col_pos:
                        st.metric(
                            label="Positive Probability",
                            value=f"{positive_prob*100:.2f}%",
                            delta=f"{positive_prob*100 - 50:.2f}%" if positive_prob > 0.5 else None
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
