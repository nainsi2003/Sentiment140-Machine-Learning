"""
Sentiment Analysis Model Training Script
=======================================
This script trains a sentiment analysis model using:
- TfidfVectorizer for text vectorization Regression classifier
- Logistic
- 80-20 train-test split
- Full preprocessing pipeline (lowercase, remove punctuation, stopwords, stemming)
"""

import pandas as pd
import numpy as np
import re
import nltk
import pickle
import os
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# --- NLTK Setup ---
print("Setting up NLTK...")
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', quiet=True)

# Define global objects
STOPWORDS = set(stopwords.words('english'))
stemmer = PorterStemmer()

# --- Expanded Dataset Creation ---
def create_expanded_dataset():
    """
    Creates a balanced dataset by:
    1. Loading the Amazon Alexa dataset
    2. Balancing positive and negative samples
    3. Adding general English sentences for better generalization
    """
    print("Loading and expanding dataset...")
    
    # Load Amazon Alexa dataset
    alexa_df = pd.read_csv('amazon_alexa.tsv', sep='\t')
    
    # Clean the data - remove empty reviews
    alexa_df = alexa_df.dropna(subset=['verified_reviews', 'feedback'])
    alexa_df = alexa_df[alexa_df['verified_reviews'].str.strip() != '']
    
    # Separate positive and negative reviews
    positive_reviews = alexa_df[alexa_df['feedback'] == 1]['verified_reviews'].tolist()
    negative_reviews = alexa_df[alexa_df['feedback'] == 0]['verified_reviews'].tolist()
    
    print(f"Amazon Alexa positive reviews: {len(positive_reviews)}")
    print(f"Amazon Alexa negative reviews: {len(negative_reviews)}")
    
    # Balance the dataset - take equal amounts
    # Use all negative (they're fewer) and sample same amount of positive
    n_samples = min(len(positive_reviews), len(negative_reviews))
    
    # Sample positive reviews to match negative count
    import random
    random.seed(42)
    positive_sampled = random.sample(positive_reviews, n_samples)
    
    print(f"Balanced samples: {n_samples} each")
    
    # Add general English sentences for better generalNaNlbfgs'
    )
    
    model.fit(X_train_tfidf, y_train)
    
    # Step 6: Evaluate the model
    print("\n" + "=" * 60)
    print("MODEL EVALUATION")
    print("=" * 60)
    
    # Training accuracy
    train_predictions = model.predict(X_train_tfidf)
    train_accuracy = accuracy_score(y_train, train_predictions)
    print(f"\nTraining Accuracy: {train_accuracy:.4f} ({train_accuracy*100:.2f}%)")
    
    # Testing accuracy
    test_predictions = model.predict(X_test_tfidf)
    test_accuracy = accuracy_score(y_test, test_predictions)
    print(f"Testing Accuracy: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")
    
    # Detailed classification report
    print("\nClassification Report:")
    print(classification_report(y_test, test_predictions, target_names=['Negative', 'Positive']))
    
    # Confusion Matrix
    print("Confusion Matrix:")
    cm = confusion_matrix(y_test, test_predictions)
    print(f"  True Negatives: {cm[0][0]}")
    print(f"  False Positives: {cm[0][1]}")
    print(f"  False Negatives: {cm[1][0]}")
    print(f"  True Positives: {cm[1][1]}")
    
    # Step 7: Save the model and vectorizer
    print("\n" + "=" * 60)
    print("SAVING MODEL")
    print("=" * 60)
    
    # Get the script directory and create backend/model path
    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_dir = os.path.join(script_dir, 'model')
    
    # Create the model directory if it doesn't exist
    os.makedirs(model_dir, exist_ok=True)
    print(f"Model directory: {model_dir}")
    
    # Save the model
    model_path = os.path.join(model_dir, 'sentiment_model.pkl')
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    print(f"Model saved to: {model_path}")
    
    # Save the vectorizer
    vectorizer_path = os.path.join(model_dir, 'tfidf_vectorizer.pkl')
    with open(vectorizer_path, 'wb') as f:
        pickle.dump(tfidf_vectorizer, f)
    print(f"Vectorizer saved to: {vectorizer_path}")
    
    # Save preprocessing info
    preprocessing_info = {
        'stemmer': 'PorterStemmer',
        'stopwords': 'english',
        'vectorizer': 'TfidfVectorizer',
        'max_features': 5000,
        'ngram_range': (1, 2),
        'train_test_split': '80-20',
        'model': 'LogisticRegression'
    }
    
    info_path = os.path.join(model_dir, 'preprocessing_info.pkl')
    with open(info_path, 'wb') as f:
        pickle.dump(preprocessing_info, f)
    print(f"Preprocessing info saved to: {info_path}")
    
    print("\n" + "=" * 60)
    print("TRAINING COMPLETE!")
    print("=" * 60)
    print(f"\nFinal Test Accuracy: {test_accuracy*100:.2f}%")
    print("\nYou can now run the Streamlit app with:")
    print("  streamlit run app.py")
    
    return model, tfidf_vectorizer


if __name__ == "__main__":
    main()
