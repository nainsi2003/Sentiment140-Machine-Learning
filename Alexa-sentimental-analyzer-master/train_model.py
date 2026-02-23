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
    
    # Add general English sentences for better generalization
    # More negative to balance the Alexa dataset
    general_positive = [
        "This is amazing! I love it so much.",
        "Great experience, highly recommend!",
        "Excellent quality and fast delivery.",
        "Very satisfied with my purchase.",
        "Wonderful product, works perfectly!",
        "Best decision I ever made.",
        "Fantastic! Exceeded my expectations.",
        "Love everything about this product.",
        "Very happy, will buy again!",
        "Outstanding quality and great value.",
        "Perfect in every way possible.",
        "Absolutely brilliant and wonderful.",
        "Superb quality and excellent service.",
        "Very impressive and highly recommend.",
        "Delighted with this purchase!",
        "Absolutely fantastic! Five stars!",
        "Great product, works as advertised.",
        "Very useful and convenient.",
        "Excellent choice, very pleased.",
        "Love it! Highly recommended!",
        "This made my day so much better.",
        "Incredibly happy with this buy.",
        "Quality is top notch.",
        "Can't live without this now.",
        "Simply the best purchase ever.",
        "Very helpful and efficient.",
        "Really enjoy using this every day.",
        "Perfect solution to my needs.",
        "Amazing quality and design.",
        "Very reliable and durable.",
        "Exceeded all my expectations!",
        "Absolutely wonderful experience.",
        "Highly satisfied customer here.",
        "Great features and easy to use.",
        "Fantastic value for money.",
        "Very comfortable and practical.",
        "Super happy with this product.",
        "Exactly what I was looking for.",
        "Beautiful design and works great.",
        "Impressive performance daily.",
        "Love the functionality.",
        "Very responsive customer service.",
        "Best in its category.",
        "Remarkable and outstanding.",
        "So glad I purchased this.",
        "Remarkable quality.",
        "Very resourceful and helpful.",
        "Excellent performance daily.",
        "Wonderful experience overall.",
        "Great investment.",
        "Highly functional and useful.",
        "Perfect for everyday use.",
        "Exceptional quality.",
    ] * 10  # Multiply to get more samples
    
    general_negative = [
        "This is terrible! I hate it.",
        "Poor quality, very disappointed.",
        "Worst purchase I've ever made.",
        "Don't waste your money on this.",
        "Very bad experience overall.",
        "Extremely poor quality.",
        "Not worth the price at all.",
        "Completely useless and broken.",
        "Terrible! Want my money back.",
        "Horrible product, avoid this!",
        "Very unhappy with this purchase.",
        "Total waste of time and money.",
        "Extremely dissatisfied with this.",
        "Poor craftsmanship and design.",
        "Not recommended at all.",
        "Very disappointing quality.",
        "This product is a scam!",
        "Absolutely terrible experience.",
        "Failed to meet expectations.",
        "Very poor performance.",
        "Not worth the investment.",
        "Completely dissatisfied.",
        "Terrible quality and service.",
        "Avoid this product completely.",
        "Very frustrating experience.",
        "Poor quality, falls apart easily.",
        "Not as described at all.",
        "Extremely unreliable product.",
        "Very disappointed and upset.",
        "Waste of money and time.",
        "Terrible customer support.",
        "Product broke after one use.",
        "Not recommended whatsoever.",
        "Very low quality materials.",
        "Extremely unhappy customer.",
        "Poor design and build.",
        "Not worth the hassle.",
        "Very bad and frustrating.",
        "Disappointed with quality.",
        "Doesn't work as promised.",
        "Terrible experience all around.",
        "Would not recommend to anyone.",
        "Very poor overall quality.",
        "Completely unsatisfied.",
        "Not worth your time.",
        "Extremely poor experience.",
        "Very dissatisfied indeed.",
        "Bad quality and design.",
        "Terrible choice, regret it.",
        "Very unhappy overall.",
        "Not recommended at all.",
        "Poor quality control.",
        "Very disappointing purchase.",
    ] * 10  # Multiply to get more samples
    
    # Combine all reviews
    all_reviews = positive_sampled + negative_reviews + general_positive + general_negative
    all_labels = [1] * len(positive_sampled) + [0] * len(negative_reviews) + [1] * len(general_positive) + [0] * len(general_negative)
    
    print(f"Total reviews after expansion: {len(all_reviews)}")
    print(f"  - Balanced Alexa (positive): {len(positive_sampled)}")
    print(f"  - Balanced Alexa (negative): {len(negative_reviews)}")
    print(f"  - General positive: {len(general_positive)}")
    print(f"  - General negative: {len(general_negative)}")
    print(f"  - Total positive: {all_labels.count(1)}")
    print(f"  - Total negative: {all_labels.count(0)}")
    
    return all_reviews, all_labels


# --- Text Preprocessing ---
def preprocess_text(text):
    """
    Apply full text preprocessing:
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


def preprocess_corpus(reviews):
    """
    Preprocess a list of reviews.
    """
    print("Preprocessing reviews...")
    processed_reviews = [preprocess_text(review) for review in reviews]
    return processed_reviews


# --- Main Training Pipeline ---
def main():
    print("=" * 60)
    print("SENTIMENT ANALYSIS MODEL TRAINING")
    print("=" * 60)
    
    # Create model directory if it doesn't exist
    os.makedirs('model', exist_ok=True)
    
    # Step 1: Create expanded dataset
    reviews, labels = create_expanded_dataset()
    
    # Step 2: Preprocess the reviews
    processed_reviews = preprocess_corpus(reviews)
    
    # Step 3: Split the data (80-20)
    print("\nSplitting data (80% train, 20% test)...")
    X_train, X_test, y_train, y_test = train_test_split(
        processed_reviews, 
        labels, 
        test_size=0.2, 
        random_state=42,
        stratify=labels
    )
    
    print(f"Training samples: {len(X_train)}")
    print(f"Testing samples: {len(X_test)}")
    
    # Step 4: Vectorize using TfidfVectorizer
    print("\nVectorizing with TfidfVectorizer...")
    tfidf_vectorizer = TfidfVectorizer(
        max_features=5000,
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.95
    )
    
    X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
    X_test_tfidf = tfidf_vectorizer.transform(X_test)
    
    print(f"Vocabulary size: {len(tfidf_vectorizer.vocabulary_)}")
    print(f"Feature matrix shape: {X_train_tfidf.shape}")
    
    # Step 5: Train Logistic Regression model
    print("\nTraining Logistic Regression model...")
    model = LogisticRegression(
        max_iter=1000,
        random_state=42,
        C=1.0,
        solver='lbfgs'
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
    
    # Save the model
    model_path = 'model/sentiment_model.pkl'
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    print(f"Model saved to: {model_path}")
    
    # Save the vectorizer
    vectorizer_path = 'model/tfidf_vectorizer.pkl'
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
    
    info_path = 'model/preprocessing_info.pkl'
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
