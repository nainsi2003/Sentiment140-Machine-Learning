"""
Sentiment Analysis Model Training Script
=======================================
This script trains a sentiment analysis model using:
- TfidfVectorizer for text vectorization
- Logistic Regression classifier
- 80-20 train-test split
- Full preprocessing pipeline (lowercase, remove punctuation, stopwords, stemming)
"""

import pandas as pd
import numpy as np
import re
import nltk
import pickle
import os
import random
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


def preprocess_text(text):
    """
    Apply preprocessing to text:
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


# --- Expanded Dataset Creation ---
def download_dataset():
    """Download the Amazon Alexa dataset if it doesn't exist."""
    import urllib.request
    import zipfile
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    dataset_path = os.path.join(project_root, 'amazon_alexa.tsv')
    
    if os.path.exists(dataset_path):
        print(f"Dataset found at: {dataset_path}")
        return dataset_path
    
    print("Dataset not found. Downloading...")
    
    # Kaggle dataset URL (direct download)
    url = "https://raw.githubusercontent.com/sid321axn/amazon-alexa-reviews/master/amazon_alexa.tsv"
    
    try:
        urllib.request.urlretrieve(url, dataset_path)
        print(f"Dataset downloaded to: {dataset_path}")
        return dataset_path
    except Exception as e:
        print(f"Failed to download: {e}")
        return None


def create_expanded_dataset():
    """
    Creates a balanced dataset by:
    1. Loading the Amazon Alexa dataset
    2. Balancing positive and negative samples
    3. Adding general English sentences for better generalization
    """
    print("Loading and expanding dataset...")
    
    # Get the script directory and look for dataset in project root
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    dataset_path = os.path.join(project_root, 'amazon_alexa.tsv')
    
    # Try to download if not exists
    if not os.path.exists(dataset_path):
        dataset_path = download_dataset()
        if dataset_path is None:
            # Create sample dataset as fallback
            print("Creating sample dataset...")
            data = {
                'verified_reviews': [
                    "I love my Alexa! The sound quality is amazing",
                    "Great product, highly recommend",
                    "Perfect for my smart home",
                    "Terrible product, stopped working after one week",
                    "Poor quality, very disappointed",
                    "Waste of money, not worth it",
                    "Amazing device, best purchase ever",
                    "Love all the features",
                    "Bad customer service",
                    "Excellent sound quality"
                ],
                'feedback': [1, 1, 1, 0, 0, 0, 1, 1, 0, 1]
            }
            alexa_df = pd.DataFrame(data)
            # Save for future use
            alexa_df.to_csv(dataset_path, sep='\t', index=False)
            print(f"Sample dataset created at: {dataset_path}")
        else:
            alexa_df = pd.read_csv(dataset_path, sep='\t')
    else:
        # Load Amazon Alexa dataset
        print(f"Loading dataset from: {dataset_path}")
        alexa_df = pd.read_csv(dataset_path, sep='\t')
    
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
    random.seed(42)
    positive_sampled = random.sample(positive_reviews, n_samples)
    
    print(f"Balanced samples: {n_samples} each")
    
    # Add general English sentences for better generalization
    general_sentences = [
        # Positive sentences
        "This is a wonderful day", "I am so happy today", "Great job well done",
        "Excellent performance", "I love this so much", "Amazing experience",
        "Fantastic result", "Very pleased with this", "Wonderful service",
        "Highly recommend this product", "Best purchase ever", "So satisfied",
        "Perfect quality", "Outstanding work", "Brilliant idea",
        # Negative sentences  
        "This is terrible", "I hate this so much", "Very bad experience",
        "Poor quality", "Disappointed with this", "Awful service",
        "Worst purchase ever", "Not recommended at all", "Terrible result",
        "Very unhappy", "Poor performance", "Bad quality",
        "Disappointing experience", "Not worth the money", "Horrible"
    ]
    
    # Add general sentences to balance
    positive_sampled.extend(general_sentences[:15])
    negative_reviews.extend(general_sentences[15:])
    
    # Combine all data
    all_reviews = positive_sampled + negative_reviews
    labels = [1] * len(positive_sampled) + [0] * len(negative_reviews)
    
    # Create DataFrame
    df = pd.DataFrame({
        'review': all_reviews,
        'label': labels
    })
    
    # Shuffle the dataset
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    print(f"Total dataset size: {len(df)}")
    print(f"Positive samples: {sum(df['label'] == 1)}")
    print(f"Negative samples: {sum(df['label'] == 0)}")
    
    return df


def main():
    """
    Main function to train the sentiment analysis model.
    """
    print("=" * 60)
    print("ALEXA SENTIMENT ANALYZER - MODEL TRAINING")
    print("=" * 60)
    
    # Step 1: Create expanded dataset
    print("\n" + "=" * 60)
    print("STEP 1: LOADING AND PREPARING DATA")
    print("=" * 60)
    
    df = create_expanded_dataset()
    
    # Step 2: Preprocess text
    print("\n" + "=" * 60)
    print("STEP 2: PREPROCESSING TEXT")
    print("=" * 60)
    
    print("Applying preprocessing to all reviews...")
    df['processed_review'] = df['review'].apply(preprocess_text)
    
    print(f"Sample original: '{df['review'].iloc[0]}'")
    print(f"Sample processed: '{df['processed_review'].iloc[0]}'")
    
    # Step 3: Split data
    print("\n" + "=" * 60)
    print("STEP 3: SPLITTING DATA")
    print("=" * 60)
    
    X = df['processed_review']
    y = df['label']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=0.2, 
        random_state=42,
        stratify=y
    )
    
    print(f"Training samples: {len(X_train)}")
    print(f"Testing samples: {len(X_test)}")
    
    # Step 4: Vectorize text
    print("\n" + "=" * 60)
    print("STEP 4: VECTORIZING TEXT (TF-IDF)")
    print("=" * 60)
    
    tfidf_vectorizer = TfidfVectorizer(
        max_features=5000,
        ngram_range=(1, 2)
    )
    
    X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
    X_test_tfidf = tfidf_vectorizer.transform(X_test)
    
    print(f"Vocabulary size: {len(tfidf_vectorizer.vocabulary_)}")
    print(f"Training matrix shape: {X_train_tfidf.shape}")
    print(f"Testing matrix shape: {X_test_tfidf.shape}")
    
    # Step 5: Train model
    print("\n" + "=" * 60)
    print("STEP 5: TRAINING MODEL")
    print("=" * 60)
    
    model = LogisticRegression(
        max_iter=1000,
        random_state=42,
        solver='lbfgs'
    )
    
    print("Training Logistic Regression...")
    model.fit(X_train_tfidf, y_train)
    print("Training complete!")
    
    # Step 6: Evaluate the model
    print("\n" + "=" * 60)
    print("STEP 6: MODEL EVALUATION")
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
    print("STEP 7: SAVING MODEL")
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
    print("  streamlit run frontend/app.py")
    
    return model, tfidf_vectorizer


if __name__ == "__main__":
    main()
