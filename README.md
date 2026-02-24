# Alexa Sentiment Analyzer

A Streamlit-based web application that uses Machine Learning to analyze sentiment in Amazon Alexa product reviews. The model is trained using Logistic Regression with TfidfVectorizer for text classification.

![Python](https://img.shields.io/badge/Python-3.11-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28-red)
![scikit-learn](https://img.shields.io/badge/scikit-learn-1.3-orange)

## 🌟 Features

- **Sentiment Analysis**: Predicts whether a review is Positive or Negative
- **Confidence Score**: Shows prediction probability as percentage
- **Real-time Prediction**: Instant analysis as you type
- **Balanced Model**: Trained on balanced dataset for accurate predictions
- **General English Support**: Works with general English sentences, not just product reviews

## 📊 Model Performance

| Metric | Score |
|--------|-------|
| Training Accuracy | 99.43% |
| Testing Accuracy | 94.79% |
| Precision (Positive) | 95% |
| Precision (Negative) | 94% |
| Recall (Positive) | 94% |
| Recall (Negative) | 95% |

## 🏗️ Project Structure

```
Alexa-sentimental-analyzer/
├── app.py                    # Streamlit web application
├── train_model.py            # Model training script
├── README.md                 # Project documentation
├── amazon_alexa.tsv         # Original dataset
├── requirements.txt          # Python dependencies
└── model/                   # Saved model files
    ├── sentiment_model.pkl
    ├── tfidf_vectorizer.pkl
    └── preprocessing_info.pkl
```

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- pip or conda

### Installation

1. **Clone the repository:**
```
bash
git clone https://github.com/yourusername/Alexa-sentimental-analyzer.git
cd Alexa-sentimental-analyzer
```

2. **Create a virtual environment (optional but recommended):**
```
bash
# Using venv
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Using conda
conda create -n sentiment python=3.11
conda activate sentiment
```

3. **Install dependencies:**
```
bash
pip install -r requirements.txt
```

### Training the Model

To train the model from scratch:
```
bash
python train_model.py
```

This will:
- Load and balance the dataset
- Preprocess text (lowercase, remove punctuation, stopwords, stemming)
- Split data 80-20
- Train Logistic Regression with TfidfVectorizer
- Save model to `model/` directory
- Display accuracy metrics in terminal

### Running the App

```
bash
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`

## 📖 How to Use

1. Enter a product review in the text area
2. Click "Analyze Sentiment" button
3. View the prediction (Positive/Negative) with confidence score
4. Try the sample reviews in the expandable section

### Example Reviews

**Positive:**
- "I love my Alexa! The sound quality is amazing and it's so easy to use."
- "Great product, works perfectly. Highly recommend to everyone!"

**Negative:**
- "Terrible product, stopped working after one week. Very disappointed."
- "Poor quality and bad customer service. Waste of money."

## 🔧 Technical Details

### Text Preprocessing Pipeline
1. **Lowercasing**: Convert all text to lowercase
2. **Remove Punctuation**: Remove all non-alphabetic characters
3. **Stopword Removal**: Remove common English stopwords
4. **Stemming**: Apply Porter Stemmer for word normalization

### Model Architecture
- **Vectorizer**: TfidfVectorizer (max_features=5000, ngram_range=(1,2))
- **Classifier**: Logistic Regression (max_iter=1000)
- **Train-Test Split**: 80-20 with stratification

### Dataset
- Original: Amazon Alexa Reviews Dataset (3,070 reviews)
- Expanded: 1,534 balanced reviews (767 positive, 767 negative)
- Additional: 1,060 general English sentences for better generalization

## 📦 Dependencies

```
streamlit>=1.28.0
pandas>=2.0.0
numpy>=1.24.0
scikit-learn>=1.3.0
nltk>=3.8.0
```

## 🤝 Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- [Amazon Alexa Reviews Dataset](https://www.kaggle.com/datasets/sid321axn/amazon-alexa-reviews) on Kaggle
- [Streamlit](https://streamlit.io/) for the web framework
- [NLTK](https://www.nltk.org/) for text preprocessing

---

Made with ❤️ for Sentiment Analysis
