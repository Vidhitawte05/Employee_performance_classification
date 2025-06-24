"""
Model Training Script - Run this to create the required model files
"""

import pandas as pd
import numpy as np
import joblib
import string
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.metrics import classification_report, accuracy_score
import warnings
warnings.filterwarnings('ignore')

# Download NLTK data
nltk.download("punkt", quiet=True)
nltk.download("stopwords", quiet=True)
nltk.download("wordnet", quiet=True)
nltk.download("punkt_tab", quiet=True)

def create_sample_data():
    """Create sample training data if test.csv is not available"""
    sample_data = {
        'Input': [
            "Employee consistently exceeds expectations and delivers exceptional results",
            "Shows great initiative and leadership qualities in team projects",
            "Meets all deadlines and produces quality work consistently",
            "Good team player with solid technical skills",
            "Completes assigned tasks adequately but lacks innovation",
            "Performance is satisfactory but could improve communication",
            "Frequently misses deadlines and requires constant supervision",
            "Work quality is below standards and needs significant improvement"
        ],
        'Output': ['Excellent', 'Excellent', 'Good', 'Good', 'Average', 'Average', 'Needs Improvement', 'Needs Improvement']
    }
    return pd.DataFrame(sample_data)

def preprocess_text(text):
    """Text preprocessing function"""
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words("english"))
    
    # Lowercasing
    text = text.lower()
    # Remove punctuation
    text = text.translate(str.maketrans("", "", string.punctuation))
    # Remove numbers
    text = re.sub(r"\d+", "", text)
    # Tokenization
    tokens = word_tokenize(text)
    # Remove stopwords and apply lemmatization
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    return " ".join(tokens)

def train_and_save_models():
    """Train models and save them as pickle files"""
    
    # Load or create data
    try:
        df = pd.read_csv("test.csv")
        print("Loaded data from test.csv")
    except FileNotFoundError:
        print("test.csv not found. Creating sample data...")
        df = create_sample_data()
        # Expand sample data for better training
        df = pd.concat([df] * 50, ignore_index=True)  # Replicate for more samples
    
    print(f"Dataset shape: {df.shape}")
    print(f"Class distribution:\n{df['Output'].value_counts()}")
    
    # Preprocess text
    df['Processed_Input'] = df['Input'].apply(preprocess_text)
    
    # Feature extraction
    tfidf_vectorizer = TfidfVectorizer(
        max_features=150, 
        min_df=1,  # Reduced for small dataset
        ngram_range=(1,2), 
        stop_words='english', 
        norm='l2'
    )
    X_tfidf = tfidf_vectorizer.fit_transform(df['Processed_Input'])
    
    # Dimensionality reduction
    svd = TruncatedSVD(n_components=min(25, X_tfidf.shape[1]-1), random_state=42)
    X_reduced = svd.fit_transform(X_tfidf)
    
    # Scaling
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_reduced)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, df['Output'], 
        test_size=0.2, 
        random_state=42, 
        stratify=df['Output']
    )
    
    # Apply SMOTE if we have enough samples
    if len(np.unique(y_train)) > 1 and len(y_train) > 10:
        try:
            smote = SMOTE(random_state=42)
            X_train, y_train = smote.fit_resample(X_train, y_train)
            print("Applied SMOTE for class balancing")
        except ValueError as e:
            print(f"SMOTE not applied: {e}")
    
    # Train Random Forest
    rf = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        random_state=42,
        class_weight='balanced'
    )
    rf.fit(X_train, y_train)
    
    # Evaluate model
    y_pred = rf.predict(X_test)
    print(f"\nModel Performance:")
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print(f"\nClassification Report:\n{classification_report(y_test, y_pred)}")
    
    # Save all models
    joblib.dump(rf, "random_forest_model.pkl")
    joblib.dump(tfidf_vectorizer, "tfidf_vectorizer.pkl")
    joblib.dump(svd, "svd_model.pkl")
    joblib.dump(scaler, "scaler.pkl")
    
    print("\n✅ All models saved successfully!")
    print("Files created:")
    print("- random_forest_model.pkl")
    print("- tfidf_vectorizer.pkl")
    print("- svd_model.pkl")
    print("- scaler.pkl")

if __name__ == "__main__":
    train_and_save_models()
