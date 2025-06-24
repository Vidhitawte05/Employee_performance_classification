# -*- coding: utf-8 -*-
"""
Employee Performance Classifier - Streamlit App
Corrected and improved version with proper error handling
"""

import streamlit as st
import joblib
import pandas as pd
import numpy as np
import string
import re
import os
from pathlib import Path

# NLTK imports with error handling
try:
    import nltk
    from nltk.corpus import stopwords
    from nltk.tokenize import word_tokenize
    from nltk.stem import WordNetLemmatizer
    
    # Download required NLTK data
    nltk.download("punkt", quiet=True)
    nltk.download("stopwords", quiet=True)
    nltk.download("wordnet", quiet=True)
    nltk.download("punkt_tab", quiet=True)
    
    stop_words = set(stopwords.words("english"))
    lemmatizer = WordNetLemmatizer()
    
except ImportError as e:
    st.error(f"NLTK import error: {e}")
    st.stop()

class PerformanceClassifier:
    def __init__(self):
        self.rf_model = None
        self.tfidf_vectorizer = None
        self.svd_model = None
        self.scaler = None
        self.models_loaded = False
        
    def load_models(self):
        """Load all required models and preprocessors"""
        try:
            model_files = {
                'random_forest_model.pkl': 'rf_model',
                'tfidf_vectorizer.pkl': 'tfidf_vectorizer',
                'svd_model.pkl': 'svd_model',
                'scaler.pkl': 'scaler'
            }
            
            missing_files = []
            for filename, attr_name in model_files.items():
                if os.path.exists(filename):
                    setattr(self, attr_name, joblib.load(filename))
                else:
                    missing_files.append(filename)
            
            if missing_files:
                st.error(f"Missing model files: {', '.join(missing_files)}")
                st.info("Please ensure all model files are in the same directory as the app.")
                return False
                
            self.models_loaded = True
            return True
            
        except Exception as e:
            st.error(f"Error loading models: {str(e)}")
            return False
    
    def preprocess_text(self, text):
        """Clean and preprocess input text"""
        if not text or not isinstance(text, str):
            return ""
            
        try:
            # Lowercasing
            text = text.lower()
            
            # Remove punctuation
            text = text.translate(str.maketrans("", "", string.punctuation))
            
            # Remove numbers
            text = re.sub(r"\d+", "", text)
            
            # Tokenization
            tokens = word_tokenize(text)
            
            # Remove stopwords and apply lemmatization
            tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words and len(word) > 2]
            
            return " ".join(tokens)
            
        except Exception as e:
            st.error(f"Error in text preprocessing: {str(e)}")
            return text.lower()
    
    def preprocess_input(self, text):
        """Complete preprocessing pipeline for prediction"""
        try:
            # Clean text
            cleaned = self.preprocess_text(text)
            
            if not cleaned.strip():
                raise ValueError("Text preprocessing resulted in empty string")
            
            # Transform using TF-IDF
            tfidf = self.tfidf_vectorizer.transform([cleaned])
            
            # Apply dimensionality reduction
            reduced = self.svd_model.transform(tfidf)
            
            # Scale features
            scaled = self.scaler.transform(reduced)
            
            return scaled
            
        except Exception as e:
            st.error(f"Error in input preprocessing: {str(e)}")
            return None
    
    def predict(self, text):
        """Make prediction on input text"""
        if not self.models_loaded:
            return None, "Models not loaded"
            
        try:
            processed_input = self.preprocess_input(text)
            
            if processed_input is None:
                return None, "Failed to preprocess input"
            
            # Make prediction
            prediction = self.rf_model.predict(processed_input)
            
            # Get prediction probabilities for confidence
            try:
                probabilities = self.rf_model.predict_proba(processed_input)
                confidence = np.max(probabilities)
            except:
                confidence = None
            
            return prediction[0], confidence
            
        except Exception as e:
            return None, f"Prediction error: {str(e)}"

def main():
    # Page configuration
    st.set_page_config(
        page_title="Employee Performance Classifier",
        page_icon="🧠",
        layout="centered",
        initial_sidebar_state="collapsed"
    )
    
    # Custom CSS for better styling
    st.markdown("""
    <style>
    .main-header {
        text-align: center;
        color: #2E86AB;
        margin-bottom: 2rem;
    }
    .prediction-box {
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    .success-box {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        color: #155724;
    }
    .info-box {
        background-color: #e2f3ff;
        border: 1px solid #b8daff;
        color: #004085;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Header
    st.markdown('<h1 class="main-header">🧠 Employee Performance Classifier</h1>', unsafe_allow_html=True)
    
    # Initialize classifier
    if 'classifier' not in st.session_state:
        st.session_state.classifier = PerformanceClassifier()
    
    classifier = st.session_state.classifier
    
    # Load models if not already loaded
    if not classifier.models_loaded:
        with st.spinner("Loading ML models..."):
            if not classifier.load_models():
                st.error("Failed to load required models. Please check if all model files are present.")
                st.stop()
        st.success("Models loaded successfully!")
    
    # App description
    st.markdown("""
    <div class="info-box">
    <strong>How to use:</strong><br>
    Enter a description of an employee's performance, behavior, or feedback in the text area below. 
    The AI model will analyze the text and predict the performance category.
    
    <strong>Performance Categories:</strong>
    • Excellent • Good • Average • Needs Improvement
    </div>
    """, unsafe_allow_html=True)
    
    # Input section
    st.subheader("📝 Enter Performance Description")
    
    # Sample examples
    with st.expander("📋 View Sample Inputs"):
        st.markdown("""
        **Excellent Performance:**
        "John consistently exceeds expectations, delivers high-quality work ahead of deadlines, and demonstrates exceptional leadership skills."
        
        **Good Performance:**
        "Sarah meets all her targets, collaborates well with the team, and shows initiative in problem-solving."
        
        **Average Performance:**
        "Mike completes assigned tasks on time but lacks creativity and requires occasional guidance."
        
        **Needs Improvement:**
        "The employee frequently misses deadlines, makes errors, and requires constant supervision."
        """)
    
    # Text input
    user_input = st.text_area(
        "Describe the employee's performance:",
        height=150,
        placeholder="Enter detailed performance description here...",
        help="Provide specific examples of behavior, achievements, or areas of concern."
    )
    
    # Prediction section
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        predict_button = st.button("🔍 Predict Performance", use_container_width=True)
    
    if predict_button:
        if not user_input.strip():
            st.warning("⚠️ Please enter a valid performance description.")
        else:
            with st.spinner("Analyzing performance..."):
                prediction, confidence = classifier.predict(user_input)
                
                if prediction is None:
                    st.error(f"❌ Prediction failed: {confidence}")
                else:
                    # Display prediction
                    st.markdown(f"""
                    <div class="prediction-box success-box">
                    <h3 style="margin: 0; text-align: center;">
                    🎯 Predicted Performance: <strong>{prediction}</strong>
                    </h3>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Display confidence if available
                    if confidence is not None:
                        confidence_percent = confidence * 100
                        st.progress(confidence)
                        st.caption(f"Confidence: {confidence_percent:.1f}%")
                    
                    # Performance interpretation
                    interpretations = {
                        "Excellent": "🌟 Outstanding performance with exceptional results",
                        "Good": "✅ Solid performance meeting or exceeding expectations",
                        "Average": "📊 Satisfactory performance with room for growth",
                        "Needs Improvement": "⚠️ Performance requires attention and development"
                    }
                    
                    if prediction in interpretations:
                        st.info(interpretations[prediction])
    
    # Additional features
    st.markdown("---")
    
    # Batch processing section
    with st.expander("📊 Batch Processing"):
        st.markdown("Upload a CSV file with employee performance descriptions for batch prediction.")
        uploaded_file = st.file_uploader("Choose CSV file", type=['csv'])
        
        if uploaded_file is not None:
            try:
                df = pd.read_csv(uploaded_file)
                
                if 'Input' in df.columns:
                    st.write("Preview of uploaded data:")
                    st.dataframe(df.head())
                    
                    if st.button("Process Batch Predictions"):
                        predictions = []
                        progress_bar = st.progress(0)
                        
                        for i, text in enumerate(df['Input']):
                            pred, _ = classifier.predict(str(text))
                            predictions.append(pred if pred else "Error")
                            progress_bar.progress((i + 1) / len(df))
                        
                        df['Predicted_Performance'] = predictions
                        st.success("Batch processing completed!")
                        st.dataframe(df)
                        
                        # Download results
                        csv = df.to_csv(index=False)
                        st.download_button(
                            label="📥 Download Results",
                            data=csv,
                            file_name="performance_predictions.csv",
                            mime="text/csv"
                        )
                else:
                    st.error("CSV file must contain an 'Input' column with performance descriptions.")
                    
            except Exception as e:
                st.error(f"Error processing file: {str(e)}")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666; margin-top: 2rem;">
    <p>Created by Vidhi,Amey,Nandan and Ananya| Employee Performance Classification System</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
