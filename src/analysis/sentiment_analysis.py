"""
Sentiment analysis for newspaper articles.
"""

import os
import pickle
from typing import List, Dict, Any, Optional, Union

import numpy as np
import pandas as pd
from nltk.sentiment import SentimentIntensityAnalyzer
from transformers import pipeline, AutoModelForSequenceClassification, AutoTokenizer


class SentimentAnalyzer:
    """Class for sentiment analysis on newspaper articles."""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the SentimentAnalyzer with configuration settings.
        
        Args:
            config: Dictionary containing sentiment analysis configuration
        """
        self.config = config
        self.model_name = config.get('model', 'vader')
        self.transformer_model = config.get(
            'transformer_model', 
            'distilbert-base-uncased-finetuned-sst-2-english'
        )
        
        # Initialize analyzer
        self.analyzer = None
        self._initialize_analyzer()
    
    def _initialize_analyzer(self):
        """Initialize the appropriate sentiment analyzer based on configuration."""
        if self.model_name == 'vader':
            try:
                self.analyzer = SentimentIntensityAnalyzer()
            except Exception as e:
                import nltk
                nltk.download('vader_lexicon')
                self.analyzer = SentimentIntensityAnalyzer()
        
        elif self.model_name == 'transformers':
            self.analyzer = pipeline(
                "sentiment-analysis", 
                model=self.transformer_model,
                truncation=True
            )
    
    def analyze_text_vader(self, text: str) -> Dict[str, float]:
        """
        Analyze sentiment of text using VADER.
        
        Args:
            text: Input text
            
        Returns:
            Dictionary with sentiment scores
        """
        scores = self.analyzer.polarity_scores(text)
        return {
            'negative': scores['neg'],
            'neutral': scores['neu'],
            'positive': scores['pos'],
            'compound': scores['compound']
        }
    
    def analyze_text_transformers(self, text: str) -> Dict[str, float]:
        """
        Analyze sentiment of text using transformers.
        
        Args:
            text: Input text
            
        Returns:
            Dictionary with sentiment scores
        """
        # Truncate text if it's too long (most transformer models have a limit)
        max_length = 512
        if len(text.split()) > max_length:
            text = ' '.join(text.split()[:max_length])
        
        result = self.analyzer(text)[0]
        
        # Convert to standardized format
        if result['label'] == 'POSITIVE':
            return {
                'negative': 1 - result['score'],
                'positive': result['score'],
                'compound': result['score'] * 2 - 1  # Scale to [-1, 1]
            }
        else:
            return {
                'negative': result['score'],
                'positive': 1 - result['score'],
                'compound': -1 * result['score'] * 2 + 1  # Scale to [-1, 1]
            }
    
    def analyze_text(self, text: str) -> Dict[str, float]:
        """
        Analyze sentiment of text.
        
        Args:
            text: Input text
            
        Returns:
            Dictionary with sentiment scores
        """
        if not text or len(text.strip()) == 0:
            return {
                'negative': 0.0,
                'neutral': 1.0,
                'positive': 0.0,
                'compound': 0.0
            }
        
        if self.model_name == 'vader':
            return self.analyze_text_vader(text)
        elif self.model_name == 'transformers':
            return self.analyze_text_transformers(text)
        else:
            raise ValueError(f"Unsupported model: {self.model_name}")
    
    def analyze_document(self, document: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze sentiment of a document.
        
        Args:
            document: Document dictionary
            
        Returns:
            Document with added sentiment scores
        """
        # Use cleaned_text if available, otherwise use text or content
        if 'cleaned_text' in document:
            text = document['cleaned_text']
        elif 'text' in document:
            text = document['text']
        elif 'content' in document:
            text = document['content']
        else:
            raise KeyError("Document must contain either 'cleaned_text', 'text', or 'content' key")
        
        sentiment = self.analyze_text(text)
        document['sentiment'] = sentiment
        
        return document
    
    def analyze_documents(self, documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Analyze sentiment of multiple documents.
        
        Args:
            documents: List of document dictionaries
            
        Returns:
            List of documents with added sentiment scores
        """
        return [self.analyze_document(doc) for doc in documents]
    
    def get_sentiment_summary(self, documents: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Get summary statistics of sentiment across documents.
        
        Args:
            documents: List of document dictionaries with sentiment scores
            
        Returns:
            Dictionary with sentiment summary statistics
        """
        # Ensure documents have sentiment scores
        documents_with_sentiment = []
        for doc in documents:
            if 'sentiment' not in doc:
                doc = self.analyze_document(doc)
            documents_with_sentiment.append(doc)
        
        # Extract compound scores
        compound_scores = [doc['sentiment']['compound'] for doc in documents_with_sentiment]
        
        # Calculate statistics
        summary = {
            'mean_compound': np.mean(compound_scores),
            'median_compound': np.median(compound_scores),
            'std_compound': np.std(compound_scores),
            'min_compound': np.min(compound_scores),
            'max_compound': np.max(compound_scores),
            'positive_count': sum(1 for score in compound_scores if score > 0.05),
            'neutral_count': sum(1 for score in compound_scores if -0.05 <= score <= 0.05),
            'negative_count': sum(1 for score in compound_scores if score < -0.05),
        }
        
        # Calculate percentages
        total_count = len(compound_scores)
        summary['positive_percentage'] = summary['positive_count'] / total_count * 100
        summary['neutral_percentage'] = summary['neutral_count'] / total_count * 100
        summary['negative_percentage'] = summary['negative_count'] / total_count * 100
        
        return summary
    
    def save_analyzer(self, output_dir: str, filename: str = 'sentiment_analyzer.pkl') -> str:
        """
        Save the sentiment analyzer.
        
        Args:
            output_dir: Directory to save the analyzer
            filename: Filename for saved analyzer
            
        Returns:
            Path to the saved analyzer
        """
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        output_path = os.path.join(output_dir, filename)
        
        # For VADER, we can pickle the analyzer
        if self.model_name == 'vader':
            with open(output_path, 'wb') as f:
                pickle.dump({
                    'analyzer': self.analyzer,
                    'config': self.config
                }, f)
        
        # For transformers, we just save the configuration
        else:
            with open(output_path, 'wb') as f:
                pickle.dump({
                    'config': self.config
                }, f)
        
        return output_path
    
    @classmethod
    def load_analyzer(cls, model_path: str) -> 'SentimentAnalyzer':
        """
        Load a saved sentiment analyzer.
        
        Args:
            model_path: Path to the saved analyzer
            
        Returns:
            Loaded SentimentAnalyzer instance
        """
        with open(model_path, 'rb') as f:
            saved_data = pickle.load(f)
        
        # Create new instance
        sentiment_analyzer = cls(saved_data['config'])
        
        # For VADER, we can restore the pickled analyzer
        if sentiment_analyzer.model_name == 'vader' and 'analyzer' in saved_data:
            sentiment_analyzer.analyzer = saved_data['analyzer']
        
        return sentiment_analyzer
