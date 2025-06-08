"""
Sentiment analysis for newspaper articles.
"""

import os
import pickle
from typing import List, Dict, Any, Optional, Union

import numpy as np
import tiktoken
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
                print("Initializing VADER sentiment analyzer...")
                self.analyzer = SentimentIntensityAnalyzer()
                print("VADER sentiment analyzer initialized successfully.")
            except Exception as e:
                print(f"Error initializing VADER, downloading lexicon: {e}")
                import nltk
                nltk.download('vader_lexicon')
                self.analyzer = SentimentIntensityAnalyzer()
                print("VADER sentiment analyzer initialized after downloading lexicon.")
        
        elif self.model_name == 'transformers':
            try:
                # Check for GPU availability
                import torch
                device = "cuda" if torch.cuda.is_available() else "cpu"
                print(f"Using device: {device}")
                if device == "cuda":
                    print(f"GPU: {torch.cuda.get_device_name(0)}")
                    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024 / 1024 / 1024:.2f} GB")
                    print(f"CUDA Version: {torch.version.cuda}")
                
                # For CamemBERT models, we need special handling
                if 'camembert' in self.transformer_model.lower():
                    print(f"Initializing CamemBERT model: {self.transformer_model}")
                    from transformers import AutoModelForSequenceClassification, AutoTokenizer, logging
                    
                    # Set transformers logging level to get more information
                    logging.set_verbosity_info()
                    
                    print("Loading tokenizer with use_fast=False...")
                    # Use the slow tokenizer to avoid conversion issues
                    self.tokenizer = AutoTokenizer.from_pretrained(
                        self.transformer_model,
                        use_fast=False  # Use the slow tokenizer to avoid conversion issues
                    )
                    print("Tokenizer loaded successfully.")
                    
                    print(f"Loading model {self.transformer_model}...")
                    self.model = AutoModelForSequenceClassification.from_pretrained(
                        self.transformer_model
                    )
                    # Move model to GPU if available
                    if device == "cuda":
                        self.model = self.model.to("cuda")
                    print(f"Model loaded successfully to {device}.")
                    
                    # Create pipeline with pre-loaded components
                    print("Creating sentiment analysis pipeline...")
                    self.analyzer = pipeline(
                        "sentiment-analysis",
                        model=self.model,
                        tokenizer=self.tokenizer,
                        device=0 if device == "cuda" else -1,  # Use GPU if available
                        truncation=True
                    )
                    print("Pipeline created successfully.")
                else:
                    print(f"Initializing standard transformer model: {self.transformer_model}")
                    # For other models, use the standard pipeline approach
                    self.analyzer = pipeline(
                        "sentiment-analysis", 
                        model=self.transformer_model,
                        device=0 if device == "cuda" else -1,  # Use GPU if available
                        truncation=True
                    )
                    print("Standard pipeline created successfully.")
            except Exception as e:
                import traceback
                print(f"Error initializing transformer model: {e}")
                print(traceback.format_exc())
                raise
    
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
        Compatible with French models like CamemBERT fine-tuned for sentiment.
        
        Args:
            text: Input text
            
        Returns:
            Dictionary with sentiment scores
        """
        # Truncate text if it's too long (most transformer models have a limit)
        max_length = 512
        if len(text.split()) > max_length:
            text = ' '.join(text.split()[:max_length])
        
        # Log the first few analyses to debug label issues
        static_counter = getattr(self, '_debug_counter', 0)
        debug_mode = static_counter < 5  # Only log the first 5 analyses
        
        try:
            result = self.analyzer(text)[0]
            label = result['label']
            score = result['score']
            
            if debug_mode:
                print(f"DEBUG: Raw model output: {result}")
                print(f"DEBUG: Label={label}, Score={score}")
                self._debug_counter = static_counter + 1
            
            # cmarkea/distilcamembert-base-sentiment uses a 5-star rating system:
            # 1 star: très négatif, 2 stars: négatif, 3 stars: neutre, 4 stars: positif, 5 stars: très positif
            if label == "1 star":
                return {"negative": score, "neutral": 0.0, "positive": 0.0, "compound": -score}
            elif label == "2 stars":
                return {"negative": score * 0.7, "neutral": score * 0.3, "positive": 0.0, "compound": -score * 0.5}
            elif label == "3 stars":
                return {"negative": 0.0, "neutral": score, "positive": 0.0, "compound": 0.0}
            elif label == "4 stars":
                return {"negative": 0.0, "neutral": score * 0.3, "positive": score * 0.7, "compound": score * 0.5}
            elif label == "5 stars":
                return {"negative": 0.0, "neutral": 0.0, "positive": score, "compound": score}
            # Fallback for other label formats
            elif 'neg' in label.lower():
                return {"negative": score, "neutral": 1 - score, "positive": 0.0, "compound": -score}
            elif 'neu' in label.lower():
                return {"negative": 0.0, "neutral": score, "positive": 0.0, "compound": 0.0}
            elif 'pos' in label.lower():
                return {"negative": 0.0, "neutral": 1 - score, "positive": score, "compound": score}
            else:
                # If we get here, we have an unexpected label format
                if debug_mode:
                    print(f"WARNING: Unknown label format: {label}. Using default neutral sentiment.")
                return {"negative": 0.0, "neutral": 1.0, "positive": 0.0, "compound": 0.0}
        except Exception as e:
            print(f"Error in analyze_text_transformers: {e}")
            return {"negative": 0.0, "neutral": 1.0, "positive": 0.0, "compound": 0.0}
    
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
        import time
        start_time = time.time()
        total_docs = len(documents)
        print(f"Starting sentiment analysis on {total_docs} documents...")
        
        results = []
        for i, doc in enumerate(documents):
            if i % 10 == 0 or i == total_docs - 1:
                elapsed = time.time() - start_time
                docs_per_second = (i + 1) / elapsed if elapsed > 0 else 0
                remaining = (total_docs - i - 1) / docs_per_second if docs_per_second > 0 else 0
                print(f"Processing document {i+1}/{total_docs} ({(i+1)/total_docs*100:.1f}%) - "
                      f"Speed: {docs_per_second:.2f} docs/sec - "
                      f"Elapsed: {elapsed:.1f}s - "
                      f"Remaining: {remaining:.1f}s")
            
            # Get document ID or title for logging
            doc_id = doc.get('id', doc.get('title', f"doc_{i}"))
            doc_len = len(doc.get('cleaned_text', doc.get('text', doc.get('content', '')))) 
            
            try:
                start_doc = time.time()
                result = self.analyze_document(doc)
                doc_time = time.time() - start_doc
                
                # Log detailed info for every 50th document or if processing took unusually long
                if i % 50 == 0 or doc_time > 1.0:
                    sentiment = result.get('sentiment', {})
                    compound = sentiment.get('compound', 0)
                    print(f"  - Doc {doc_id[:30]}... ({doc_len} chars): "
                          f"sentiment={compound:.2f} ({doc_time:.3f}s)")
                
                results.append(result)
            except Exception as e:
                print(f"Error processing document {doc_id[:30]}...: {e}")
                # Add the document without sentiment to avoid data loss
                doc['sentiment'] = {
                    'negative': 0.0,
                    'neutral': 1.0,
                    'positive': 0.0,
                    'compound': 0.0,
                    'error': str(e)
                }
                results.append(doc)
        
        total_time = time.time() - start_time
        print(f"Completed sentiment analysis on {total_docs} documents in {total_time:.2f}s ")
        print(f"Average processing time: {total_time/total_docs:.4f}s per document")
        
        return results
    
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
