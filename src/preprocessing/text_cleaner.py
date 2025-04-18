"""
Text cleaning and preprocessing utilities for OCR-processed newspaper articles.
"""

import re
import string
import unicodedata
from typing import List, Dict, Any, Optional

import nltk
import spacy
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

# Download necessary NLTK resources
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('punkt')
    nltk.download('stopwords')
    nltk.download('wordnet')


class TextCleaner:
    """Class for cleaning and preprocessing OCR text data."""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the TextCleaner with configuration settings.
        
        Args:
            config: Dictionary containing preprocessing configuration
        """
        self.config = config
        self.min_word_length = config.get('min_word_length', 3)
        self.remove_stopwords = config.get('remove_stopwords', True)
        self.lemmatize = config.get('lemmatize', True)
        self.language = config.get('language', 'english')
        
        # Initialize NLP tools
        if self.remove_stopwords:
            self.stop_words = set(stopwords.words(self.language))
        
        if self.lemmatize:
            self.lemmatizer = WordNetLemmatizer()
            
        # Load spaCy model if needed
        self.nlp = None
    
    def load_spacy_model(self, model_name: str = 'en_core_web_sm'):
        """
        Load a spaCy language model.
        
        Args:
            model_name: Name of the spaCy model to load
        """
        try:
            self.nlp = spacy.load(model_name)
        except OSError:
            print(f"SpaCy model {model_name} not found. Please install it with:")
            print(f"python -m spacy download {model_name}")
            raise
    
    def normalize_unicode(self, text: str) -> str:
        """
        Normalize Unicode characters in text.
        
        Args:
            text: Input text
            
        Returns:
            Normalized text
        """
        return unicodedata.normalize('NFKD', text)
    
    def remove_special_chars(self, text: str) -> str:
        """
        Remove special characters and replace with space.
        
        Args:
            text: Input text
            
        Returns:
            Text with special characters removed
        """
        # Keep alphanumeric, spaces, and basic punctuation
        pattern = r'[^a-zA-Z0-9\s.,!?;:\-\'"]'
        return re.sub(pattern, ' ', text)
    
    def fix_common_ocr_errors(self, text: str) -> str:
        """
        Fix common OCR errors.
        
        Args:
            text: Input text
            
        Returns:
            Text with common OCR errors fixed
        """
        # Common OCR error patterns
        ocr_fixes = {
            r'\bI([^a-zA-Z])': r'I\1',  # Fix standalone I
            r'\b0\b': 'O',              # 0 -> O
            r'\b1\b': 'I',              # 1 -> I
            r'vv': 'w',                 # vv -> w
            r'rn': 'm',                 # rn -> m
            # Add more patterns as needed
        }
        
        for pattern, replacement in ocr_fixes.items():
            text = re.sub(pattern, replacement, text)
        
        return text
    
    def tokenize(self, text: str) -> List[str]:
        """
        Tokenize text into words.
        
        Args:
            text: Input text
            
        Returns:
            List of tokens
        """
        return word_tokenize(text, language=self.language)
    
    def filter_tokens(self, tokens: List[str]) -> List[str]:
        """
        Filter tokens based on configuration.
        
        Args:
            tokens: List of tokens
            
        Returns:
            Filtered list of tokens
        """
        # Remove punctuation and convert to lowercase
        tokens = [token.lower() for token in tokens 
                 if token not in string.punctuation]
        
        # Filter by length
        tokens = [token for token in tokens 
                 if len(token) >= self.min_word_length]
        
        # Remove stopwords if configured
        if self.remove_stopwords:
            tokens = [token for token in tokens 
                     if token not in self.stop_words]
        
        # Lemmatize if configured
        if self.lemmatize:
            tokens = [self.lemmatizer.lemmatize(token) for token in tokens]
        
        return tokens
    
    def clean_text(self, text: str) -> str:
        """
        Clean and preprocess text.
        
        Args:
            text: Input text
            
        Returns:
            Cleaned and preprocessed text
        """
        # Normalize Unicode
        text = self.normalize_unicode(text)
        
        # Fix common OCR errors
        text = self.fix_common_ocr_errors(text)
        
        # Remove special characters
        text = self.remove_special_chars(text)
        
        # Tokenize
        tokens = self.tokenize(text)
        
        # Filter tokens
        filtered_tokens = self.filter_tokens(tokens)
        
        # Join tokens back into text
        cleaned_text = ' '.join(filtered_tokens)
        
        return cleaned_text
    
    def process_document(self, document: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process a document dictionary.
        
        Args:
            document: Dictionary containing document data with 'text' key
            
        Returns:
            Document with cleaned text
        """
        if 'text' not in document:
            raise KeyError("Document must contain a 'text' key")
        
        document['cleaned_text'] = self.clean_text(document['text'])
        return document
