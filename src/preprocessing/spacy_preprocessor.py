"""
SpaCy-based text preprocessing for topic modeling.

This module provides functions for preprocessing text using SpaCy,
particularly optimized for topic modeling techniques like LDA, NMF, and BERTopic.
"""

import spacy
from typing import List, Set, Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)

def load_spacy_model(model_name: str = "fr_core_news_md") -> spacy.language.Language:
    """
    Load and return a SpaCy language model.
    
    Args:
        model_name: Name of the SpaCy model to load (default: fr_core_news_md)
        
    Returns:
        Loaded SpaCy language model
        
    Raises:
        OSError: If the model is not installed
    """
    try:
        nlp = spacy.load(model_name)
        logger.info(f"Loaded SpaCy model: {model_name}")
        return nlp
    except OSError:
        logger.error(f"SpaCy model {model_name} not found. Please install it.")
        raise


def preprocess_with_spacy(text: str, 
                          nlp: Optional[spacy.language.Language] = None,
                          allowed_pos: Set[str] = {"NOUN", "PROPN", "ADJ"},
                          min_token_length: int = 3) -> List[str]:
    """
    Nettoie et tokenize un texte pour du topic modeling (LDA, NMF, BERTopic).
    
    Args:
        text: le texte original
        nlp: modèle SpaCy préchargé (si None, charge fr_core_news_md)
        allowed_pos: les classes grammaticales qu'on garde (NOUN, PROPN, ADJ, VERB...)
        min_token_length: longueur minimale des tokens à conserver
        
    Returns:
        Liste de tokens lemmatisés, nettoyés
    """
    if nlp is None:
        nlp = load_spacy_model()
    logger.debug(f"SpaCy preprocessing started for text: {text[:50]}...")
    doc = nlp(text.lower())
    tokens = [
        token.lemma_
        for token in doc
        if token.is_alpha
        and token.lemma_ not in nlp.Defaults.stop_words
        and token.pos_ in allowed_pos
        and len(token.lemma_) >= min_token_length
    ]
    logger.debug(f"SpaCy preprocessing finished. {len(tokens)} tokens extracted. Sample: {tokens[:10]}")
    return tokens


def preprocess_documents(documents: List[Dict[str, Any]], 
                         text_key: str = "cleaned_text",
                         output_key: str = "tokens",
                         nlp: Optional[spacy.language.Language] = None,
                         allowed_pos: Set[str] = {"NOUN", "PROPN", "ADJ"},
                         min_token_length: int = 3) -> List[Dict[str, Any]]:
    """
    Prétraite une liste de documents avec SpaCy pour du topic modeling.
    
    Args:
        documents: liste de dictionnaires représentant les documents
        text_key: clé contenant le texte à prétraiter
        output_key: clé où stocker les tokens résultants
        nlp: modèle SpaCy préchargé (si None, charge fr_core_news_md)
        allowed_pos: les classes grammaticales à conserver
        min_token_length: longueur minimale des tokens à conserver
        
    Returns:
        Liste de documents avec les tokens ajoutés
    """
    if nlp is None:
        nlp = load_spacy_model()
    
    processed_docs = []
    for doc in documents:
        if text_key not in doc:
            raise KeyError(f"Document must contain a '{text_key}' key")
        
        doc[output_key] = preprocess_with_spacy(
            doc[text_key], 
            nlp=nlp,
            allowed_pos=allowed_pos,
            min_token_length=min_token_length
        )
        processed_docs.append(doc)
    
    return processed_docs


class SpacyPreprocessor:
    """Class for SpaCy-based preprocessing of text data for topic modeling."""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the SpacyPreprocessor with configuration settings.
        
        Args:
            config: Dictionary containing preprocessing configuration
        """
        self.config = config
        self.model_name = config.get('spacy_model', 'fr_core_news_md')
        self.allowed_pos = set(config.get('allowed_pos', ["NOUN", "PROPN", "ADJ"]))
        self.min_token_length = config.get('min_token_length', 3)
        
        # Load SpaCy model
        self.nlp = load_spacy_model(self.model_name)
        logger.info(f"SpacyPreprocessor initialized with model={self.model_name}, allowed_pos={self.allowed_pos}, min_token_length={self.min_token_length}")
    
    def preprocess_text(self, text: str) -> List[str]:
        logger.debug(f"Preprocessing single document with SpaCy. Text sample: {text[:50]}...")
        tokens = preprocess_with_spacy(
            text, 
            nlp=self.nlp,
            allowed_pos=self.allowed_pos,
            min_token_length=self.min_token_length
        )
        logger.debug(f"Tokens for single document: {tokens[:10]}")
        return tokens
    
    def process_document(self, document: Dict[str, Any], 
                         text_key: str = "cleaned_text",
                         output_key: str = "tokens") -> Dict[str, Any]:
        if text_key not in document:
            logger.warning(f"Document missing key '{text_key}' for SpaCy preprocessing.")
            raise KeyError(f"Document must contain a '{text_key}' key")
        logger.debug(f"Processing document with key '{text_key}' using SpaCy.")
        document[output_key] = self.preprocess_text(document[text_key])
        return document
    
    def process_documents(self, documents: List[Dict[str, Any]],
                          text_key: str = "cleaned_text",
                          output_key: str = "tokens") -> List[Dict[str, Any]]:
        logger.info(f"SpaCy batch preprocessing started for {len(documents)} documents (text_key='{text_key}').")
        processed_docs = [self.process_document(doc, text_key, output_key) for doc in documents]
        logger.info(f"SpaCy batch preprocessing finished. Sample tokens from first doc: {processed_docs[0][output_key][:10] if processed_docs else []}")
        return processed_docs
