"""
SpaCy-based text preprocessing for topic modeling.

This module provides functions for preprocessing text using SpaCy,
particularly optimized for topic modeling techniques like LDA, NMF, and BERTopic.
"""

import spacy
from typing import List, Set, Dict, Any, Optional
import logging
import time
from datetime import datetime
import os
import sys
from tqdm import tqdm

# Initialiser le logger avec notre utilitaire centralisé
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
    start_time = time.time()
    text_sample = text[:100] + "..." if len(text) > 100 else text
    text_length = len(text)
    
    logger.info(f"SpaCy preprocessing started for text of length {text_length} chars. Sample: {text_sample}")
    
    if nlp is None:
        logger.info("No SpaCy model provided, loading default model")
        nlp = load_spacy_model()
    
    # Mesurer le temps de parsing
    parse_start = time.time()
    doc = nlp(text.lower())
    parse_time = time.time() - parse_start
    logger.info(f"SpaCy parsing completed in {parse_time:.2f} seconds for {len(doc)} tokens")
    
    # Compter les tokens par catégorie POS avant filtrage
    pos_counts = {}
    for token in doc:
        pos = token.pos_
        pos_counts[pos] = pos_counts.get(pos, 0) + 1
    
    logger.info(f"POS distribution before filtering: {pos_counts}")
    
    # Filtrer les tokens
    filter_start = time.time()
    tokens = []
    filtered_counts = {"non_alpha": 0, "stopword": 0, "wrong_pos": 0, "too_short": 0}
    
    for token in doc:
        if not token.is_alpha:
            filtered_counts["non_alpha"] += 1
        elif token.lemma_ in nlp.Defaults.stop_words:
            filtered_counts["stopword"] += 1
        elif token.pos_ not in allowed_pos:
            filtered_counts["wrong_pos"] += 1
        elif len(token.lemma_) < min_token_length:
            filtered_counts["too_short"] += 1
        else:
            tokens.append(token.lemma_)
    
    filter_time = time.time() - filter_start
    total_time = time.time() - start_time
    
    logger.info(f"Token filtering completed in {filter_time:.2f} seconds")
    logger.info(f"Filtering stats: {filtered_counts}")
    logger.info(f"SpaCy preprocessing finished in {total_time:.2f} seconds. {len(tokens)} tokens extracted from {len(doc)} original tokens")
    logger.info(f"Sample tokens: {tokens[:20]}")
    
    return tokens


def preprocess_documents(documents: List[Dict[str, Any]], 
                          text_key: str = "cleaned_text",
                          output_key: str = "tokens",
                          nlp: Optional[spacy.language.Language] = None,
                          allowed_pos: Set[str] = {"NOUN", "PROPN", "ADJ"},
                          min_token_length: int = 3,
                          show_progress: bool = True) -> List[Dict[str, Any]]:
    """
    Prétraite une liste de documents avec SpaCy pour du topic modeling.
    
    Args:
        documents: liste de dictionnaires représentant les documents
        text_key: clé contenant le texte à prétraiter
        output_key: clé où stocker les tokens résultants
        nlp: modèle SpaCy préchargé (si None, charge fr_core_news_md)
        allowed_pos: les classes grammaticales à conserver
        min_token_length: longueur minimale des tokens à conserver
        show_progress: afficher une barre de progression
        
    Returns:
        Liste de documents avec les tokens ajoutés
    """
    start_time = time.time()
    logger.info(f"Starting batch preprocessing of {len(documents)} documents with SpaCy")
    logger.info(f"Parameters: text_key='{text_key}', output_key='{output_key}', allowed_pos={allowed_pos}, min_token_length={min_token_length}")
    
    if nlp is None:
        logger.info("No SpaCy model provided, loading default model")
        nlp = load_spacy_model()
    
    processed_docs = []
    missing_key_count = 0
    empty_text_count = 0
    token_stats = {"min": float('inf'), "max": 0, "total": 0}
    
    # Créer une barre de progression
    progress_bar = None
    if show_progress:
        progress_bar = tqdm(total=len(documents), desc="SpaCy preprocessing", 
                           unit="doc", ncols=100, file=sys.stdout)
        # Ajouter une représentation de la barre dans les logs
        logger.info("Progress: [" + "-" * 50 + "] 0%")
    
    # Fonction pour mettre à jour la barre de progression dans les logs
    def update_log_progress(current, total):
        if not show_progress:
            return
        progress = int(50 * current / total)
        percent = int(100 * current / total)
        progress_bar = "[" + "#" * progress + "-" * (50 - progress) + f"] {percent}%"
        logger.info(f"Progress: {progress_bar} - {current}/{total} documents")
    
    for i, doc in enumerate(documents):
        doc_id = doc.get('_id', doc.get('id', f"doc_{i}"))
        
        # Mettre à jour la barre de progression
        if progress_bar:
            progress_bar.update(1)
        
        # Log progress every 100 documents or at specific percentages
        if i % max(1, len(documents) // 20) == 0:  # Log at 5% intervals
            progress = (i / len(documents)) * 100
            elapsed = time.time() - start_time
            docs_per_sec = i / max(elapsed, 0.001)
            estimated_remaining = (len(documents) - i) / max(docs_per_sec, 0.001)
            update_log_progress(i, len(documents))
            logger.info(f"Stats: {docs_per_sec:.2f} docs/sec - Est. remaining: {estimated_remaining:.1f} sec")
        
        # Check if the document has the required key
        if text_key not in doc:
            missing_key_count += 1
            logger.warning(f"Document {doc_id} missing key '{text_key}' for SpaCy preprocessing")
            raise KeyError(f"Document must contain a '{text_key}' key")
        
        # Check if the text is empty
        if not doc[text_key]:
            empty_text_count += 1
            logger.warning(f"Document {doc_id} has empty text for key '{text_key}'")
            doc[output_key] = []
            processed_docs.append(doc)
            continue
        
        # Process the document
        logger.debug(f"Processing document with key '{text_key}' using SpaCy.")
        doc[output_key] = preprocess_with_spacy(
            doc[text_key], 
            nlp=nlp,
            allowed_pos=allowed_pos,
            min_token_length=min_token_length
        )
        
        # Update token statistics
        num_tokens = len(doc[output_key])
        token_stats["min"] = min(token_stats["min"], num_tokens)
        token_stats["max"] = max(token_stats["max"], num_tokens)
        token_stats["total"] += num_tokens
        
        processed_docs.append(doc)
    
    # Fermer la barre de progression
    if progress_bar:
        progress_bar.close()
        # Afficher une barre de progression complète dans les logs
        logger.info("Progress: [" + "#" * 50 + "] 100%")
    
    # Calculate average tokens per document
    if processed_docs:
        token_stats["avg"] = token_stats["total"] / len(processed_docs)
        if token_stats["min"] == float('inf'):
            token_stats["min"] = 0
    else:
        token_stats["avg"] = 0
        token_stats["min"] = 0
    
    total_time = time.time() - start_time
    docs_per_sec = len(documents) / max(total_time, 0.001)
    
    logger.info(f"SpaCy batch preprocessing completed in {total_time:.2f} seconds ({docs_per_sec:.2f} docs/sec)")
    logger.info(f"Documents with missing key: {missing_key_count}, Documents with empty text: {empty_text_count}")
    logger.info(f"Token statistics: min={token_stats['min']}, max={token_stats['max']}, avg={token_stats['avg']:.2f}, total={token_stats['total']}")
    
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
                           output_key: str = "tokens",
                           alternative_keys: List[str] = None,
                           show_progress: bool = True) -> List[Dict[str, Any]]:
        """Process a list of documents with SpaCy.
        
        Args:
            documents: List of document dictionaries
            text_key: Key in documents containing text to process
            output_key: Key to store processed tokens in documents
            alternative_keys: List of alternative keys to try if text_key is not found
            show_progress: Whether to show a progress bar
            
        Returns:
            List of processed documents with tokens added
        """
        if alternative_keys is None:
            alternative_keys = ["content", "text", "original_content"]
        
        # Initialiser les compteurs pour le résumé final
        total_docs = len(documents)
        processed_success = 0
        processed_with_alt_key = 0
        processed_empty = 0
        error_docs = 0
        error_details = {}
        
        logger.info(f"========== DÉBUT DU PRÉTRAITEMENT SPACY ==========")
        logger.info(f"Traitement de {total_docs} documents avec SpaCy (modèle: {self.model_name})")
        logger.info(f"Paramètres: text_key='{text_key}', output_key='{output_key}', POS tags={self.allowed_pos}, min_token_length={self.min_token_length}")
        
        # Vérifier la disponibilité des clés dans les documents
        key_availability = {key: 0 for key in [text_key] + alternative_keys}
        for doc in documents:
            for key in key_availability.keys():
                if key in doc:
                    key_availability[key] += 1
        
        logger.info(f"Disponibilité des clés dans les documents: {key_availability}")
        
        # Initialiser la barre de progression
        start_time = time.time()
        progress_bar = None
        if show_progress:
            progress_bar = tqdm(total=total_docs, desc=f"SpaCy ({self.model_name})",
                                unit="doc", ncols=100, file=sys.stdout)
        
        # Fonction pour mettre à jour la barre de progression dans les logs (moins fréquemment)
        def update_log_progress(current, total):
            if not show_progress:
                return
            progress = int(50 * current / total)
            percent = int(100 * current / total)
            progress_bar = "[" + "#" * progress + "-" * (50 - progress) + f"] {percent}%"
            logger.info(f"Progression: {progress_bar} - {current}/{total} documents")
        
        # Traiter les documents
        processed_docs = []
        for i, doc in enumerate(documents):
            doc_id = doc.get('_id', doc.get('id', f"doc_{i}"))
            
            # Mettre à jour la barre de progression
            if progress_bar:
                progress_bar.update(1)
            
            # Log progress at specific intervals (10% of total) pour réduire le défilement
            if i % max(1, total_docs // 10) == 0:
                elapsed = time.time() - start_time
                docs_per_sec = i / max(elapsed, 0.001)
                estimated_remaining = (total_docs - i) / max(docs_per_sec, 0.001)
                update_log_progress(i, total_docs)
                # Convertir le temps restant en format plus lisible
                remaining_min = int(estimated_remaining // 60)
                remaining_sec = int(estimated_remaining % 60)
                logger.info(f"Vitesse: {docs_per_sec:.2f} docs/sec - Temps restant estimé: {remaining_min}m {remaining_sec}s")
            
            # Essayer de traiter le document
            try:
                processed_doc = self.process_document(doc, text_key, output_key)
                processed_docs.append(processed_doc)
                processed_success += 1
            except KeyError as e:
                # Ne pas logger chaque erreur pour éviter le défilement
                error_type = str(e)
                error_details[error_type] = error_details.get(error_type, 0) + 1
                
                # Essayer avec des clés alternatives
                alt_key_success = False
                for alt_key in alternative_keys:
                    if alt_key in doc and doc[alt_key]:
                        try:
                            processed_doc = self.process_document(doc, alt_key, output_key)
                            processed_docs.append(processed_doc)
                            processed_with_alt_key += 1
                            alt_key_success = True
                            break
                        except Exception as alt_e:
                            error_type = f"Alt key '{alt_key}': {str(alt_e)}"
                            error_details[error_type] = error_details.get(error_type, 0) + 1
                
                # Si aucune clé alternative ne fonctionne, ajouter le document sans tokens
                if not alt_key_success:
                    doc[output_key] = []
                    processed_docs.append(doc)
                    processed_empty += 1
            except Exception as e:
                # Capturer toutes les autres exceptions et ajouter quand même le document
                error_type = f"Unexpected: {str(e)}"
                error_details[error_type] = error_details.get(error_type, 0) + 1
                doc[output_key] = []
                processed_docs.append(doc)
                error_docs += 1
        
        # Fermer la barre de progression
        if progress_bar:
            progress_bar.close()
            # Afficher une barre de progression complète dans les logs
            logger.info("Progression: [" + "#" * 50 + "] 100% - Terminé")
        
        # Calculer les statistiques sur les tokens
        token_stats = {"min": float('inf'), "max": 0, "total": 0, "empty": 0}
        for doc in processed_docs:
            num_tokens = len(doc[output_key])
            token_stats["min"] = min(token_stats["min"], num_tokens) if num_tokens > 0 else token_stats["min"]
            token_stats["max"] = max(token_stats["max"], num_tokens)
            token_stats["total"] += num_tokens
            if num_tokens == 0:
                token_stats["empty"] += 1
        
        # Calculer la moyenne des tokens par document
        non_empty_docs = len(processed_docs) - token_stats["empty"]
        token_stats["avg"] = token_stats["total"] / max(non_empty_docs, 1)
        if token_stats["min"] == float('inf'):
            token_stats["min"] = 0
        
        total_time = time.time() - start_time
        docs_per_sec = len(processed_docs) / max(total_time, 0.001)
        
        # Afficher un résumé complet à la fin
        logger.info(f"\n========== RÉSUMÉ DU PRÉTRAITEMENT SPACY ==========")
        logger.info(f"Temps total: {total_time:.2f} secondes ({docs_per_sec:.2f} docs/sec)")
        logger.info(f"Documents traités: {len(processed_docs)}/{total_docs} ({len(processed_docs)/total_docs*100:.1f}%)")
        logger.info(f"  - Succès avec clé principale: {processed_success} ({processed_success/total_docs*100:.1f}%)")
        logger.info(f"  - Succès avec clé alternative: {processed_with_alt_key} ({processed_with_alt_key/total_docs*100:.1f}%)")
        logger.info(f"  - Documents sans tokens: {processed_empty} ({processed_empty/total_docs*100:.1f}%)")
        logger.info(f"  - Erreurs non récupérables: {error_docs} ({error_docs/total_docs*100:.1f}%)")
        
        if error_details:
            logger.info(f"\nDétail des erreurs rencontrées:")
            for error_type, count in sorted(error_details.items(), key=lambda x: x[1], reverse=True):
                logger.info(f"  - {error_type}: {count} occurrences")
        
        logger.info(f"\nStatistiques des tokens:")
        logger.info(f"  - Minimum par document: {token_stats['min']}")
        logger.info(f"  - Maximum par document: {token_stats['max']}")
        logger.info(f"  - Moyenne par document: {token_stats['avg']:.2f}")
        logger.info(f"  - Total de tokens: {token_stats['total']}")
        logger.info(f"  - Documents sans tokens: {token_stats['empty']} ({token_stats['empty']/len(processed_docs)*100:.1f}%)")
        
        if processed_docs and processed_docs[0].get(output_key):
            logger.info(f"\nExemple de tokens (premier document): {processed_docs[0][output_key][:10]}...")
        
        logger.info(f"========== FIN DU PRÉTRAITEMENT SPACY ==========\n")
        
        return processed_docs
