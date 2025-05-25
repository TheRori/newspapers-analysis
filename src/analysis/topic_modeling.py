"""
Topic modeling for newspaper articles.
"""

import os
import sys
import time
import json
import pickle
import logging
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
from collections import Counter
from tqdm import tqdm
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation, NMF
import gensim
from gensim.corpora import Dictionary
from gensim.models import LdaModel, HdpModel, CoherenceModel, LdaMulticore
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
from src.preprocessing import SpacyPreprocessor
from .utils import get_stopwords

# Configure logger
logger = logging.getLogger(__name__)

def print_important(message, symbol='='):
    """Displays a prominent message in the terminal."""
    border = symbol * 80
    tqdm.write(f"\n{border}\n{message}\n{border}\n")

class TopicModeler:
    """Class for topic modeling on newspaper articles."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.algorithm = config.get('algorithm', 'lda')
        self.num_topics = config.get('num_topics', 10)
        self.workers = config.get('workers', -1)
        self.model = None
        self.vectorizer = None
        self.feature_names = None
        self.doc_ids = []
        preproc_config = config.get('preprocessing', {})
        self.spacy_preprocessor = SpacyPreprocessor(preproc_config)
        logger.info("SpacyPreprocessor initialized. It will be used as needed by the selected algorithm.")

    def fit_transform(self, documents, preprocessed_data: Optional[Dict] = None) -> Dict[str, Any]:
        # Handle case where documents is a list of tokens (for Gensim) or a list of dictionaries (for other algorithms)
        if documents and isinstance(documents[0], list):
            # For Gensim: documents is a list of token lists
            self.doc_ids = [f"doc_{i}" for i in range(len(documents))]
            logger.info(f"Starting topic modeling for {len(documents)} tokenized documents using '{self.algorithm}'.")
            raw_texts = None  # Not needed for pre-tokenized input
        else:
            # For other algorithms: documents is a list of dictionaries
            self.doc_ids = [doc.get('id', f"doc_{i}") for i, doc in enumerate(documents)]
            logger.info(f"Starting topic modeling for {len(documents)} documents using '{self.algorithm}'.")
            raw_texts = [doc.get('content', doc.get('text', doc.get('cleaned_text', ''))) for doc in documents]

        if self.algorithm in ['lda', 'nmf']:
            doc_topic_matrix = self._fit_transform_sklearn(raw_texts)
            top_terms = self._get_top_terms_sklearn()
        elif self.algorithm in ['gensim_lda', 'hdp']:
            # For Gensim, documents should already be tokenized lists
            if isinstance(documents[0], list):
                tokenized_texts = documents
            elif preprocessed_data and 'tokenized_texts' in preprocessed_data:
                tokenized_texts = preprocessed_data['tokenized_texts']
            else:
                raise ValueError("Gensim models require preprocessed and tokenized text.")
                
            self.gensim_dictionary = Dictionary(tokenized_texts)
            self.gensim_corpus = [self.gensim_dictionary.doc2bow(text) for text in tokenized_texts]
            doc_topic_matrix = self._fit_transform_gensim(tokenized_texts)
            top_terms = self._get_top_terms_gensim()
        elif self.algorithm == 'bertopic':
            doc_topic_matrix = self._fit_transform_bertopic(raw_texts)
            top_terms = self.get_top_terms_bertopic()
        else:
            raise ValueError(f"Unsupported algorithm: {self.algorithm}")

        doc_topics = {}
        dominant_topics = np.argmax(doc_topic_matrix, axis=1)
        
        for i, doc_id in enumerate(self.doc_ids):
            topic_index = self.bertopic_topics[i] if self.algorithm == 'bertopic' else int(dominant_topics[i])
            doc_topics[doc_id] = {
                'topic_distribution': doc_topic_matrix[i].tolist(),
                'dominant_topic': topic_index
            }
        
        return {'doc_topics': doc_topics, 'top_terms': top_terms, 'num_topics': self.num_topics, 'algorithm': self.algorithm}

    def _fit_transform_sklearn(self, texts: List[str]) -> np.ndarray:
        if self.algorithm == 'nmf':
            self.vectorizer = TfidfVectorizer(max_df=0.7, min_df=5)
        else:
            self.vectorizer = CountVectorizer(max_df=0.7, min_df=5)
        dtm = self.vectorizer.fit_transform(texts)
        self.feature_names = self.vectorizer.get_feature_names_out()
        if self.algorithm == 'nmf':
            self.model = NMF(n_components=self.num_topics, random_state=42)
        else:
            self.model = LatentDirichletAllocation(n_components=self.num_topics, random_state=42, n_jobs=self.workers)
        return self.model.fit_transform(dtm)

    def _fit_transform_gensim(self, tokenized_texts: List[List[str]]) -> np.ndarray:
        # Handle 'auto' value for num_topics
        if isinstance(self.num_topics, str) and self.num_topics.lower() == 'auto':
            logger.info("Auto topic detection requested for Gensim LDA. Searching for optimal number of topics...")
            # Search for optimal number of topics using coherence scores
            best_num_topics, best_coherence = self._find_optimal_num_topics(tokenized_texts)
            logger.info(f"Auto topic detection complete. Optimal number of topics: {best_num_topics} (coherence: {best_coherence:.4f})")
            actual_num_topics = best_num_topics
            # Update num_topics to the optimal value found
            self.num_topics = best_num_topics
        else:
            actual_num_topics = int(self.num_topics)
            
        if self.algorithm == 'hdp':
            self.model = HdpModel(self.gensim_corpus, id2word=self.gensim_dictionary)
            self.num_topics = len(self.model.show_topics(formatted=False))
        else:
            logger.info(f"Training LDA model with {actual_num_topics} topics")
            self.model = LdaMulticore(self.gensim_corpus, num_topics=actual_num_topics, id2word=self.gensim_dictionary, 
                                     passes=10, workers=self.workers, random_state=42)
                
        return gensim.matutils.corpus2dense(self.model[self.gensim_corpus], num_terms=self.num_topics).T
        
    def _find_optimal_num_topics(self, tokenized_texts: List[List[str]]) -> Tuple[int, float]:
        """Find the optimal number of topics by evaluating coherence scores."""
        coherence_scores = []
        
        # Get topic range parameters from config
        topic_min = self.config.get('topic_range_min', 2)
        topic_max = self.config.get('topic_range_max', 20)
        topic_step = self.config.get('topic_range_step', 2)
        
        # Create the range of topics to test
        topic_range = range(topic_min, topic_max + 1, topic_step)
        
        print_important(f"SEARCHING FOR OPTIMAL NUMBER OF TOPICS\nTesting {len(topic_range)} different values: {list(topic_range)}\nRange: min={topic_min}, max={topic_max}, step={topic_step}")
        
        for num_topics in topic_range:
            logger.info(f"Testing model with {num_topics} topics...")
            model = LdaModel(self.gensim_corpus, num_topics=num_topics, id2word=self.gensim_dictionary, passes=2, random_state=42)
            
            # Calculate coherence score
            coherence_model = CoherenceModel(model=model, texts=tokenized_texts, dictionary=self.gensim_dictionary, coherence='c_v')
            coherence_score = coherence_model.get_coherence()
            coherence_scores.append(coherence_score)
            
            logger.info(f"Topics: {num_topics}, Coherence: {coherence_score:.4f}")
        
        # Find the elbow point (where coherence score starts to decrease significantly)
        # First, find the maximum coherence score as a reference
        max_index = coherence_scores.index(max(coherence_scores))
        max_coherence = coherence_scores[max_index]
        max_num_topics = topic_range[max_index]
        
        # Then, find the elbow point - the point where adding more topics doesn't improve coherence much
        # We'll use a simple approach: find where the rate of improvement drops below a threshold
        elbow_index = 0
        elbow_found = False
        improvement_threshold = self.config.get('coherence_improvement_threshold', 0.05)  # 5% improvement threshold
        
        # Start by assuming the first point is the best
        best_index = 0
        best_coherence = coherence_scores[0]
        best_num_topics = topic_range[0]
        
        # Look for significant improvements in coherence
        for i in range(1, len(coherence_scores)):
            # Calculate improvement over previous point
            improvement = coherence_scores[i] - coherence_scores[i-1]
            improvement_percent = improvement / abs(coherence_scores[i-1]) if coherence_scores[i-1] != 0 else 0
            
            # If we see a significant improvement, update the best point
            if improvement_percent > improvement_threshold:
                best_index = i
                best_coherence = coherence_scores[i]
                best_num_topics = topic_range[i]
            # If improvement is minimal or negative, we may have found the elbow
            elif not elbow_found and i > 1 and i > best_index:
                elbow_index = i - 1  # The previous point was the elbow
                elbow_found = True
        
        # If we didn't find a clear elbow, use the point with maximum coherence
        if not elbow_found:
            logger.info("No clear elbow point found. Using the maximum coherence score.")
            best_index = max_index
            best_coherence = max_coherence
            best_num_topics = max_num_topics
        
        # Log all coherence scores for reference
        for i, (num_topics, score) in enumerate(zip(topic_range, coherence_scores)):
            status = ""
            if i == max_index:
                status = " (MAX)"
            if i == best_index:
                status = " (BEST)"
            if elbow_found and i == elbow_index:
                status += " (ELBOW)"
            logger.info(f"Topics: {num_topics}, Coherence: {score:.4f}{status}")
        
        selection_method = "elbow point" if elbow_found else "maximum coherence"
        print_important(f"OPTIMAL NUMBER OF TOPICS: {best_num_topics}\nCoherence score: {best_coherence:.4f}\nSelection method: {selection_method}")
        
        return best_num_topics, best_coherence

    def _fit_transform_bertopic(self, texts: List[str]) -> np.ndarray:
        bertopic_config = self.config.get('bertopic', {})
        embedding_model_name = bertopic_config.get('embedding_model', 'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')
        
        if getattr(self, 'using_cache', False):
            logger.warning("BERTopic: Cache detected, using default vectorizer WITHOUT SpaCy preprocessor.")
            vectorizer_model = CountVectorizer(stop_words=list(get_stopwords()), min_df=5, max_df=0.7)
        else:
            logger.info("BERTopic will use SpaCy-based vectorizer for topic term representation.")
            def spacy_preprocessor_func(text):
                truncated_text = text[:1_000_000]
                return " ".join(self.spacy_preprocessor.preprocess_text(truncated_text))
            vectorizer_model = CountVectorizer(preprocessor=spacy_preprocessor_func, stop_words=list(get_stopwords()), min_df=5, max_df=0.7)

        embedding_model = SentenceTransformer(embedding_model_name)
        self.model = BERTopic(language="french", embedding_model=embedding_model, vectorizer_model=vectorizer_model, calculate_probabilities=True, nr_topics=self.num_topics if self.num_topics != "auto" else "auto", verbose=True)
        
        print_important(f"STARTING BERTOPIC TRAINING\nDOCUMENTS: {len(texts)}")
        self.bertopic_topics, self.bertopic_probs = self.model.fit_transform(texts)
        self.num_topics = len(self.model.get_topic_info()) - 1
        print_important(f"BERTOPIC TRAINING COMPLETE\nTOPICS FOUND: {self.num_topics}")
        return self.bertopic_probs

    def _get_top_terms_sklearn(self, n_terms: int = 10) -> Dict[int, List[str]]:
        if self.model is None or self.feature_names is None: return {}
        top_terms = {}
        for topic_idx, topic in enumerate(self.model.components_):
            top_indices = topic.argsort()[:-n_terms - 1:-1]
            top_terms[topic_idx] = [self.feature_names[i] for i in top_indices]
        return top_terms

    def _get_top_terms_gensim(self, n_terms: int = 10) -> Dict[int, List[str]]:
        if self.model is None: return {}
        return {i: [word for word, _ in self.model.show_topic(i, n_terms)] for i in range(self.num_topics)}

    def get_top_terms_bertopic(self, n_terms: int = 10) -> Dict[int, List[str]]:
        if self.model is None: raise ValueError("BERTopic model has not been fit yet.")
        all_topics = self.model.get_topics()
        top_terms = {}
        for topic_id, topic_words_with_scores in all_topics.items():
            if topic_id == -1: continue
            words = [word for word, score in topic_words_with_scores[:n_terms]]
            top_terms[int(topic_id)] = words
        return top_terms

    def save_model(self, output_dir: str, prefix: str = 'topic_model') -> str:
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        model_path = os.path.join(output_dir, f"{prefix}_{self.algorithm}.pkl")
        if self.algorithm == 'bertopic':
            self.model.save(model_path)
        else:
            with open(model_path, 'wb') as f: pickle.dump(self.model, f)
        logger.info(f"Model saved to {model_path}")
        return model_path

    @classmethod
    def load_model(cls, model_path: str, config: Dict) -> 'TopicModeler':
        if not os.path.exists(model_path): raise FileNotFoundError(f"Model file not found at {model_path}")
        modeler = cls(config)
        algorithm = config.get('algorithm')
        if algorithm == 'bertopic':
            modeler.model = BERTopic.load(model_path)
        else:
            with open(model_path, 'rb') as f: modeler.model = pickle.load(f)
        logger.info(f"Model loaded successfully from {model_path}")
        return modeler

    # --- AJOUT DES MÉTHODES D'ANALYSE AVANCÉE POUR BERTOPIC ---

    def get_topic_coherence(self, texts: List[List[str]], coherence_type: str = "c_v") -> float:
        """Calculate the coherence score of the fitted model (works for any algorithm)."""
        if self.model is None: raise ValueError("Model has not been fit yet")
        
        if self.algorithm == 'bertopic':
            return self.get_bertopic_coherence(texts, coherence_type)
        elif self.algorithm in ['gensim_lda', 'hdp']:
            # For Gensim models
            if not hasattr(self, 'gensim_dictionary'):
                raise ValueError("Gensim dictionary not found. Model may not be properly initialized.")
                
            cm = CoherenceModel(model=self.model, texts=texts, dictionary=self.gensim_dictionary, coherence=coherence_type)
            return cm.get_coherence()
        else:
            # For sklearn models
            raise NotImplementedError(f"Coherence calculation not implemented for {self.algorithm}")
    
    def get_topic_distribution(self) -> List[float]:
        """Get the global distribution of topics in the corpus (works for any algorithm)."""
        if self.model is None: raise ValueError("Model has not been fit yet")
        
        if self.algorithm == 'bertopic':
            return self.get_bertopic_topic_distribution()
        elif self.algorithm in ['gensim_lda', 'hdp']:
            # For Gensim models
            if not hasattr(self, 'gensim_corpus') or not hasattr(self, 'model'):
                raise ValueError("Gensim corpus or model not found. Model may not be properly initialized.")
                
            # Get topic distribution across all documents
            topic_counts = [0] * self.num_topics
            for doc_bow in self.gensim_corpus:
                doc_topics = self.model.get_document_topics(doc_bow)
                for topic_id, prob in doc_topics:
                    if topic_id < self.num_topics:  # Ensure we don't go out of bounds
                        topic_counts[topic_id] += prob
            
            # Normalize to get distribution
            total = sum(topic_counts)
            if total > 0:
                return [count / total for count in topic_counts]
            return topic_counts
        else:
            # For sklearn models
            raise NotImplementedError(f"Topic distribution calculation not implemented for {self.algorithm}")
    
    def get_topic_word_weights(self, n_terms: int = 10) -> Dict[str, List[Tuple[str, float]]]:
        """Get the top weighted words for each topic (works for any algorithm)."""
        if self.model is None: raise ValueError("Model has not been fit yet")
        
        if self.algorithm == 'bertopic':
            return self.get_bertopic_word_weights(n_terms)
        elif self.algorithm in ['gensim_lda', 'hdp']:
            # For Gensim models
            if not hasattr(self, 'gensim_dictionary') or not hasattr(self, 'model'):
                raise ValueError("Gensim dictionary or model not found. Model may not be properly initialized.")
                
            weighted_words = {}
            for topic_id in range(self.num_topics):
                # Get topic terms with weights
                topic_terms = self.model.show_topic(topic_id, n_terms)
                weighted_words[str(topic_id)] = topic_terms
            
            return weighted_words
        else:
            # For sklearn models
            raise NotImplementedError(f"Word weights calculation not implemented for {self.algorithm}")
    
    def get_bertopic_coherence(self, texts: List[List[str]], coherence_type: str = "c_v") -> float:
        """Calculate the coherence score of the fitted BERTopic model."""
        if self.model is None: raise ValueError("Model has not been fit yet")
        
        topics = self.model.get_topics()
        # BERTopic's get_topics() returns a dict where keys are topic IDs
        # and values are lists of (word, score) tuples. We just need the words.
        # We also filter out the outlier topic (-1).
        topics_for_coherence = [[word for word, score in values] for key, values in topics.items() if key != -1]
        
        dictionary = Dictionary(texts)
        cm = CoherenceModel(topics=topics_for_coherence, texts=texts, dictionary=dictionary, coherence=coherence_type)
        return cm.get_coherence()

    def get_bertopic_topic_distribution(self) -> List[float]:
        """Get the global distribution of topics in the corpus."""
        if self.model is None or self.bertopic_topics is None: raise ValueError("Model has not been fit yet")
        
        topic_assignments = np.array(self.bertopic_topics)
        valid_topics = topic_assignments[topic_assignments != -1]
        
        unique, counts = np.unique(valid_topics, return_counts=True)
        distribution = np.zeros(self.num_topics)
        
        for topic_idx, count in zip(unique, counts):
            if 0 <= topic_idx < self.num_topics:
                distribution[topic_idx] = count
        
        if distribution.sum() > 0:
            distribution = distribution / distribution.sum()
        
        return distribution.tolist()

    def get_bertopic_word_weights(self, n_terms: int = 10) -> Dict[str, List[Tuple[str, float]]]:
        """Get the top weighted words for each topic from BERTopic model."""
        if self.model is None: raise ValueError("Model has not been fit yet")
        
        all_topics = self.model.get_topics()
        weighted_words = {}
        for topic_id, topic_words_with_scores in all_topics.items():
            if topic_id == -1: continue
            weighted_words[str(topic_id)] = topic_words_with_scores[:n_terms]
            
        return weighted_words
    
    def get_bertopic_representative_docs(self) -> Dict[str, List[int]]:
        """Get the most representative documents for each topic."""
        if self.model is None: raise ValueError("Model has not been fit yet")
        
        # This returns a dict of {topic_id: [list_of_doc_ids]}
        return self.model.get_representative_docs()