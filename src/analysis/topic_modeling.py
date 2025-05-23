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

    def fit_transform(self, documents: List[Dict[str, Any]], preprocessed_data: Optional[Dict] = None) -> Dict[str, Any]:
        self.doc_ids = [doc.get('id', f"doc_{i}") for i, doc in enumerate(documents)]
        logger.info(f"Starting topic modeling for {len(documents)} documents using '{self.algorithm}'.")
        raw_texts = [doc.get('content', doc.get('text', doc.get('cleaned_text', ''))) for doc in documents]

        if self.algorithm in ['lda', 'nmf']:
            doc_topic_matrix = self._fit_transform_sklearn(raw_texts)
            top_terms = self._get_top_terms_sklearn()
        elif self.algorithm in ['gensim_lda', 'hdp']:
            if not preprocessed_data or 'tokenized_texts' not in preprocessed_data:
                raise ValueError("Gensim models require preprocessed and tokenized text.")
            tokenized_texts = preprocessed_data['tokenized_texts']
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
        if self.algorithm == 'hdp':
            self.model = HdpModel(self.gensim_corpus, id2word=self.gensim_dictionary)
            self.num_topics = len(self.model.show_topics(formatted=False))
        else:
            self.model = LdaMulticore(self.gensim_corpus, num_topics=self.num_topics, id2word=self.gensim_dictionary, passes=10, workers=self.workers, random_state=42)
        return gensim.matutils.corpus2dense(self.model[self.gensim_corpus], num_terms=self.num_topics).T

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