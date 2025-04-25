"""
Topic modeling for newspaper articles.
"""

import os
import pickle
from typing import List, Dict, Any, Tuple, Optional
import logging
import re
import requests
import json

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.feature_extraction import text as sklearn_text
from sklearn.decomposition import LatentDirichletAllocation, NMF
import gensim
from gensim.corpora import Dictionary
from gensim.models import LdaModel, HdpModel, CoherenceModel
from bertopic import BERTopic
from bertopic.vectorizers import ClassTfidfTransformer
from .utils import get_stopwords
from .llm_utils import LLMClient
from sentence_transformers import SentenceTransformer
from src.preprocessing import preprocess_with_spacy, SpacyPreprocessor

class TopicModeler:
    """Class for topic modeling on newspaper articles."""
    
    logger = logging.getLogger(__name__)
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the TopicModeler with configuration settings.
        
        Args:
            config: Dictionary containing topic modeling configuration
        """
        self.config = config
        self.algorithm = config.get('algorithm', 'lda')
        self.num_topics = config.get('num_topics', 10)
        self.max_df = config.get('max_df', 0.7)
        self.min_df = config.get('min_df', 5)
        
        # Initialize model and vectorizer
        self.model = None
        self.vectorizer = None
        self.feature_names = None
        self.gensim_dictionary = None
        self.gensim_corpus = None
        
        # BERTopic specific attributes
        self.bertopic_topics = None
        self.bertopic_probs = None

        # Initialize SpaCy preprocessor if needed
        self.spacy_preprocessor = None
        if config.get('use_spacy_preprocessing', True):
            preproc_config = config.get('preprocessing', {})
            self.spacy_preprocessor = SpacyPreprocessor(preproc_config)
        
    def preprocess_for_sklearn(self, documents: List[Dict[str, Any]]) -> Tuple[List[str], List[str]]:
        """
        Preprocess documents for sklearn-based topic modeling.
        
        Args:
            documents: List of document dictionaries
            
        Returns:
            Tuple of (document_ids, document_texts)
        """
        doc_ids = [doc.get('doc_id', doc.get('id', f"doc_{i}")) for i, doc in enumerate(documents)]
        
        # Use cleaned_text, text, or content if available
        texts = []
        for doc in documents:
            if 'cleaned_text' in doc:
                texts.append(doc['cleaned_text'])
            elif 'text' in doc:
                texts.append(doc['text'])
            elif 'content' in doc:
                texts.append(doc['content'])
            else:
                raise KeyError("Documents must contain 'cleaned_text', 'text', or 'content' key")
        
        return doc_ids, texts
    
    def preprocess_for_gensim(self, documents: List[Dict[str, Any]]) -> Tuple[List[str], List[List[str]]]:
        """
        Preprocess documents for gensim-based topic modeling.
        
        Args:
            documents: List of document dictionaries
            
        Returns:
            Tuple of (document_ids, tokenized_texts)
        """
        doc_ids = [doc.get('doc_id', doc.get('id', f"doc_{i}")) for i, doc in enumerate(documents)]
        
        # Get tokenized texts - prioritize tokens field which should be populated by SpaCy preprocessor
        tokenized_texts = []
        for doc in documents:
            if 'tokens' in doc:
                tokenized_texts.append(doc['tokens'])
            elif 'cleaned_text' in doc:
                # If we have a SpaCy preprocessor, use it to tokenize the text
                if self.spacy_preprocessor:
                    tokenized_texts.append(
                        self.spacy_preprocessor.preprocess_text(doc['cleaned_text'])
                    )
                else:
                    # Fallback to simple splitting (not recommended)
                    tokenized_texts.append(doc['cleaned_text'].split())
            elif 'text' in doc:
                # If we have a SpaCy preprocessor, use it to tokenize the text
                if self.spacy_preprocessor:
                    tokenized_texts.append(
                        self.spacy_preprocessor.preprocess_text(doc['text'])
                    )
                else:
                    # Fallback to simple splitting (not recommended)
                    tokenized_texts.append(doc['text'].split())
            elif 'content' in doc:
                # If we have a SpaCy preprocessor, use it to tokenize the text
                if self.spacy_preprocessor:
                    tokenized_texts.append(
                        self.spacy_preprocessor.preprocess_text(doc['content'])
                    )
                else:
                    # Fallback to simple splitting (not recommended)
                    tokenized_texts.append(doc['content'].split())
            else:
                raise KeyError("Documents must contain 'tokens', 'cleaned_text', 'text', or 'content' key")
        
        return doc_ids, tokenized_texts
    
    def fit_transform_sklearn(self, texts: List[str]) -> np.ndarray:
        """
        Fit and transform texts using sklearn vectorizer and topic model.
        
        Args:
            texts: List of document texts
            
        Returns:
            Document-topic matrix
        """
        # Create vectorizer without stopwords (handled by SpaCy preprocessing)
        if self.algorithm == 'nmf':
            self.vectorizer = TfidfVectorizer(
                max_df=self.max_df,
                min_df=self.min_df
            )
        else:  # lda
            self.vectorizer = CountVectorizer(
                max_df=self.max_df,
                min_df=self.min_df
            )
        
        # Create document-term matrix
        dtm = self.vectorizer.fit_transform(texts)
        
        # Store feature names
        self.feature_names = self.vectorizer.get_feature_names_out()
        
        # Create and fit topic model
        if self.algorithm == 'nmf':
            self.model = NMF(n_components=self.num_topics, random_state=42)
        else:  # lda
            self.model = LatentDirichletAllocation(
                n_components=self.num_topics,
                random_state=42,
                learning_method='online'
            )
        
        # Transform to document-topic matrix
        doc_topic_matrix = self.model.fit_transform(dtm)
        
        return doc_topic_matrix
    
    def fit_transform_gensim(self, tokenized_texts: List[List[str]]) -> np.ndarray:
        """
        Fit and transform texts using gensim LDA or HDP.
        
        Args:
            tokenized_texts: List of tokenized documents (list of lists of tokens)
        
        Returns:
            Document-topic matrix (list of topic distributions per doc)
        """
        # Use the tokenized texts directly - preprocessing is done by SpaCy
        self.tokenized_texts = tokenized_texts
        
        # Ensure all items in tokenized_texts are lists of tokens, not strings
        for i, tokens in enumerate(tokenized_texts):
            if isinstance(tokens, str):
                self.logger.warning(f"Found string instead of token list at index {i}, converting...")
                # If it's a string, convert it to a list of tokens using SpaCy
                if self.spacy_preprocessor:
                    tokenized_texts[i] = self.spacy_preprocessor.preprocess_text(tokens)
                else:
                    # Fallback to simple splitting if no SpaCy preprocessor
                    tokenized_texts[i] = tokens.split()
        
        self.gensim_dictionary = Dictionary(tokenized_texts)
        self.gensim_corpus = [self.gensim_dictionary.doc2bow(text) for text in tokenized_texts]
        
        if self.algorithm == 'hdp':
            self.model = HdpModel(
                self.gensim_corpus,
                id2word=self.gensim_dictionary
            )
            # Estimate number of topics from HDP
            self.num_topics = len(self.model.show_topics(formatted=False))
        else:
            self.model = LdaModel(
                self.gensim_corpus,
                num_topics=self.num_topics,
                id2word=self.gensim_dictionary,
                passes=10,
                alpha='auto',
                random_state=42
            )
        
        # Store feature names (gensim: id2word mapping)
        self.feature_names = [self.gensim_dictionary[i] for i in range(len(self.gensim_dictionary))]
        
        # Get dense doc-topic matrix (list of lists, shape [n_docs, n_topics])
        doc_topic_matrix = []
        if self.algorithm == 'hdp':
            for doc_bow in self.gensim_corpus:
                topic_dist = [0.0] * self.num_topics
                for topic_id, prob in self.model[doc_bow]:
                    if topic_id < self.num_topics:
                        topic_dist[topic_id] = prob
                doc_topic_matrix.append(topic_dist)
        else:
            for doc_bow in self.gensim_corpus:
                topic_dist = [0.0] * self.num_topics
                for topic_id, prob in self.model.get_document_topics(doc_bow, minimum_probability=0.0):
                    topic_dist[topic_id] = prob
                doc_topic_matrix.append(topic_dist)
        return doc_topic_matrix
    
    def fit_transform_bertopic(self, texts: List[str]) -> np.ndarray:
        """
        Fit and transform texts using BERTopic.
        
        Args:
            texts: List of document texts
        
        Returns:
            Document-topic matrix (list of topic distributions per doc)
        """
        from sklearn.feature_extraction.text import CountVectorizer
        import torch
        
        # Vérifier si un GPU est disponible
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.logger.info(f"Using device for BERTopic: {device}")
        
        # No need for stopwords as preprocessing is done by SpaCy
        vectorizer_model = CountVectorizer()
        
        # Initialiser le modèle d'embedding avec le device spécifié
        embedding_model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2", device=device)
        
        # Initialiser le modèle BERTopic
        self.model = BERTopic(
            language="french", 
            embedding_model=embedding_model,
            vectorizer_model=vectorizer_model,
            calculate_probabilities=True,
            nr_topics=self.num_topics if self.num_topics != "auto" else "auto",
            verbose=True
        )
        
        # Fit the model and get topics and probabilities
        self.bertopic_topics, self.bertopic_probs = self.model.fit_transform(texts)
        
        # Get the actual number of topics found (may differ from requested)
        topic_info = self.model.get_topic_info()
        self.num_topics = len(topic_info) - 1  # Exclude the -1 outlier topic
        
        # Convert sparse probabilities to a full document-topic matrix
        # Each row is a document, each column is a topic
        import numpy as np
        doc_topic_matrix = np.zeros((len(texts), self.num_topics))
        
        # Fill in the matrix with available probabilities
        for i, (topic_idx, probs) in enumerate(zip(self.bertopic_topics, self.bertopic_probs)):
            if topic_idx != -1:  # Skip outlier topic
                for j, prob in enumerate(probs):
                    topic_id = j if j < topic_idx else j + 1
                    if topic_id < self.num_topics:
                        doc_topic_matrix[i, topic_id] = prob
        
        return doc_topic_matrix
    
    def fit_transform(self, documents: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Fit topic model and transform documents.
        
        Args:
            documents: List of document dictionaries
            
        Returns:
            Dictionary with topic model results
        """
        self.logger.info(f"Fitting topic model with algorithm: {self.algorithm}")
        
        # Preprocess documents with SpaCy if needed and not already done
        if self.spacy_preprocessor and not any('tokens' in doc for doc in documents[:10]):
            self.logger.info("Preprocessing documents with SpaCy...")
            documents = self.spacy_preprocessor.process_documents(
                documents, 
                text_key="cleaned_text" if "cleaned_text" in documents[0] else "text"
            )
            self.logger.info(f"SpaCy preprocessing complete. First document tokens: {documents[0]['tokens'][:10]}")
        
        if self.algorithm in ['lda', 'nmf']:
            # Sklearn-based modeling
            doc_ids, texts = self.preprocess_for_sklearn(documents)
            doc_topic_matrix = self.fit_transform_sklearn(texts)
            
            # Convert to more usable format
            doc_topics = {}
            for i, doc_id in enumerate(doc_ids):
                doc_topics[doc_id] = {
                    'topic_distribution': doc_topic_matrix[i],
                    'dominant_topic': int(np.argmax(doc_topic_matrix[i]))
                }
            
            # Get top terms for each topic
            top_terms = self.get_top_terms_sklearn(n_terms=10)
        elif self.algorithm == 'hdp':
            doc_ids, tokenized_texts = self.preprocess_for_gensim(documents)
            doc_topic_matrix = self.fit_transform_gensim(tokenized_texts)
            doc_topics = {}
            for i, doc_id in enumerate(doc_ids):
                doc_topics[doc_id] = {
                    'topic_distribution': doc_topic_matrix[i],
                    'dominant_topic': int(np.argmax(doc_topic_matrix[i]))
                }
            top_terms = self.get_top_terms_gensim(n_terms=10)
        elif self.algorithm == 'bertopic':
            doc_ids, texts = self.preprocess_for_sklearn(documents)
            doc_topic_matrix = self.fit_transform_bertopic(texts)
            doc_topics = {}
            for i, doc_id in enumerate(doc_ids):
                doc_topics[doc_id] = {
                    'topic_distribution': doc_topic_matrix[i].tolist() if isinstance(doc_topic_matrix[i], np.ndarray) else doc_topic_matrix[i],
                    'dominant_topic': self.bertopic_topics[i] if self.bertopic_topics[i] != -1 else -1
                }
            top_terms = self.get_top_terms_bertopic(n_terms=10)
        else:
            doc_ids, tokenized_texts = self.preprocess_for_gensim(documents)
            doc_topic_matrix = self.fit_transform_gensim(tokenized_texts)
            doc_topics = {}
            for i, doc_id in enumerate(doc_ids):
                doc_topics[doc_id] = {
                    'topic_distribution': doc_topic_matrix[i],
                    'dominant_topic': int(np.argmax(doc_topic_matrix[i]))
                }
            top_terms = self.get_top_terms_gensim(n_terms=10)
        
        return {
            'doc_topics': doc_topics,
            'top_terms': top_terms,
            'num_topics': self.num_topics,
            'algorithm': self.algorithm
        }
    
    def get_top_terms_sklearn(self, n_terms: int = 10) -> Dict[int, List[str]]:
        """
        Get top terms for each topic from sklearn model.
        
        Args:
            n_terms: Number of terms to return per topic
            
        Returns:
            Dictionary mapping topic IDs to lists of top terms
        """
        if self.model is None or self.feature_names is None:
            raise ValueError("Model has not been fit yet")
        
        top_terms = {}
        for topic_idx, topic in enumerate(self.model.components_):
            top_indices = topic.argsort()[:-n_terms-1:-1]
            top_terms[topic_idx] = [self.feature_names[i] for i in top_indices]
        
        return top_terms
    
    def get_top_terms_gensim(self, n_terms: int = 10) -> Dict[int, List[str]]:
        """
        Get top terms for each topic from gensim model.
        
        Args:
            n_terms: Number of terms to return per topic
            
        Returns:
            Dictionary mapping topic IDs to lists of top terms
        """
        if self.model is None:
            raise ValueError("Model has not been fit yet")
        
        top_terms = {}
        for topic_idx in range(self.num_topics):
            topic_terms = self.model.show_topic(topic_idx, n_terms)
            top_terms[topic_idx] = [term for term, _ in topic_terms]
        
        return top_terms
    
    def get_top_terms_bertopic(self, n_terms: int = 10) -> Dict[int, List[str]]:
        """
        Get the top terms for each topic from BERTopic.
        Args:
            n_terms: Number of terms to return per topic
        Returns:
            Dictionary mapping topic index to list of top terms
        """
        if self.model is None:
            raise ValueError("Model has not been fit yet")
        
        top_terms = {}
        topic_info = self.model.get_topic_info()
        
        # Skip the outlier topic (-1)
        for topic_id in topic_info[topic_info['Topic'] != -1]['Topic'].values:
            topic_words = self.model.get_topic(topic_id)
            # Robust: handle tuple or str
            cleaned_terms = []
            for item in topic_words[:n_terms]:
                if isinstance(item, (tuple, list)) and len(item) >= 1:
                    cleaned_terms.append(item[0])
                else:
                    cleaned_terms.append(str(item))
            top_terms[int(topic_id)] = cleaned_terms
        return top_terms
    
    def get_topic_coherence(self, texts: Optional[List[List[str]]] = None, coherence_type: str = "c_v") -> float:
        """
        Calculate the coherence score of the fitted Gensim topic model.
        Args:
            texts: Tokenized texts (if None, use self.gensim_corpus and self.gensim_dictionary)
            coherence_type: Type of coherence (e.g., 'c_v', 'u_mass', 'c_uci', 'c_npmi')
        Returns:
            Coherence score (float)
        """
        if self.model is None or self.gensim_dictionary is None:
            raise ValueError("Gensim model and dictionary must be fit before computing coherence.")
        if texts is None:
            # Try to use corpus texts if available
            if hasattr(self, 'tokenized_texts'):
                texts = self.tokenized_texts
            else:
                raise ValueError("Tokenized texts must be provided or available on the object.")
        from gensim.models import CoherenceModel
        cm = CoherenceModel(model=self.model, texts=texts, dictionary=self.gensim_dictionary, coherence=coherence_type)
        return cm.get_coherence()

    def get_topic_word_weights(self, n_terms: int = 10) -> Dict[int, List[Tuple[str, float]]]:
        """
        Get top terms and their weights for each topic.
        Args:
            n_terms: Number of terms per topic
        Returns:
            Dict mapping topic index to list of (word, weight)
        """
        if self.model is None:
            raise ValueError("Model has not been fit yet")
        topic_word_weights = {}
        for topic_idx in range(self.num_topics):
            topic_terms = self.model.show_topic(topic_idx, n_terms)
            topic_word_weights[topic_idx] = topic_terms
        return topic_word_weights

    def get_topic_distribution(self, max_topics: int = 20) -> List[float]:
        """
        Get the overall topic distribution (importance/frequency) in the corpus.
        Returns:
            List of topic weights (sum of probabilities across all documents, normalized)
        """
        if self.model is None or self.gensim_corpus is None:
            raise ValueError("Model and corpus must be fit before getting distribution.")
        if hasattr(self.model, 'get_document_topics'):
            # LDA
            topic_sums = [0.0] * self.num_topics
            for doc_bow in self.gensim_corpus:
                for topic_id, prob in self.model.get_document_topics(doc_bow, minimum_probability=0.0):
                    if topic_id < self.num_topics:
                        topic_sums[topic_id] += prob
            total = sum(topic_sums)
            return [s / total for s in topic_sums] if total > 0 else topic_sums
        else:
            # HDP
            topic_sums = []
            for _ in range(max_topics):
                topic_sums.append(0.0)
            for doc_bow in self.gensim_corpus:
                for topic_id, prob in self.model[doc_bow]:
                    if topic_id < max_topics:
                        topic_sums[topic_id] += prob
            total = sum(topic_sums)
            return [s / total for s in topic_sums] if total > 0 else topic_sums

    def get_topic_article_counts(self, threshold: float = 0.2, max_topics: int = 20) -> Dict[int, int]:
        """
        Count the number of articles (documents) for which each topic is dominant above a threshold.
        Args:
            threshold: Minimum probability for a topic to be considered as present in a document
        Returns:
            Dict mapping topic index to count of articles
        """
        if self.model is None or self.gensim_corpus is None:
            raise ValueError("Model and corpus must be fit before getting article counts.")
        if hasattr(self.model, 'get_document_topics'):
            doc_topic_matrix = []
            for doc_bow in self.gensim_corpus:
                topic_dist = [0.0] * self.num_topics
                for topic_id, prob in self.model.get_document_topics(doc_bow, minimum_probability=0.0):
                    if topic_id < self.num_topics:
                        topic_dist[topic_id] = prob
                doc_topic_matrix.append(topic_dist)
            doc_topic_matrix = np.array(doc_topic_matrix)
            topic_counts = {}
            for topic_idx in range(self.num_topics):
                topic_counts[topic_idx] = int((doc_topic_matrix[:, topic_idx] >= threshold).sum())
            return topic_counts
        else:
            # HDP
            doc_topic_matrix = []
            for doc_bow in self.gensim_corpus:
                topic_dist = [0.0] * max_topics
                for topic_id, prob in self.model[doc_bow]:
                    if topic_id < max_topics:
                        topic_dist[topic_id] = prob
                doc_topic_matrix.append(topic_dist)
            doc_topic_matrix = np.array(doc_topic_matrix)
            topic_counts = {}
            for topic_idx in range(max_topics):
                topic_counts[topic_idx] = int((doc_topic_matrix[:, topic_idx] >= threshold).sum())
            return topic_counts

    def get_representative_docs(self, n_docs: int = 3, max_topics: int = 20) -> Dict[int, List[int]]:
        """
        For each topic, return indices of the most representative documents (highest topic probability).
        Args:
            n_docs: Number of top documents per topic
        Returns:
            Dict mapping topic index to list of document indices
        """
        if self.model is None or self.gensim_corpus is None:
            raise ValueError("Model and corpus must be fit before getting representative documents.")
        if hasattr(self.model, 'get_document_topics'):
            doc_topic_matrix = []
            for doc_bow in self.gensim_corpus:
                topic_dist = [0.0] * self.num_topics
                for topic_id, prob in self.model.get_document_topics(doc_bow, minimum_probability=0.0):
                    if topic_id < self.num_topics:
                        topic_dist[topic_id] = prob
                doc_topic_matrix.append(topic_dist)
            doc_topic_matrix = np.array(doc_topic_matrix)
            rep_docs = {}
            for topic_idx in range(self.num_topics):
                top_doc_indices = doc_topic_matrix[:, topic_idx].argsort()[::-1][:n_docs]
                rep_docs[topic_idx] = top_doc_indices.tolist()
            return rep_docs
        else:
            # HDP
            doc_topic_matrix = []
            for doc_bow in self.gensim_corpus:
                topic_dist = [0.0] * max_topics
                for topic_id, prob in self.model[doc_bow]:
                    if topic_id < max_topics:
                        topic_dist[topic_id] = prob
                doc_topic_matrix.append(topic_dist)
            doc_topic_matrix = np.array(doc_topic_matrix)
            rep_docs = {}
            for topic_idx in range(max_topics):
                top_doc_indices = doc_topic_matrix[:, topic_idx].argsort()[::-1][:n_docs]
                rep_docs[topic_idx] = top_doc_indices.tolist()
            return rep_docs
    
    def get_topic_article_counts(self, threshold: float = 0.2, max_topics: int = 20) -> Dict[int, int]:
        """
        Count the number of articles (documents) for which each topic is dominant above a threshold.
        Args:
            threshold: Minimum probability for a topic to be considered as present in a document
        Returns:
            Dict mapping topic index to count of articles
        """
        if self.model is None or self.gensim_corpus is None:
            raise ValueError("Model and corpus must be fit before getting article counts.")
        if hasattr(self.model, 'get_document_topics'):
            doc_topic_matrix = []
            for doc_bow in self.gensim_corpus:
                topic_dist = [0.0] * self.num_topics
                for topic_id, prob in self.model.get_document_topics(doc_bow, minimum_probability=0.0):
                    if topic_id < self.num_topics:
                        topic_dist[topic_id] = prob
                doc_topic_matrix.append(topic_dist)
            doc_topic_matrix = np.array(doc_topic_matrix)
            topic_counts = {}
            for topic_idx in range(self.num_topics):
                topic_counts[topic_idx] = int((doc_topic_matrix[:, topic_idx] >= threshold).sum())
            return topic_counts
        else:
            # HDP
            doc_topic_matrix = []
            for doc_bow in self.gensim_corpus:
                topic_dist = [0.0] * max_topics
                for topic_id, prob in self.model[doc_bow]:
                    if topic_id < max_topics:
                        topic_dist[topic_id] = prob
                doc_topic_matrix.append(topic_dist)
            doc_topic_matrix = np.array(doc_topic_matrix)
            topic_counts = {}
            for topic_idx in range(max_topics):
                topic_counts[topic_idx] = int((doc_topic_matrix[:, topic_idx] >= threshold).sum())
            return topic_counts
    
    def get_bertopic_coherence(self, texts: Optional[List[List[str]]] = None, coherence_type: str = "c_v") -> float:
        """
        Calculate the coherence score of the fitted BERTopic model.
        
        Args:
            texts: List of tokenized texts (list of lists of tokens)
            coherence_type: Type of coherence to calculate
            
        Returns:
            Coherence score
        """
        if self.model is None:
            raise ValueError("Model has not been fit yet")
        
        # Ensure we have tokenized texts
        if texts is None:
            raise ValueError("texts must be provided for coherence calculation")
        
        # Get topics as lists of top words
        topics = []
        topic_info = self.model.get_topic_info()
        
        # Skip the outlier topic (-1)
        for topic_id in topic_info[topic_info['Topic'] != -1]['Topic'].values:
            topic_words = [word for word, _ in self.model.get_topic(topic_id)]
            topics.append(topic_words)
        
        # Calculate coherence using gensim
        from gensim.models.coherencemodel import CoherenceModel
        cm = CoherenceModel(topics=topics, texts=texts, dictionary=None, coherence=coherence_type)
        return cm.get_coherence()
    
    def get_bertopic_topic_distribution(self) -> List[float]:
        """
        Get the global distribution of topics in the corpus.
        
        Returns:
            List of topic proportions
        """
        if self.model is None or self.bertopic_topics is None:
            raise ValueError("Model has not been fit yet")
        
        import numpy as np
        # Get topic assignments (excluding outlier topic -1)
        topic_assignments = np.array(self.bertopic_topics)
        valid_topics = topic_assignments[topic_assignments != -1]
        
        # Count occurrences of each topic
        unique, counts = np.unique(valid_topics, return_counts=True)
        
        # Create a distribution array with zeros
        distribution = np.zeros(self.num_topics)
        
        # Fill in the counts for topics that appear in the corpus
        for topic_idx, count in zip(unique, counts):
            if 0 <= topic_idx < self.num_topics:
                distribution[topic_idx] = count
        
        # Normalize to get proportions
        if distribution.sum() > 0:
            distribution = distribution / distribution.sum()
        
        return distribution.tolist()
    
    def get_bertopic_word_weights(self, n_terms: int = 10) -> Dict[str, List[Tuple[str, float]]]:
        """
        Get the top weighted words for each topic from BERTopic model.
        
        Args:
            n_terms: Number of terms to return per topic
            
        Returns:
            Dictionary mapping topic IDs to lists of (word, weight) tuples
        """
        if self.model is None:
            raise ValueError("Model has not been fit yet")
        
        weighted_words = {}
        topic_info = self.model.get_topic_info()
        
        # Skip the outlier topic (-1)
        for topic_id in topic_info[topic_info['Topic'] != -1]['Topic'].values:
            topic_words = self.model.get_topic(topic_id)
            weighted_words[str(int(topic_id))] = [(word, float(weight)) for word, weight in topic_words[:n_terms]]
        
        return weighted_words
    
    def get_bertopic_representative_docs(self, n_docs: int = 3) -> Dict[str, List[int]]:
        """
        Get the most representative documents for each topic.
        
        Args:
            n_docs: Number of documents to return per topic
            
        Returns:
            Dictionary mapping topic IDs to lists of document indices
        """
        if self.model is None:
            raise ValueError("Model has not been fit yet")
        
        representative_docs = {}
        topic_info = self.model.get_topic_info()
        
        # Skip the outlier topic (-1)
        for topic_id in topic_info[topic_info['Topic'] != -1]['Topic'].values:
            # Get representative documents for this topic
            doc_indices = self.model.get_representative_docs(topic_id)
            # Always treat as list
            if not isinstance(doc_indices, (list, tuple, np.ndarray)):
                doc_indices = [doc_indices]
            # Limit to n_docs
            representative_docs[str(int(topic_id))] = list(doc_indices)[:n_docs]
        
        return representative_docs
    
    def get_bertopic_article_counts(self) -> Dict[str, int]:
        """
        Get the number of articles assigned to each topic.
        
        Returns:
            Dictionary mapping topic IDs to article counts
        """
        if self.model is None or self.bertopic_topics is None:
            raise ValueError("Model has not been fit yet")
        
        from collections import Counter
        # Count occurrences of each topic
        counts = Counter(self.bertopic_topics)
        # Skip the outlier topic (-1)
        topic_article_counts = {
            str(topic_id): counts[topic_id] 
            for topic_id in range(self.num_topics)
        }
        
        return topic_article_counts
    
    def save_model(self, output_dir: str, prefix: str = 'topic_model') -> str:
        """
        Save the topic model and related objects.
        
        Args:
            output_dir: Directory to save the model
            prefix: Prefix for saved files
            
        Returns:
            Path to the saved model
        """
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        model_path = os.path.join(output_dir, f"{prefix}_{self.algorithm}.pkl")
        
        # Save different components based on algorithm
        if self.algorithm in ['lda', 'nmf']:
            # Save sklearn model and vectorizer
            with open(model_path, 'wb') as f:
                pickle.dump({
                    'model': self.model,
                    'vectorizer': self.vectorizer,
                    'feature_names': self.feature_names,
                    'config': self.config
                }, f)
        elif self.algorithm == 'bertopic':
            # Save BERTopic model
            self.model.save(model_path)
            # Save config used for training
            config_path = os.path.join(output_dir, f"{prefix}_{self.algorithm}_config.json")
            with open(config_path, 'w', encoding='utf-8') as f:
                json.dump({'num_topics': self.num_topics, **self.config}, f, ensure_ascii=False, indent=2)
        else:
            # Save gensim model and dictionary
            self.model.save(model_path)
            dict_path = os.path.join(output_dir, f"{prefix}_dictionary.pkl")
            self.gensim_dictionary.save(dict_path)
        
        return model_path
    
    @classmethod
    def load_model(cls, model_path: str, algorithm: str = 'lda') -> 'TopicModeler':
        """
        Load a saved topic model.
        
        Args:
            model_path: Path to the saved model
            algorithm: Algorithm type ('lda', 'nmf', or 'gensim_lda')
            
        Returns:
            Loaded TopicModeler instance
        """
        if algorithm in ['lda', 'nmf']:
            # Load sklearn model
            with open(model_path, 'rb') as f:
                saved_data = pickle.load(f)
            
            # Create new instance
            topic_modeler = cls(saved_data['config'])
            topic_modeler.model = saved_data['model']
            topic_modeler.vectorizer = saved_data['vectorizer']
            topic_modeler.feature_names = saved_data['feature_names']
            
        elif algorithm == 'bertopic':
            # Load BERTopic model
            topic_modeler = cls({'algorithm': 'bertopic'})
            topic_modeler.model = BERTopic.load(model_path)
            # Load config used for training if available
            config_path = model_path.replace('.pkl', '_config.json')
            if os.path.exists(config_path):
                with open(config_path, 'r', encoding='utf-8') as f:
                    topic_modeler.training_config = json.load(f)
            else:
                topic_modeler.training_config = {'num_topics': 'auto'}
        else:
            # Load gensim model
            # Assume dictionary is in same directory with _dictionary.pkl suffix
            dict_path = model_path.replace('.pkl', '_dictionary.pkl')
            
            # Create new instance with empty config
            topic_modeler = cls({'algorithm': 'gensim_lda'})
            topic_modeler.model = LdaModel.load(model_path)
            topic_modeler.gensim_dictionary = Dictionary.load(dict_path)
            topic_modeler.num_topics = topic_modeler.model.num_topics
        
        return topic_modeler

    def get_topic_names_with_llm(self, top_words_per_topic, llm_config=None):
        """
        Génère des noms de topics automatiquement via un LLM externe (configurable).
        Args:
            top_words_per_topic: liste de listes de mots (top words par topic)
            llm_config: dictionnaire de configuration (clé API, modèle, etc.)
        Returns:
            Liste des noms de topics (str)
        """
        if llm_config is None:
            # Essaye de charger depuis self.config si disponible
            llm_config = getattr(self, 'llm_config', {})
        client = LLMClient(llm_config)
        return client.get_topic_names(top_words_per_topic)

    def get_topic_names_llm_direct(self, top_words_per_topic, llm_config=None):
        """
        Génère des noms de topics automatiquement via l'API Mistral (ou autre LLM compatible), sans dépendance externe.
        Args:
            top_words_per_topic: liste de listes de mots (top words par topic)
            llm_config: dict avec au moins api_key, endpoint, model, language
        Returns:
            Liste des noms de topics (str)
        """
        if llm_config is None:
            raise ValueError("llm_config requis (clé api_key, endpoint, model, language)")
        api_key = llm_config.get("api_key")
        endpoint = llm_config.get("endpoint", "https://api.mistral.ai/v1/chat/completions")
        model = llm_config.get("model", "mistral-small")
        language = llm_config.get("language", "fr")
        if not api_key:
            raise ValueError("Clé API LLM manquante dans la config!")
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        topic_names = []
        for words in top_words_per_topic:
            prompt = (
                f"Donne uniquement un titre court, explicite, nominal (sans verbe), en français, "
                f"de moins de 6 mots (maximum 6 mots), qui résume le thème principal représenté par ces mots-clés extraits d'un topic modeling. "
                f"Les mots-clés du topic sont : {', '.join(words)}. "
                f"Réponds uniquement par le titre proposé, sans phrase introductive, sans ponctuation finale."
            )
            data = {
                "model": model,
                "messages": [
                    {"role": "user", "content": prompt}
                ],
                "max_tokens": 20,
                "temperature": 0.3
            }
            response = requests.post(endpoint, headers=headers, json=data)
            if response.status_code == 200:
                content = response.json()["choices"][0]["message"]["content"]
                topic_names.append(content.strip())
            else:
                topic_names.append(f"(Erreur API: {response.status_code})")
        return topic_names

    def transform_with_bertopic(self, texts: List[str]) -> Dict[str, Any]:
        """
        Transform texts using a pre-trained BERTopic model.
        
        Args:
            texts: List of document texts
            
        Returns:
            Dictionary with document topics and other results
        """
        if self.model is None:
            raise ValueError("No BERTopic model available. Load or fit a model first.")
        
        # Transform documents using the loaded model
        topics, probs = self.model.transform(texts)
        self.bertopic_topics = topics
        self.bertopic_probs = probs
        
        # Get the actual number of topics found
        topic_info = self.model.get_topic_info()
        self.num_topics = len(topic_info[topic_info['Topic'] != -1])
        
        # Prepare document-topic matrix
        doc_topics = {}
        for i, (doc_id, text) in enumerate(zip(range(len(texts)), texts)):
            topic_idx = topics[i]
            # Skip -1 (outlier) topics
            if topic_idx == -1:
                topic_idx = self.model.get_topic_info().iloc[1]['Topic']  # Use the first non-outlier topic
            
            # Create topic distribution (mostly zeros, with probability at the assigned topic)
            topic_distribution = np.zeros(self.num_topics)
            if topic_idx >= 0 and topic_idx < self.num_topics:
                topic_distribution[topic_idx] = 1.0
            
            # Store document topic info
            doc_topics[str(doc_id)] = {
                'topic': int(topic_idx),
                'topic_distribution': topic_distribution.tolist()
            }
        
        # Get top terms per topic
        top_terms = self.get_top_terms_bertopic()
        
        return {
            'doc_topics': doc_topics,
            'top_terms': top_terms
        }
