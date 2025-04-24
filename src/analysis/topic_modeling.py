"""
Topic modeling for newspaper articles.
"""

import os
import pickle
from typing import List, Dict, Any, Tuple, Optional
import logging
import re
import requests

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.feature_extraction import text as sklearn_text
from sklearn.decomposition import LatentDirichletAllocation, NMF
import gensim
from gensim.corpora import Dictionary
from gensim.models import LdaModel, HdpModel, CoherenceModel
from .utils import get_stopwords
from .llm_utils import LLMClient

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
        
    def preprocess_for_sklearn(self, documents: List[Dict[str, Any]]) -> Tuple[List[str], List[str]]:
        """
        Preprocess documents for sklearn-based topic modeling.
        
        Args:
            documents: List of document dictionaries
            
        Returns:
            Tuple of (document_ids, document_texts)
        """
        doc_ids = [doc.get('doc_id', f"doc_{i}") for i, doc in enumerate(documents)]
        
        # Use cleaned_text if available, otherwise use text
        texts = []
        for doc in documents:
            if 'cleaned_text' in doc:
                texts.append(doc['cleaned_text'])
            elif 'text' in doc:
                texts.append(doc['text'])
            else:
                raise KeyError("Documents must contain either 'cleaned_text' or 'text' key")
        
        return doc_ids, texts
    
    def preprocess_for_gensim(self, documents: List[Dict[str, Any]]) -> Tuple[List[str], List[List[str]]]:
        """
        Preprocess documents for gensim-based topic modeling.
        
        Args:
            documents: List of document dictionaries
            
        Returns:
            Tuple of (document_ids, tokenized_texts)
        """
        doc_ids = [doc.get('doc_id', f"doc_{i}") for i, doc in enumerate(documents)]
        
        # Get tokenized texts
        tokenized_texts = []
        for doc in documents:
            if 'tokens' in doc:
                tokenized_texts.append(doc['tokens'])
            elif 'cleaned_text' in doc:
                # Simple tokenization by splitting on whitespace
                tokenized_texts.append(doc['cleaned_text'].split())
            elif 'text' in doc:
                tokenized_texts.append(doc['text'].split())
            else:
                raise KeyError("Documents must contain 'tokens', 'cleaned_text', or 'text' key")
        
        return doc_ids, tokenized_texts
    
    def fit_transform_sklearn(self, texts: List[str]) -> np.ndarray:
        """
        Fit and transform texts using sklearn vectorizer and topic model.
        
        Args:
            texts: List of document texts
            
        Returns:
            Document-topic matrix
        """
        # Exclude numbers but keep French/Unicode words
        token_pattern = r"(?u)\b[^\d\W]{2,}\b"  # words of at least 2 letters, no digits
        stopwords = list(get_stopwords("fr"))
        self.logger.info(f"Using {len(stopwords)} French stopwords for vectorizer.")
        self.logger.debug(f"French stopwords: {sorted(stopwords)}")
        self.logger.info(f"First 3 texts before vectorization: {[t[:200] for t in texts[:3]]}")
        if self.algorithm == 'nmf':
            self.vectorizer = TfidfVectorizer(
                max_df=self.max_df,
                min_df=self.min_df,
                stop_words=stopwords,
                token_pattern=token_pattern  # Exclude numbers
            )
        else:  # lda
            self.vectorizer = CountVectorizer(
                stop_words=stopwords,
                token_pattern=token_pattern,
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
    
    def fit_transform_gensim(self, texts: List[str]) -> np.ndarray:
        """
        Fit and transform texts using gensim LDA or HDP.
        
        Args:
            texts: List of document texts (untokenized)
        
        Returns:
            Document-topic matrix (list of topic distributions per doc)
        """
        # Utilisation uniquement des stopwords français
        stopwords = get_stopwords(("fr",))
        self.logger.info(f"Using {len(stopwords)} French stopwords for Gensim.")
        # Tokenize texts with better cleaning: remove punctuation, digits, short tokens, lowercase
        tokenized_texts = [
            [
                word for word in re.findall(r"\b\w{2,}\b", text.lower())
                if word not in stopwords and not word.isdigit()
            ]
            for text in texts
        ]
        self.tokenized_texts = tokenized_texts
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
    
    def fit_transform(self, documents: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Fit topic model and transform documents.
        
        Args:
            documents: List of document dictionaries
            
        Returns:
            Dictionary with topic model results
        """
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
            doc_topic_matrix = self.fit_transform_gensim([' '.join(toks) for toks in tokenized_texts])
            doc_topics = {}
            for i, doc_id in enumerate(doc_ids):
                doc_topics[doc_id] = {
                    'topic_distribution': doc_topic_matrix[i],
                    'dominant_topic': int(np.argmax(doc_topic_matrix[i]))
                }
            top_terms = self.get_top_terms_gensim(n_terms=10)
        else:
            doc_ids, tokenized_texts = self.preprocess_for_gensim(documents)
            doc_topic_matrix = self.fit_transform_gensim([' '.join(toks) for toks in tokenized_texts])
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
