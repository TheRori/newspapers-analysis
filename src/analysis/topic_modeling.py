"""
Topic modeling for newspaper articles.
"""

import os
import pickle
from typing import List, Dict, Any, Tuple, Optional
import logging
import re

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.feature_extraction import text as sklearn_text
from sklearn.decomposition import LatentDirichletAllocation, NMF
import gensim
from gensim.corpora import Dictionary
from gensim.models import LdaModel, HdpModel, CoherenceModel


class TopicModeler:
    """Class for topic modeling on newspaper articles."""
    
    logger = logging.getLogger(__name__)
    
    # French stopwords fallback (minimal list, can be expanded)
    FRENCH_STOPWORDS = set([
        'au', 'aux', 'avec', 'ce', 'ces', 'dans', 'de', 'des', 'du', 'elle', 'en', 'et', 'eux', 'il', 'je', 'la', 'le', 'leur', 'lui', 'ma', 'mais', 'me', 'même', 'mes', 'moi', 'mon', 'ne', 'nos', 'notre', 'nous', 'on', 'ou', 'par', 'pas', 'pour', 'qu', 'que', 'qui', 'sa', 'se', 'ses', 'son', 'sur', 'ta', 'te', 'tes', 'toi', 'ton', 'tu', 'un', 'une', 'vos', 'votre', 'vous', 'c', 'd', 'j', 'l', 'à', 'm', 'n', 's', 't', 'y', 'été', 'étée', 'étées', 'étés', 'étant', 'suis', 'es', 'est', 'sommes', 'êtes', 'sont', 'serai', 'seras', 'sera', 'serons', 'serez', 'seront', 'serais', 'serait', 'serions', 'seriez', 'seraient', 'étais', 'était', 'étions', 'étiez', 'étaient', 'fus', 'fut', 'fûmes', 'fûtes', 'furent', 'sois', 'soit', 'soyons', 'soyez', 'soient', 'fusse', 'fusses', 'fût', 'fussions', 'fussiez', 'fussent', 'ayant', 'eu', 'eue', 'eues', 'eus', 'ai', 'as', 'avons', 'avez', 'ont', 'aurai', 'auras', 'aura', 'aurons', 'aurez', 'auront', 'aurais', 'aurait', 'aurions', 'auriez', 'auraient', 'avais', 'avait', 'avions', 'aviez', 'avaient', 'eut', 'eûmes', 'eûtes', 'eurent', 'aie', 'aies', 'ait', 'ayons', 'ayez', 'aient', 'eusse', 'eusses', 'eût', 'eussions', 'eussiez', 'eussent', 'ceci', 'cela', 'celà', 'cet', 'cette', 'ici', 'ils', 'les', 'leurs', 'quel', 'quels', 'quelle', 'quelles', 'sans', 'soi'
    ])
    
    # German stopwords fallback (minimal list, can be expanded)
    GERMAN_STOPWORDS = set([
        'aber', 'alle', 'allem', 'allen', 'aller', 'alles', 'als', 'also', 'am', 'an', 'ander', 'andere', 'anderem', 'anderen', 'anderer', 'anderes', 'anderm', 'andern', 'anderr', 'anders', 'auch', 'auf', 'aus', 'bei', 'bin', 'bis', 'bist', 'da', 'damit', 'dann', 'der', 'den', 'des', 'dem', 'die', 'das', 'daß', 'derselbe', 'derselben', 'denselben', 'desselben', 'demselben', 'dieselbe', 'dieselben', 'dasselbe', 'dazu', 'dein', 'deine', 'deinem', 'deinen', 'deiner', 'deines', 'denn', 'derer', 'dessen', 'dich', 'dir', 'du', 'dies', 'diese', 'diesem', 'diesen', 'dieser', 'dieses', 'doch', 'dort', 'durch', 'ein', 'eine', 'einem', 'einen', 'einer', 'eines', 'einig', 'einige', 'einigem', 'einigen', 'einiger', 'einiges', 'einmal', 'er', 'ihn', 'ihm', 'es', 'etwas', 'euer', 'eure', 'eurem', 'euren', 'eurer', 'eures', 'für', 'gegen', 'gewesen', 'hab', 'habe', 'haben', 'hat', 'hatte', 'hatten', 'hier', 'hin', 'hinter', 'ich', 'mich', 'mir', 'ihr', 'ihre', 'ihrem', 'ihren', 'ihrer', 'ihres', 'euch', 'im', 'in', 'indem', 'ins', 'ist', 'jede', 'jedem', 'jeden', 'jeder', 'jedes', 'jene', 'jenem', 'jenen', 'jener', 'jenes', 'jetzt', 'kann', 'kein', 'keine', 'keinem', 'keinen', 'keiner', 'keines', 'können', 'könnte', 'machen', 'man', 'manche', 'manchem', 'manchen', 'mancher', 'manches', 'mein', 'meine', 'meinem', 'meinen', 'meiner', 'meines', 'mit', 'muss', 'musste', 'nach', 'nicht', 'nichts', 'noch', 'nun', 'nur', 'ob', 'oder', 'ohne', 'sehr', 'sein', 'seine', 'seinem', 'seinen', 'seiner', 'seines', 'selbst', 'sich', 'sie', 'ihnen', 'sind', 'so', 'solche', 'solchem', 'solchen', 'solcher', 'solches', 'soll', 'sollte', 'sondern', 'sonst', 'über', 'um', 'und', 'uns', 'unser', 'unserem', 'unseren', 'unserer', 'unseres', 'unter', 'viel', 'vom', 'von', 'vor', 'während', 'war', 'waren', 'warst', 'was', 'weg', 'weil', 'weiter', 'welche', 'welchem', 'welchen', 'welcher', 'welches', 'wenn', 'werde', 'werden', 'wie', 'wieder', 'will', 'wir', 'wird', 'wirst', 'wo', 'wollen', 'wollte', 'würde', 'würden', 'zu', 'zum', 'zur', 'zwar', 'zwischen'
    ])
    
    @staticmethod
    def get_stopwords(langs=("fr", "de")):
        """
        Get stopwords for specified languages (default: French + German).
        """
        stopwords = set()
        try:
            import stopwordsiso as stopwordsiso
            for lang in langs:
                stopwords.update(stopwordsiso.stopwords(lang))
        except Exception:
            # Fallback to NLTK/hardcoded if stopwordsiso unavailable
            try:
                import nltk
                from nltk.corpus import stopwords as nltk_stopwords
                for lang in langs:
                    try:
                        nltk.data.find('corpora/stopwords')
                    except LookupError:
                        nltk.download('stopwords')
                    stopwords.update(nltk_stopwords.words({'fr': 'french', 'de': 'german'}.get(lang, lang)))
            except Exception:
                stopwords = TopicModeler.FRENCH_STOPWORDS.union(TopicModeler.GERMAN_STOPWORDS)
        # Ajout manuel de mots fréquents si besoin
        extra_fr = {'plus', 'être', 'fait', 'sans', 'tout', 'bien', 'très', 'comme', 'peut', 'cette', 'dont', 'ainsi', 'encore', 'entre', 'aussi', 'ans', 'depuis', 'avant', 'après', 'lors', 'chez'}
        extra_de = {'auch', 'noch', 'schon', 'immer', 'gibt', 'kann', 'sich', 'wird', 'wurden', 'werden', 'diese', 'dieser', 'dieses', 'uns', 'unser', 'euch', 'euer', 'eure', 'ihnen', 'ihre', 'ihrem', 'ihren', 'ihres', 'ihm', 'ihn', 'ihr', 'sein', 'seine', 'seinem', 'seinen', 'seiner', 'seines', 'mein', 'meine', 'meinem', 'meinen', 'meiner', 'meines', 'dein', 'deine', 'deinem', 'deinen', 'deiner', 'deines'}
        return stopwords.union(extra_fr).union(extra_de)

    @staticmethod
    def get_french_stopwords():
        return TopicModeler.get_stopwords(langs=("fr",))
    
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
        stopwords = self.get_french_stopwords()
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
                max_df=self.max_df,
                min_df=self.min_df,
                stop_words=stopwords,
                token_pattern=token_pattern  # Exclude numbers
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
        stopwords = self.get_stopwords(langs=("fr", "de"))
        self.logger.info(f"Using {len(stopwords)} French+German stopwords for Gensim.")
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
