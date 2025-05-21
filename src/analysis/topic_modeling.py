"""
Topic modeling for newspaper articles.
"""

import os
import sys
import time
import json
import pickle
import logging
import numpy as np
from typing import List, Dict, Any, Optional, Tuple, Union
from pathlib import Path
from collections import Counter
from tqdm import tqdm
from datetime import datetime

def print_important(message, symbol='=', width=80, color='green'):
    """
    Affiche un message important de manière très visible dans le terminal.
    
    Args:
        message (str): Le message à afficher
        symbol (str): Le symbole à utiliser pour la bordure (défaut: '=')
        width (int): La largeur de la bordure (défaut: 80)
        color (str): La couleur du message ('green', 'yellow', 'red', 'blue')
    """
    colors = {
        'green': '\033[1;32m',
        'yellow': '\033[1;33m',
        'red': '\033[1;31m',
        'blue': '\033[1;34m',
        'reset': '\033[0m'
    }
    
    color_code = colors.get(color, colors['green'])
    reset = colors['reset']
    
    border = symbol * width
    # Utiliser tqdm.write pour éviter de perturber les barres de progression
    tqdm.write(f"\n{color_code}{border}{reset}")
    
    # Diviser le message en lignes et centrer chaque ligne
    for line in message.split('\n'):
        padding = max(0, (width - len(line)) // 2)
        tqdm.write(f"{color_code}{' ' * padding}{line}{reset}")
    
    tqdm.write(f"{color_code}{border}{reset}\n")

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

# Configuration du logger
logger = logging.getLogger(__name__)
# Définir le niveau de log global pour ce logger
logger.setLevel(logging.DEBUG)

# Créer un gestionnaire de fichier pour les logs spécifiques au topic modeling
log_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "logs")
os.makedirs(log_dir, exist_ok=True)
topic_modeling_log_file = os.path.join(log_dir, f"topic_modeling_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")

# Créer un gestionnaire de fichier pour les logs
file_handler = logging.FileHandler(topic_modeling_log_file, encoding='utf-8')
file_handler.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

# Créer un gestionnaire de console pour afficher les logs dans la console (visible dans Dash)
# Utiliser stderr pour éviter les conflits avec tqdm qui utilise stdout
console_handler = logging.StreamHandler(sys.stderr)
console_handler.setLevel(logging.INFO)  # Niveau INFO pour la console
# Utiliser un format plus visible pour les logs de console
console_formatter = logging.Formatter('\033[1;36m%(levelname)s\033[0m - %(message)s')  # Format coloré et simplifié pour la console
console_handler.setFormatter(console_formatter)
logger.addHandler(console_handler)

# Fonction pour afficher des messages importants dans le terminal
def print_important(message, symbol='='):
    """Affiche un message important dans le terminal avec une bordure visible."""
    border = symbol * 80
    # Utiliser tqdm.write pour éviter de perturber les barres de progression
    tqdm.write(f"\n{border}\n{message}\n{border}\n")

# Importer tqdm pour les barres de progression
from tqdm import tqdm

class TopicModeler:
    """Class for topic modeling on newspaper articles."""
    
    logger = logging.getLogger(__name__)
    
    def _get_device(self):
        """
        Détecte le meilleur dispositif disponible pour les calculs tensoriels.
        Retourne 'cuda' si un GPU NVIDIA est disponible, 'mps' pour Apple Silicon, sinon 'cpu'.
        
        Returns:
            str: Nom du dispositif ('cuda', 'mps', ou 'cpu')
        """
        import torch
        
        # Vérifier si CUDA est disponible (GPU NVIDIA)
        if torch.cuda.is_available():
            device = "cuda"
            gpu_name = torch.cuda.get_device_name(0)
            cuda_version = torch.version.cuda if hasattr(torch.version, 'cuda') else "Unknown"
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)  # en GB
            gpu_memory_allocated = torch.cuda.memory_allocated(0) / (1024**3)  # en GB
            gpu_memory_reserved = torch.cuda.memory_reserved(0) / (1024**3)  # en GB
            gpu_memory_free = gpu_memory - gpu_memory_allocated
            
            # Afficher un message très visible dans le terminal
            gpu_info = f"GPU DÉTECTÉ: {gpu_name}\n" \
                      f"CUDA VERSION: {cuda_version}\n" \
                      f"MÉMOIRE GPU TOTALE: {gpu_memory:.2f} GB\n" \
                      f"MÉMOIRE GPU DISPONIBLE: {gpu_memory_free:.2f} GB"
            print_important(gpu_info, symbol='*')
            
            # Logs normaux pour le fichier de log
            self.logger.info(f"BERTopic: GPU détecté: {gpu_name}")
            self.logger.info(f"BERTopic: Version CUDA: {cuda_version}")
            self.logger.info(f"BERTopic: Mémoire GPU totale: {gpu_memory:.2f} GB")
            self.logger.info(f"BERTopic: Mémoire GPU allouée: {gpu_memory_allocated:.2f} GB")
            self.logger.info(f"BERTopic: Mémoire GPU réservée: {gpu_memory_reserved:.2f} GB")
            self.logger.info(f"BERTopic: Mémoire GPU libre: {gpu_memory_free:.2f} GB")
            
            # Vérifier si CUDA est réellement utilisable avec un petit test
            try:
                test_tensor = torch.tensor([1.0, 2.0, 3.0], device='cuda')
                test_result = test_tensor * 2
                self.logger.info(f"BERTopic: Test CUDA réussi, GPU opérationnel: {test_result}")
                tqdm.write(f"\033[1;32m[GPU] Test CUDA réussi, GPU opérationnel\033[0m")
            except Exception as e:
                self.logger.warning(f"BERTopic: CUDA disponible mais test échoué: {str(e)}")
                self.logger.warning("BERTopic: Utilisation du CPU comme fallback")
                tqdm.write(f"\033[1;31m[GPU] CUDA disponible mais test échoué: {str(e)}\033[0m")
                tqdm.write(f"\033[1;31m[GPU] Utilisation du CPU comme fallback\033[0m")
                return "cpu"
                
            return device
        
        # Vérifier si MPS est disponible (Apple Silicon M1/M2)
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            self.logger.info("BERTopic: Apple Silicon (MPS) détecté, utilisation de MPS")
            # Test MPS
            try:
                test_tensor = torch.tensor([1.0, 2.0, 3.0], device='mps')
                test_result = test_tensor * 2
                self.logger.info(f"BERTopic: Test MPS réussi, Apple Silicon opérationnel: {test_result}")
            except Exception as e:
                self.logger.warning(f"BERTopic: MPS disponible mais test échoué: {str(e)}")
                self.logger.warning("BERTopic: Utilisation du CPU comme fallback")
                return "cpu"
            return "mps"
        
        # Fallback sur CPU
        else:
            self.logger.info("BERTopic: Aucun GPU détecté, utilisation du CPU")
            return "cpu"
            
    def _log_gpu_usage(self, message=""):
        """
        Enregistre l'utilisation actuelle du GPU dans les logs.
        Utile pour suivre l'utilisation de la mémoire pendant le traitement.
        
        Args:
            message: Message optionnel à inclure dans le log
            
        Returns:
            dict: Statistiques d'utilisation du GPU ou None si pas de GPU
        """
        import torch
        if not torch.cuda.is_available():
            return {"utilization": 0, "allocated": 0, "total": 0, "free": 0}
            
        # Collecter les statistiques d'utilisation du GPU
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)  # en GB
        gpu_memory_allocated = torch.cuda.memory_allocated(0) / (1024**3)  # en GB
        gpu_memory_reserved = torch.cuda.memory_reserved(0) / (1024**3)  # en GB
        gpu_memory_free = gpu_memory - gpu_memory_allocated
        gpu_utilization = (gpu_memory_allocated / gpu_memory) * 100 if gpu_memory > 0 else 0
        
        # Construire le message de log
        prefix = f"{message} - " if message else ""
        self.logger.info(f"{prefix}GPU utilisation: {gpu_utilization:.1f}% ({gpu_memory_allocated:.2f}/{gpu_memory:.2f} GB)")
        
        # Afficher un message plus visible dans le terminal pour les utilisations élevées
        if gpu_utilization > 50:
            tqdm.write(f"\033[1;33m[GPU] {prefix}Utilisation: {gpu_utilization:.1f}% ({gpu_memory_allocated:.2f}/{gpu_memory:.2f} GB)\033[0m")
        
        # Libérer la mémoire cache si l'utilisation est élevée
        if gpu_utilization > 80:
            self.logger.warning(f"GPU utilisation élevée ({gpu_utilization:.1f}%), nettoyage du cache")
            tqdm.write(f"\033[1;31m[GPU] UTILISATION ÉLEVÉE ({gpu_utilization:.1f}%), NETTOYAGE DU CACHE\033[0m")
            torch.cuda.empty_cache()
            new_gpu_memory_allocated = torch.cuda.memory_allocated(0) / (1024**3)
            self.logger.info(f"Après nettoyage: {new_gpu_memory_allocated:.2f} GB utilisés")
            tqdm.write(f"\033[1;32m[GPU] Après nettoyage: {new_gpu_memory_allocated:.2f} GB utilisés\033[0m")
        
        return {"utilization": gpu_utilization, "allocated": gpu_memory_allocated, "total": gpu_memory, "free": gpu_memory_free}
    
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
        self.workers = config.get('workers', 2)
        
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
            # Vérifier si on veut utiliser le multi-threading
            if self.workers > 1:
                # Utiliser LdaMulticore avec une valeur fixe pour alpha
                from gensim.models import LdaMulticore
                
                self.model = LdaMulticore(
                    self.gensim_corpus,
                    num_topics=self.num_topics,
                    id2word=self.gensim_dictionary,
                    passes=10,
                    alpha='symmetric',  # Valeur fixe pour alpha, car 'auto' n'est pas supporté
                    workers=self.workers,  # Utiliser le nombre de workers spécifié
                    random_state=42
                )
            else:
                # Utiliser LdaModel avec alpha='auto'
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
    
    def fit_transform_bertopic(self, texts: List[str], preprocessed_docs=None) -> np.ndarray:
        """
        Fit and transform texts using BERTopic.
        
        Args:
            texts: List of document texts
            preprocessed_docs: Optional, prétraités par SpaCy (avec tokens)
        
        Returns:
            Document-topic matrix (list of topic distributions per doc)
        """
        from sklearn.feature_extraction.text import CountVectorizer
        import umap
        import hdbscan
        import numpy as np
        from sklearn.decomposition import PCA
        import time
        
        start_time = time.time()
        
        # Récupérer le meilleur dispositif disponible
        device = self._get_device()
        self.logger.info(f"BERTopic: Using device: {device}")
        
        # Afficher un message très visible dans le terminal
        print_important(f"DÉMARRAGE DU FITTING BERTOPIC SUR {device.upper()}\nNOMBRE DE DOCUMENTS: {len(texts)}")
        
        # Récupérer les paramètres de configuration pour UMAP et HDBSCAN
        bertopic_config = self.config.get('bertopic', {})
        umap_n_neighbors = bertopic_config.get('umap_n_neighbors', 15)
        umap_n_components = bertopic_config.get('umap_n_components', 5)
        umap_min_dist = bertopic_config.get('umap_min_dist', 0.0)
        hdbscan_min_cluster_size = bertopic_config.get('hdbscan_min_cluster_size', 15)
        hdbscan_min_samples = bertopic_config.get('hdbscan_min_samples', 10)
        embedding_model_name = bertopic_config.get('embedding_model', 'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')
        batch_size = bertopic_config.get('batch_size', 64)  # Taille de batch pour les embeddings
        use_pca = bertopic_config.get('use_pca', True)  # Utiliser PCA pour réduire la dimensionnalité
        pca_components = bertopic_config.get('pca_components', 50)  # Nombre de composantes PCA
        
        # Si nous avons des documents prétraités, utilisons-les
        if preprocessed_docs is not None:
            self.logger.info("BERTopic: Using preprocessed documents with SpaCy tokens")
            # Convertir les tokens en textes pour BERTopic
            preprocessed_texts = [' '.join(doc.get('tokens', [])) for doc in preprocessed_docs]
            self.logger.info(f"BERTopic: Example preprocessed text: {preprocessed_texts[0][:100]}...")
            
            # Utiliser un vectorizer qui ne fait pas de prétraitement supplémentaire
            vectorizer_model = CountVectorizer(lowercase=False, token_pattern=r'\b\w+\b')
        else:
            self.logger.info("BERTopic: No preprocessed documents provided, using raw texts")
            preprocessed_texts = texts
            # Vectorizer standard avec prétraitement minimal
            vectorizer_model = CountVectorizer()
        
        # Initialiser le modèle d'embedding avec le device spécifié
        from sentence_transformers import SentenceTransformer
        embedding_model = SentenceTransformer(embedding_model_name, device=device)
        
        # Optimiser les paramètres du modèle d'embedding pour le GPU
        if device == 'cuda':
            # Augmenter la taille du batch pour le GPU
            embedding_model.max_seq_length = min(512, embedding_model.max_seq_length)  # Limiter la longueur des séquences
            self.logger.info(f"BERTopic: Set embedding batch size to {batch_size} and max_seq_length to {embedding_model.max_seq_length}")
        
        # Créer une classe d'embedding personnalisée pour le traitement par lots avec tqdm
        class BatchEmbedder:
            def __init__(self, model, batch_size=64, logger=None):
                self.model = model
                self.batch_size = batch_size
                self.logger = logger
            
            def __call__(self, documents):
                # Traiter les documents par lots pour économiser la mémoire
                embeddings = []
                
                # Calculer le nombre de lots
                num_batches = (len(documents) + self.batch_size - 1) // self.batch_size
                total_docs = len(documents)
                
                # Afficher un message unique au début avec tqdm.write
                tqdm.write(f"\033[1;36m[EMBEDDING] Traitement de {total_docs} documents en {num_batches} lots (batch size: {self.batch_size})\033[0m")
                
                # Créer une barre de progression unique avec tqdm
                with tqdm(total=total_docs, desc="Embedding", unit="docs", ncols=100, file=sys.stdout) as pbar:
                    last_gpu_log = 0
                    
                    for i in range(0, total_docs, self.batch_size):
                        batch = documents[i:i+self.batch_size]
                        batch_size = len(batch)
                        batch_embeddings = self.model.encode(batch, show_progress_bar=False)
                        embeddings.append(batch_embeddings)
                        
                        # Mettre à jour la barre de progression avec le nombre exact de documents traités
                        pbar.update(batch_size)
                        
                        # Log GPU usage seulement toutes les 10 batches ou tous les 5000 documents
                        current_docs = i + batch_size
                        if (current_docs - last_gpu_log > 5000 or i == 0) and self.logger:
                            last_gpu_log = current_docs
                            percent_done = min(100, int(current_docs / total_docs * 100))
                            
                            # Obtenir les statistiques GPU
                            gpu_stats = None
                            if hasattr(self.logger, '_log_gpu_usage'):
                                gpu_stats = self.logger._log_gpu_usage()
                            
                            if gpu_stats:
                                # Mettre à jour la barre de progression avec les infos GPU
                                pbar.set_postfix({"GPU": f"{gpu_stats['utilization']:.1f}%", "Mem": f"{gpu_stats['allocated']:.1f}GB", "Docs": f"{current_docs}/{total_docs}"})
                                
                                # Afficher un message pour les utilisations élevées uniquement avec tqdm.write
                                if gpu_stats['utilization'] > 70:
                                    tqdm.write(f"\033[1;33m[EMBEDDING] {percent_done}% - GPU: {gpu_stats['utilization']:.1f}% ({gpu_stats['allocated']:.2f}/{gpu_stats['total']:.2f} GB)\033[0m")
                            else:
                                # Fallback si nous n'avons pas accès à _log_gpu_usage
                                import torch
                                if torch.cuda.is_available():
                                    mem_allocated = torch.cuda.memory_allocated(0) / (1024**3)
                                    mem_total = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                                    utilization = (mem_allocated / mem_total) * 100 if mem_total > 0 else 0
                                    pbar.set_postfix({"GPU": f"{utilization:.1f}%", "Mem": f"{mem_allocated:.1f}GB", "Docs": f"{current_docs}/{total_docs}"})
                                    
                                    # Afficher un message pour les utilisations élevées uniquement avec tqdm.write
                                    if utilization > 70:
                                        tqdm.write(f"\033[1;33m[EMBEDDING] {percent_done}% - GPU: {utilization:.1f}% ({mem_allocated:.2f}/{mem_total:.2f} GB)\033[0m")
                        
                        # Libérer la mémoire GPU périodiquement
                        if i > 0 and i % (20 * self.batch_size) == 0:
                            import torch
                            if torch.cuda.is_available():
                                torch.cuda.empty_cache()
                
                return np.vstack(embeddings)
        
        # Optimiser la taille de batch en fonction de la mémoire GPU disponible
        if device == 'cuda':
            import torch
            total_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)  # en GB
            # Ajuster la taille du batch en fonction de la mémoire disponible
            if total_memory > 7:  # Pour les GPU avec beaucoup de mémoire (comme la RTX 4060 Ti 8GB)
                batch_size = 1024
                tqdm.write(f"\033[1;32m[OPTIMISATION] GPU avec {total_memory:.1f} GB détecté, utilisation d'un batch size de {batch_size} pour l'embedding\033[0m")
            elif total_memory > 5:  # Pour les GPU avec une mémoire moyenne
                batch_size = 128
                tqdm.write(f"\033[1;32m[OPTIMISATION] GPU avec {total_memory:.1f} GB détecté, utilisation d'un batch size de {batch_size} pour l'embedding\033[0m")
            elif total_memory > 3:  # Pour les GPU avec une mémoire limitée
                batch_size = 64
                tqdm.write(f"\033[1;33m[OPTIMISATION] GPU avec {total_memory:.1f} GB détecté, utilisation d'un batch size de {batch_size} pour l'embedding\033[0m")
            else:  # Pour les GPU avec peu de mémoire
                batch_size = 32
                tqdm.write(f"\033[1;31m[OPTIMISATION] GPU avec seulement {total_memory:.1f} GB détecté, utilisation d'un batch size réduit de {batch_size} pour l'embedding\033[0m")
            
            # Forcer l'allocation de mémoire GPU pour mieux utiliser la carte
            tqdm.write("Pré-allocation de mémoire GPU pour optimiser les performances d'embedding...")
            try:
                dummy_tensor = torch.rand(2000, 2000, device='cuda')
                dummy_result = torch.matmul(dummy_tensor, dummy_tensor)
                del dummy_tensor, dummy_result
                torch.cuda.empty_cache()
                tqdm.write("\033[1;32m[GPU] Pré-allocation réussie\033[0m")
            except Exception as e:
                self.logger.warning(f"Erreur lors de la pré-allocation de mémoire GPU: {e}")
                tqdm.write(f"\033[1;31m[GPU] Erreur lors de la pré-allocation de mémoire GPU: {e}\033[0m")
        
        # Créer l'embedder par lots avec la taille de batch optimisée
        batch_embedder = BatchEmbedder(embedding_model, batch_size=batch_size, logger=self)
        
        # Afficher un message explicite pour l'utilisation du GPU pendant l'embedding
        if device == 'cuda':
            print_important(f"UTILISATION DU GPU POUR L'EMBEDDING\nMODÈLE: {embedding_model_name}\nBATCH SIZE: {batch_size}\nMAX SEQ LENGTH: {embedding_model.max_seq_length}", symbol='-')
        
        # Tenter d'utiliser cuML pour UMAP et HDBSCAN si CUDA est disponible
        use_cuml = False
        if device == 'cuda':
            try:
                from cuml.manifold import UMAP as cumlUMAP
                from cuml.cluster import HDBSCAN as cumlHDBSCAN
                
                umap_model = cumlUMAP(
                    n_neighbors=umap_n_neighbors,
                    n_components=umap_n_components,
                    min_dist=umap_min_dist,
                    metric='cosine',
                    random_state=42
                )
                
                hdbscan_model = cumlHDBSCAN(
                    min_cluster_size=hdbscan_min_cluster_size,
                    min_samples=hdbscan_min_samples,
                    metric='euclidean',
                    cluster_selection_method='eom',
                    prediction_data=True
                )
                
                self.logger.info("BERTopic: Successfully initialized cuML for UMAP and HDBSCAN on GPU")
                use_cuml = True
            except ImportError:
                self.logger.warning("BERTopic: cuML not found. Using CPU versions for UMAP/HDBSCAN.")
                use_cuml = False
        
        # Si cuML n'est pas disponible ou si nous ne sommes pas sur CUDA, utiliser les versions CPU
        if not use_cuml:
            umap_model = umap.UMAP(
                n_neighbors=umap_n_neighbors,
                n_components=umap_n_components,
                min_dist=umap_min_dist,
                metric='cosine',
                low_memory=True,
                random_state=42
            )
            
            hdbscan_model = hdbscan.HDBSCAN(
                min_cluster_size=hdbscan_min_cluster_size,
                min_samples=hdbscan_min_samples,
                metric='euclidean',
                cluster_selection_method='eom',
                prediction_data=True
            )
            
            self.logger.info("BERTopic: Using CPU versions for UMAP and HDBSCAN")
        
        # Configurer le pipeline de dimensionnalité
        dimension_reduction_pipeline = None
        if use_pca and len(preprocessed_texts) > 1000:  # Utiliser PCA seulement pour les grands corpus
            # Créer un pipeline de réduction de dimensionnalité avec PCA avant UMAP
            from sklearn.pipeline import Pipeline
            
            dimension_reduction_pipeline = Pipeline([
                ("pca", PCA(n_components=pca_components)),
                ("umap", umap_model)
            ])
            
            self.logger.info(f"BERTopic: Using PCA ({pca_components} components) before UMAP for dimensionality reduction")
        
        # Initialiser le modèle BERTopic avec les modèles optimisés
        self.model = BERTopic(
            language="french", 
            embedding_model=batch_embedder,  # Utiliser notre embedder par lots
            umap_model=dimension_reduction_pipeline if dimension_reduction_pipeline else umap_model,
            hdbscan_model=hdbscan_model,
            vectorizer_model=vectorizer_model,
            calculate_probabilities=True,
            nr_topics=self.num_topics if self.num_topics != "auto" else "auto",
            verbose=True
        )
        
        # Fit the model and get topics and probabilities
        self.logger.info("BERTopic: Starting model fitting...")
        fit_start_time = time.time()
        
        # Vérifier l'utilisation du GPU avant de commencer le fitting
        if device == 'cuda':
            gpu_stats = self._log_gpu_usage("Avant fitting BERTopic")
            
            # Afficher un message explicite pour indiquer que le modèle va commencer à s'entraîner
            start_info = f"DÉMARRAGE DE L'ENTRAÎNEMENT DU MODÈLE BERTOPIC SUR GPU\n" \
                         f"NOMBRE DE DOCUMENTS: {len(preprocessed_texts)}\n" \
                         f"NOMBRE DE TOPICS DEMANDÉS: {self.num_topics if self.num_topics != 'auto' else 'auto (détection automatique)'}\n" \
                         f"UTILISATION GPU INITIALE: {gpu_stats['utilization']:.1f}% ({gpu_stats['allocated']:.2f}/{gpu_stats['total']:.2f} GB)"
            print_important(start_info, symbol='*')
        else:
            # Afficher un message explicite pour indiquer que le modèle va commencer à s'entraîner sur CPU
            print_important(f"DÉMARRAGE DE L'ENTRAÎNEMENT DU MODÈLE BERTOPIC SUR CPU\n" \
                          f"NOMBRE DE DOCUMENTS: {len(preprocessed_texts)}\n" \
                          f"NOMBRE DE TOPICS DEMANDÉS: {self.num_topics if self.num_topics != 'auto' else 'auto (détection automatique)'}")
        
        # Fit et transform
        self.bertopic_topics, self.bertopic_probs = self.model.fit_transform(preprocessed_texts)
        
        # Vérifier l'utilisation du GPU après le fitting
        if device == 'cuda':
            gpu_stats = self._log_gpu_usage("Après fitting BERTopic")
            
            # Afficher un message explicite pour indiquer que l'entraînement est terminé
            end_info = f"ENTRAÎNEMENT BERTOPIC TERMINÉ\n" \
                       f"TEMPS TOTAL: {time.time() - fit_start_time:.2f} secondes\n" \
                       f"NOMBRE DE TOPICS TROUVÉS: {len(self.model.get_topic_info()[self.model.get_topic_info()['Topic'] != -1])}\n" \
                       f"UTILISATION GPU FINALE: {gpu_stats['utilization']:.1f}% ({gpu_stats['allocated']:.2f}/{gpu_stats['total']:.2f} GB)"
            print_important(end_info, symbol='#')
        else:
            # Afficher un message explicite pour indiquer que l'entraînement est terminé sur CPU
            print_important(f"ENTRAÎNEMENT BERTOPIC TERMINÉ SUR CPU\n" \
                          f"TEMPS TOTAL: {time.time() - fit_start_time:.2f} secondes\n" \
                          f"NOMBRE DE TOPICS TROUVÉS: {len(self.model.get_topic_info()[self.model.get_topic_info()['Topic'] != -1])}")
            
        fit_time = time.time() - fit_start_time
        
        # Get the actual number of topics found (may differ from requested)
        topic_info = self.model.get_topic_info()
        self.num_topics = len(topic_info[topic_info['Topic'] != -1])
        self.logger.info(f"BERTopic: Found {self.num_topics} topics in {fit_time:.2f} seconds")
        
        # Log topic distribution
        topic_counts = topic_info[topic_info['Topic'] != -1]['Count'].values
        topic_percentages = topic_counts / topic_counts.sum() * 100
        self.logger.info(f"BERTopic: Top 5 topics by size: {topic_info[topic_info['Topic'] != -1].head(5)[['Topic', 'Count', 'Name']].to_dict('records')}")
        self.logger.info(f"BERTopic: Largest topic contains {topic_percentages[0]:.1f}% of documents")
        
        # Get the number of outliers and log it
        outlier_count = (np.array(self.bertopic_topics) == -1).sum()
        outlier_percentage = outlier_count / len(self.bertopic_topics) * 100
        self.logger.info(f"BERTopic: Found {outlier_count} outliers ({outlier_percentage:.1f}%)")
        
        # Create a dedicated 'outliers' topic if there are outliers
        has_outlier_topic = outlier_count > 0
        if has_outlier_topic:
            self.logger.info(f"BERTopic: Creating a dedicated 'outliers' topic (index {self.num_topics})")
            
        # The total number of topics including the outlier topic
        total_topics = self.num_topics + (1 if has_outlier_topic else 0)
        self.logger.info(f"BERTopic: Final number of topics = {total_topics}")
        
        # Log the topic distribution for debugging
        topic_distribution = {}
        
        for topic_idx in self.bertopic_topics:
            topic_distribution[topic_idx] = topic_distribution.get(topic_idx, 0) + 1
        self.logger.info(f"BERTopic: Topic distribution: {topic_distribution}")
        
        # Convert sparse probabilities to a full document-topic matrix
        # Each row is a document, each column is a topic (including outlier topic if present)
        doc_topic_matrix = np.zeros((len(texts), total_topics))
        
        # Fill in the matrix with available probabilities
        for i, (topic_idx, probs) in enumerate(zip(self.bertopic_topics, self.bertopic_probs)):
            if topic_idx != -1:  # Regular topic
                # For regular topics, use the probability distribution from BERTopic
                for j, prob in enumerate(probs):
                    topic_id = j if j < topic_idx else j + 1
                    if topic_id < self.num_topics:
                        doc_topic_matrix[i, topic_id] = prob
            else:  # Outlier topic (-1)
                if has_outlier_topic:
                    # For outliers, set a high probability for the outlier topic
                    doc_topic_matrix[i, self.num_topics] = 1.0
        
        # Mesurer le temps total d'exécution
        execution_time = time.time() - start_time
        self.logger.info(f"BERTopic: Processing completed in {execution_time:.2f} seconds")
        
        # Afficher un résumé des résultats
        outlier_count = (np.array(self.bertopic_topics) == -1).sum()
        outlier_percentage = outlier_count / len(self.bertopic_topics) * 100
        self.logger.info(f"BERTopic: Summary: {self.num_topics} topics, {outlier_count} outliers ({outlier_percentage:.1f}%)")
        
        # Afficher la distribution des probabilités (pour détecter le problème mentionné)
        if len(self.bertopic_probs) > 0:
            max_probs = [max(probs) if len(probs) > 0 else 0 for probs in self.bertopic_probs]
            avg_max_prob = sum(max_probs) / len(max_probs) if max_probs else 0
            self.logger.info(f"BERTopic: Average max probability: {avg_max_prob:.4f}")
            if avg_max_prob > 0.9:
                self.logger.warning("BERTopic: High average max probability detected (>0.9). This may indicate topic assignment issues.")
        
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
        
        # Store document IDs for later reference
        self.doc_ids = [doc.get('doc_id', doc.get('id', f"doc_{i}")) for i, doc in enumerate(documents)]
        
        # Preprocess documents with SpaCy if needed and not already done
        # Only preprocess if tokens aren't already present in most documents
        tokens_present = sum(1 for doc in documents[:min(10, len(documents))] if 'tokens' in doc)
        tokens_ratio = tokens_present / min(10, len(documents))
        
        if self.spacy_preprocessor and tokens_ratio < 0.8:  # Process only if less than 80% have tokens
            self.logger.info(f"Preprocessing documents with SpaCy... (only {tokens_ratio*100:.1f}% have tokens)")
            tqdm.write(f"[PREPROCESSING] Documents need preprocessing with SpaCy ({tokens_ratio*100:.1f}% have tokens)")
            documents = self.spacy_preprocessor.process_documents(
                documents, 
                text_key="cleaned_text" if "cleaned_text" in documents[0] else "text"
            )
            self.logger.info(f"SpaCy preprocessing complete. First document tokens: {documents[0]['tokens'][:10]}")
        else:
            if tokens_ratio >= 0.8:
                self.logger.info(f"Skipping SpaCy preprocessing - {tokens_ratio*100:.1f}% of documents already have tokens")
                tqdm.write(f"[PREPROCESSING] Using cached tokens ({tokens_ratio*100:.1f}% of documents have tokens)")
        
        if self.algorithm in ['lda', 'nmf']:
            # Sklearn-based modeling
            doc_ids, texts = self.preprocess_for_sklearn(documents)
            doc_topic_matrix = self.fit_transform_sklearn(texts)
            
            # Convert to more usable format
            doc_topics = {}
            for i, doc_id in enumerate(doc_ids):
                doc_topics[doc_id] = {
                    'topic_distribution': doc_topic_matrix[i].tolist() if isinstance(doc_topic_matrix[i], np.ndarray) else doc_topic_matrix[i],
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
                    'topic_distribution': doc_topic_matrix[i].tolist() if isinstance(doc_topic_matrix[i], np.ndarray) else doc_topic_matrix[i],
                    'dominant_topic': int(np.argmax(doc_topic_matrix[i]))
                }
            top_terms = self.get_top_terms_gensim(n_terms=10)
        elif self.algorithm == 'bertopic':
            doc_ids, texts = self.preprocess_for_sklearn(documents)
            # Utiliser les documents prétraités par SpaCy si disponibles
            tokens_present = sum(1 for doc in documents[:min(10, len(documents))] if 'tokens' in doc)
            tokens_ratio = tokens_present / min(10, len(documents))
            
            if tokens_ratio >= 0.8:  # Si au moins 80% des documents ont des tokens
                self.logger.info(f"Using SpaCy preprocessed documents for BERTopic ({tokens_ratio*100:.1f}% have tokens)")
                tqdm.write(f"[BERTOPIC] Using cached tokens for {len(documents)} documents")
                doc_topic_matrix = self.fit_transform_bertopic(texts, preprocessed_docs=documents)
            else:
                self.logger.info(f"Insufficient preprocessed tokens found ({tokens_ratio*100:.1f}%), using raw texts for BERTopic")
                tqdm.write(f"[BERTOPIC] Using raw texts (only {tokens_ratio*100:.1f}% have tokens)")
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
                    'topic_distribution': doc_topic_matrix[i].tolist() if isinstance(doc_topic_matrix[i], np.ndarray) else doc_topic_matrix[i],
                    'dominant_topic': int(np.argmax(doc_topic_matrix[i]))
                }
            top_terms = self.get_top_terms_gensim(n_terms=10)
        
        # Store doc_topics for later use
        self.doc_topics = doc_topics
        
        # Perform clustering on document-topic matrix
        n_clusters = self.config.get('n_clusters', 5)
        clustering_method = self.config.get('clustering_method', 'kmeans')
        
        # Convert doc_topic_matrix to numpy array if it's not already
        if isinstance(doc_topic_matrix, list):
            doc_topic_matrix_np = np.array(doc_topic_matrix)
        else:
            doc_topic_matrix_np = doc_topic_matrix
        
        # Cluster documents
        doc_clusters = self.cluster_documents(doc_topic_matrix_np, n_clusters=n_clusters, method=clustering_method)
        
        # Prepare top words per topic for cluster naming
        top_words_per_topic = {}
        for topic_id, words in top_terms.items():
            if isinstance(topic_id, str) and topic_id.startswith('topic_'):
                topic_id = int(topic_id.split('_')[1])
            top_words_per_topic[topic_id] = words
        
        # Generate cluster names if LLM config is available
        cluster_names = {}
        llm_config = self.config.get('llm_config', None)
        if llm_config:
            cluster_names = self.get_cluster_names_with_llm(doc_clusters, top_words_per_topic, llm_config)
        
        return {
            'doc_topics': doc_topics,
            'top_terms': top_terms,
            'num_topics': self.num_topics,
            'algorithm': self.algorithm,
            'clusters': doc_clusters,
            'cluster_names': cluster_names,
            'n_clusters': n_clusters,
            'clustering_method': clustering_method
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
            try:
                # Get topic words with error handling
                topic_words = self.model.get_topic(topic_id)
                
                # Handle different return formats from BERTopic
                cleaned_terms = []
                for item in topic_words[:n_terms]:
                    try:
                        # If item is a tuple with word and score (most common format)
                        if isinstance(item, (tuple, list)) and len(item) == 2:
                            word, _ = item
                            cleaned_terms.append(word)
                        # If item is a tuple with more than 2 elements
                        elif isinstance(item, (tuple, list)):
                            cleaned_terms.append(str(item[0]))
                        # If item is a string
                        elif isinstance(item, str):
                            cleaned_terms.append(item)
                        # Any other type
                        else:
                            cleaned_terms.append(str(item))
                    except Exception as e:
                        self.logger.warning(f"Problème lors du traitement d'un terme pour le topic {topic_id}: {e}")
                        self.logger.warning(f"Type de l'élément: {type(item)}, Valeur: {item}")
                        # Add a placeholder if we can't process this term
                        cleaned_terms.append(f"terme_{len(cleaned_terms)+1}")
                        
                top_terms[int(topic_id)] = cleaned_terms
            except Exception as e:
                self.logger.warning(f"Erreur lors de la récupération des mots pour le topic {topic_id}: {e}")
                # Add placeholder terms if we can't get the real ones
                top_terms[int(topic_id)] = [f"terme_{i+1}" for i in range(n_terms)]
                
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

    def get_topic_article_counts(self, threshold: float = 0.2, max_topics: int = 20) -> Dict[str, int]:
        """
        Count the number of articles (documents) for which each topic is dominant above a threshold.
        Args:
            threshold: Minimum probability for a topic to be considered as present in a document
        Returns:
            Dict mapping topic index to count of articles
        """
        if not hasattr(self, 'doc_topics') or not self.doc_topics:
            return {}
        
        counts = {}
        for doc_id, doc_info in self.doc_topics.items():
            if 'dominant_topic' in doc_info:
                topic_id = str(doc_info['dominant_topic'])
                counts[topic_id] = counts.get(topic_id, 0) + 1
        
        # Sort by topic ID and limit to max_topics
        return {k: counts[k] for k in sorted(counts.keys())[:max_topics] if k in counts}
    
    def cluster_documents(self, doc_topic_matrix: np.ndarray, n_clusters: int = 5, 
                          method: str = 'kmeans') -> Dict[str, int]:
        """
        Cluster documents based on their topic distributions.
        
        Args:
            doc_topic_matrix: Document-topic matrix (each row is a document, each column is a topic)
            n_clusters: Number of clusters to create
            method: Clustering method ('kmeans', 'agglomerative', 'dbscan')
            
        Returns:
            Dictionary mapping document IDs to cluster assignments
        """
        from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
        
        if not hasattr(self, 'doc_ids'):
            raise ValueError("Document IDs not available. Run fit_transform first.")
        
        # Choose clustering algorithm
        if method == 'kmeans':
            clustering = KMeans(n_clusters=n_clusters, random_state=42)
        elif method == 'agglomerative':
            clustering = AgglomerativeClustering(n_clusters=n_clusters)
        elif method == 'dbscan':
            clustering = DBSCAN(eps=0.5, min_samples=5)
        else:
            raise ValueError(f"Unsupported clustering method: {method}")
        
        # Fit clustering model
        cluster_labels = clustering.fit_predict(doc_topic_matrix)
        
        # Map document IDs to cluster assignments
        doc_clusters = {}
        for i, doc_id in enumerate(self.doc_ids):
            doc_clusters[doc_id] = int(cluster_labels[i])
        
        return doc_clusters
    
    def get_cluster_names_with_llm(self, doc_clusters: Dict[str, int], 
                                   top_words_per_topic: Dict[int, List[str]],
                                   llm_config: Optional[Dict[str, Any]] = None) -> Dict[str, str]:
        """
        Generate cluster names based on the dominant topics in each cluster.
        
        Args:
            doc_clusters: Dictionary mapping document IDs to cluster assignments
            top_words_per_topic: Dictionary mapping topic IDs to lists of top words
            llm_config: LLM configuration for naming
            
        Returns:
            Dictionary mapping cluster IDs to cluster names
        """
        # Count topics in each cluster
        cluster_topics = {}
        for doc_id, cluster in doc_clusters.items():
            cluster_str = str(cluster)
            if cluster_str not in cluster_topics:
                cluster_topics[cluster_str] = {}
            
            # Get dominant topic for this document
            if doc_id in self.doc_topics and 'dominant_topic' in self.doc_topics[doc_id]:
                topic = self.doc_topics[doc_id]['dominant_topic']
                cluster_topics[cluster_str][topic] = cluster_topics[cluster_str].get(topic, 0) + 1
        
        # For each cluster, find the most common topics and their top words
        cluster_keywords = {}
        for cluster, topics in cluster_topics.items():
            # Sort topics by frequency (descending)
            sorted_topics = sorted(topics.items(), key=lambda x: x[1], reverse=True)
            # Take top 3 topics or fewer if there aren't 3
            top_topics = sorted_topics[:3]
            
            # Collect top words from these topics
            keywords = []
            for topic, _ in top_topics:
                if topic in top_words_per_topic:
                    keywords.extend(top_words_per_topic[topic][:5])  # Top 5 words from each topic
            
            # Remove duplicates while preserving order
            seen = set()
            unique_keywords = [x for x in keywords if not (x in seen or seen.add(x))]
            cluster_keywords[cluster] = unique_keywords[:10]  # Limit to 10 keywords
        
        # Use LLM to generate names based on keywords
        if llm_config:
            try:
                # Convert to format expected by LLM naming function
                keyword_lists = [words for cluster, words in sorted(cluster_keywords.items(), key=lambda x: int(x[0]))]
                cluster_names_list = self.get_topic_names_llm_direct(keyword_lists, llm_config)
                
                # Map back to cluster IDs
                cluster_names = {}
                for i, (cluster, _) in enumerate(sorted(cluster_keywords.items(), key=lambda x: int(x[0]))):
                    if i < len(cluster_names_list):
                        cluster_names[cluster] = cluster_names_list[i]
                
                return cluster_names
            except Exception as e:
                self.logger.error(f"Error generating cluster names with LLM: {e}")
                # Fall back to generic names
                return {str(i): f"Cluster {i}" for i in range(len(cluster_keywords))}
        else:
            # Generate generic names
            return {str(i): f"Cluster {i}" for i in range(len(cluster_keywords))}
    
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
            try:
                topic_words_data = self.model.get_topic(topic_id)
                topic_words = []
                
                # Extract words with robust error handling
                for item in topic_words_data:
                    try:
                        if isinstance(item, (tuple, list)) and len(item) >= 1:
                            topic_words.append(str(item[0]))
                        elif isinstance(item, str):
                            topic_words.append(item)
                        else:
                            topic_words.append(str(item))
                    except Exception as e:
                        self.logger.warning(f"Problème lors de l'extraction d'un mot pour la cohérence du topic {topic_id}: {e}")
                
                if topic_words:  # Only add if we have words
                    topics.append(topic_words)
            except Exception as e:
                self.logger.warning(f"Erreur lors de la récupération des mots pour la cohérence du topic {topic_id}: {e}")
        
        # If we don't have any valid topics, return 0
        if not topics:
            self.logger.warning("Aucun topic valide pour le calcul de cohérence, retour de 0.0")
            return 0.0
        
        # Calculate coherence using gensim
        try:
            from gensim.models.coherencemodel import CoherenceModel
            from gensim.corpora import Dictionary
            
            # Create a dictionary from the texts
            dictionary = Dictionary(texts)
            
            # Now create the coherence model with the dictionary
            cm = CoherenceModel(topics=topics, texts=texts, dictionary=dictionary, coherence=coherence_type)
            return cm.get_coherence()
        except Exception as e:
            self.logger.error(f"Erreur lors du calcul de la cohérence: {e}")
            return 0.0
    
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
            try:
                # Get topic words with error handling
                topic_words = self.model.get_topic(topic_id)
                
                # Process the topic words with robust error handling
                processed_words = []
                for item in topic_words[:n_terms]:
                    try:
                        # Handle different return formats from BERTopic
                        if isinstance(item, (tuple, list)):
                            if len(item) == 2:  # Normal case: (word, weight)
                                word, weight = item
                                processed_words.append((str(word), float(weight)))
                            elif len(item) > 0:  # Tuple with more elements
                                word = str(item[0])
                                weight = 1.0 if len(item) == 1 else float(item[1]) if len(item) > 1 else 1.0
                                processed_words.append((word, weight))
                        elif isinstance(item, str):  # Just a string
                            processed_words.append((item, 1.0))
                        else:  # Any other type
                            processed_words.append((str(item), 1.0))
                    except Exception as e:
                        self.logger.warning(f"Problème lors du traitement d'un terme pour le topic {topic_id} dans get_bertopic_word_weights: {e}")
                        # Add a placeholder with weight 0.5
                        processed_words.append((f"terme_{len(processed_words)+1}", 0.5))
                
                weighted_words[str(int(topic_id))] = processed_words
            except Exception as e:
                self.logger.warning(f"Erreur lors de la récupération des mots pour le topic {topic_id} dans get_bertopic_word_weights: {e}")
                # Add placeholder terms with weights
                weighted_words[str(int(topic_id))] = [(f"terme_{i+1}", 1.0 - (i * 0.1)) for i in range(n_terms)]
        
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

    def transform_with_bertopic(self, texts: List[str]) -> Dict[str, Any]:
        """
        Transforme des textes en utilisant un modèle BERTopic pré-entraîné.
        
        Args:
            texts: Liste de textes à transformer
            
        Returns:
            Dict contenant la matrice document-topic et d'autres informations
        """
        if not hasattr(self, 'model') or self.model is None:
            raise ValueError("Le modèle BERTopic n'a pas été initialisé ou entraîné.")
        
        transform_start_time = time.time()
        
        # Déterminer le meilleur appareil (GPU ou CPU)
        device = self._get_device()
        self.logger.info(f"BERTopic: Using device for transformation: {device}")
        
        # Importer torch pour la gestion de la mémoire GPU
        import torch
        
        # Récupérer les paramètres de configuration pour le batch processing
        bertopic_config = self.config.get('bertopic', {})
        batch_size = bertopic_config.get('batch_size', 128)  # Taille de batch par défaut augmentée à 128
        
        # Optimiser la taille des batchs et forcer l'utilisation du GPU
        if device == 'cuda':
            # Forcer le modèle à utiliser le GPU
            if hasattr(self.model, 'embedding_model') and hasattr(self.model.embedding_model, 'embedding_model'):
                try:
                    self.model.embedding_model.embedding_model = self.model.embedding_model.embedding_model.to('cuda')
                    self.logger.info("BERTopic: Embedding model moved to GPU")
                    
                    # Créer un tenseur test pour vérifier que le GPU est utilisé
                    test_tensor = torch.tensor([1.0, 2.0, 3.0], device='cuda')
                    test_result = test_tensor * 2
                    print(f"Test GPU avec tenseur: {test_result}")
                    
                    # Forcer l'allocation de mémoire GPU pour mieux utiliser la carte
                    print("Pré-allocation de mémoire GPU pour optimiser les performances...")
                    dummy_tensor = torch.rand(2000, 2000, device='cuda')
                    dummy_result = torch.matmul(dummy_tensor, dummy_tensor)
                    del dummy_tensor, dummy_result
                    torch.cuda.empty_cache()
                except Exception as e:
                    self.logger.warning(f"Erreur lors du déplacement du modèle sur GPU: {e}")
            
            # Optimiser la taille de batch en fonction de la mémoire GPU disponible
            total_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)  # en GB
            # Ajuster la taille du batch en fonction de la mémoire disponible
            if total_memory > 7:  # Pour les GPU avec beaucoup de mémoire (comme la RTX 4060 Ti 8GB)
                batch_size = 1024
                print(f"\033[1;32m[OPTIMISATION] GPU avec {total_memory:.1f} GB détecté, utilisation d'un batch size de {batch_size}\033[0m")
            elif total_memory > 5:  # Pour les GPU avec une mémoire moyenne
                batch_size = 64
                print(f"\033[1;32m[OPTIMISATION] GPU avec {total_memory:.1f} GB détecté, utilisation d'un batch size de {batch_size}\033[0m")
            elif total_memory > 3:  # Pour les GPU avec une mémoire limitée
                batch_size = 32
                print(f"\033[1;33m[OPTIMISATION] GPU avec {total_memory:.1f} GB détecté, utilisation d'un batch size de {batch_size}\033[0m")
            else:  # Pour les GPU avec peu de mémoire
                batch_size = 16
                print(f"\033[1;31m[OPTIMISATION] GPU avec seulement {total_memory:.1f} GB détecté, utilisation d'un batch size réduit de {batch_size}\033[0m")
                
            # Libérer la mémoire cache avant de commencer
            torch.cuda.empty_cache()
            
            # Forcer l'allocation de mémoire GPU pour éviter les problèmes de fragmentation
            try:
                dummy_tensor = torch.zeros(100, 100, device='cuda')
                del dummy_tensor
            except Exception as e:
                self.logger.warning(f"Erreur lors de l'allocation de mémoire GPU: {e}")
            
            gpu_stats = self._log_gpu_usage("Avant transformation")
            # Afficher un message très visible dans le terminal
            start_info = f"TRANSFORMATION AVEC BERTOPIC SUR GPU (CUDA)\n" \
                         f"NOMBRE DE DOCUMENTS: {len(texts)}\n" \
                         f"TAILLE DE BATCH: {batch_size}\n" \
                         f"UTILISATION GPU INITIALE: {gpu_stats['utilization']:.1f}% ({gpu_stats['allocated']:.2f}/{gpu_stats['total']:.2f} GB)"
            print_important(start_info)
        
        # Transform documents using the loaded model with batch processing
        self.logger.info(f"BERTopic: Transforming {len(texts)} documents with pre-trained model (batch size: {batch_size})")
        
        # Initialiser les listes pour stocker les résultats
        all_topics = []
        all_probs = []
        
        # Traitement par lots avec barre de progression
        num_batches = (len(texts) + batch_size - 1) // batch_size
        
        # Pour les petits corpus, pas besoin de traitement par lots
        if len(texts) <= batch_size:
            topics, probs = self.model.transform(texts)
        else:
            # Traiter par lots pour économiser la mémoire
            # Afficher un message unique au début avec les informations importantes
            print(f"\033[1;36m[BERTOPIC] Transformation de {len(texts)} documents en {num_batches} lots (batch size: {batch_size})\033[0m")
            
            # Initialiser les variables pour le suivi des logs
            last_log_time = time.time()
            log_interval = 30  # Secondes entre les logs
            
            with tqdm(total=num_batches, desc="BERTopic Batches", unit="batch") as pbar:
                for i in range(0, len(texts), batch_size):
                    batch = texts[i:i + batch_size]
                    current_batch = i // batch_size
                    
                    # Log de progression uniquement au début, à la fin, et périodiquement (toutes les 30 secondes)
                    current_time = time.time()
                    if current_batch == 0 or current_batch == num_batches - 1 or current_time - last_log_time > log_interval:
                        last_log_time = current_time
                        progress_percent = int(i/len(texts)*100)
                        
                        # Log GPU uniquement si on utilise CUDA et seulement périodiquement
                        if device == 'cuda':
                            # Ne pas appeler _log_gpu_usage qui génère des logs supplémentaires
                            try:
                                mem_allocated = torch.cuda.memory_allocated(0) / (1024**3)
                                mem_total = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                                utilization = (mem_allocated / mem_total) * 100 if mem_total > 0 else 0
                                
                                # Mettre à jour la barre de progression avec les infos GPU
                                pbar.set_postfix({"GPU": f"{utilization:.1f}%", "Mem": f"{mem_allocated:.1f}GB", "Docs": f"{i+len(batch)}/{len(texts)}"})
                                
                                # Afficher un message d'alerte pour faible utilisation uniquement au début
                                if current_batch < 3 and utilization < 10:
                                    print(f"\033[1;33m[INFO] Utilisation GPU: {utilization:.1f}% - Considérez d'augmenter la taille du batch pour la prochaine exécution\033[0m")
                            except Exception:
                                pass  # Ignorer les erreurs lors de la récupération des stats GPU
                    
                    # Transformer le lot sans logs supplémentaires
                    try:
                        if device == 'cuda':
                            with torch.cuda.amp.autocast(enabled=True):
                                batch_topics, batch_probs = self.model.transform(batch)
                        else:
                            batch_topics, batch_probs = self.model.transform(batch)
                    except Exception as e:
                        self.logger.warning(f"Erreur lors de la transformation du lot {current_batch}: {e}")
                        batch_topics, batch_probs = self.model.transform(batch)  # Réessayer sans autocast
                    
                    # Ajouter aux résultats
                    all_topics.extend(batch_topics)
                    all_probs.extend(batch_probs)
                    
                    # Mettre à jour la barre de progression sans logs supplémentaires
                    pbar.update(1)
                    
                    # Libérer la mémoire GPU périodiquement sans logs
                    if device == 'cuda' and i > 0 and i % (20 * batch_size) == 0:
                        torch.cuda.empty_cache()
            
            # Convertir les résultats en arrays numpy
            topics, probs = np.array(all_topics), all_probs
        
        # Vérifier l'utilisation du GPU après la transformation
        if device == 'cuda':
            gpu_stats = self._log_gpu_usage("Après transformation")
            # Afficher un message très visible dans le terminal
            end_info = f"TRANSFORMATION BERTOPIC TERMINÉE\n" \
                       f"TEMPS TOTAL: {time.time() - transform_start_time:.2f} secondes\n" \
                       f"UTILISATION GPU FINALE: {gpu_stats['utilization']:.1f}% ({gpu_stats['allocated']:.2f}/{gpu_stats['total']:.2f} GB)"
            print_important(end_info, symbol='#')
        
        transform_time = time.time() - transform_start_time
        self.bertopic_topics = topics
        self.bertopic_probs = probs
        self.logger.info(f"BERTopic: Transformation completed in {transform_time:.2f} seconds")
        
        # Get the actual number of topics found
        topic_info = self.model.get_topic_info()
        self.num_topics = len(topic_info[topic_info['Topic'] != -1])
        
        # Compter les outliers pour décider s'il faut créer un topic spécifique
        outlier_count = (np.array(topics) == -1).sum()
        outlier_percentage = outlier_count / len(topics) * 100
        self.logger.info(f"BERTopic: Found {outlier_count} outliers ({outlier_percentage:.1f}%)")
        
        # Créer un topic spécifique pour les outliers si leur pourcentage est significatif
        create_outlier_topic = outlier_percentage > 10  # Seuil arbitraire, à ajuster selon les besoins
        outlier_topic_idx = self.num_topics  # Utiliser l'indice après le dernier topic normal
        
        if create_outlier_topic:
            self.logger.info(f"BERTopic: Creating a dedicated 'outliers' topic (index {outlier_topic_idx})")
            # Ajuster le nombre total de topics pour inclure le topic des outliers
            self.num_topics += 1
        else:
            self.logger.info("BERTopic: Outlier percentage is low, no dedicated outlier topic will be created")
        
        # Afficher des informations de débogage dans les logs
        self.logger.info(f"BERTopic: Final number of topics = {self.num_topics}")
        import collections
        topic_distribution = collections.Counter(topics)
        self.logger.info(f"BERTopic: Topic distribution: {dict(topic_distribution)}")
        
        # Prepare document-topic matrix
        doc_topics = {}
        for i, (doc_id, text) in enumerate(zip(range(len(texts)), texts)):
            doc_topics[i] = {}
            doc_topics[i]['text'] = text[:200] + '...' if len(text) > 200 else text  # Tronquer pour l'affichage
            doc_topics[i]['topic'] = int(topics[i])
            doc_topics[i]['topic_probs'] = probs[i].tolist() if isinstance(probs[i], np.ndarray) else probs[i]
            
            # Si c'est un outlier et qu'on crée un topic spécifique, remplacer -1 par l'index du topic d'outliers
            if create_outlier_topic and doc_topics[i]['topic'] == -1:
                doc_topics[i]['topic'] = outlier_topic_idx
        
        # Générer les noms de topics
        topic_words = {}
        for topic_id in range(-1, self.num_topics):
            if topic_id == -1:
                # Topic spécial pour les outliers dans BERTopic
                topic_words[topic_id] = {'words': ['outlier'], 'weights': [1.0]}
            elif topic_id == outlier_topic_idx and create_outlier_topic:
                # Notre topic dédié aux outliers
                topic_words[topic_id] = {'words': ['outlier', 'divers', 'inclassable'], 'weights': [1.0, 0.8, 0.6]}
            else:
                # Topics normaux
                try:
                    words, weights = self.model.get_topic(topic_id)
                    topic_words[topic_id] = {'words': words, 'weights': weights}
                except Exception as e:
                    self.logger.warning(f"Erreur lors de la récupération des mots pour le topic {topic_id}: {e}")
                    topic_words[topic_id] = {'words': [f'topic_{topic_id}'], 'weights': [1.0]}
        
        # Créer un résultat structuré
        result = {
            'doc_topics': doc_topics,
            'topic_words': topic_words,
            'num_topics': self.num_topics,
            'model_type': 'bertopic',
            'has_outlier_topic': create_outlier_topic,
            'outlier_topic_idx': outlier_topic_idx if create_outlier_topic else None,
            'processing_time': transform_time
        }
        
        return result
        
        # Compter les outliers pour décider s'il faut créer un topic spécifique
        outlier_count = (np.array(topics) == -1).sum()
        outlier_percentage = outlier_count / len(topics) * 100
        self.logger.info(f"BERTopic: Found {outlier_count} outliers ({outlier_percentage:.1f}%)")
        
        # Créer un topic spécifique pour les outliers si leur pourcentage est significatif
        create_outlier_topic = outlier_percentage > 10  # Seuil arbitraire, à ajuster selon les besoins
        outlier_topic_idx = self.num_topics  # Utiliser l'indice après le dernier topic normal
        
        if create_outlier_topic:
            self.logger.info(f"BERTopic: Creating a dedicated 'outliers' topic (index {outlier_topic_idx})")
            # Ajuster le nombre total de topics pour inclure le topic des outliers
            self.num_topics += 1
        else:
            self.logger.info("BERTopic: Outlier percentage is low, no dedicated outlier topic will be created")
        
        # Afficher des informations de débogage dans les logs
        self.logger.info(f"BERTopic: Final number of topics = {self.num_topics}")
        import collections
        topic_distribution = collections.Counter(topics)
        self.logger.info(f"BERTopic: Topic distribution: {dict(topic_distribution)}")
        
        # Prepare document-topic matrix
        doc_topics = {}
        for i, (doc_id, text) in enumerate(zip(range(len(texts)), texts)):
            original_topic = topics[i]  # Conserver la valeur originale
            
            # Initialiser la distribution des topics avec des zéros
            topic_distribution = np.zeros(self.num_topics)
            
            # Gérer les outliers différemment selon notre stratégie
            if original_topic == -1:  # C'est un outlier
                if create_outlier_topic:
                    # Assigner au topic spécifique des outliers
                    topic_idx = outlier_topic_idx
                    # Mettre une probabilité de 1.0 pour le topic des outliers
                    topic_distribution[outlier_topic_idx] = 1.0
                    self.logger.debug(f"BERTopic: Document {i} assigned to outlier topic {outlier_topic_idx}")
                else:
                    # Si pas assez d'outliers pour justifier un topic dédié, utiliser le comportement par défaut
                    # qui est de l'assigner au premier topic non-outlier
                    topic_idx = self.model.get_topic_info().iloc[1]['Topic']
                    # Utiliser les probabilités réelles si disponibles
                    if i < len(probs) and len(probs[i]) > 0:
                        # Distribuer les probabilités selon les topics existants
                        for j, prob in enumerate(probs[i]):
                            real_topic_idx = topic_info.iloc[j+1]['Topic'] if j+1 < len(topic_info) else -1
                            if 0 <= real_topic_idx < self.num_topics - (1 if create_outlier_topic else 0):
                                topic_distribution[real_topic_idx] = prob
                    else:
                        # Fallback: assigner une probabilité de 1.0 au topic choisi
                        if 0 <= topic_idx < self.num_topics - (1 if create_outlier_topic else 0):
                            topic_distribution[topic_idx] = 1.0
                    self.logger.debug(f"BERTopic: Outlier document {i} reassigned to topic {topic_idx}")
            else:  # Document avec un topic normal
                topic_idx = original_topic
                # Utiliser les probabilités réelles si disponibles
                if i < len(probs) and len(probs[i]) > 0:
                    for j, prob in enumerate(probs[i]):
                        real_topic_idx = topic_info.iloc[j+1]['Topic'] if j+1 < len(topic_info) else -1
                        if 0 <= real_topic_idx < self.num_topics - (1 if create_outlier_topic else 0):
                            topic_distribution[real_topic_idx] = prob
                else:
                    # Fallback: assigner une probabilité de 1.0 au topic détecté
                    if 0 <= topic_idx < self.num_topics - (1 if create_outlier_topic else 0):
                        topic_distribution[topic_idx] = 1.0
                    else:
                        self.logger.warning(f"BERTopic: Doc index {i}: topic_idx {topic_idx} out of bounds for num_topics {self.num_topics}")
            
            # Store document topic info
            doc_topics[str(doc_id)] = {
                'topic': int(topic_idx),
                'topic_distribution': topic_distribution.tolist(),
                'is_outlier': original_topic == -1  # Ajouter un flag pour indiquer si c'était un outlier à l'origine
            }
        
        # Get top terms per topic
        top_terms = self.get_top_terms_bertopic()
        
        # Si nous avons créé un topic outlier, ajouter des termes descriptifs
        if create_outlier_topic:
            top_terms[outlier_topic_idx] = ["outlier", "divers", "non-classé", "hors-sujet", "inclassable", 
                                           "atypique", "marginal", "exceptionnel", "particulier", "unique"]
        
        return {
            'doc_topics': doc_topics,
            'top_terms': top_terms,
            'has_outlier_topic': create_outlier_topic,
            'outlier_topic_idx': outlier_topic_idx if create_outlier_topic else None,
            'outlier_percentage': outlier_percentage
        }
