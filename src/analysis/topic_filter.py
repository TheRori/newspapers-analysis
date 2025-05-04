"""
Module pour filtrer les analyses en fonction des résultats de topic modeling et clustering.
"""

import os
import json
from typing import List, Dict, Any, Optional, Union, Set
import logging
import numpy as np
import pandas as pd
from pathlib import Path

logger = logging.getLogger(__name__)

class TopicFilter:
    """
    Classe pour filtrer les documents en fonction des topics et clusters.
    Permet d'intégrer les résultats de topic modeling avec d'autres analyses.
    """
    
    def __init__(self, topic_results_path: Optional[str] = None):
        """
        Initialise le filtre de topics.
        
        Args:
            topic_results_path: Chemin vers le fichier de résultats de topic modeling
        """
        self.topic_results = None
        self.doc_topics = {}
        self.doc_clusters = {}
        self.topic_names = {}
        self.cluster_names = {}
        
        if topic_results_path:
            self.load_topic_results(topic_results_path)
    
    def load_topic_results(self, topic_results_path: str) -> None:
        """
        Charge les résultats de topic modeling à partir d'un fichier.
        
        Args:
            topic_results_path: Chemin vers le fichier de résultats de topic modeling
        """
        try:
            with open(topic_results_path, 'r', encoding='utf-8') as f:
                self.topic_results = json.load(f)
            
            # Extraire les topics par document
            if 'doc_topics' in self.topic_results:
                self.doc_topics = self.topic_results['doc_topics']
                logger.info(f"Loaded topic information for {len(self.doc_topics)} documents")
            
            # Extraire les noms de topics s'ils existent
            if 'topic_names' in self.topic_results:
                self.topic_names = self.topic_results['topic_names']
                logger.info(f"Loaded {len(self.topic_names)} topic names")
            
            # Extraire les clusters s'ils existent
            if 'clusters' in self.topic_results:
                self.doc_clusters = self.topic_results['clusters']
                logger.info(f"Loaded cluster information for {len(self.doc_clusters)} documents")
            
            # Extraire les noms de clusters s'ils existent
            if 'cluster_names' in self.topic_results:
                self.cluster_names = self.topic_results['cluster_names']
                logger.info(f"Loaded {len(self.cluster_names)} cluster names")
                
        except Exception as e:
            logger.error(f"Error loading topic results: {e}")
            raise
    
    def get_dominant_topics(self, threshold: float = 0.2) -> Dict[str, int]:
        """
        Obtient le topic dominant pour chaque document.
        
        Args:
            threshold: Seuil minimal de probabilité pour qu'un topic soit considéré dominant
            
        Returns:
            Dictionnaire associant chaque ID de document à son topic dominant
        """
        dominant_topics = {}
        
        for doc_id, doc_info in self.doc_topics.items():
            if 'dominant_topic' in doc_info:
                dominant_topics[doc_id] = doc_info['dominant_topic']
            elif 'topic_distribution' in doc_info:
                # Trouver le topic avec la probabilité la plus élevée
                topic_dist = doc_info['topic_distribution']
                max_prob = max(topic_dist)
                if max_prob >= threshold:
                    dominant_topics[doc_id] = np.argmax(topic_dist)
        
        return dominant_topics
    
    def get_documents_by_topic(self, topic_id: int, threshold: float = 0.2) -> List[str]:
        """
        Obtient tous les documents associés à un topic spécifique.
        
        Args:
            topic_id: ID du topic à filtrer
            threshold: Seuil minimal de probabilité pour qu'un document soit associé au topic
            
        Returns:
            Liste des IDs de documents associés au topic
        """
        documents = []
        
        for doc_id, doc_info in self.doc_topics.items():
            if 'dominant_topic' in doc_info and doc_info['dominant_topic'] == topic_id:
                documents.append(doc_id)
            elif 'topic_distribution' in doc_info:
                topic_dist = doc_info['topic_distribution']
                if len(topic_dist) > topic_id and topic_dist[topic_id] >= threshold:
                    documents.append(doc_id)
        
        return documents
    
    def get_documents_by_cluster(self, cluster_id: int) -> List[str]:
        """
        Obtient tous les documents associés à un cluster spécifique.
        
        Args:
            cluster_id: ID du cluster à filtrer
            
        Returns:
            Liste des IDs de documents associés au cluster
        """
        documents = []
        
        for doc_id, cluster in self.doc_clusters.items():
            if cluster == cluster_id:
                documents.append(doc_id)
        
        return documents
    
    def filter_documents(self, 
                        documents: List[Dict[str, Any]], 
                        topic_id: Optional[int] = None, 
                        cluster_id: Optional[int] = None,
                        exclude_topic_id: Optional[int] = None,
                        exclude_cluster_id: Optional[int] = None,
                        threshold: float = 0.2) -> List[Dict[str, Any]]:
        """
        Filtre une liste de documents en fonction des topics et/ou clusters.
        
        Args:
            documents: Liste de documents à filtrer
            topic_id: ID du topic à inclure (optionnel)
            cluster_id: ID du cluster à inclure (optionnel)
            exclude_topic_id: ID du topic à exclure (optionnel)
            exclude_cluster_id: ID du cluster à exclure (optionnel)
            threshold: Seuil minimal de probabilité pour les topics
            
        Returns:
            Liste filtrée de documents
        """
        # Déterminer les IDs de documents à inclure
        include_doc_ids = set()
        
        if topic_id is not None:
            include_doc_ids.update(self.get_documents_by_topic(topic_id, threshold))
        
        if cluster_id is not None:
            if topic_id is not None:
                # Intersection: documents qui sont à la fois dans le topic ET le cluster
                include_doc_ids = include_doc_ids.intersection(self.get_documents_by_cluster(cluster_id))
            else:
                # Juste les documents du cluster
                include_doc_ids.update(self.get_documents_by_cluster(cluster_id))
        
        # Déterminer les IDs de documents à exclure
        exclude_doc_ids = set()
        
        if exclude_topic_id is not None:
            exclude_doc_ids.update(self.get_documents_by_topic(exclude_topic_id, threshold))
        
        if exclude_cluster_id is not None:
            exclude_doc_ids.update(self.get_documents_by_cluster(exclude_cluster_id))
        
        # Si aucun filtre d'inclusion n'est spécifié, inclure tous les documents
        if topic_id is None and cluster_id is None:
            include_doc_ids = {doc.get('id', doc.get('doc_id', '')) for doc in documents}
        
        # Appliquer les filtres
        filtered_documents = []
        
        for doc in documents:
            doc_id = doc.get('id', doc.get('doc_id', ''))
            
            # Vérifier si le document doit être inclus
            if (not include_doc_ids or doc_id in include_doc_ids) and doc_id not in exclude_doc_ids:
                filtered_documents.append(doc)
        
        return filtered_documents
    
    def get_topic_summary(self) -> Dict[str, Any]:
        """
        Obtient un résumé des topics et de leur distribution.
        
        Returns:
            Dictionnaire avec des informations sur les topics
        """
        if not self.topic_results:
            return {}
        
        # Compter les documents par topic
        topic_counts = {}
        dominant_topics = self.get_dominant_topics()
        
        for topic in dominant_topics.values():
            topic_counts[topic] = topic_counts.get(topic, 0) + 1
        
        # Créer le résumé
        summary = {
            'num_topics': self.topic_results.get('num_topics', 0),
            'algorithm': self.topic_results.get('algorithm', ''),
            'topic_counts': topic_counts,
            'topic_names': self.topic_names
        }
        
        return summary
    
    def get_cluster_summary(self) -> Dict[str, Any]:
        """
        Obtient un résumé des clusters et de leur distribution.
        
        Returns:
            Dictionnaire avec des informations sur les clusters
        """
        if not self.doc_clusters:
            return {}
        
        # Compter les documents par cluster
        cluster_counts = {}
        
        for cluster in self.doc_clusters.values():
            cluster_counts[cluster] = cluster_counts.get(cluster, 0) + 1
        
        # Créer le résumé
        summary = {
            'num_clusters': len(set(self.doc_clusters.values())),
            'cluster_counts': cluster_counts,
            'cluster_names': self.cluster_names
        }
        
        return summary
    
    def add_topic_info_to_documents(self, documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Ajoute les informations de topic et cluster aux documents.
        
        Args:
            documents: Liste de documents
            
        Returns:
            Liste de documents avec informations de topic et cluster ajoutées
        """
        for doc in documents:
            doc_id = doc.get('id', doc.get('doc_id', ''))
            
            # Ajouter les informations de topic
            if doc_id in self.doc_topics:
                doc['topic_info'] = self.doc_topics[doc_id]
                
                # Ajouter le nom du topic dominant si disponible
                if 'dominant_topic' in self.doc_topics[doc_id] and str(self.doc_topics[doc_id]['dominant_topic']) in self.topic_names:
                    doc['topic_name'] = self.topic_names[str(self.doc_topics[doc_id]['dominant_topic'])]
            
            # Ajouter les informations de cluster
            if doc_id in self.doc_clusters:
                doc['cluster'] = self.doc_clusters[doc_id]
                
                # Ajouter le nom du cluster si disponible
                if str(self.doc_clusters[doc_id]) in self.cluster_names:
                    doc['cluster_name'] = self.cluster_names[str(self.doc_clusters[doc_id])]
        
        return documents


def run_filtered_analysis(analysis_type: str, 
                         topic_results_path: str, 
                         output_path: str,
                         topic_id: Optional[int] = None,
                         cluster_id: Optional[int] = None,
                         exclude_topic_id: Optional[int] = None,
                         exclude_cluster_id: Optional[int] = None,
                         config_path: Optional[str] = None,
                         articles_path: Optional[str] = None,
                         **kwargs) -> Dict[str, Any]:
    """
    Exécute une analyse filtrée par topic ou cluster.
    
    Args:
        analysis_type: Type d'analyse ('sentiment', 'entity', 'lexical', etc.)
        topic_results_path: Chemin vers les résultats de topic modeling
        output_path: Chemin pour sauvegarder les résultats
        topic_id: ID du topic à inclure (optionnel)
        cluster_id: ID du cluster à inclure (optionnel)
        exclude_topic_id: ID du topic à exclure (optionnel)
        exclude_cluster_id: ID du cluster à exclure (optionnel)
        config_path: Chemin vers le fichier de configuration (optionnel)
        articles_path: Chemin vers le fichier d'articles (optionnel)
        **kwargs: Arguments supplémentaires pour l'analyse
        
    Returns:
        Résultats de l'analyse
    """
    from src.utils.config_loader import load_config
    
    # Déterminer les chemins par défaut si non spécifiés
    if not config_path:
        project_root = Path(__file__).parent.parent.parent
        config_path = str(project_root / 'config' / 'config.yaml')
    
    if not articles_path:
        project_root = Path(__file__).parent.parent.parent
        articles_path = str(project_root / 'data' / 'processed' / 'articles.json')
    
    # Charger la configuration
    config = load_config(config_path)
    
    # Charger les articles
    with open(articles_path, 'r', encoding='utf-8') as f:
        articles = json.load(f)
    
    # Initialiser le filtre de topics
    topic_filter = TopicFilter(topic_results_path)
    
    # Filtrer les documents
    filtered_articles = topic_filter.filter_documents(
        articles,
        topic_id=topic_id,
        cluster_id=cluster_id,
        exclude_topic_id=exclude_topic_id,
        exclude_cluster_id=exclude_cluster_id
    )
    
    logger.info(f"Filtered {len(filtered_articles)}/{len(articles)} articles for analysis")
    
    # Exécuter l'analyse appropriée
    results = None
    
    if analysis_type == 'sentiment':
        from src.analysis.sentiment_analysis import SentimentAnalyzer
        analyzer = SentimentAnalyzer(config.get('analysis', {}).get('sentiment', {}))
        articles_with_results = analyzer.analyze_documents(filtered_articles)
        summary = analyzer.get_sentiment_summary(articles_with_results)
        results = {
            'articles': articles_with_results,
            'summary': summary,
            'filter_info': {
                'topic_id': topic_id,
                'cluster_id': cluster_id,
                'exclude_topic_id': exclude_topic_id,
                'exclude_cluster_id': exclude_cluster_id,
                'total_articles': len(articles),
                'filtered_articles': len(filtered_articles)
            }
        }
    
    elif analysis_type == 'entity':
        from src.analysis.entity_recognition import EntityRecognizer
        analyzer = EntityRecognizer(config.get('analysis', {}).get('entity_recognition', {}))
        articles_with_results = analyzer.process_documents(filtered_articles)
        summary = analyzer.get_entity_summary(articles_with_results)
        results = {
            'articles': articles_with_results,
            'summary': summary,
            'filter_info': {
                'topic_id': topic_id,
                'cluster_id': cluster_id,
                'exclude_topic_id': exclude_topic_id,
                'exclude_cluster_id': exclude_cluster_id,
                'total_articles': len(articles),
                'filtered_articles': len(filtered_articles)
            }
        }
    
    elif analysis_type == 'lexical':
        from src.analysis.lexical_analysis import LexicalAnalyzer
        analyzer = LexicalAnalyzer(config.get('analysis', {}).get('lexical', {}))
        results = analyzer.analyze_documents(filtered_articles)
        results['filter_info'] = {
            'topic_id': topic_id,
            'cluster_id': cluster_id,
            'exclude_topic_id': exclude_topic_id,
            'exclude_cluster_id': exclude_cluster_id,
            'total_articles': len(articles),
            'filtered_articles': len(filtered_articles)
        }
    
    # Sauvegarder les résultats
    if results and output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        logger.info(f"Saved filtered analysis results to {output_path}")
    
    return results
