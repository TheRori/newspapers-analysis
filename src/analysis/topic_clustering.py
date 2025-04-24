"""
Clustering documents using topic distributions from Gensim (LDA/HDP).

Usage example:
    from analysis.topic_clustering import cluster_documents_from_matrix
    clusters, model = cluster_documents_from_matrix(doc_topic_matrix, n_clusters=6)

Or to cluster from a saved doc_topic_matrix JSON file:
    clusters, model = cluster_documents_from_json(json_path, n_clusters=6)
"""
import json
import numpy as np
from sklearn.cluster import KMeans
from typing import List, Tuple, Any

def cluster_documents_from_matrix(doc_topic_matrix: List[List[float]], n_clusters: int = 6, random_state: int = 42) -> Tuple[np.ndarray, KMeans]:
    """
    Cluster documents using KMeans on topic distribution matrix.
    Args:
        doc_topic_matrix: List of lists or np.ndarray, shape [n_docs, n_topics]
        n_clusters: Number of clusters to find
        random_state: Random seed
    Returns:
        clusters: np.ndarray of cluster labels
        model: fitted KMeans instance
    """
    X = np.array(doc_topic_matrix)
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state)
    clusters = kmeans.fit_predict(X)
    return clusters, kmeans

def cluster_documents_from_json(json_path: str, n_clusters: int = 6, random_state: int = 42) -> Tuple[np.ndarray, KMeans]:
    """
    Load doc_topic_matrix from JSON and cluster documents.
    Args:
        json_path: Path to JSON file containing doc_topic_matrix (list of lists)
        n_clusters: Number of clusters
        random_state: Random seed
    Returns:
        clusters: np.ndarray of cluster labels
        model: fitted KMeans instance
    """
    with open(json_path, 'r', encoding='utf-8') as f:
        matrix = json.load(f)
    return cluster_documents_from_matrix(matrix, n_clusters=n_clusters, random_state=random_state)
