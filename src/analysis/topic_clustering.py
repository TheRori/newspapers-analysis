"""
Clustering documents using topic distributions from Gensim (LDA/HDP).

Usage example:
    from analysis.topic_clustering import cluster_documents_from_matrix
    clusters, model = cluster_documents_from_matrix(doc_topic_matrix, n_clusters=6)

Or to cluster from a saved doc_topic_matrix JSON file:
    clusters, model = cluster_documents_from_json(json_path, n_clusters=6)
    
Or to automatically find the optimal number of clusters:
    optimal_n, clusters, model = find_optimal_clusters(doc_topic_matrix, k_min=2, k_max=15)
"""
import json
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from typing import List, Tuple, Any, Dict, Optional
import matplotlib.pyplot as plt
import logging
import os
os.environ['LOKY_MAX_CPU_COUNT'] = '8' 

logger = logging.getLogger(__name__)

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


def find_optimal_clusters(doc_topic_matrix: List[List[float]], k_min: int = 2, k_max: int = 15, 
                         method: str = 'silhouette', random_state: int = 42) -> Tuple[int, np.ndarray, KMeans]:
    """
    Automatically find the optimal number of clusters using various metrics.
    
    Args:
        doc_topic_matrix: List of lists or np.ndarray, shape [n_docs, n_topics]
        k_min: Minimum number of clusters to try
        k_max: Maximum number of clusters to try
        method: Method to use for determining optimal clusters ('silhouette', 'calinski_harabasz', 'davies_bouldin', or 'all')
        random_state: Random seed
        
    Returns:
        optimal_n: Optimal number of clusters
        clusters: np.ndarray of cluster labels for the optimal number of clusters
        model: fitted KMeans instance for the optimal number of clusters
    """
    X = np.array(doc_topic_matrix)
    
    # Ensure we have enough samples
    if len(X) < k_max:
        logger.warning(f"Only {len(X)} samples available, reducing k_max from {k_max} to {len(X) - 1}")
        k_max = max(k_min, min(k_max, len(X) - 1))
    
    # Initialize scores
    silhouette_scores = []
    ch_scores = []
    db_scores = []
    k_values = range(k_min, k_max + 1)
    
    # Calculate scores for each k
    for k in k_values:
        logger.info(f"Evaluating clustering with k={k}")
        kmeans = KMeans(n_clusters=k, random_state=random_state)
        labels = kmeans.fit_predict(X)
        
        # Only calculate the requested metrics to save time
        if method in ['silhouette', 'all']:
            if len(np.unique(labels)) > 1:  # Silhouette requires at least 2 clusters with samples
                s_score = silhouette_score(X, labels)
                silhouette_scores.append(s_score)
                logger.info(f"  Silhouette score: {s_score:.4f}")
            else:
                silhouette_scores.append(-1)  # Invalid score
                
        if method in ['calinski_harabasz', 'all']:
            ch_score = calinski_harabasz_score(X, labels)
            ch_scores.append(ch_score)
            logger.info(f"  Calinski-Harabasz score: {ch_score:.4f}")
            
        if method in ['davies_bouldin', 'all']:
            if len(np.unique(labels)) > 1:  # Davies-Bouldin requires at least 2 clusters with samples
                db_score = davies_bouldin_score(X, labels)
                db_scores.append(db_score)
                logger.info(f"  Davies-Bouldin score: {db_score:.4f}")
            else:
                db_scores.append(float('inf'))  # Invalid score (higher is worse)
    
    # Determine optimal k based on the chosen method
    optimal_k = k_min
    if method == 'silhouette' or (method == 'all' and silhouette_scores):
        # Higher silhouette score is better
        optimal_k = k_values[np.argmax(silhouette_scores)]
        logger.info(f"Optimal k based on silhouette score: {optimal_k}")
    elif method == 'calinski_harabasz' or (method == 'all' and not silhouette_scores and ch_scores):
        # Higher Calinski-Harabasz score is better
        optimal_k = k_values[np.argmax(ch_scores)]
        logger.info(f"Optimal k based on Calinski-Harabasz score: {optimal_k}")
    elif method == 'davies_bouldin' or (method == 'all' and not silhouette_scores and not ch_scores and db_scores):
        # Lower Davies-Bouldin score is better
        optimal_k = k_values[np.argmin(db_scores)]
        logger.info(f"Optimal k based on Davies-Bouldin score: {optimal_k}")
    
    # Run clustering with the optimal k
    logger.info(f"Running final clustering with optimal k={optimal_k}")
    final_kmeans = KMeans(n_clusters=optimal_k, random_state=random_state)
    final_clusters = final_kmeans.fit_predict(X)
    
    return optimal_k, final_clusters, final_kmeans


def cluster_documents_auto(doc_topic_matrix: List[List[float]], k_min: int = 2, k_max: int = 15, 
                          method: str = 'silhouette', random_state: int = 42) -> Tuple[int, np.ndarray, KMeans]:
    """
    Convenience function to automatically cluster documents with optimal number of clusters.
    
    Args:
        doc_topic_matrix: List of lists or np.ndarray, shape [n_docs, n_topics]
        k_min: Minimum number of clusters to try
        k_max: Maximum number of clusters to try
        method: Method to use for determining optimal clusters
        random_state: Random seed
        
    Returns:
        optimal_n: Optimal number of clusters
        clusters: np.ndarray of cluster labels
        model: fitted KMeans instance
    """
    return find_optimal_clusters(doc_topic_matrix, k_min, k_max, method, random_state)


def visualize_cluster_metrics(doc_topic_matrix: List[List[float]], k_min: int = 2, k_max: int = 15, 
                             random_state: int = 42, save_path: Optional[str] = None) -> Dict[str, List[float]]:
    """
    Visualize different clustering metrics for a range of k values to help determine the optimal number of clusters.
    
    Args:
        doc_topic_matrix: List of lists or np.ndarray, shape [n_docs, n_topics]
        k_min: Minimum number of clusters to try
        k_max: Maximum number of clusters to try
        random_state: Random seed
        save_path: Optional path to save the plot
        
    Returns:
        Dictionary containing the scores for each metric
    """
    X = np.array(doc_topic_matrix)
    
    # Ensure we have enough samples
    if len(X) < k_max:
        logger.warning(f"Only {len(X)} samples available, reducing k_max from {k_max} to {len(X) - 1}")
        k_max = max(k_min, min(k_max, len(X) - 1))
    
    # Initialize scores
    silhouette_scores = []
    ch_scores = []
    db_scores = []
    inertia_values = []
    k_values = list(range(k_min, k_max + 1))
    
    # Calculate scores for each k
    for k in k_values:
        logger.info(f"Evaluating clustering with k={k}")
        kmeans = KMeans(n_clusters=k, random_state=random_state)
        labels = kmeans.fit_predict(X)
        inertia_values.append(kmeans.inertia_)
        
        # Calculate metrics
        if len(np.unique(labels)) > 1:  # Silhouette requires at least 2 clusters with samples
            silhouette_scores.append(silhouette_score(X, labels))
        else:
            silhouette_scores.append(-1)  # Invalid score
            
        ch_scores.append(calinski_harabasz_score(X, labels))
        
        if len(np.unique(labels)) > 1:  # Davies-Bouldin requires at least 2 clusters with samples
            db_scores.append(davies_bouldin_score(X, labels))
        else:
            db_scores.append(float('inf'))  # Invalid score (higher is worse)
    
    # Create a 2x2 subplot
    fig, axs = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Clustering Metrics for Different Numbers of Clusters', fontsize=16)
    
    # Plot silhouette scores (higher is better)
    axs[0, 0].plot(k_values, silhouette_scores, 'o-', color='blue')
    axs[0, 0].set_title('Silhouette Score (higher is better)')
    axs[0, 0].set_xlabel('Number of Clusters (k)')
    axs[0, 0].set_ylabel('Silhouette Score')
    axs[0, 0].grid(True)
    
    # Plot Calinski-Harabasz scores (higher is better)
    axs[0, 1].plot(k_values, ch_scores, 'o-', color='green')
    axs[0, 1].set_title('Calinski-Harabasz Score (higher is better)')
    axs[0, 1].set_xlabel('Number of Clusters (k)')
    axs[0, 1].set_ylabel('Calinski-Harabasz Score')
    axs[0, 1].grid(True)
    
    # Plot Davies-Bouldin scores (lower is better)
    axs[1, 0].plot(k_values, db_scores, 'o-', color='red')
    axs[1, 0].set_title('Davies-Bouldin Score (lower is better)')
    axs[1, 0].set_xlabel('Number of Clusters (k)')
    axs[1, 0].set_ylabel('Davies-Bouldin Score')
    axs[1, 0].grid(True)
    
    # Plot inertia (Elbow method - look for the "elbow" point)
    axs[1, 1].plot(k_values, inertia_values, 'o-', color='purple')
    axs[1, 1].set_title('Inertia (Elbow Method)')
    axs[1, 1].set_xlabel('Number of Clusters (k)')
    axs[1, 1].set_ylabel('Inertia')
    axs[1, 1].grid(True)
    
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    
    # Save the plot if a path is provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Plot saved to {save_path}")
    
    plt.show()
    
    # Return the scores for further analysis
    return {
        'k_values': k_values,
        'silhouette': silhouette_scores,
        'calinski_harabasz': ch_scores,
        'davies_bouldin': db_scores,
        'inertia': inertia_values
    }
