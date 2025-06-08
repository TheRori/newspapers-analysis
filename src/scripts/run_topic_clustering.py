"""
Script pour effectuer un clustering KMeans sur les documents à partir de la matrice doc-topic (issue de Gensim LDA/HDP).

Usage :
    python src/scripts/run_topic_clustering.py --input data/results/doc_topic_matrix.json --n-clusters 6 --output data/results/doc_clusters.json

- Le fichier d'entrée doit être un JSON (liste de listes) ou un fichier avancé (advanced_topic_analysis*.json)
- Le script sauvegarde les labels de cluster et les centres dans un fichier JSON
- Les filtres appliqués au topic modeling sont automatiquement utilisés pour le clustering
  (pas besoin de les réappliquer, ils sont inclus pour documentation)
"""
import argparse
import os
import json
import numpy as np
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from analysis.topic_clustering import cluster_documents_from_matrix, find_optimal_clusters, visualize_cluster_metrics
from utils.filter_utils import get_filter_summary


def get_parser():
    parser = argparse.ArgumentParser(description="Clustering de documents à partir d'une matrice doc-topic.")
    parser.add_argument('--input', type=str, required=True, help='Chemin du fichier JSON doc_topic_matrix ou advanced_topic_analysis')
    parser.add_argument('--n-clusters', type=str, default='6', help='Nombre de clusters KMeans ou "auto" pour détermination automatique')
    parser.add_argument('--k-min', type=int, default=2, help='Nombre minimum de clusters à tester (défaut: 2)')
    parser.add_argument('--k-max', type=int, default=15, help='Nombre maximum de clusters à tester (défaut: 15)')
    parser.add_argument('--force-k', action='store_true', help='Forcer le nombre de clusters spécifié par --n-clusters plutôt que de déterminer automatiquement le nombre optimal')
    parser.add_argument('--metric', type=str, choices=['silhouette', 'calinski_harabasz', 'davies_bouldin', 'all'], 
                        default='silhouette', help='Métrique à utiliser pour déterminer le nombre optimal de clusters')
    parser.add_argument('--visualize', action='store_true', help='Visualiser les métriques pour différents nombres de clusters')
    parser.add_argument('--output', type=str, default=None, help='Fichier de sortie pour les labels (JSON)')
    
    # Add filtering options (same as in run_topic_modeling.py for documentation purposes)
    # Note: These filters are not applied here, they're just for documentation
    parser.add_argument('--start-date', type=str, help='Filter used in topic modeling (format: YYYY-MM-DD)')
    parser.add_argument('--end-date', type=str, help='Filter used in topic modeling (format: YYYY-MM-DD)')
    parser.add_argument('--newspaper', type=str, help='Filter used in topic modeling')
    parser.add_argument('--canton', type=str, help='Filter used in topic modeling (e.g., FR, VD)')
    parser.add_argument('--topic', type=str, help='Filter used in topic modeling')
    parser.add_argument('--min-words', type=int, help='Filter used in topic modeling')
    parser.add_argument('--max-words', type=int, help='Filter used in topic modeling')
    
    return parser


def main():
    parser = get_parser()
    args = parser.parse_args()

    # Charger la matrice
    if args.input.endswith('.json'):
        with open(args.input, 'r', encoding='utf-8') as f:
            data = json.load(f)
        # Cas dict avec clé 'doc_topic_matrix'
        if isinstance(data, dict) and 'doc_topic_matrix' in data:
            doc_topic_matrix = [row['topic_distribution'] for row in data['doc_topic_matrix']]
            doc_ids = [row.get('doc_id', i) for i, row in enumerate(data['doc_topic_matrix'])]
        # Cas advanced_topic_analysis : dict avec 'doc_topics'
        elif isinstance(data, dict) and 'doc_topics' in data:
            doc_topic_matrix = [v['topic_distribution'] for v in data['doc_topics'].values()]
            doc_ids = list(data['doc_topics'].keys())
        # Cas liste de dicts avec 'topic_distribution'
        elif isinstance(data, list) and all(isinstance(row, dict) and 'topic_distribution' in row for row in data):
            doc_topic_matrix = [row['topic_distribution'] for row in data]
            doc_ids = list(range(len(doc_topic_matrix)))
        # Cas liste de listes déjà prête
        elif isinstance(data, list) and all(isinstance(row, list) for row in data):
            doc_topic_matrix = data
            doc_ids = list(range(len(doc_topic_matrix)))
        else:
            raise ValueError("Format de fichier non reconnu ou non supporté pour le clustering.")
    else:
        raise ValueError('Format de fichier non supporté')

    # Check if we have enough documents for clustering
    if len(doc_topic_matrix) < 2:
        print("Error: Not enough documents for clustering.")
        sys.exit(1)
        
    # Ajuster k_max si nécessaire
    if args.k_max > len(doc_topic_matrix) - 1:
        print(f"Warning: k_max ({args.k_max}) is greater than the number of documents - 1 ({len(doc_topic_matrix) - 1}).")
        args.k_max = max(args.k_min, min(args.k_max, len(doc_topic_matrix) - 1))
        print(f"Adjusted k_max to {args.k_max}")
    
    # Visualiser les métriques si demandé
    if args.visualize:
        print(f"Visualizing clustering metrics for k in range [{args.k_min}, {args.k_max}]...")
        # Créer un dossier pour les visualisations
        viz_dir = os.path.join('data', 'results', 'clusters', 'visualizations')
        os.makedirs(viz_dir, exist_ok=True)
        viz_path = os.path.join(viz_dir, f'cluster_metrics_{Path(args.input).stem}.png')
        
        # Visualiser les métriques
        metrics = visualize_cluster_metrics(doc_topic_matrix, k_min=args.k_min, k_max=args.k_max, 
                                         random_state=42, save_path=viz_path)
        print(f"Visualization saved to {viz_path}")
    
    # Déterminer le nombre de clusters à utiliser
    force_k = None
    auto_detection = False
    original_n_clusters = None
    
    # Convert n_clusters to int if it's not 'auto'
    if args.n_clusters.lower() == 'auto':
        print("Using automatic cluster detection (--n-clusters=auto)")
        auto_detection = True
        n_clusters_int = None
    else:
        try:
            n_clusters_int = int(args.n_clusters)
            original_n_clusters = n_clusters_int
            if args.force_k and n_clusters_int:
                print(f"Forcing the use of {n_clusters_int} clusters as specified")
                force_k = n_clusters_int
        except ValueError:
            print(f"Warning: Invalid value for --n-clusters: {args.n_clusters}. Using default value of 6.")
            n_clusters_int = 6
            original_n_clusters = 6
    
    # Update args.n_clusters with the integer value
    args.n_clusters = n_clusters_int
    
    # Auto-détection du nombre optimal de clusters si demandé
    if auto_detection:
        print(f"Finding optimal number of clusters using {args.metric} metric...")
        optimal_k, clusters, kmeans = find_optimal_clusters(
            doc_topic_matrix, 
            k_min=args.k_min, 
            k_max=args.k_max, 
            method=args.metric, 
            random_state=42,
            force_k=force_k
        )
        print(f"=== CLUSTERING RESULTS ====")
        print(f"Optimal number of clusters found: {optimal_k}")
        
        # Log cluster distribution
        unique_clusters, cluster_counts = np.unique(clusters, return_counts=True)
        cluster_sizes = {int(cluster): int(count) for cluster, count in zip(unique_clusters, cluster_counts)}
        print(f"Cluster distribution: {cluster_sizes}")
        print(f"Total documents: {len(clusters)}")
        
        # Mettre à jour args.n_clusters pour la sortie
        args.n_clusters = optimal_k
        print(f"Using {optimal_k} clusters for output file generation")
        print(f"==========================")
    else:
        # Vérifier si le nombre de clusters est valide
        if len(doc_topic_matrix) < args.n_clusters:
            print(f"Warning: Only {len(doc_topic_matrix)} documents available, but {args.n_clusters} clusters requested.")
            print("Reducing number of clusters to match document count.")
            args.n_clusters = max(2, len(doc_topic_matrix) // 2)
            print(f"New number of clusters: {args.n_clusters}")
        
        # Clustering standard
        print(f"Running standard clustering with k={args.n_clusters}...")
        clusters, kmeans = cluster_documents_from_matrix(doc_topic_matrix, n_clusters=args.n_clusters)
        
        # Log cluster distribution
        print(f"=== CLUSTERING RESULTS ====")
        unique_clusters, cluster_counts = np.unique(clusters, return_counts=True)
        cluster_sizes = {int(cluster): int(count) for cluster, count in zip(unique_clusters, cluster_counts)}
        print(f"Cluster distribution: {cluster_sizes}")
        print(f"Total documents: {len(clusters)}")
        print(f"Using {args.n_clusters} clusters for output file generation")
        print(f"==========================")

    # Détermination du chemin de sortie
    output_dir = os.path.join('data', 'results', 'clusters')
    os.makedirs(output_dir, exist_ok=True)
    
    # Get filter information using the utility function
    filter_summary = get_filter_summary(
        len(doc_topic_matrix),  # We don't have the original count here
        len(doc_topic_matrix),  # Same as above since we don't filter here
        start_date=args.start_date,
        end_date=args.end_date,
        newspaper=args.newspaper,
        canton=args.canton,
        topic=args.topic,
        min_words=args.min_words,
        max_words=args.max_words
    )
    
    # Add filter information to the output filename
    filter_suffix = ""
    for filter_name, filter_value in filter_summary['filters_applied'].items():
        if filter_name == 'start_date':
            filter_suffix += f"_from{filter_value}"
        elif filter_name == 'end_date':
            filter_suffix += f"_to{filter_value}"
        elif filter_name == 'newspaper':
            filter_suffix += f"_{filter_value}"
        elif filter_name == 'canton':
            filter_suffix += f"_{filter_value}"
        elif filter_name == 'topic':
            filter_suffix += f"_topic{filter_value}"
    
    # Créer deux fichiers de sortie: un avec le nombre de clusters demandé et un avec le nombre optimal
    # Store the originally requested number of clusters (for compatibility file naming)
    original_n_clusters = 'auto' if args.n_clusters == 'auto' else args.n_clusters
    requested_output_path = args.output or os.path.join(output_dir, f'doc_clusters_k{original_n_clusters}{filter_suffix}.json')
    
    # Si le chemin de sortie est spécifié explicitement, l'utiliser tel quel
    if args.output:
        output_path = args.output
    else:
        # Sinon, utiliser le chemin par défaut avec le nombre de clusters
        output_path = requested_output_path
        
        # Si auto-detection est activée et qu'on n'a pas forcé le nombre de clusters,
        # créer un fichier d'information sur le clustering optimal
        if auto_detection and not args.force_k:
            # Créer un fichier d'information sur le clustering optimal
            info_path = os.path.join(output_dir, f'optimal_clustering_info.json')
            
            # Sauvegarder les informations sur le clustering optimal
            try:
                # Préparer les informations à sauvegarder
                optimal_info = {
                    'optimal_k': args.n_clusters,
                    'original_request': original_n_clusters,
                    'metric_used': args.metric,
                    'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
                    'output_file': output_path,
                    'cluster_sizes': cluster_sizes
                }
                
                # Sauvegarder les informations
                with open(info_path, 'w', encoding='utf-8') as f:
                    json.dump(optimal_info, f, indent=2, ensure_ascii=False)
                    
                print(f"=== OPTIMAL CLUSTERING INFO =====")
                print(f"Saved optimal clustering info to: {info_path}")
                print(f"Optimal k: {args.n_clusters}")
                print(f"Original request: {original_n_clusters}")
                print(f"Output file: {output_path}")
                print(f"=================================")
            except Exception as e:
                print(f"Warning: Could not save optimal clustering info: {e}")
                
            # Ne plus créer de lien symbolique, la webapp va maintenant chercher le meilleur fichier directement
    
    # Sauvegarder les embeddings originaux (doc_topic_matrix) pour une meilleure visualisation
    result = {
        'doc_ids': doc_ids,
        'labels': clusters.tolist(),
        'cluster_centers': kmeans.cluster_centers_.tolist(),
        'n_clusters': args.n_clusters,
        'filters_applied': filter_summary['filters_applied'],
        'embeddings': doc_topic_matrix  # Ajout des embeddings originaux
    }
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(f"=== OUTPUT FILES ===")
    print(f"Main clusters file saved to: {output_path}")
    print(f"Number of clusters in file: {args.n_clusters}")
    print(f"Total documents clustered: {len(clusters)}")
    print(f"===================")
    
    # Si on a demandé un nombre spécifique de clusters mais que l'auto-détection a trouvé un nombre différent,
    # créer également un fichier avec le nombre demandé pour assurer la compatibilité
    if auto_detection and not args.force_k and original_n_clusters != args.n_clusters and not hasattr(os, 'symlink'):
        original_output_path = os.path.join(output_dir, f'doc_clusters_k{original_n_clusters}{filter_suffix}.json')
        
        # Copier le fichier si on ne peut pas créer de lien symbolique
        try:
            import shutil
            shutil.copy2(output_path, original_output_path)
            print(f"=== COMPATIBILITY FILE ====")
            print(f"Created compatibility file: {original_output_path}")
            print(f"This file contains {args.n_clusters} clusters but is named with k={original_n_clusters}")
            print(f"Purpose: Ensures visualization works when requesting {original_n_clusters} clusters")
            print(f"===========================")
        except Exception as e:
            print(f"Warning: Could not create compatibility file for requested clusters: {e}")

if __name__ == '__main__':
    main()
