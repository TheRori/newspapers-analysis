"""
Script pour effectuer un clustering KMeans sur les documents à partir de la matrice doc-topic (issue de Gensim LDA/HDP).

Usage :
    python src/scripts/run_topic_clustering.py --input data/results/doc_topic_matrix.json --n-clusters 6 --output data/results/doc_clusters.json

- Le fichier d'entrée doit être un JSON (liste de listes) ou un fichier avancé (advanced_topic_analysis*.json)
- Le script sauvegarde les labels de cluster et les centres dans un fichier JSON
"""
import argparse
import os
import json
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from analysis.topic_clustering import cluster_documents_from_matrix


def main():
    parser = argparse.ArgumentParser(description="Clustering de documents à partir d'une matrice doc-topic.")
    parser.add_argument('--input', type=str, required=True, help='Chemin du fichier JSON doc_topic_matrix ou advanced_topic_analysis')
    parser.add_argument('--n-clusters', type=int, default=6, help='Nombre de clusters KMeans')
    parser.add_argument('--output', type=str, default=None, help='Fichier de sortie pour les labels (JSON)')
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

    # Clustering
    clusters, kmeans = cluster_documents_from_matrix(doc_topic_matrix, n_clusters=args.n_clusters)

    # Détermination du chemin de sortie
    output_dir = os.path.join('data', 'results', 'clusters')
    os.makedirs(output_dir, exist_ok=True)
    output_path = args.output or os.path.join(output_dir, f'doc_clusters_k{args.n_clusters}.json')
    result = {
        'doc_ids': doc_ids,
        'labels': clusters.tolist(),
        'cluster_centers': kmeans.cluster_centers_.tolist(),
        'n_clusters': args.n_clusters
    }
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(f"Clusters sauvegardés dans {output_path}")

if __name__ == '__main__':
    main()
