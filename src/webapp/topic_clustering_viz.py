"""
Dash page for Topic Clustering visualization, inspired by topic_modeling_viz.py.
"""
import json
import os
import re
import argparse
import sys
from dash import dcc, html, Input, Output, State, dash_table, callback_context, no_update, ALL
from dash.dependencies import MATCH
import dash
import dash_bootstrap_components as dbc
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
from pathlib import Path
import pathlib
import yaml
import threading
import time
import subprocess
from src.webapp.topic_modeling_viz import get_topic_name
from src.webapp.cluster_map_viz import create_cluster_map, generate_map_coordinates
from src.webapp.export_component import (
    create_export_button,
    create_export_modal,
    create_feedback_toast,
    register_export_callbacks
)
import pandas as pd
import numpy as np

# Variable globale pour stocker le résultat de clustering actuellement sélectionné
current_selected_cluster = None

# NOUVELLE LOGIQUE : Fonction pour trouver et charger les noms de topics les plus récents
def find_and_load_latest_topic_names():
    """
    Finds the most recent topic_names_llm_*.json file and loads the names.
    """
    try:
        project_root = Path(__file__).resolve().parents[2]
        results_dir = project_root / 'data' / 'results'
        
        topic_names_files = list(results_dir.glob('topic_names_llm_*.json'))
        if not topic_names_files:
            print("Aucun fichier de noms de topics (topic_names_llm_*.json) trouvé.")
            return {}
            
        # Trouver le fichier le plus récent
        latest_file = max(topic_names_files, key=os.path.getmtime)
        print(f"Chargement des noms de topics depuis le fichier le plus récent : {latest_file}")
        
        with open(latest_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Retourner le dictionnaire de noms de topics
        return data.get('topic_names', {})
    except Exception as e:
        print(f"Erreur lors du chargement des noms de topics les plus récents : {e}")
        return {}

# Helper to load clustering results (assume similar structure to topic modeling)
def load_clustering_stats(path):
    """
    Load clustering statistics from a JSON file.
    
    Args:
        path (str): Path to the clustering results JSON file
        
    Returns:
        dict: Processed clustering statistics or None if an error occurred
    """
    print(f"[DEBUG] Début de load_clustering_stats avec path={path}")
    
    if not os.path.exists(path):
        print(f"Erreur: Le fichier {path} n'existe pas")
        return None
    
    # Nous n'utilisons plus les liens symboliques, mais nous gardons cette partie pour la compatibilité
    # avec les anciens fichiers qui pourraient être des liens symboliques
    try:
        if os.path.islink(path):
            print(f"Le fichier {path} est un lien symbolique, nous allons essayer de le résoudre")
            # Nous allons plutôt chercher le fichier directement
            if 'kauto' in path:
                # Chercher le fichier de clustering optimal
                optimal_path = find_best_cluster_file(os.path.dirname(path))
                if optimal_path:
                    print(f"Utilisation du fichier de clustering optimal: {optimal_path}")
                    path = optimal_path
                else:
                    print(f"Aucun fichier de clustering optimal trouvé, utilisation du fichier original")
    except Exception as e:
        print(f"Erreur lors de la résolution du fichier de clustering: {e}")
    
    try:
        print(f"[DEBUG] Ouverture du fichier {path}")
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        
        print(f"[DEBUG] Fichier chargé, type de données: {type(data)}")
        
        # Vérifier que le format est correct
        if not isinstance(data, dict):
            print(f"Erreur: Le fichier {path} n'est pas au format attendu (dict), type actuel: {type(data)}")
            print(f"[DEBUG] Contenu du fichier (premiers 100 caractères): {str(data)[:100]}...")
            return None
            
        # Vérifier que les clés nécessaires sont présentes
        print(f"[DEBUG] Clés présentes dans le fichier: {list(data.keys())}")
        required_keys = ['doc_ids', 'labels']
        for key in required_keys:
            if key not in data:
                print(f"Erreur: La clé '{key}' est manquante dans le fichier {path}")
                return None
        
        # Extraire les données
        doc_ids = data['doc_ids']
        labels = data['labels']
        
        print(f"[DEBUG] doc_ids: type={type(doc_ids)}, len={len(doc_ids)}")
        print(f"[DEBUG] labels: type={type(labels)}, len={len(labels)}")
        
        # Vérifier que les données sont cohérentes
        if len(doc_ids) != len(labels):
            print(f"Erreur: Le nombre de documents ({len(doc_ids)}) ne correspond pas au nombre de labels ({len(labels)})")
            return None
            
        # Calculer les statistiques de clustering
        unique_labels = set(labels)
        n_clusters = len(unique_labels)
        print(f"[DEBUG] Nombre de clusters uniques: {n_clusters}")
        
        # Calculer la taille de chaque cluster
        cluster_sizes = {}
        for label in labels:
            if label not in cluster_sizes:
                cluster_sizes[label] = 0
            cluster_sizes[label] += 1
            
        # Trier les clusters par taille
        sorted_clusters = sorted(cluster_sizes.items(), key=lambda x: x[1], reverse=True)
        print(f"[DEBUG] Tailles des clusters: {cluster_sizes}")
        
        # Préparer les statistiques
        stats = {
            'n_clusters': n_clusters,
            'n_documents': len(doc_ids),
            'cluster_sizes': cluster_sizes,
            'sorted_clusters': sorted_clusters,
            'doc_ids': doc_ids,
            'labels': labels
        }
        
        # Ajouter les centres de clusters s'ils sont disponibles
        if 'cluster_centers' in data:
            stats['cluster_centers'] = data['cluster_centers']
            print(f"[DEBUG] Centres de clusters ajoutés: {len(data['cluster_centers'])}")
            
        # Ajouter les embeddings s'ils sont disponibles
        if 'embeddings' in data:
            stats['embeddings'] = data['embeddings']
            print(f"[DEBUG] Embeddings ajoutés: {len(data['embeddings'])}")
            
        # 3. Documents représentatifs par cluster
        representative_docs = {}
        
        # Pour chaque cluster, prendre les 3 premiers documents
        for cluster in range(n_clusters):
            cluster_docs = [doc_id for doc_id, label in zip(doc_ids, labels) if label == cluster]
            representative_docs[str(cluster)] = cluster_docs[:3]  # Prendre les 3 premiers
        
        stats['representative_docs'] = representative_docs
        
        # 4. Extraire les informations temporelles des articles
        temporal_info = {}
        journal_info = {}
        for cluster in range(stats['n_clusters']):
            cluster_docs = [doc_id for doc_id, label in zip(doc_ids, labels) if label == cluster]
            dates = []
            journals = {}
            
            for doc in cluster_docs:
                if isinstance(doc, str) and '_' in doc:
                    # Format attendu: article_YYYY-MM-DD_journal_id
                    parts = doc.split('_')
                    if len(parts) >= 3 and '-' in parts[1]:
                        try:
                            # Extraire la date
                            date_str = parts[1]
                            year = int(date_str.split('-')[0])
                            month = int(date_str.split('-')[1])
                            dates.append((year, month))
                            
                            # Extraire le journal
                            journal = parts[2]
                            if journal not in journals:
                                journals[journal] = 0
                            journals[journal] += 1
                        except (ValueError, IndexError):
                            pass
            
            if dates:
                # Calculer la moyenne des années/mois
                avg_year = sum(d[0] for d in dates) / len(dates)
                avg_month = sum(d[1] for d in dates) / len(dates)
                temporal_info[cluster] = {'year': avg_year, 'month': avg_month}
            
            if journals:
                # Trouver le journal le plus fréquent
                main_journal = max(journals.items(), key=lambda x: x[1])[0]
                journal_info[cluster] = {'main_journal': main_journal, 'distribution': journals}
        
        if temporal_info:
            stats['temporal_info'] = temporal_info
        if journal_info:
            stats['journal_info'] = journal_info
        
        # 5. Centres des clusters (si disponibles)
        if 'cluster_centers' in data:
            stats['cluster_centers'] = data['cluster_centers']
            
            # Générer des "top words" fictifs pour chaque cluster basés sur les centres
            # (dans un vrai cas, il faudrait utiliser les vrais mots)
            top_words = {}
            for i, center in enumerate(data['cluster_centers']):
                # Prendre les 10 dimensions les plus importantes
                top_indices = sorted(range(len(center)), key=lambda i: center[i], reverse=True)[:10]
                words = [("Topic_" + str(idx), center[idx]) for idx in top_indices]
                top_words[str(i)] = words
            
            stats['top_words'] = top_words
        
        # 6. Essayer de charger les statistiques avancées du topic modeling
        try:
            project_root = Path(__file__).resolve().parents[2]
            advanced_topic_path = project_root / 'data' / 'results' / 'advanced_topic' / 'advanced_topic_analysis.json'
            if os.path.exists(advanced_topic_path):
                with open(advanced_topic_path, encoding="utf-8") as f:
                    advanced_stats = json.load(f)
                
                # Fusionner les informations pertinentes
                if 'topic_names_llm' in advanced_stats:
                    stats['topic_names_llm'] = advanced_stats['topic_names_llm']
                if 'coherence_score' in advanced_stats:
                    stats['coherence_score'] = advanced_stats['coherence_score']
        except Exception as e:
            print(f"Erreur lors du chargement des stats avancées: {e}")
        
        return stats
        
    except json.JSONDecodeError as e:
        print(f"Erreur: Le fichier {path} n'est pas un JSON valide: {e}")
        return None
    except Exception as e:
        import traceback
        print(f"Exception lors du chargement des stats de clustering: {str(e)}")
        print(f"[DEBUG] Traceback complet:")
        traceback.print_exc()
        return None

# Helper to find the best cluster file in a directory
def find_best_cluster_file(directory):
    """
    Find the best cluster file in a directory.
    The best file is the most recent one with a valid k number in the filename.
    
    Args:
        directory (str): Directory to search in
        
    Returns:
        str: Path to the best cluster file, or None if no valid file is found
    """
    if not os.path.exists(directory):
        print(f"Le répertoire {directory} n'existe pas")
        return None
        
    # Chercher tous les fichiers de clustering
    cluster_files = []
    for filename in os.listdir(directory):
        if filename.startswith('doc_clusters_k') and filename.endswith('.json'):
            # Extraire le numéro de cluster du nom de fichier
            try:
                # Format attendu: doc_clusters_k{number}.json ou doc_clusters_k{number}_filter.json
                k_str = filename.replace('doc_clusters_k', '').split('_')[0].split('.')[0]
                if k_str.isdigit():
                    k = int(k_str)
                    file_path = os.path.join(directory, filename)
                    cluster_files.append((file_path, k, os.path.getmtime(file_path)))
            except (ValueError, IndexError):
                pass
    
    if not cluster_files:
        print(f"Aucun fichier de clustering valide trouvé dans {directory}")
        return None
        
    # Trier par date de modification (plus récent d'abord)
    cluster_files.sort(key=lambda x: x[2], reverse=True)
    
    # Retourner le chemin du fichier le plus récent
    best_file = cluster_files[0][0]
    print(f"Meilleur fichier de clustering trouvé: {best_file} (k={cluster_files[0][1]})")
    return best_file

# Helper to render clustering stats
def render_clustering_stats(stats):
    print(f"[DEBUG] render_clustering_stats called with stats type: {type(stats)}")
    print(f"[DEBUG] Stats keys: {list(stats.keys()) if stats else 'None'}")
    
    # NOUVELLE LOGIQUE : Charger les noms de topics les plus récents de manière indépendante
    all_topic_names = find_and_load_latest_topic_names()
    
    children = []
    # 1. Score de clustering (silhouette, etc.)
    if 'silhouette_score' in stats:
        children.append(dbc.Alert(f"Score de silhouette : {stats['silhouette_score']:.3f}", color="info", className="mb-3"))
    if 'n_clusters' in stats:
        children.append(dbc.Alert(f"Nombre de clusters : {stats['n_clusters']}", color="secondary", className="mb-3"))
    if 'coherence_score' in stats:
        children.append(dbc.Alert(f"Score de cohérence des topics : {stats['coherence_score']:.3f}", color="info", className="mb-3"))
    
    # Bouton pour accéder au browser d'articles - TOUJOURS AFFICHER
    children.append(
        dbc.Button(
            "Explorer les articles par cluster", 
            id="btn-explore-articles",
            color="primary", 
            className="mb-4 mt-2"
        )
    )
    
    # Visualisation temporelle des clusters (année/mois) - PRIORITAIRE
    if 'temporal_info' in stats:
        print(f"[DEBUG] temporal_info found in stats: {len(stats['temporal_info'])} clusters")
        
        # Debug cluster_sizes
        print(f"[DEBUG] cluster_sizes type: {type(stats.get('cluster_sizes'))}")
        print(f"[DEBUG] cluster_sizes content: {stats.get('cluster_sizes')}")
        
        # Créer un DataFrame pour la visualisation temporelle avec gestion d'erreurs
        temporal_data = []
        for cluster, info in stats['temporal_info'].items():
            print(f"[DEBUG] Processing cluster {cluster}, info: {info}")
            try:
                # Convert cluster to int safely
                try:
                    cluster_int = int(cluster)
                    print(f"[DEBUG] Converted cluster {cluster} to int: {cluster_int}")
                except (ValueError, TypeError) as e:
                    print(f"[DEBUG] Could not convert cluster {cluster} to int: {e}")
                    cluster_int = 0
                
                # Get cluster size safely
                cluster_sizes = stats.get('cluster_sizes', {})
                if cluster_sizes is None:
                    print(f"[DEBUG] cluster_sizes is None!")
                    cluster_size = 10  # Default value
                else:
                    cluster_size = cluster_sizes.get(cluster_int, 10)
                    print(f"[DEBUG] Got cluster size for {cluster_int}: {cluster_size}")
                
                temporal_data.append({
                    'Cluster': f"Cluster {cluster}",
                    'Année': info['year'],
                    'Mois': info['month'],
                    'Taille': cluster_size
                })
            except Exception as e:
                import traceback
                print(f"[DEBUG] Error processing cluster {cluster}: {e}")
                traceback.print_exc()
        
        print(f"[DEBUG] Created temporal_data with {len(temporal_data)} entries")
        temporal_df = pd.DataFrame(temporal_data)
        
        # Visualisation temporelle avec noms de topics
        time_fig = px.scatter(temporal_df, x='Année', y='Mois', 
                             size='Taille', color='Cluster', hover_name='Cluster',
                             size_max=60, title="Distribution temporelle des clusters")
        time_fig.update_layout(height=600)
        children.append(dcc.Graph(figure=time_fig, id='cluster-temporal-plot'))
    
    # 2. Répartition des clusters
    if 'cluster_distribution' in stats:
        dist = stats['cluster_distribution']
        df_dist = pd.DataFrame({
            'Cluster': [f"Cluster {i}" for i in range(len(dist))],
            'Proportion': dist
        })
        children.append(dcc.Graph(
            figure=px.bar(df_dist, x='Cluster', y='Proportion', title='Distribution des clusters', text_auto='.2f'),
            id='cluster-distribution-plot'
        ))
    
    # Visualisation des journaux par cluster
    if 'journal_info' in stats:
        # Créer un DataFrame pour la visualisation des journaux
        journal_data = []
        for cluster, info in stats['journal_info'].items():
            for journal, count in info['distribution'].items():
                journal_data.append({
                    'Cluster': f"Cluster {cluster}",
                    'Journal': journal,
                    'Articles': count
                })
        
        if journal_data:
            journal_df = pd.DataFrame(journal_data)
            journal_fig = px.bar(journal_df, x='Cluster', y='Articles', 
                                color='Journal', barmode='stack',
                                title="Distribution des journaux par cluster")
            journal_fig.update_layout(height=500)
            children.append(dcc.Graph(figure=journal_fig, id='cluster-journal-plot'))
    
    # NOUVELLE LOGIQUE : La condition pour la heatmap ne dépend plus de `stats['topic_names_llm']`
    if 'cluster_centers' in stats:
        try:
            print("[DEBUG] Démarrage de la visualisation de la heatmap")
            centers = stats['cluster_centers']
            
            heatmap_data = []
            
            for i, center in enumerate(centers):
                for j, weight in enumerate(center):
                    topic_id = j
                    # Utiliser les noms de topics chargés indépendamment
                    topic_name = get_topic_name(topic_id, all_topic_names, default=f"Topic {topic_id}")
                    
                    heatmap_data.append({
                        'Cluster': f"Cluster {i}",
                        'Topic': topic_name,
                        'Poids': weight
                    })
            
            if heatmap_data:
                heatmap_df = pd.DataFrame(heatmap_data)
                heatmap_fig = px.density_heatmap(
                    heatmap_df, x='Cluster', y='Topic', z='Poids',
                    title="Poids des topics dans chaque cluster",
                    color_continuous_scale='Viridis'
                )
                
                num_topics = len(heatmap_df['Topic'].unique())
                dynamic_height = max(600, 400 + num_topics * 30)
                
                unique_topics = heatmap_df['Topic'].unique()
                topic_indices = {topic: int(re.search(r'\d+', topic).group()) if re.search(r'\d+', topic) else i for i, topic in enumerate(unique_topics)}
                sorted_topics = sorted(unique_topics, key=lambda x: topic_indices.get(x, 0))
                
                heatmap_fig.update_layout(
                    height=dynamic_height,
                    margin=dict(l=250, r=50, t=50, b=100),
                    yaxis=dict(
                        tickmode='array',
                        tickvals=sorted_topics, # Utiliser directement les noms pour les tickvals
                        ticktext=sorted_topics,
                        tickfont=dict(size=10),
                        title_font=dict(size=12),
                        autorange="reversed" # Assure que le topic 0 est en haut
                    )
                )
                children.append(dcc.Graph(figure=heatmap_fig, id='cluster-heatmap-plot'))

        except Exception as e:
            import traceback
            print(f"Erreur lors de la création de la heatmap: {str(e)}")
            print(traceback.print_exc())

    # Store pour stocker les données des clusters
    children.append(dcc.Store(id='cluster-data-store', data=stats))
    
    return html.Div(children)


# Page de navigation des articles par cluster
def get_article_browser_layout(cluster_data=None):
    if not cluster_data or 'labels' not in cluster_data or 'doc_ids' not in cluster_data:
        return html.Div([
            html.H3("Explorateur d'articles par cluster"),
            dbc.Alert("Aucune donnée de clustering disponible. Veuillez d'abord lancer un clustering.", color="warning"),
            dbc.Button("Retour aux visualisations", id="btn-back-to-viz", color="secondary")
        ])
    
    # Créer un dictionnaire cluster -> liste d'articles
    cluster_articles = {}
    if 'labels' in cluster_data and 'doc_ids' in cluster_data:
        for doc_id, label in zip(cluster_data['doc_ids'], cluster_data['labels']):
            if label not in cluster_articles:
                cluster_articles[label] = []
            cluster_articles[label].append(doc_id)
    
    # Vérifier si nous avons des clusters
    if not cluster_articles:
        return html.Div([
            html.H3("Explorateur d'articles par cluster"),
            dbc.Alert("Aucun cluster trouvé dans les données. Veuillez relancer le clustering.", color="warning"),
            dbc.Button("Retour aux visualisations", id="btn-back-to-viz", color="secondary")
        ])
    
    # Créer les onglets pour chaque cluster
    tabs = []
    article_modal = dbc.Modal(
        [
            dbc.ModalHeader(dbc.ModalTitle("Contenu de l'article")),
            dbc.ModalBody(id="article-details-body"),
            dbc.ModalFooter(
                dbc.Button("Fermer", id="close-article-modal", className="ms-auto", n_clicks=0)
            ),
        ],
        id="article-modal",
        is_open=False,
        size="xl",
    )
    for cluster in sorted(cluster_articles.keys()):
        # Préparer les données pour la table
        table_data = []
        for article_id in cluster_articles[cluster]:
            # Extraire les informations de l'ID
            date = "N/A"
            journal = "N/A"
            parts = article_id.split("_")
            if len(parts) > 1:
                date = parts[1]
            if len(parts) > 2:
                journal = parts[2]
            
            table_data.append({
                'article_id': article_id,
                'date': date,
                'journal': journal
            })
        
        # Créer un ID unique pour chaque table de cluster
        cluster_id = f"cluster-{cluster}"
        
        # Limiter le nombre d'articles affichés à la fois (pagination)
        items_per_page = 20
        
        # Créer la table avec des boutons explicites pour chaque article
        table = html.Div([
            # Pagination controls
            html.Div([
                dbc.Pagination(
                    id={'type': 'cluster-pagination', 'index': cluster_id},
                    max_value=max(1, (len(table_data) + items_per_page - 1) // items_per_page),
                    first_last=True,
                    previous_next=True,
                    active_page=1,
                    className="mt-3 mb-3 justify-content-center"
                )
            ]),
            # Table container with fixed height for better performance
            html.Div([
                html.Table([
                    # En-tête de la table
                    html.Thead(
                        html.Tr([
                            html.Th("ID Article", style={'width': '40%', 'textAlign': 'left', 'padding': '10px', 'backgroundColor': '#e6e6e6', 'fontWeight': 'bold'}),
                            html.Th("Date", style={'width': '20%', 'textAlign': 'left', 'padding': '10px', 'backgroundColor': '#e6e6e6', 'fontWeight': 'bold'}),
                            html.Th("Journal", style={'width': '20%', 'textAlign': 'left', 'padding': '10px', 'backgroundColor': '#e6e6e6', 'fontWeight': 'bold'}),
                            html.Th("Action", style={'width': '20%', 'textAlign': 'center', 'padding': '10px', 'backgroundColor': '#e6e6e6', 'fontWeight': 'bold'})
                        ])
                    ),
                    # Corps de la table (limité aux premiers articles)
                    html.Tbody([
                        html.Tr([
                            html.Td(row['article_id'], style={'padding': '10px', 'borderBottom': '1px solid #ddd'}),
                            html.Td(row['date'], style={'padding': '10px', 'borderBottom': '1px solid #ddd'}),
                            html.Td(row['journal'], style={'padding': '10px', 'borderBottom': '1px solid #ddd'}),
                            html.Td(
                                dbc.Button(
                                    "Voir", 
                                    id={'type': 'view-article-btn', 'index': row['article_id']},
                                    color="primary",
                                    size="sm",
                                    className="me-1",
                                    n_clicks=0
                                ),
                                style={'textAlign': 'center', 'padding': '10px', 'borderBottom': '1px solid #ddd'}
                            )
                        ]) for row in table_data[:items_per_page]  # Afficher seulement la première page
                    ], id={'type': 'cluster-table-body', 'index': cluster_id})
                ], className="table table-hover", style={'width': '100%', 'borderCollapse': 'collapse'})
            ], style={'overflowX': 'auto', 'maxHeight': '600px', 'overflowY': 'auto'}),
            # Store pour les données de la table complète
            dcc.Store(id={'type': 'cluster-table-data', 'index': cluster_id}, data=table_data)
        ])
        
        # Créer l'onglet pour ce cluster
        tabs.append(
            dbc.Tab(
                [
                    html.H4(f"Cluster {cluster} - {len(cluster_articles[cluster])} articles", className="mt-3"),
                    table,
                    html.Div(id=f'cluster-{cluster}-article-content', className="mt-3")
                ],
                label=f"Cluster {cluster}",
                tab_id=f"tab-cluster-{cluster}"
            )
        )
    
    # Déterminer l'onglet actif (avec gestion d'erreur)
    active_tab = f"tab-cluster-{min(cluster_articles.keys())}" if cluster_articles else None
    
    # Créer un Store pour stocker l'ID de l'article sélectionné
    article_id_store = dcc.Store(id='selected-article-id-store')
    
    # Ajouter le Store et la modale à la mise en page
    return html.Div([
        html.H3("Explorateur d'articles par cluster"),
        dbc.Button("Retour aux visualisations", id="btn-back-to-viz", color="secondary", className="mb-3"),
        dbc.Tabs(tabs, id="cluster-tabs", active_tab=active_tab) if tabs else html.Div(),
        article_modal,
        # Store pour stocker les données des clusters
        dcc.Store(id='browser-cluster-data-store', data=cluster_data),
        # Store pour l'ID de l'article sélectionné
        article_id_store
    ])

# Dash parser extraction (comme topic_modeling_viz)
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
    return parser

def get_clustering_args():
    parser = get_parser()
    parser_args = []
    for action in parser._actions:
        if action.dest == 'help':
            continue
        # Set default for --input to doc_topic_matrix.json in data/results
        default = action.default
        if action.dest == 'input':
            project_root = Path(__file__).resolve().parents[2]
            default = str(project_root / 'data' / 'results' / 'doc_topic_matrix.json')
            
        # Déterminer le type de l'argument
        arg_type = 'str'  # Type par défaut
        if hasattr(action, 'type') and action.type is not None:
            arg_type = action.type.__name__ if hasattr(action.type, '__name__') else str(action.type)
        elif isinstance(action, argparse._StoreAction) and action.nargs == 0:
            arg_type = 'bool'
        elif isinstance(action, argparse._StoreTrueAction) or isinstance(action, argparse._StoreFalseAction):
            arg_type = 'bool'
            default = False if isinstance(action, argparse._StoreTrueAction) else True
            
        arg = {
            'name': action.dest,
            'help': action.help,
            'default': default,
            'type': arg_type,
            'required': action.required,
        }
        parser_args.append(arg)
    return parser_args

from dash import html as _html

def generate_dash_controls_for_clustering(args_list):
    controls = []
    for arg in args_list:
        name = arg['name']
        label = arg['help'] or name
        default = arg['default']
        required = arg['required']
        typ = arg['type']
        
        # Special handling for n-clusters parameter
        if name == 'n_clusters':
            # Create a dropdown with numeric options and 'auto' option
            control = dcc.Dropdown(
                id=f'input-{name}',
                options=[
                    {'label': 'Auto (détection automatique)', 'value': 'auto'},
                    {'label': '2 clusters', 'value': '2'},
                    {'label': '3 clusters', 'value': '3'},
                    {'label': '4 clusters', 'value': '4'},
                    {'label': '5 clusters', 'value': '5'},
                    {'label': '6 clusters', 'value': '6'},
                    {'label': '8 clusters', 'value': '8'},
                    {'label': '10 clusters', 'value': '10'},
                    {'label': '12 clusters', 'value': '12'},
                    {'label': '15 clusters', 'value': '15'},
                    {'label': '20 clusters', 'value': '20'},
                ],
                value=default,
                clearable=False
            )
        # String, int, bool
        elif typ == 'int':
            control = dbc.Input(id=f'input-{name}', type='number', value=default, placeholder=label, required=required, min=1)
        elif typ == 'bool':
            control = dbc.Checkbox(id=f'input-{name}', value=default, className='ms-2')
        else:
            control = dbc.Input(id=f'input-{name}', type='text', value=default if default is not None else '', placeholder=label, required=required)
        
        # Utiliser la nouvelle structure recommandée (Row + Col) au lieu de FormGroup
        controls.append(
            dbc.Row([
                dbc.Col(dbc.Label(label, html_for=f'input-{name}'), width=4),
                dbc.Col(control, width=8)
            ], className="mb-3")
        )
    return controls

# Layout for clustering page (dynamic controls + results)
def get_clustering_layout():
    # Charger les arguments du script
    args_list = get_clustering_args()
    
    # Récupérer la liste des fichiers de clustering disponibles
    project_root = Path(__file__).resolve().parents[2]
    clusters_dir = os.path.join(project_root, 'data', 'results', 'clusters')
    cluster_files = []
    if os.path.exists(clusters_dir):
        cluster_files = [f for f in os.listdir(clusters_dir) if f.startswith('doc_clusters_k')]
        # Trier par date de modification (le plus récent d'abord)
        cluster_files.sort(key=lambda x: os.path.getmtime(os.path.join(clusters_dir, x)), reverse=True)
    
    return dbc.Container([
        dbc.Row([
            dbc.Col([
                html.H2("Analyse de Clustering Thématique", className="text-center mb-2"),
                dbc.Alert([
                    html.H4("Qu'est-ce que le clustering thématique ?", className="alert-heading"),
                    html.P(
                        "Cette page regroupe les articles en clusters (groupes) en se basant sur leurs similarités thématiques. "
                        "L'analyse utilise les résultats de la modélisation de topics pour identifier des ensembles de documents qui traitent de sujets similaires. "
                        "Vous pouvez choisir le nombre de clusters ou laisser l'algorithme le déterminer automatiquement. "
                        "Les visualisations produites incluent une carte interactive des clusters, des graphiques montrant la répartition temporelle et par journal, "
                        "ainsi qu'une heatmap détaillant la composition thématique de chaque cluster. Un explorateur d'articles est également disponible pour examiner le contenu de chaque groupe."
                    )
                ], color="info", className="mb-4"),
                
                # Bouton d'exportation pour la médiation
                create_export_button("clustering", button_id="export-clustering-button"),
                
                # Système d'onglets
                dbc.Tabs([
                    # Onglet Paramètres
                    dbc.Tab([
                        dbc.Card([
                            dbc.CardHeader(html.H3("Paramètres du Clustering", className="mb-0")),
                            dbc.CardBody([
                                html.P("Configurez les paramètres du clustering ci-dessous, puis cliquez sur 'Lancer'.", className="text-muted mb-3"),
                                html.P("Pour détecter automatiquement le nombre optimal de clusters, sélectionnez 'Auto' dans le menu déroulant.", className="text-info mb-3"),
                                html.Div(generate_dash_controls_for_clustering(args_list), id="clustering-controls"),
                                dbc.Button("Lancer le Clustering", id="btn-run-clustering", color="primary", className="mt-3"),
                                dbc.Button("Voir la Carte des Clusters", id="btn-view-cluster-map", color="success", className="mt-3 ms-3"),
                            ])
                        ], className="mb-4")
                    ], label="Paramètres", tab_id="tab-params"),
                    
                    # Onglet Résultats
                    dbc.Tab([
                        dbc.Card([
                            dbc.CardHeader(html.H3("Résultats du Clustering", className="mb-0")),
                            dbc.CardBody([
                                html.P("Sélectionnez un fichier de résultats de clustering pour visualiser les analyses.", className="text-muted mb-3"),
                                dbc.Row([
                                    dbc.Col([
                                        dbc.Label("Fichier de résultats"),
                                        dcc.Dropdown(
                                            id="results-file-dropdown",
                                            options=[{"label": f, "value": f} for f in cluster_files],
                                            value=cluster_files[0] if cluster_files else None,
                                            clearable=False
                                        )
                                    ], width=6),
                                    dbc.Col([
                                        dbc.Button(
                                            "Charger les résultats", 
                                            id="btn-load-results", 
                                            color="primary", 
                                            className="mt-4"
                                        )
                                    ], width=6)
                                ]),
                                html.Div(id="results-stats-output", className="mt-4")
                            ])
                        ], className="mb-4")
                    ], label="Résultats", tab_id="tab-results")
                ], id="clustering-tabs", active_tab="tab-params"),
                
                # Résultats du clustering (pour l'onglet Paramètres)
                html.Div(id="clustering-stats-output"),
                
                # Container pour l'explorateur d'articles (initialement caché)
                html.Div(id="article-browser-container", style={"display": "none"}),
                
                # Store pour les données de clustering
                dcc.Store(id="cluster-data-store"),
                # Store pour les données de résultats chargés
                dcc.Store(id="results-data-store"),
                # Store pour stocker le résultat de clustering sélectionné
                dcc.Store(id="clustering-selected-result-store", data=None),
                
                # Composants pour l'exportation
                create_export_modal("clustering", modal_id="export-clustering-modal"),
                create_feedback_toast("export-clustering-feedback-toast")
            ], width=12)
        ])
    ])

# Callback registration for clustering (to be called from app.py)
def get_clustering_source_data():
    """
    Récupère les données source pour l'exportation du clustering.
    Utilise le résultat de clustering actuellement sélectionné dans le dropdown.
    
    Returns:
        dict: Données source pour l'exportation
    """
    global current_selected_cluster
    project_root = Path(__file__).resolve().parents[2]
    
    # Chemins vers les répertoires
    doc_topic_matrix_path = project_root / 'data' / 'results' / 'doc_topic_matrix'
    advanced_analysis_path = project_root / 'data' / 'results' / 'advanced_analysis'
    clusters_dir = project_root / 'data' / 'results' / 'clusters'
    
    # Initialiser les fichiers à None
    doc_topic_file = None
    advanced_analysis_file = None
    cluster_file = None
    
    # Si un résultat de clustering est sélectionné
    if current_selected_cluster:
        try:
            # Utiliser le fichier de clustering sélectionné
            cluster_file = current_selected_cluster
            
            # Extraire l'ID du modèle à partir du nom du fichier
            file_name = os.path.basename(cluster_file)
            
            # Format typique: doc_clusters_k5_gensim_lda_20250609-191600_08f53591.json
            match = re.search(r'doc_clusters_k\d+_(\w+)_(\d{8}-\d{6})_(\w+)\.json', file_name)
            if match:
                model_type = match.group(1)
                timestamp = match.group(2)
                model_id = match.group(3)
                
                # Trouver le fichier de matrice document-topic correspondant
                doc_topic_file_path = doc_topic_matrix_path / f"doc_topic_matrix_{model_type}_{timestamp}_{model_id}.csv"
                if doc_topic_file_path.exists():
                    doc_topic_file = str(doc_topic_file_path)
                
                # Trouver le fichier d'analyse avancée correspondant
                advanced_analysis_file_path = advanced_analysis_path / f"advanced_analysis_{model_type}_{timestamp}_{model_id}.json"
                if advanced_analysis_file_path.exists():
                    advanced_analysis_file = str(advanced_analysis_file_path)
            else:
                # Si le format ne correspond pas, utiliser les fichiers les plus récents
                print(f"Format de nom de fichier de clustering non reconnu: {file_name}. Utilisation des fichiers les plus récents.")
                
                # Fichier de matrice document-topic
                doc_topic_files = list(doc_topic_matrix_path.glob('*.csv'))
                if doc_topic_files:
                    doc_topic_file = str(max(doc_topic_files, key=os.path.getmtime))
                
                # Fichier d'analyse avancée
                advanced_analysis_files = list(advanced_analysis_path.glob('*.json'))
                if advanced_analysis_files:
                    advanced_analysis_file = str(max(advanced_analysis_files, key=os.path.getmtime))
        except Exception as e:
            print(f"Erreur lors de la récupération des fichiers source pour le clustering: {e}")
    else:
        # Si aucun résultat n'est sélectionné, utiliser les fichiers les plus récents
        # Fichier de matrice document-topic
        doc_topic_files = list(doc_topic_matrix_path.glob('*.csv'))
        if doc_topic_files:
            doc_topic_file = str(max(doc_topic_files, key=os.path.getmtime))
        
        # Fichier d'analyse avancée
        advanced_analysis_files = list(advanced_analysis_path.glob('*.json'))
        if advanced_analysis_files:
            advanced_analysis_file = str(max(advanced_analysis_files, key=os.path.getmtime))
        
        # Fichier de clustering
        cluster_files = list(clusters_dir.glob('doc_clusters_k*.json'))
        if cluster_files:
            cluster_file = str(max(cluster_files, key=os.path.getmtime))
    
    return {
        "doc_topic_matrix_file": doc_topic_file,
        "advanced_analysis_file": advanced_analysis_file,
        "results_file": cluster_file
    }

def get_clustering_figure():
    """
    Récupère la figure actuelle pour l'exportation du clustering.
    
    Returns:
        dict: Données de la figure pour l'exportation
    """
    # Pour le clustering, nous utilisons la figure de la carte des clusters si disponible
    try:
        from src.webapp.cluster_map_viz import get_current_cluster_map
        figure = get_current_cluster_map()
        if figure:
            return figure
    except Exception as e:
        print(f"Erreur lors de la récupération de la figure de clustering: {e}")
    
    # Si pas de figure disponible, retourner None
    return None

def register_clustering_callbacks(app):
    import threading
    import time
    import json
    import pathlib
    import yaml
    import os
    from dash.dependencies import Input, Output, State, ALL
    from dash import callback_context, no_update
    import subprocess
    
    # Enregistrer les callbacks d'exportation pour le clustering
    register_export_callbacks(
        app,
        analysis_type="clustering",
        get_source_data_function=get_clustering_source_data,
        get_figure_function=get_clustering_figure,
        button_id="export-clustering-button",
        modal_id="export-clustering-modal",
        toast_id="export-clustering-feedback-toast"
    )
    
    @app.callback(
        [Output("clustering-stats-output", "children"),
         Output("cluster-data-store", "data"),
         Output("clustering-tabs", "active_tab")],
        [Input("btn-run-clustering", "n_clicks")],
        [State(f"input-{arg['name']}", "value") for arg in get_clustering_args()]
    )
    def run_clustering(n_clicks, *args):
        if not n_clicks:
            return [], None, no_update
        
        # Récupérer les arguments
        arg_list = get_clustering_args()
        arg_values = {arg['name']: val for arg, val in zip(arg_list, args)}
        
        # Construire la commande
        script_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "scripts", "run_topic_clustering.py")
        cmd = [sys.executable, script_path]
        for arg, val in arg_values.items():
            if val is None or val == '':
                continue
                
            # Vérifier le type de l'argument
            arg_info = next((a for a in arg_list if a['name'] == arg), None)
            if not arg_info:
                continue
                
            # Gérer les arguments booléens (flags)
            if arg_info['type'] == 'bool':
                # Pour les booléens, on ajoute juste le flag si la valeur est True
                if val in [True, 'true', 'True', 1, '1']:
                    cmd.append(f"--{arg.replace('_','-')}")
            else:
                # Pour les autres types, on ajoute le nom de l'argument et sa valeur
                cmd.append(f"--{arg.replace('_','-')}")
                cmd.append(str(val))
        # Threaded execution to avoid blocking Dash
        result_holder = {'output': None, 'data': None}
        def run_subprocess():
            try:
                # Afficher la commande qui va être exécutée
                print(f"Exécution du clustering: {sys.executable} {script_path} {' '.join(str(c) for c in cmd[2:])}")
                result = subprocess.run(cmd, capture_output=True, text=True, check=True)
                print(f"Résultat du clustering: {result.stdout}")
                
                # Charger les résultats du clustering
                project_root = Path(__file__).resolve().parents[2]
                
                # Récupérer le chemin de sortie du script de clustering
                output_arg = arg_values.get('output')
                n_clusters = arg_values.get('n_clusters', '6')
                
                # Handle 'auto' value for n_clusters
                if n_clusters == 'auto':
                    # When using auto, we'll need to find the actual number of clusters from the output files
                    print("Using automatic cluster detection (--n-clusters=auto)")
                
                if output_arg:
                    # Si un chemin de sortie est spécifié explicitement
                    if not os.path.isabs(output_arg):
                        output_path = os.path.join(project_root, output_arg)
                    else:
                        output_path = output_arg
                else:
                    # Utiliser le chemin par défaut du script de clustering
                    clusters_dir = os.path.join(project_root, 'data', 'results', 'clusters')
                    
                    if n_clusters == 'auto':
                        # For auto clustering, we need to find the most recent cluster file
                        if os.path.exists(clusters_dir):
                            # First try to find a specific auto file
                            auto_file = os.path.join(clusters_dir, 'doc_clusters_kauto.json')
                            if os.path.exists(auto_file):
                                output_path = auto_file
                                print(f"Auto clustering: Using dedicated auto file: {output_path}")
                            else:
                                # Otherwise look for any cluster file
                                cluster_files = [f for f in os.listdir(clusters_dir) if f.startswith('doc_clusters_k')]
                                if cluster_files:
                                    # Sort by modification time (most recent first)
                                    cluster_files.sort(key=lambda x: os.path.getmtime(os.path.join(clusters_dir, x)), reverse=True)
                                    output_path = os.path.join(clusters_dir, cluster_files[0])
                                    print(f"Auto clustering: Using most recent cluster file: {output_path}")
                                else:
                                    # Fallback if no files found
                                    output_path = os.path.join(clusters_dir, 'doc_clusters_kauto.json')
                        else:
                            output_path = os.path.join(clusters_dir, 'doc_clusters_kauto.json')
                    else:
                        # For specific number of clusters
                        output_path = os.path.join(clusters_dir, f'doc_clusters_k{n_clusters}.json')
                
                print(f"Recherche du fichier de résultats: {output_path}")
                if os.path.exists(output_path):
                    print(f"Fichier trouvé, chargement des stats...")
                    try:
                        # Essayer de charger le fichier directement
                        stats = load_clustering_stats(output_path)
                        
                        # Si le chargement a échoué et que c'est un mode auto, chercher le meilleur fichier de clustering
                        if stats is None and n_clusters == 'auto':
                            # Chercher le fichier de clustering optimal
                            clusters_dir = os.path.dirname(output_path)
                            best_file = find_best_cluster_file(clusters_dir)
                            if best_file:
                                print(f"Tentative de chargement du meilleur fichier de clustering: {best_file}")
                                stats = load_clustering_stats(best_file)
                        
                        if stats:
                            result_holder['output'] = render_clustering_stats(stats)
                            result_holder['data'] = stats
                            result_holder['tab'] = "tab-params"
                        else:
                            result_holder['output'] = dbc.Alert("Fichier de clustering trouvé mais impossible de charger les données.", color="warning")
                            result_holder['data'] = None
                            result_holder['tab'] = "tab-params"
                    except Exception as e:
                        print(f"Exception lors du chargement des stats de clustering: {e}")
                        result_holder['output'] = dbc.Alert(f"Erreur lors du chargement des données de clustering: {str(e)}", color="danger")
                        result_holder['data'] = None
                        result_holder['tab'] = "tab-params"
                else:
                    # Si le chemin spécifié n'existe pas, essayons de trouver le fichier dans le dossier clusters
                    clusters_dir = os.path.join(project_root, 'data', 'results', 'clusters')
                    print(f"Fichier non trouvé, recherche dans: {clusters_dir}")
                    if os.path.exists(clusters_dir):
                        # Chercher le fichier le plus récent
                        cluster_files = [f for f in os.listdir(clusters_dir) if f.startswith('doc_clusters_k')]
                        if cluster_files:
                            # Trier par date de modification (le plus récent d'abord)
                            cluster_files.sort(key=lambda x: os.path.getmtime(os.path.join(clusters_dir, x)), reverse=True)
                            latest_file = os.path.join(clusters_dir, cluster_files[0])
                            print(f"Fichier le plus récent trouvé: {latest_file}")
                            stats = load_clustering_stats(latest_file)
                            if stats:
                                result_holder['output'] = render_clustering_stats(stats)
                                result_holder['data'] = stats
                                result_holder['tab'] = "tab-params"
                            else:
                                result_holder['output'] = dbc.Alert(f"Impossible de charger les données depuis {latest_file}", color="warning")
                                result_holder['data'] = None
                                result_holder['tab'] = "tab-params"
                        else:
                            result_holder['output'] = dbc.Alert("Aucun fichier de clustering trouvé dans le dossier clusters.", color="warning")
                            result_holder['data'] = None
                            result_holder['tab'] = "tab-params"
                    else:
                        result_holder['output'] = dbc.Alert(f"Dossier de clustering non trouvé: {clusters_dir}", color="warning")
                        result_holder['data'] = None
                        result_holder['tab'] = "tab-params"
            except subprocess.CalledProcessError as e:
                print(f"Erreur lors de l'exécution du clustering: {e}")
                result_holder['output'] = dbc.Alert(f"Erreur lors de l'exécution : {e.stderr}", color="danger")
                result_holder['data'] = None
                result_holder['tab'] = "tab-params"
            except Exception as e:
                print(f"Exception lors du clustering: {e}")
                result_holder['output'] = dbc.Alert(f"Exception: {str(e)}", color="danger")
                result_holder['data'] = None
                result_holder['tab'] = "tab-params"
        thread = threading.Thread(target=run_subprocess)
        thread.start()
        # Attendre la fin du thread (ou afficher un loading...)
        while thread.is_alive():
            time.sleep(0.2)
        # S'assurer que la clé 'tab' existe dans result_holder
        if 'tab' not in result_holder:
            result_holder['tab'] = "tab-params"
        return result_holder['output'], result_holder['data'], result_holder['tab']
    
    # Callback pour mettre à jour la variable globale current_selected_cluster
    @app.callback(
        Output("clustering-selected-result-store", "data"),
        [Input("results-file-dropdown", "value")]
    )
    def update_selected_cluster(selected_file):
        global current_selected_cluster
        if selected_file:
            # Construire le chemin complet vers le fichier de résultats
            project_root = Path(__file__).resolve().parents[2]
            current_selected_cluster = os.path.join(project_root, 'data', 'results', 'clusters', selected_file)
        else:
            current_selected_cluster = None
        return selected_file
    
    # CORRECTION : Callback pour charger les résultats depuis un fichier existant
    @app.callback(
        [Output("results-stats-output", "children"),
         Output("results-data-store", "data")],
        [Input("btn-load-results", "n_clicks")],
        [State("results-file-dropdown", "value")]
    )
    def load_results_file(n_clicks, selected_file):
        if not n_clicks or not selected_file:
            return no_update, no_update
        
        # Construire le chemin complet vers le fichier de résultats
        project_root = Path(__file__).resolve().parents[2]
        file_path = os.path.join(project_root, 'data', 'results', 'clusters', selected_file)
        
        if os.path.exists(file_path):
            try:
                # Charger les statistiques du fichier
                stats = load_clustering_stats(file_path)
                if stats:
                    # Préparer les éléments à afficher
                    output_elements = []
                    
                    # CORRECTION : Charger les noms de topics les plus récents
                    all_topic_names = find_and_load_latest_topic_names()
                    
                    # 1. Générer la carte des clusters
                    if 'map_coordinates' in stats:
                        df = pd.DataFrame(stats['map_coordinates'])
                    elif 'embeddings' in stats and 'labels' in stats and 'doc_ids' in stats:
                        print(f"Génération des coordonnées de la carte à partir des embeddings pour {len(stats['doc_ids'])} documents")
                        df = generate_map_coordinates(stats, method='tsne')
                        
                        if df.empty:
                            print("Tentative de génération des coordonnées avec PCA (méthode de secours)")
                            from sklearn.decomposition import PCA
                            embeddings = np.array(stats['embeddings'])
                            if len(embeddings) > 0:
                                pca = PCA(n_components=2, random_state=42)
                                coords = pca.fit_transform(embeddings)
                                df = pd.DataFrame({
                                    'doc_id': stats['doc_ids'], 'cluster': stats['labels'],
                                    'x': coords[:, 0], 'y': coords[:, 1],
                                    'is_anomaly': [False] * len(stats['doc_ids']),
                                    'intensity': [0.8] * len(stats['doc_ids']),
                                    'anomaly_reason': [""] * len(stats['doc_ids'])
                                })
                    else:
                        return dbc.Alert("Le fichier de clustering ne contient pas les données nécessaires (embeddings, labels ou doc_ids).", color="warning"), None
                    
                    if not df.empty:
                        fig = create_cluster_map(df)
                        output_elements.append(html.H3("Carte des clusters", className="mt-4"))
                        output_elements.append(dcc.Graph(figure=fig, config={'displayModeBar': True}, style={"height": "600px"}))
                    
                    # 2. Générer la heatmap de répartition des topics
                    # CORRECTION : La condition ne dépend plus de `stats['topic_names_llm']`
                    if 'cluster_centers' in stats:
                        try:
                            centers = stats['cluster_centers']
                            heatmap_data = []
                            
                            for i, center in enumerate(centers):
                                for j, weight in enumerate(center):
                                    topic_id = j
                                    # CORRECTION : Appel correct à get_topic_name avec le dictionnaire de noms
                                    topic_name = get_topic_name(topic_id, all_topic_names, default=f"Topic {topic_id}")
                                    
                                    heatmap_data.append({
                                        'Cluster': f"Cluster {i}",
                                        'Topic': topic_name,
                                        'Poids': weight
                                    })
                            
                            if heatmap_data:
                                heatmap_df = pd.DataFrame(heatmap_data)
                                heatmap_fig = px.density_heatmap(
                                    heatmap_df, x='Cluster', y='Topic', z='Poids',
                                    title="Poids des topics dans chaque cluster",
                                    color_continuous_scale='Viridis'
                                )
                                
                                num_topics = len(heatmap_df['Topic'].unique())
                                dynamic_height = max(600, 400 + num_topics * 30)
                                
                                unique_topics = heatmap_df['Topic'].unique()
                                topic_indices = {topic: int(re.search(r'\d+', topic).group()) if re.search(r'\d+', topic) else i for i, topic in enumerate(unique_topics)}
                                sorted_topics = sorted(unique_topics, key=lambda x: topic_indices.get(x, 0))
                                
                                heatmap_fig.update_layout(
                                    height=dynamic_height,
                                    margin=dict(l=250, r=50, t=50, b=100),
                                    yaxis=dict(
                                        tickmode='array',
                                        tickvals=sorted_topics,
                                        ticktext=sorted_topics,
                                        tickfont=dict(size=10),
                                        title_font=dict(size=12)
                                    )
                                )
                                
                                output_elements.append(html.H3("Répartition des topics", className="mt-4"))
                                output_elements.append(dcc.Graph(figure=heatmap_fig, id='cluster-heatmap-plot'))
                        except Exception as e:
                            import traceback
                            print(f"Erreur lors de la création de la heatmap: {str(e)}")
                            print(traceback.format_exc())
                    
                    return html.Div(output_elements), stats
                else:
                    return dbc.Alert("Impossible de charger les données du fichier sélectionné.", color="warning"), None
            except Exception as e:
                import traceback
                print(f"Erreur lors du chargement ou de la génération des visualisations: {str(e)}")
                print(traceback.format_exc())
                return dbc.Alert(f"Erreur lors du traitement du fichier: {str(e)}", color="danger"), None
        else:
            return dbc.Alert(f"Le fichier {selected_file} n'existe pas.", color="warning"), None
    
    # Callback pour mettre à jour le dropdown des fichiers de résultats
    @app.callback(
        Output("results-file-dropdown", "options"),
        [Input("clustering-tabs", "active_tab")]
    )
    def update_results_dropdown(active_tab):
        if active_tab != "tab-results":
            return no_update
            
        # Récupérer la liste des fichiers de clustering disponibles
        project_root = Path(__file__).resolve().parents[2]
        clusters_dir = os.path.join(project_root, 'data', 'results', 'clusters')
        cluster_files = []
        if os.path.exists(clusters_dir):
            cluster_files = [f for f in os.listdir(clusters_dir) if f.startswith('doc_clusters_k')]
            # Trier par date de modification (le plus récent d'abord)
            cluster_files.sort(key=lambda x: os.path.getmtime(os.path.join(clusters_dir, x)), reverse=True)
        
        return [{"label": f, "value": f} for f in cluster_files]
    
    # Callback pour détecter les clics sur les boutons "Voir" des articles
    @app.callback(
        Output('selected-article-id-store', 'data', allow_duplicate=True),
        [Input({'type': 'view-article-btn', 'index': ALL}, 'n_clicks')],
        prevent_initial_call=True
    )
    def handle_article_button_click(n_clicks_list):
        ctx = callback_context
        if not ctx.triggered:
            return no_update
        
        # Vérifier si un bouton a été cliqué
        if not any(n_clicks_list):
            return no_update
            
        # Identifier quel bouton a été cliqué
        button_id = ctx.triggered[0]['prop_id'].split('.')[0]
        if button_id:
            # Extraire l'ID de l'article du bouton
            try:
                button_dict = json.loads(button_id)
                article_id = button_dict.get('index')
                if article_id:
                    print(f"Bouton cliqué pour l'article: {article_id}")
                    return article_id
            except Exception as e:
                print(f"Erreur lors de l'extraction de l'ID de l'article: {e}")
        
        return no_update
    
    # Callback pour ouvrir la modale lorsqu'un article est sélectionné
    @app.callback(
        Output('article-modal', 'is_open', allow_duplicate=True),
        [Input('selected-article-id-store', 'data')],
        [State('browser-cluster-data-store', 'data'),
         State('article-modal', 'is_open')],
        prevent_initial_call=True
    )
    def open_article_modal(selected_article_id, stored_data, is_open):
        # Vérifier si l'ID d'article est valide
        if not selected_article_id:
            return no_update
            
        # Vérifier si l'appel vient du callback_context
        ctx = callback_context
        if not ctx.triggered or ctx.triggered[0]['prop_id'] != 'selected-article-id-store.data':
            return no_update
        
        print(f"Ouverture de l'article: {selected_article_id}")
        return True
    
    # Callback pour mettre à jour le contenu de l'article dans la modale
    @app.callback(
        Output('article-details-body', 'children'),
        [Input('selected-article-id-store', 'data'),
         Input('page-content', 'children')],
        [State('browser-cluster-data-store', 'data')],
        prevent_initial_call=True
    )
    def update_article_content(selected_article_id, page_content, stored_data):
        # Vérifier si l'ID d'article est valide
        if not selected_article_id:
            return no_update
            
        # Vérifier si l'appel vient du callback_context
        ctx = callback_context
        if not ctx.triggered:
            return no_update
            
        # Si le déclencheur est page-content, ne rien faire
        trigger_id = ctx.triggered[0]['prop_id'].split('.')[0]
        if trigger_id == 'page-content':
            return no_update
            
        # Vérifier si nous sommes sur la page de carte des clusters
        # Si oui, ne pas exécuter ce callback pour éviter l'erreur
        try:
            if 'cluster-map-graph' in str(page_content):
                return no_update
        except:
            pass
        
        print(f"Mise à jour du contenu de l'article: {selected_article_id}")
        article_details = get_article_details(selected_article_id, stored_data)
        if article_details:
            return article_details
        return html.Div("Article non trouvé")
    
    # Callback pour fermer la modale
    @app.callback(
        Output("article-modal", "is_open", allow_duplicate=True),
        [Input("close-article-modal", "n_clicks")],
        [State("article-modal", "is_open")],
        prevent_initial_call=True
    )
    def close_modal(n_clicks, is_open):
        if n_clicks:
            return False
        return is_open
    
    # Callback pour fermer automatiquement la modale lors d'un changement d'onglet ou d'un retour aux visualisations
    @app.callback(
        [Output('article-modal', 'is_open', allow_duplicate=True),
         Output('selected-article-id-store', 'data', allow_duplicate=True)],
        [Input('cluster-tabs', 'active_tab'),
         Input('btn-back-to-viz', 'n_clicks')],
        [State('article-modal', 'is_open')],
        prevent_initial_call=True
    )
    def close_modal_on_tab_change(active_tab, back_clicks, is_open):
        # Seulement fermer la modale si elle est ouverte
        if (active_tab or back_clicks) and is_open:
            # Fermer la modale et réinitialiser l'article sélectionné
            return False, None
        return no_update, no_update

    # Callback pour la pagination des tables de cluster
    @app.callback(
        Output({'type': 'cluster-table-body', 'index': MATCH}, 'children'),
        [Input({'type': 'cluster-pagination', 'index': MATCH}, 'active_page')],
        [State({'type': 'cluster-table-data', 'index': MATCH}, 'data')]
    )
    def update_table_page(page, table_data):
        if not page or not table_data:
            return no_update
        
        # Calculer les indices de début et de fin pour la pagination
        items_per_page = 20
        start_idx = (page - 1) * items_per_page
        end_idx = start_idx + items_per_page
        
        # Extraire les données pour la page actuelle
        page_data = table_data[start_idx:end_idx]
        
        # Générer les lignes de la table pour la page actuelle
        rows = [
            html.Tr([
                html.Td(row['article_id'], style={'padding': '10px', 'borderBottom': '1px solid #ddd'}),
                html.Td(row['date'], style={'padding': '10px', 'borderBottom': '1px solid #ddd'}),
                html.Td(row['journal'], style={'padding': '10px', 'borderBottom': '1px solid #ddd'}),
                html.Td(
                    dbc.Button(
                        "Voir", 
                        id={'type': 'view-article-btn', 'index': row['article_id']},
                        color="primary",
                        size="sm",
                        className="me-1",
                        n_clicks=0
                    ),
                    style={'textAlign': 'center', 'padding': '10px', 'borderBottom': '1px solid #ddd'}
                )
            ]) for row in page_data
        ]
        
        return rows
    
    # Fonction pour récupérer les détails d'un article (optimisée)
    def get_article_details(article_id, stored_data):
        try:
            # Si article_id est un dictionnaire (cellRendererData), extraire la valeur
            if isinstance(article_id, dict) and 'value' in article_id:
                article_id = article_id['value']
            
            # Chemin vers le fichier articles.json en utilisant config.yaml
            project_root = pathlib.Path(__file__).resolve().parents[2]
            
            # Utiliser un cache pour éviter de recharger config.yaml à chaque fois
            if not hasattr(get_article_details, 'config_cache'):
                with open(project_root / 'config' / 'config.yaml', encoding='utf-8') as f:
                    get_article_details.config_cache = yaml.safe_load(f)
            
            config = get_article_details.config_cache
            
            def resolve_path_from_config(path_from_config):
                if os.path.isabs(path_from_config):
                    return path_from_config
                return str(project_root / path_from_config)
            
            processed_dir = resolve_path_from_config(config['data']['processed_dir'])
            articles_json_path = os.path.join(processed_dir, 'articles.json')
            
            print(f"Recherche de l'article {article_id} dans: {articles_json_path}")
            
            # Utiliser un cache pour éviter de recharger articles.json à chaque fois
            if not hasattr(get_article_details, 'articles_cache'):
                if os.path.exists(articles_json_path):
                    with open(articles_json_path, 'r', encoding='utf-8') as f:
                        get_article_details.articles_cache = json.load(f)
                else:
                    get_article_details.articles_cache = []
            
            articles_data = get_article_details.articles_cache
            
            # Extraire l'ID de base du document (sans le suffixe éventuel)
            base_id = article_id
            
            # S'assurer que article_id est une chaîne de caractères
            article_id = str(article_id)
            base_id = str(base_id)
            
            # Extraire l'ID de base sans le suffixe "_mistral" ou similaire
            if '_' in article_id:
                parts = article_id.split("_")
                # Si le dernier segment est court (comme "mistral"), l'ignorer
                if len(parts) > 1 and len(parts[-1]) < 10:
                    base_id = '_'.join(parts[:-1])
            
            print(f"Recherche avec article_id={article_id}, base_id={base_id}")
            
            # Chercher l'article correspondant
            article = None
            for art in articles_data:
                # Convertir les identifiants en chaînes pour la comparaison
                art_id = str(art.get('id', ''))
                art_base_id = str(art.get('base_id', ''))
                
                # Vérifier correspondance exacte avec id ou base_id
                if art_id == article_id or art_base_id == article_id:
                    article = art
                    break
                
                # Vérifier avec base_id calculé
                elif art_id == base_id or art_base_id == base_id:
                    article = art
                    break
                
                # Vérifier correspondance partielle (si l'ID contient l'article_id)
                elif article_id in art_id or article_id in art_base_id:
                    article = art
                    break
                
                # Vérifier correspondance partielle inverse (si l'article_id contient l'ID)
                elif art_id in article_id or (art_base_id and art_base_id in article_id):
                    article = art
                    break
            
            if article:
                print(f"Article trouvé: {article.get('title', 'Sans titre')}")
                # Récupérer le contenu et le titre
                content = article.get('content', 'Contenu non disponible')
                title = article.get('title', 'Sans titre')
                source = article.get('url', 'Source inconnue')
                date = article.get('date', 'Date inconnue')
                
                # Trouver le cluster de l'article
                cluster = None
                if stored_data and 'labels' in stored_data and 'doc_ids' in stored_data:
                    for i, doc_id in enumerate(stored_data['doc_ids']):
                        doc_id_str = str(doc_id)
                        if doc_id_str == article_id or doc_id_str == base_id or article_id in doc_id_str or base_id in doc_id_str:
                            cluster = stored_data['labels'][i]
                            break
                
                # Construire l'affichage des détails
                details = [
                    html.H4(title, className="mb-3"),
                    html.Div([
                        html.Strong("Date: "), 
                        html.Span(f"{date}"),
                        html.Br(),
                        html.Strong("Source: "), 
                        html.A(source, href=source, target="_blank"),
                        html.Br(),
                        html.Strong("Cluster: "), 
                        html.Span(f"Cluster {cluster}" if cluster is not None else "Non classifié"),
                    ], className="mb-3"),
                    html.Hr(),
                    html.Div([
                        html.H5("Contenu:"),
                        html.P(content.replace('\n', '\n\n'), style={"whiteSpace": "pre-wrap"})
                    ])
                ]
                return details
            else:
                print(f"Article non trouvé: {article_id}")
                return [
                    html.H4(f"Article {article_id}"),
                    html.P("Cet article n'a pas été trouvé dans la base de données.")
                ]
        except Exception as e:
            print(f"Erreur lors de la récupération de l'article: {e}")
            return [
                html.H4(f"Article {article_id}"),
                html.P(f"Erreur lors de la récupération des détails: {str(e)}")
            ]
    
    # Callback pour afficher la page d'exploration des articles
    @app.callback(
        [Output("clustering-stats-output", "style"),
         Output("article-browser-container", "style"),
         Output("article-browser-container", "children")],
        [Input("btn-explore-articles", "n_clicks")],
        [State("cluster-data-store", "data"),
         State("results-data-store", "data")]
    )
    def toggle_article_browser(explore_clicks, cluster_data, results_data):
        if not explore_clicks:
            return {"display": "block"}, {"display": "none"}, []
        
        # Utiliser les données de résultats si disponibles, sinon utiliser les données de clustering
        data_to_use = results_data if results_data else cluster_data
        
        if not data_to_use:
            return html.Div("Aucune donnée de clustering disponible"), {"display": "block"}, no_update
        
        # Afficher la page d'exploration des articles
        return {"display": "none"}, {"display": "block"}, get_article_browser_layout(data_to_use)
    
    # Callback pour revenir à la visualisation depuis la page d'exploration
    @app.callback(
        [Output("clustering-stats-output", "style", allow_duplicate=True),
         Output("article-browser-container", "style", allow_duplicate=True),
         Output('article-modal', 'is_open', allow_duplicate=True)],
        [Input("btn-back-to-viz", "n_clicks")],
        prevent_initial_call=True
    )
    def back_to_visualization(back_clicks):
        if back_clicks:
            # Revenir à la page de visualisation et s'assurer que la modale est fermée
            return {"display": "block"}, {"display": "none"}, False
        return no_update, no_update, no_update
    
    # Callback pour naviguer vers la carte des clusters
    @app.callback(
        Output("page-content", "children", allow_duplicate=True),
        [Input("btn-view-cluster-map", "n_clicks")],
        [State("cluster-data-store", "data"),
         State("results-data-store", "data")],
        prevent_initial_call=True
    )
    def navigate_to_cluster_map(view_map_clicks, cluster_data, results_data):
        from src.webapp.cluster_map_viz import get_cluster_map_layout, register_cluster_map_callbacks
        
        if view_map_clicks:
            # Utiliser les données de résultats si disponibles, sinon utiliser les données de clustering
            data_to_use = results_data if results_data else cluster_data
            
            # Enregistrer les callbacks de la carte des clusters
            register_cluster_map_callbacks(app)
            
            # Retourner le layout de la carte des clusters
            return get_cluster_map_layout()
            
        return no_update