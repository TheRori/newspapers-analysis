"""
Cluster Map Visualization Page for Dash app
Shows a 2D map of articles in clusters with distances between them
"""

import json
import os
import sys
from dash import dcc, html, Input, Output, State, callback_context, no_update, MATCH
import dash_bootstrap_components as dbc
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from pathlib import Path
import yaml
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import random
import pickle

# Essayer d'importer UMAP (optionnel)
try:
    from umap import UMAP
    UMAP_AVAILABLE = True
except ImportError:
    UMAP_AVAILABLE = False
    print("UMAP non disponible. Pour l'installer: pip install umap-learn")

# Cache pour stocker les coordonnées générées
_map_coordinates_cache = {}

# Helper to get config and paths
def get_config_and_paths():
    project_root = Path(__file__).resolve().parents[2]
    config_path = project_root / 'config' / 'config.yaml'
    with open(config_path, encoding='utf-8') as f:
        config = yaml.safe_load(f)
    results_dir = project_root / config['data']['results_dir']
    processed_dir = project_root / config['data']['processed_dir']
    return project_root, config, results_dir, processed_dir

# Function to load the articles data
def load_articles_data():
    project_root, config, results_dir, processed_dir = get_config_and_paths()
    articles_path = processed_dir / config['data'].get('articles_file', 'articles.json')
    
    if not os.path.exists(articles_path):
        return None
    
    with open(articles_path, encoding='utf-8') as f:
        articles = json.load(f)
    
    return articles

# Function to load clustering results
def load_clustering_results():
    project_root, config, results_dir, _ = get_config_and_paths()
    clusters_dir = results_dir / 'clusters'
    
    if not os.path.exists(clusters_dir):
        return None
    
    # Find the most recent clustering file
    cluster_files = [f for f in os.listdir(clusters_dir) if f.startswith('doc_clusters_k')]
    if not cluster_files:
        return None
    
    # Sort by modification time (most recent first)
    cluster_files.sort(key=lambda x: os.path.getmtime(os.path.join(clusters_dir, x)), reverse=True)
    latest_file = os.path.join(clusters_dir, cluster_files[0])
    
    with open(latest_file, encoding='utf-8') as f:
        clustering_data = json.load(f)
    
    return clustering_data

# Function to generate 2D coordinates from doc-topic matrix
def generate_map_coordinates(clustering_data, method='tsne'):
    global _map_coordinates_cache
    
    # Create a cache key based on the clustering data and method
    cache_key = f"{hash(str(clustering_data))}-{method}"
    
    # Check if we have cached results
    if cache_key in _map_coordinates_cache:
        print("Using cached map coordinates")
        return _map_coordinates_cache[cache_key]
    
    # Check if we have a cache file
    project_root, _, results_dir, _ = get_config_and_paths()
    cache_dir = project_root / 'cache'
    os.makedirs(cache_dir, exist_ok=True)
    cache_file = cache_dir / f"map_coordinates_{method}.pkl"
    
    if os.path.exists(cache_file):
        try:
            with open(cache_file, 'rb') as f:
                cached_data = pickle.load(f)
                if cached_data.get('cache_key') == cache_key:
                    print(f"Loading map coordinates from cache file: {cache_file}")
                    _map_coordinates_cache[cache_key] = cached_data.get('data')
                    return _map_coordinates_cache[cache_key]
        except Exception as e:
            print(f"Error loading cache file: {e}")
    
    if not clustering_data:
        return pd.DataFrame()
    
    # Get the document IDs and their cluster labels
    doc_ids = clustering_data.get('doc_ids', [])
    labels = clustering_data.get('labels', [])
    
    # Get cluster centers
    cluster_centers = clustering_data.get('cluster_centers', [])
    
    # Try to load advanced topic modeling data for better visualization
    project_root, config, results_dir, _ = get_config_and_paths()
    advanced_topic_path = results_dir / 'advanced_topic' / 'advanced_topic_analysis.json'
    
    # Fix random seed for reproducibility
    np.random.seed(42)
    random.seed(42)
    
    # Limit the number of articles for performance (max 1000)
    max_articles = 1000
    if len(doc_ids) > max_articles:
        # Sample articles while preserving cluster distribution
        indices = list(range(len(doc_ids)))
        sampled_indices = []
        
        # Group indices by cluster
        cluster_indices = {}
        for i, label in enumerate(labels):
            if label not in cluster_indices:
                cluster_indices[label] = []
            cluster_indices[label].append(i)
        
        # Sample from each cluster proportionally
        for cluster, indices in cluster_indices.items():
            cluster_sample_size = int(len(indices) / len(doc_ids) * max_articles)
            if cluster_sample_size < 1:
                cluster_sample_size = 1
            sampled_cluster_indices = random.sample(indices, min(cluster_sample_size, len(indices)))
            sampled_indices.extend(sampled_cluster_indices)
        
        # Ensure we don't exceed max_articles
        if len(sampled_indices) > max_articles:
            sampled_indices = random.sample(sampled_indices, max_articles)
        
        # Create new lists with only sampled items
        sampled_doc_ids = [doc_ids[i] for i in sampled_indices]
        sampled_labels = [labels[i] for i in sampled_indices]
        
        doc_ids = sampled_doc_ids
        labels = sampled_labels
        
        print(f"Sampled {len(doc_ids)} articles from {len(clustering_data['doc_ids'])} for performance")
    
    # Variables to store our results
    valid_doc_ids = []
    valid_labels = []
    topic_distributions = []
    
    # Try to load the doc_topic_matrix.json file first
    doc_topic_matrix_path = results_dir / 'doc_topic_matrix.json'
    if os.path.exists(doc_topic_matrix_path):
        try:
            with open(doc_topic_matrix_path, encoding='utf-8') as f:
                doc_topic_data = json.load(f)
            
            # Create a dictionary for quick lookup
            doc_topic_dict = {}
            for item in doc_topic_data:
                if 'doc_id' in item and 'topic_distribution' in item:
                    doc_topic_dict[str(item['doc_id'])] = item['topic_distribution']
            
            print(f"Chargé {len(doc_topic_dict)} distributions de topics depuis doc_topic_matrix.json")
            
            # Get topic distributions for our documents
            for i, doc_id in enumerate(doc_ids):
                doc_id_str = str(doc_id)
                if i < len(labels) and doc_id_str in doc_topic_dict:
                    valid_doc_ids.append(doc_id)
                    valid_labels.append(labels[i])
                    topic_distributions.append(doc_topic_dict[doc_id_str])
            
            if valid_doc_ids:
                print(f"Trouvé {len(valid_doc_ids)} articles avec distributions de topics dans doc_topic_matrix.json")
            else:
                print("Aucun article correspondant trouvé dans doc_topic_matrix.json")
        except Exception as e:
            print(f"Erreur lors du chargement de doc_topic_matrix.json: {e}")
    
    # If we don't have topic distributions from doc_topic_matrix.json, try advanced topic modeling
    if not topic_distributions and os.path.exists(advanced_topic_path):
        try:
            with open(advanced_topic_path, encoding='utf-8') as f:
                advanced_data = json.load(f)
            
            if 'doc_topics' in advanced_data:
                # Get topic distributions for our documents
                doc_topics = advanced_data.get('doc_topics', {})
                
                for i, doc_id in enumerate(doc_ids):
                    doc_id_str = str(doc_id)
                    if i < len(labels) and doc_id_str in doc_topics:
                        valid_doc_ids.append(doc_id)
                        valid_labels.append(labels[i])
                        topic_distributions.append(doc_topics[doc_id_str]['topic_distribution'])
                
                if not valid_doc_ids:
                    print("No matching documents found in advanced topic data")
                else:
                    print(f"Trouvé {len(valid_doc_ids)} articles avec distributions de topics dans advanced_topic_analysis.json")
        except Exception as e:
            print(f"Error loading advanced topic data: {e}")
    
    # Si nous n'avons pas de distributions de topics, vérifier si nous avons des embeddings originaux
    if not topic_distributions and 'embeddings' in clustering_data:
        print("Using original embeddings for visualization")
        # Utiliser les embeddings originaux sauvegardés lors du clustering
        embeddings = clustering_data['embeddings']
        
        # Vérifier que nous avons le même nombre d'embeddings que de documents
        if len(embeddings) == len(doc_ids):
            for i, doc_id in enumerate(doc_ids):
                if i < len(labels):
                    topic_distributions.append(embeddings[i])
                    valid_doc_ids.append(doc_id)
                    valid_labels.append(labels[i])
            print(f"Utilisé {len(topic_distributions)} embeddings originaux pour la visualisation")
    
    # Si nous n'avons toujours pas de distributions, utiliser les centres de clusters avec du bruit
    if not topic_distributions:
        print("Using cluster centers for visualization (fallback)")
        # Create a simple representation based on cluster centers
        for i, doc_id in enumerate(doc_ids):
            if i < len(labels):
                cluster = labels[i]
                if cluster < len(cluster_centers):
                    # Add some random noise to avoid all documents in a cluster being at the same point
                    center = np.array(cluster_centers[cluster])
                    noise = np.random.normal(0, 0.05, size=center.shape)
                    doc_vector = center + noise
                    
                    topic_distributions.append(doc_vector)
                    valid_doc_ids.append(doc_id)
                    valid_labels.append(cluster)
        
        if not valid_doc_ids:
            print("No valid documents found")
            return pd.DataFrame()
    
    # Convert to numpy array for dimensionality reduction
    X = np.array(topic_distributions)
    
    # Apply dimensionality reduction
    if method == 'tsne':
        # t-SNE for visualization (perplexity should be smaller than n_samples)
        perplexity = min(30, len(X) - 1)
        if perplexity < 5:
            perplexity = 5
        
        print(f"Applying t-SNE with perplexity={perplexity}...")
        tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity)
        coords = tsne.fit_transform(X)
    elif method == 'umap' and UMAP_AVAILABLE:
        # UMAP souvent meilleur que t-SNE pour préserver la structure globale
        n_neighbors = min(30, len(X) - 1)
        if n_neighbors < 5:
            n_neighbors = 5
            
        print(f"Applying UMAP with n_neighbors={n_neighbors}...")
        umap = UMAP(n_components=2, random_state=42, n_neighbors=n_neighbors)
        coords = umap.fit_transform(X)
    else:
        # PCA for visualization (fallback)
        print("Applying PCA...")
        pca = PCA(n_components=2, random_state=42)
        coords = pca.fit_transform(X)
    
    # Calculate distance to cluster centroid for color intensity
    distances = []
    is_anomaly = []
    anomaly_threshold = 0.8  # Threshold for anomaly detection (80th percentile)
    anomaly_reasons = []
    
    if cluster_centers and len(cluster_centers) > 0:
        # Calculate distances to cluster centroids
        cluster_distances = {}
        for i, label in enumerate(valid_labels):
            if label < len(cluster_centers):
                centroid = cluster_centers[label]
                dist = np.linalg.norm(X[i] - centroid)
                if label not in cluster_distances:
                    cluster_distances[label] = []
                cluster_distances[label].append(dist)
                distances.append(dist)
            else:
                distances.append(0)
        
        # Normalize distances per cluster and detect anomalies
        if distances:
            normalized_distances = []
            for i, label in enumerate(valid_labels):
                if label in cluster_distances:
                    cluster_dists = cluster_distances[label]
                    if cluster_dists:
                        # Normalize within this cluster
                        min_dist = min(cluster_dists)
                        max_dist = max(cluster_dists)
                        if max_dist > min_dist:
                            norm_dist = (distances[i] - min_dist) / (max_dist - min_dist)
                        else:
                            norm_dist = 0
                        
                        normalized_distances.append(norm_dist)
                        
                        # Check if this is an anomaly
                        if norm_dist > anomaly_threshold:
                            is_anomaly.append(True)
                            
                            # Generate explanation for anomaly
                            reason = "Cet article est éloigné du centre de son cluster."
                            
                            # If we have topic information, explain which topics differ most
                            if advanced_data:
                                article_topics = X[i]
                                centroid_topics = cluster_centers[label]
                                
                                # Calculate differences for each topic
                                topic_diffs = []
                                for t, (article_val, centroid_val) in enumerate(zip(article_topics, centroid_topics)):
                                    topic_diffs.append((t, abs(article_val - centroid_val)))
                                
                                # Sort by difference (descending)
                                topic_diffs.sort(key=lambda x: x[1], reverse=True)
                                
                                # Get top 3 differing topics
                                top_diff_topics = topic_diffs[:3]
                                
                                # Get topic names and words
                                topic_names = advanced_data.get('topic_names', {})
                                topic_words = advanced_data.get('topic_words', {})
                                
                                diff_descriptions = []
                                for topic_idx, diff_val in top_diff_topics:
                                    topic_name = topic_names.get(str(topic_idx), f"Topic {topic_idx}")
                                    words = topic_words.get(str(topic_idx), [])[:5]
                                    words_str = ", ".join(words) if words else "pas de mots disponibles"
                                    
                                    if article_topics[topic_idx] > centroid_topics[topic_idx]:
                                        direction = "plus élevé"
                                    else:
                                        direction = "plus faible"
                                    
                                    diff_descriptions.append(f"{topic_name} ({direction}): {words_str}")
                                
                                reason = "Diffère du centre pour les sujets: " + "; ".join(diff_descriptions)
                            
                            anomaly_reasons.append(reason)
                        else:
                            is_anomaly.append(False)
                            anomaly_reasons.append(None)
                    else:
                        normalized_distances.append(0)
                        is_anomaly.append(False)
                        anomaly_reasons.append(None)
                else:
                    normalized_distances.append(0)
                    is_anomaly.append(False)
                    anomaly_reasons.append(None)
            
            # Invert for intensity (1 = closest to center)
            distances = [1 - d for d in normalized_distances]
        else:
            distances = [1] * len(valid_doc_ids)
            is_anomaly = [False] * len(valid_doc_ids)
            anomaly_reasons = [None] * len(valid_doc_ids)
    else:
        distances = [1] * len(valid_doc_ids)
        is_anomaly = [False] * len(valid_doc_ids)
        anomaly_reasons = [None] * len(valid_doc_ids)
    
    # Create DataFrame with results
    df = pd.DataFrame({
        'doc_id': valid_doc_ids,
        'cluster': valid_labels,
        'x': coords[:, 0],
        'y': coords[:, 1],
        'intensity': distances,
        'is_anomaly': is_anomaly,
        'anomaly_reason': anomaly_reasons
    })
    
    # Cache the results
    _map_coordinates_cache[cache_key] = df
    
    # Save to cache file
    try:
        with open(cache_file, 'wb') as f:
            pickle.dump({'cache_key': cache_key, 'data': df}, f)
        print(f"Saved map coordinates to cache file: {cache_file}")
    except Exception as e:
        print(f"Error saving cache file: {e}")
    
    return df

# Function to get article content by ID
def get_article_by_id(article_id):
    articles = load_articles_data()
    if not articles:
        return None
    
    # Convert article_id to string for comparison
    article_id_str = str(article_id)
    
    # Handle both dictionary and list formats
    if isinstance(articles, dict):
        # First try direct lookup
        if article_id_str in articles:
            return articles[article_id_str]
        
        # Try searching by base_id if available
        for art_id, article in articles.items():
            if str(article.get('base_id', '')) == article_id_str:
                return article
        
        # If not found, try to match by ID substring
        for art_id, article in articles.items():
            if article_id_str in art_id:
                return article
    
    elif isinstance(articles, list):
        # Search through the list of articles
        for article in articles:
            # Check if the article has an 'id' field that matches
            if str(article.get('id', '')) == article_id_str:
                return article
            
            # Check if the article has a 'base_id' field that matches
            if str(article.get('base_id', '')) == article_id_str:
                return article
            
            # Check for partial matches
            if 'id' in article and article_id_str in str(article['id']):
                return article
    
    return None

# Function to load articles data and extract statistics
def get_cluster_statistics(clustering_data):
    if not clustering_data:
        return None
    
    project_root, config, results_dir, _ = get_config_and_paths()
    
    # Get the document IDs and their cluster labels
    doc_ids = clustering_data.get('doc_ids', [])
    labels = clustering_data.get('labels', [])
    
    # Try to load the advanced topic analysis for topic information
    advanced_topic_path = results_dir / 'advanced_topic' / 'advanced_topic_analysis.json'
    advanced_data = None
    
    if os.path.exists(advanced_topic_path):
        try:
            with open(advanced_topic_path, encoding='utf-8') as f:
                advanced_data = json.load(f)
        except Exception as e:
            print(f"Error loading advanced topic data: {e}")
    
    # Group documents by cluster
    clusters = {}
    for i, doc_id in enumerate(doc_ids):
        if i < len(labels):
            cluster = labels[i]
            if cluster not in clusters:
                clusters[cluster] = []
            clusters[cluster].append(doc_id)
    
    # Calculate statistics for each cluster
    cluster_stats = {}
    
    for cluster, cluster_docs in clusters.items():
        stats = {
            "num_docs": len(cluster_docs),
            "doc_ids": cluster_docs[:10],  # Store first 10 doc IDs as examples
        }
        
        # Get topic distribution if available
        if advanced_data and 'doc_topics' in advanced_data:
            doc_topics = advanced_data.get('doc_topics', {})
            
            # Collect topic distributions for this cluster
            topic_distributions = []
            # Limit to 100 documents per cluster for performance
            for doc_id in cluster_docs[:100]:
                if doc_id in doc_topics:
                    topic_distributions.append(doc_topics[doc_id]['topic_distribution'])
            
            if topic_distributions:
                # Calculate average topic distribution for the cluster
                avg_distribution = np.mean(topic_distributions, axis=0).tolist()
                stats["avg_topic_distribution"] = avg_distribution
                
                # Find dominant topics
                top_topic_indices = np.argsort(avg_distribution)[-3:][::-1]
                stats["top_topics"] = top_topic_indices.tolist()
                
                # Get topic names and words if available
                if 'topic_names' in advanced_data:
                    topic_names = advanced_data.get('topic_names', {})
                    stats["top_topic_names"] = [topic_names.get(str(idx), f"Topic {idx}") for idx in top_topic_indices]
                
                if 'topic_words' in advanced_data:
                    topic_words = advanced_data.get('topic_words', {})
                    stats["top_topic_words"] = [topic_words.get(str(idx), [])[:10] for idx in top_topic_indices]
        
        cluster_stats[cluster] = stats
    
    return cluster_stats

# Function to create the scatter plot
def create_cluster_map(df):
    if df is None or df.empty:
        # Return empty figure if no data
        return go.Figure()
    
    # Get unique clusters
    clusters = df['cluster'].unique()
    
    # Create a discrete color map for clusters
    cluster_colors = {}
    color_scale = px.colors.qualitative.Plotly
    for i, cluster in enumerate(clusters):
        cluster_colors[cluster] = color_scale[i % len(color_scale)]
    
    # Create figure
    fig = go.Figure()
    
    # Add traces for each cluster
    for cluster in clusters:
        cluster_df = df[df['cluster'] == cluster]
        
        # Separate normal points and anomalies
        normal_df = cluster_df[~cluster_df['is_anomaly']]
        anomaly_df = cluster_df[cluster_df['is_anomaly']]
        
        # Add normal points
        if not normal_df.empty:
            fig.add_trace(go.Scatter(
                x=normal_df['x'],
                y=normal_df['y'],
                mode='markers',
                marker=dict(
                    color=cluster_colors[cluster],
                    opacity=normal_df['intensity'],
                    size=8
                ),
                text=normal_df['doc_id'],
                hovertemplate='<b>Article ID:</b> %{text}<extra></extra>',
                name=f'Cluster {cluster}'
            ))
        
        # Add anomalies with different style
        if not anomaly_df.empty:
            hover_texts = []
            for _, row in anomaly_df.iterrows():
                text = f"<b>Article ID:</b> {row['doc_id']}<br>"
                if row['anomaly_reason']:
                    text += f"<b>Anomalie:</b> {row['anomaly_reason']}"
                hover_texts.append(text)
            
            fig.add_trace(go.Scatter(
                x=anomaly_df['x'],
                y=anomaly_df['y'],
                mode='markers',
                marker=dict(
                    color='rgba(0,0,0,0)',
                    size=12,
                    line=dict(
                        color='red',
                        width=2
                    )
                ),
                text=anomaly_df['doc_id'],
                hovertemplate='%{hovertext}<extra></extra>',
                hovertext=hover_texts,
                name=f'Anomalies Cluster {cluster}'
            ))
    
    # Update layout
    fig.update_layout(
        title="Carte des articles par cluster",
        xaxis=dict(title="Dimension 1", showgrid=True),
        yaxis=dict(title="Dimension 2", showgrid=True),
        legend_title="Clusters",
        hovermode='closest',
        height=600
    )
    
    return fig
# Function to get the cluster map layout
def get_cluster_map_layout():
    # Load clustering results
    clustering_data = load_clustering_results()
    
    if not clustering_data:
        return html.Div([
            html.H3("Carte des clusters", className="mb-4"),
            # Ajouter les contrôles de visualisation
            visualization_controls,
            dcc.Graph(
                id="cluster-map-graph",
                figure=fig,
                style={"height": "600px"},
                className="mb-4 shadow"
            ),])
    
    # Générer les coordonnées avec t-SNE par défaut
    df = generate_map_coordinates(clustering_data, method='tsne')
    if df.empty:
        return html.Div("Impossible de générer les coordonnées pour la carte. Vérifiez les données de clustering.")
        
    # Créer les contrôles pour la visualisation
    visualization_controls = dbc.Card([
        dbc.CardHeader("Options de visualisation", className="bg-secondary text-white"),
        dbc.CardBody([
            dbc.Row([
                dbc.Col([
                    dbc.Label("Méthode de projection"),
                    dcc.Dropdown(
                        id="cluster-map-projection-method",
                        options=[
                            {"label": "t-SNE (préserve les structures locales)", "value": "tsne"},
                            {"label": "UMAP (préserve la structure globale)", "value": "umap"},
                            {"label": "PCA (plus rapide, moins précis)", "value": "pca"}
                        ],
                        value="tsne",
                        clearable=False
                    )
                ], width=6),
                dbc.Col([
                    dbc.Button(
                        "Appliquer", 
                        id="cluster-map-apply-projection", 
                        color="primary", 
                        className="mt-4"
                    )
                ], width=6, className="d-flex align-items-end")
            ])
        ])
    ], className="mb-4 shadow-sm")
    
    # Créer les contrôles pour la visualisation
    visualization_controls = dbc.Card([
        dbc.CardHeader("Options de visualisation", className="bg-secondary text-white"),
        dbc.CardBody([
            dbc.Row([
                dbc.Col([
                    dbc.Label("Méthode de projection"),
                    dcc.Dropdown(
                        id="cluster-map-projection-method",
                        options=[
                            {"label": "t-SNE (préserve les structures locales)", "value": "tsne"},
                            {"label": "UMAP (préserve la structure globale)", "value": "umap"},
                            {"label": "PCA (plus rapide, moins précis)", "value": "pca"}
                        ],
                        value="tsne",
                        clearable=False
                    )
                ], width=6),
                dbc.Col([
                    dbc.Button(
                        "Appliquer", 
                        id="cluster-map-apply-projection", 
                        color="primary", 
                        className="mt-4"
                    )
                ], width=6, className="d-flex align-items-end")
            ])
        ])
    ], className="mb-4 shadow-sm")
    
    # Create the cluster map
    fig = create_cluster_map(df)
    
    # Create cards for cluster statistics
    cluster_stats = get_cluster_statistics(clustering_data)
    cluster_stat_cards = []
    
    # Définir les couleurs des clusters (mêmes couleurs que dans le graphique)
    clusters = df['cluster'].unique()
    color_scale = px.colors.qualitative.Plotly
    cluster_colors = {}
    for i, cluster in enumerate(clusters):
        cluster_colors[cluster] = color_scale[i % len(color_scale)]
    
    for cluster, stats in cluster_stats.items():
        # Create a card for each cluster
        card = dbc.Card([
            dbc.CardHeader(f"Cluster {cluster} ({stats['num_docs']} articles)", className="bg-primary text-white"),
            dbc.CardBody([
                # Statistiques de base
                
                # Add a button to load detailed statistics on demand
                dbc.Button(
                    "Charger plus de statistiques",
                    id={"type": "load-cluster-stats-btn", "index": cluster},
                    color="secondary",
                    size="sm",
                    className="mt-3",
                    n_clicks=0
                ),
                # Add a loading component
                dbc.Spinner(
                    html.Div(id={"type": "cluster-detailed-stats", "index": cluster}),
                    color="primary",
                    type="grow",
                    spinner_style={"width": "2rem", "height": "2rem"}
                )
            ])
        ], className="mb-4", style={"border-left": f"5px solid {cluster_colors[cluster]}"})
        
        cluster_stat_cards.append(card)
    
    # Main layout
    return dbc.Container([
        html.H2("Carte des clusters d'articles", className="text-center mb-4"),
        dbc.Row([
            dbc.Col([
                dbc.Button("← Retour au clustering", id="back-to-clustering-btn", color="link", className="mb-3")
            ], width=12)
        ]),
        # Ajouter le store pour les données de clustering
        dcc.Store(id="browser-cluster-data-store", data=clustering_data),
        # Add loading spinner for the map
        dbc.Spinner(
            dbc.Row([
                dbc.Col([
                    dcc.Graph(
                        id='cluster-map-graph',
                        figure=fig,
                        config={'displayModeBar': True}
                    )
                ], width=12)
            ]),
            color="primary",
            type="border",
            fullscreen=False,
        ),
        
        # Article modal
        dbc.Modal([
            dbc.ModalHeader("Détails de l'article"),
            # Add loading spinner for article content
            dbc.ModalBody([
                dbc.Spinner(
                    html.Div(id="article-modal-content"),
                    color="primary",
                    type="grow",
                    spinner_style={"width": "2rem", "height": "2rem"}
                )
            ]),
            dbc.ModalFooter(
                dbc.Button("Fermer", id="close-article-modal", className="ml-auto")
            )
        ], id="article-modal", size="lg"),
        
        # Store for selected article ID
        dcc.Store(id='selected-article-id-store'),
        
        html.H3("Statistiques par cluster", className="mb-4"),
        html.Div(cluster_stat_cards),
        
        # Hidden div for article browser container (needed for callbacks)
        html.Div(id="article-browser-container", style={"display": "none"})
    ])

# Register callbacks for the cluster map page
def register_cluster_map_callbacks(app):
    @app.callback(
        Output("article-modal", "is_open"),
        [Input("cluster-map-graph", "clickData"),
         Input("close-article-modal", "n_clicks")],
        [State("article-modal", "is_open")],
        prevent_initial_call=True
    )
    def toggle_article_modal(clickData, close_clicks, is_open):
        ctx = callback_context
        if not ctx.triggered:
            return is_open
        
        trigger_id = ctx.triggered[0]["prop_id"].split(".")[0]
        
        if trigger_id == "cluster-map-graph" and clickData:
            return True
        elif trigger_id == "close-article-modal":
            return False
        
        return is_open
    
    @app.callback(
        Output("selected-article-id-store", "data"),
        Input("cluster-map-graph", "clickData"),
        prevent_initial_call=True
    )
    def store_selected_article_id(clickData):
        if not clickData:
            return no_update
        
        # Get the article ID from the clicked point
        try:
            # Format changed when we switched from px.scatter to go.Figure
            point_data = clickData["points"][0]
            article_id = point_data.get("text")  # Now using 'text' instead of 'hovertext'
            
            if article_id:
                return article_id
        except Exception as e:
            print(f"Error extracting article ID from clickData: {e}")
            print(f"clickData: {clickData}")
        
        return no_update
    
    @app.callback(
        Output("article-modal-content", "children"),
        Input("selected-article-id-store", "data")
    )
    def update_article_content(article_id):
        if not article_id:
            return "Aucun article sélectionné"
        
        # Get the article content
        article = get_article_by_id(article_id)
        
        if not article:
            return f"Article non trouvé: {article_id}"
        
        # Extract article information
        title = article.get("title", "Titre non disponible")
        date = article.get("date", "Date non disponible")
        newspaper = article.get("newspaper", "Journal non disponible")
        canton = article.get("canton", "Canton non disponible")
        
        # Try to get the text from either 'text' or 'content' field
        text = article.get("text", article.get("content", "Texte non disponible"))
        
        # Format the content
        content = [
            html.H4(title, className="mb-3"),
            dbc.Row([
                dbc.Col([
                    html.Strong("Date: "),
                    html.Span(date),
                ], width=4),
                dbc.Col([
                    html.Strong("Journal: "),
                    html.Span(newspaper),
                ], width=4),
                dbc.Col([
                    html.Strong("Canton: "),
                    html.Span(canton),
                ], width=4),
            ], className="mb-3"),
            html.Hr(),
            html.Div([
                html.P(paragraph) for paragraph in text.split('\n') if paragraph.strip()
            ], className="article-text")
        ]
        
        return content
    
    # New callback to load detailed statistics on demand
    @app.callback(
        Output({"type": "cluster-detailed-stats", "index": MATCH}, "children"),
        Input({"type": "load-cluster-stats-btn", "index": MATCH}, "n_clicks"),
        State({"type": "load-cluster-stats-btn", "index": MATCH}, "id"),
        prevent_initial_call=True
    )
    def load_detailed_cluster_stats(n_clicks, btn_id):
        if not n_clicks or n_clicks == 0:
            return no_update
        
        cluster = btn_id["index"]
        
        # Load the clustering data
        clustering_data = load_clustering_results()
        if not clustering_data:
            return html.Div("Données de clustering non disponibles", className="text-danger")
        
        # Get the document IDs and their cluster labels
        doc_ids = clustering_data.get('doc_ids', [])
        labels = clustering_data.get('labels', [])
        cluster_centers = clustering_data.get('cluster_centers', [])
        
        # Get the documents for this cluster
        cluster_docs = []
        for i, doc_id in enumerate(doc_ids):
            if i < len(labels) and str(labels[i]) == str(cluster):
                cluster_docs.append((doc_id, i))  # Store doc_id and its index
        
        if not cluster_docs:
            return html.Div("Aucun document trouvé dans ce cluster", className="text-danger")
        
        # Load articles data
        project_root, config, results_dir, _ = get_config_and_paths()
        # Utiliser le répertoire processed_dir au lieu de results_dir pour articles.json
        processed_dir = project_root / config['data']['processed_dir']
        articles_path = processed_dir / 'articles.json'
        
        try:
            with open(articles_path, encoding='utf-8') as f:
                articles_data = json.load(f)
        except Exception as e:
            return html.Div(f"Erreur lors du chargement des articles: {e}", className="text-danger")
        
        # Calculate statistics (limit to 50 articles for performance)
        text_lengths = []
        newspapers = []
        cantons = []
        
        # Sample a subset of articles for faster processing
        sample_size = min(50, len(cluster_docs))
        sampled_docs = random.sample(cluster_docs, sample_size) if len(cluster_docs) > sample_size else cluster_docs
        
        # Lists to store article info for finding typical and anomalous articles
        article_info = []
        
        # Try to load advanced topic modeling data for anomaly detection
        advanced_topic_path = results_dir / 'advanced_topic' / 'advanced_topic_analysis.json'
        advanced_data = None
        doc_topics = {}
        
        if os.path.exists(advanced_topic_path):
            try:
                with open(advanced_topic_path, encoding='utf-8') as f:
                    advanced_data = json.load(f)
                if 'doc_topics' in advanced_data:
                    doc_topics = advanced_data.get('doc_topics', {})
            except Exception as e:
                print(f"Error loading advanced topic data: {e}")
        
        # Calculate distances to centroid for anomaly detection
        distances_to_centroid = {}
        if cluster_centers and int(cluster) < len(cluster_centers):
            centroid = cluster_centers[int(cluster)]
            
            # Try to load the doc_topic_matrix.json file
            doc_topic_matrix_path = results_dir / 'doc_topic_matrix.json'
            if os.path.exists(doc_topic_matrix_path):
                try:
                    with open(doc_topic_matrix_path, encoding='utf-8') as f:
                        doc_topic_data = json.load(f)
                    
                    # Create a dictionary for quick lookup
                    doc_topic_dict = {}
                    for item in doc_topic_data:
                        if 'doc_id' in item and 'topic_distribution' in item:
                            doc_topic_dict[str(item['doc_id'])] = item['topic_distribution']
                    
                    print(f"Chargé {len(doc_topic_dict)} distributions de topics depuis doc_topic_matrix.json")
                    
                    # Calculate distances for each document in this cluster
                    for doc_id, _ in cluster_docs:
                        doc_id_str = str(doc_id)
                        if doc_id_str in doc_topic_dict:
                            topic_dist = doc_topic_dict[doc_id_str]
                            dist = np.linalg.norm(np.array(topic_dist) - np.array(centroid))
                            distances_to_centroid[doc_id_str] = dist
                    
                    print(f"Calculé {len(distances_to_centroid)} distances au centroïde avec doc_topic_matrix.json")
                except Exception as e:
                    print(f"Erreur lors du chargement de doc_topic_matrix.json: {e}")
            
            # If we still don't have distances, try advanced topic modeling data
            if not distances_to_centroid and doc_topics:
                for doc_id, _ in cluster_docs:
                    doc_id_str = str(doc_id)
                    if doc_id_str in doc_topics:
                        topic_dist = doc_topics[doc_id_str]['topic_distribution']
                        dist = np.linalg.norm(np.array(topic_dist) - np.array(centroid))
                        distances_to_centroid[doc_id_str] = dist
                
                print(f"Calculé {len(distances_to_centroid)} distances au centroïde avec les données de topic modeling avancé")
            
            # If we still don't have distances, create simple distances based on random values
            if not distances_to_centroid:
                print("Aucune donnée de distance disponible, création de distances aléatoires pour démonstration")
                # Create random distances for demonstration
                np.random.seed(42)  # For reproducibility
                for doc_id, _ in cluster_docs:
                    distances_to_centroid[str(doc_id)] = np.random.random()
        
        # Debug info
        print(f"Nombre total d'articles dans le cluster: {len(cluster_docs)}")
        print(f"Nombre d'articles avec distance au centroïde: {len(distances_to_centroid)}")
        
        for doc_id, idx in sampled_docs:
            article = get_article_by_id(doc_id)
            if article:
                # Get text length
                text = article.get("text", article.get("content", ""))
                text_length = len(text)
                text_lengths.append(text_length)
                
                # Get newspaper if available
                newspaper = article.get("newspaper", "Inconnu")
                newspapers.append(newspaper)
                
                # Get canton if available
                canton = article.get("canton", "Inconnu")
                cantons.append(canton)
                
                # Get distance to centroid if available
                distance = distances_to_centroid.get(str(doc_id), None)
                
                # Store article info for finding typical and anomalous examples
                title = article.get("title", "Sans titre")
                article_info.append({
                    "id": doc_id,
                    "title": title,
                    "text_length": text_length,
                    "newspaper": newspaper,
                    "canton": canton,
                    "distance": distance,
                    "text_preview": text[:200] + "..." if len(text) > 200 else text
                })
        
        # Debug info
        print(f"Nombre d'articles échantillonnés: {len(article_info)}")
        print(f"Nombre d'articles avec distance: {len([a for a in article_info if a['distance'] is not None])}")

        # Find typical and anomalous articles
        typical_articles = []
        anomalous_articles = []
        
        if article_info:
            # Sort by text length to find the most typical articles (closest to median length)
            median_length = sorted(text_lengths)[len(text_lengths) // 2]
            sorted_by_typicality = sorted(article_info, key=lambda x: abs(x["text_length"] - median_length))
            
            # Get top 3 most typical articles
            typical_articles = sorted_by_typicality[:3]
            
            # Get anomalous articles (those with highest distance to centroid)
            articles_with_distance = [a for a in article_info if a["distance"] is not None]
            print(f"Articles avec distance pour anomalies: {len(articles_with_distance)}")
            
            if articles_with_distance:
                # Sort by distance to centroid (descending)
                sorted_by_distance = sorted(articles_with_distance, key=lambda x: x["distance"], reverse=True)
                
                # Get top 3 most anomalous articles
                anomalous_articles = sorted_by_distance[:3]
                print(f"Nombre d'anomalies trouvées: {len(anomalous_articles)}")
                
                # Calculate threshold for anomalies (80th percentile)
                all_distances = [a["distance"] for a in articles_with_distance]
                threshold = np.percentile(all_distances, 80)
                
                # Add anomaly explanation
                for article in anomalous_articles:
                    if article["distance"] > threshold:
                        # Try to get topic explanation
                        if doc_topics and article["id"] in doc_topics and advanced_data:
                            topic_dist = doc_topics[article["id"]]['topic_distribution']
                            centroid_dist = centroid
                            
                            # Calculate differences for each topic
                            topic_diffs = []
                            for t, (article_val, centroid_val) in enumerate(zip(topic_dist, centroid_dist)):
                                topic_diffs.append((t, abs(article_val - centroid_val)))
                            
                            # Sort by difference (descending)
                            topic_diffs.sort(key=lambda x: x[1], reverse=True)
                            
                            # Get top 3 differing topics
                            top_diff_topics = topic_diffs[:3]
                            
                            # Get topic names and words
                            topic_names = advanced_data.get('topic_names', {})
                            topic_words = advanced_data.get('topic_words', {})
                            
                            diff_descriptions = []
                            for topic_idx, diff_val in top_diff_topics:
                                topic_name = topic_names.get(str(topic_idx), f"Topic {topic_idx}")
                                words = topic_words.get(str(topic_idx), [])[:5]
                                words_str = ", ".join(words) if words else "pas de mots disponibles"
                                
                                if topic_dist[topic_idx] > centroid_dist[topic_idx]:
                                    direction = "plus élevé"
                                else:
                                    direction = "plus faible"
                                
                                diff_descriptions.append(f"{topic_name} ({direction}): {words_str}")
                            
                            article["anomaly_reason"] = "Diffère du centre pour les sujets: " + "; ".join(diff_descriptions)
                        else:
                            article["anomaly_reason"] = f"Distance au centroïde: {article['distance']:.4f} (seuil: {threshold:.4f})"
            else:
                print("Aucun article avec distance au centroïde trouvé")
                # Fallback: use articles with extreme lengths as anomalies
                sorted_by_length = sorted(article_info, key=lambda x: x["text_length"])
                if len(sorted_by_length) >= 2:
                    anomalous_articles = [sorted_by_length[0], sorted_by_length[-1]]
                    anomalous_articles[0]["anomaly_reason"] = "Article exceptionnellement court (aucune donnée de distance disponible)"
                    anomalous_articles[-1]["anomaly_reason"] = "Article exceptionnellement long (aucune donnée de distance disponible)"
                    print(f"Utilisation de {len(anomalous_articles)} articles extrêmes comme fallback")
        
        # Debug info
        print(f"Nombre d'articles typiques: {len(typical_articles)}")
        print(f"Nombre d'articles anomalies: {len(anomalous_articles)}")
        
        # Create newspaper distribution pie chart if available
        newspaper_chart = None
        if newspapers:
            # Count occurrences of each newspaper
            newspaper_counts = {}
            for newspaper in newspapers:
                if newspaper not in newspaper_counts:
                    newspaper_counts[newspaper] = 0
                newspaper_counts[newspaper] += 1
            
            # Sort by count (descending) and get top 5
            sorted_newspapers = sorted(newspaper_counts.items(), key=lambda x: x[1], reverse=True)[:5]
            
            # Create pie chart
            newspaper_fig = go.Figure(go.Pie(
                labels=[newspaper for newspaper, _ in sorted_newspapers],
                values=[count for _, count in sorted_newspapers],
                marker_colors=px.colors.qualitative.Pastel
            ))
            
            newspaper_fig.update_layout(
                title="Distribution des journaux",
                margin=dict(l=20, r=20, t=40, b=20),
                height=200,
                xaxis_title="",
                yaxis_title="",
                showlegend=False
            )
            
            newspaper_chart = dcc.Graph(figure=newspaper_fig)
        
        # Create canton distribution pie chart if available
        canton_chart = None
        if cantons:
            # Count occurrences of each canton
            canton_counts = {}
            for canton in cantons:
                if canton not in canton_counts:
                    canton_counts[canton] = 0
                canton_counts[canton] += 1
            
            # Sort by count (descending) and get top 5
            sorted_cantons = sorted(canton_counts.items(), key=lambda x: x[1], reverse=True)[:5]
            
            # Create pie chart
            canton_fig = go.Figure(go.Pie(
                labels=[canton for canton, _ in sorted_cantons],
                values=[count for _, count in sorted_cantons],
                marker_colors=px.colors.qualitative.Pastel
            ))
            
            canton_fig.update_layout(
                title="Distribution des cantons",
                margin=dict(l=20, r=20, t=40, b=20),
                height=200,
                xaxis_title="",
                yaxis_title="",
                showlegend=False
            )
            
            canton_chart = dcc.Graph(figure=canton_fig)
        
        # Create text length statistics if available
        text_length_section = None
        if text_lengths:
            text_length_section = html.Div([
                html.H5("Statistiques de longueur", className="mb-3"),
                html.P([
                    html.Strong("Longueur moyenne: "), 
                    f"{sum(text_lengths) / len(text_lengths):.0f} caractères"
                ], className="mb-1"),
                html.P([
                    html.Strong("Longueur min: "), 
                    f"{min(text_lengths)} caractères"
                ], className="mb-1"),
                html.P([
                    html.Strong("Longueur max: "), 
                    f"{max(text_lengths)} caractères"
                ], className="mb-1")
            ])
        
        # Create typical articles section
        typical_articles_section = None
        if typical_articles:
            typical_articles_section = html.Div([
                html.H5("Articles typiques du cluster", className="mb-3"),
                html.Div([
                    html.Div([
                        html.H6(f"{article['title']}", className="mb-1"),
                        html.P([
                            html.Strong("ID: "), article['id'], html.Br(),
                            html.Strong("Journal: "), article['newspaper'], html.Br(),
                            html.Strong("Canton: "), article['canton'], html.Br(),
                            html.Strong("Longueur: "), f"{article['text_length']} caractères"
                        ], className="small text-muted mb-2"),
                        html.P(article['text_preview'], className="small"),
                        html.Hr()
                    ]) for article in typical_articles
                ])
            ])
        
        # Create anomalous articles section
        anomalous_articles_section = None
        if anomalous_articles:
            anomalous_articles_section = html.Div([
                html.H5("Articles anomalies (éloignés du centroïde)", className="mb-3"),
                html.Div([
                    html.Div([
                        html.H6(f"{article['title']}", className="mb-1"),
                        html.P([
                            html.Strong("ID: "), article['id'], html.Br(),
                            html.Strong("Journal: "), article['newspaper'], html.Br(),
                            html.Strong("Canton: "), article['canton'], html.Br(),
                            html.Strong("Distance au centroïde: "), 
                            f"{article['distance']:.4f}" if article.get('distance') is not None else "Non calculée"
                        ], className="small text-muted mb-2"),
                        html.P(html.Strong("Raison: "), className="small text-danger mb-1") if "anomaly_reason" in article else None,
                        html.P(article.get("anomaly_reason", ""), className="small text-danger mb-2") if "anomaly_reason" in article else None,
                        html.P(article['text_preview'], className="small"),
                        html.Hr()
                    ]) for article in anomalous_articles
                ])
            ])
        
        # Create the detailed stats layout
        return html.Div([
            html.Hr(className="mt-3 mb-3"),
            html.H5("Statistiques détaillées", className="mb-3"),
            html.P(f"Basé sur un échantillon de {sample_size} articles", className="text-muted small"),
            dbc.Row([
                dbc.Col([newspaper_chart], width=6) if newspaper_chart else html.Div(),
                dbc.Col([canton_chart], width=6) if canton_chart else html.Div()
            ]),
            text_length_section if text_length_section else html.Div(),
            html.Hr(className="mt-4 mb-4"),
            dbc.Row([
                dbc.Col([typical_articles_section], width=6) if typical_articles_section else html.Div(),
                dbc.Col([anomalous_articles_section], width=6) if anomalous_articles_section else html.Div()
            ])
        ], className="mt-3")
    
    # Callback for the back button to return to clustering page
    @app.callback(
        Output("page-content", "children", allow_duplicate=True),
        Input("back-to-clustering-btn", "n_clicks"),
        prevent_initial_call=True
    )
    def navigate_back_to_clustering(n_clicks):
        if n_clicks:
            from src.webapp.topic_clustering_viz import get_clustering_layout
            return get_clustering_layout()
        return no_update
