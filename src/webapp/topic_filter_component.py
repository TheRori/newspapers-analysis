"""
Composant de filtrage par cluster pour l'application Dash.
Ce composant permet de filtrer les analyses par cluster.
"""

import os
import json
import pathlib
from typing import Dict, List, Optional, Any, Tuple
import dash
from dash import html, dcc, Input, Output, State, callback_context
import dash_bootstrap_components as dbc
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

def get_cluster_files():
    """
    Récupère les fichiers de clusters disponibles.
    
    Returns:
        Liste de dictionnaires avec label et value pour chaque fichier
    """
    project_root = pathlib.Path(__file__).resolve().parents[2]
    
    # Chercher les fichiers de clusters dans le répertoire des résultats
    cluster_files = []
    
    # Vérifier le répertoire des résultats de clusters
    clusters_dir = project_root / 'data' / 'results' / 'clusters'
    if clusters_dir.exists():
        print(f"Recherche de fichiers de clusters dans {clusters_dir}")
        # Chercher les fichiers JSON qui contiennent "cluster" dans leur nom
        cluster_files_found = list(clusters_dir.glob('*cluster*.json'))
        print(f"Trouvé {len(cluster_files_found)} fichiers de clusters: {[f.name for f in cluster_files_found]}")
        cluster_files.extend(cluster_files_found)
    else:
        print(f"Répertoire de clusters non trouvé: {clusters_dir}")
    
    # Vérifier aussi le répertoire des résultats de topic modeling
    topic_dir = project_root / 'data' / 'results' / 'doc_topic_matrix'
    if topic_dir.exists():
        print(f"Recherche de fichiers de clusters dans {topic_dir}")
        # Chercher les fichiers JSON qui contiennent "cluster" dans leur nom
        topic_cluster_files = list(topic_dir.glob('*cluster*.json'))
        print(f"Trouvé {len(topic_cluster_files)} fichiers de clusters: {[f.name for f in topic_cluster_files]}")
        cluster_files.extend(topic_cluster_files)
    else:
        print(f"Répertoire de topic modeling non trouvé: {topic_dir}")
    
    # Trier par date de modification (le plus récent en premier)
    cluster_files.sort(key=lambda x: os.path.getmtime(x), reverse=True)
    
    # Convertir en options pour le dropdown
    options = []
    for file_path in cluster_files:
        # Créer un label plus lisible
        label = f"{file_path.name} ({file_path.parent.name})"
        options.append({
            "label": label,
            "value": str(file_path)
        })
    
    return options

def get_cluster_ids(cluster_file):
    """
    Récupère les IDs de clusters disponibles dans un fichier de clusters.
    
    Args:
        cluster_file: Chemin vers le fichier de clusters
        
    Returns:
        Liste de dictionnaires avec label et value pour chaque ID de cluster
    """
    if not cluster_file:
        return []
    
    try:
        with open(cluster_file, 'r', encoding='utf-8') as f:
            cluster_data = json.load(f)
        
        options = []
        
        # Vérifier le format du fichier
        if "doc_ids" in cluster_data and "labels" in cluster_data and "n_clusters" in cluster_data:
            # Format avec doc_ids, labels et n_clusters
            n_clusters = cluster_data.get("n_clusters", 0)
            labels = cluster_data.get("labels", [])
            
            # Compter le nombre d'articles par cluster
            cluster_counts = {}
            for label in labels:
                if label not in cluster_counts:
                    cluster_counts[label] = 0
                cluster_counts[label] += 1
            
            # Créer les options pour chaque cluster
            for i in range(n_clusters):
                count = cluster_counts.get(i, 0)
                options.append({
                    "label": f"Cluster {i} ({count} articles)",
                    "value": str(i)
                })
        elif "clusters" in cluster_data:
            # Format avec une liste de clusters
            clusters = cluster_data.get("clusters", [])
            for cluster in clusters:
                cluster_id = cluster.get("id", "")
                articles = cluster.get("articles", [])
                options.append({
                    "label": f"Cluster {cluster_id} ({len(articles)} articles)",
                    "value": str(cluster_id)
                })
        else:
            # Format inconnu, essayer de détecter les clusters
            print(f"Format de fichier de clusters inconnu: {list(cluster_data.keys())}")
            
            # Si le fichier contient des clés numériques, supposer que ce sont des clusters
            numeric_keys = [k for k in cluster_data.keys() if str(k).isdigit()]
            if numeric_keys:
                for key in numeric_keys:
                    articles = cluster_data.get(key, [])
                    if isinstance(articles, list):
                        options.append({
                            "label": f"Cluster {key} ({len(articles)} articles)",
                            "value": str(key)
                        })
        
        return options
    
    except Exception as e:
        print(f"Erreur lors de la lecture du fichier de clusters: {e}")
        return []

def get_topic_filter_component(id_prefix: str = "topic-filter"):
    """
    Crée un composant de filtrage par cluster.
    
    Args:
        id_prefix: Préfixe pour les IDs des éléments du composant
        
    Returns:
        Composant Dash
    """
    # Créer le composant
    component = dbc.Card([
        dbc.CardHeader(html.H5("Filtrage par Cluster", className="mb-0")),
        dbc.CardBody([
            # Sélection du fichier de cluster
            dbc.Row([
                dbc.Col([
                    dbc.Label("Fichier de clusters"),
                    dcc.Dropdown(
                        id=f"{id_prefix}-cluster-file-dropdown",
                        options=[],  # Sera rempli dynamiquement
                        placeholder="Sélectionnez un fichier de clusters",
                        clearable=True
                    )
                ], width=12),
            ], className="mb-3"),
            
            # Sélection du cluster
            dbc.Row([
                dbc.Col([
                    dbc.Label("Cluster"),
                    dcc.Dropdown(
                        id=f"{id_prefix}-cluster-id-dropdown",
                        options=[],  # Sera rempli dynamiquement
                        placeholder="Sélectionnez un cluster",
                        clearable=True
                    )
                ], width=12),
            ], className="mb-3"),
            
            # Bouton d'application des filtres
            dbc.Row([
                dbc.Col([
                    dbc.Button(
                        "Appliquer les filtres",
                        id=f"{id_prefix}-apply-button",
                        color="primary",
                        className="w-100"
                    )
                ], width=12)
            ]),
            
            # Indicateur de filtres actifs
            dbc.Row([
                dbc.Col([
                    html.Div(
                        id=f"{id_prefix}-active-filters",
                        className="mt-3"
                    )
                ], width=12)
            ])
        ])
    ])
    
    return component

def register_topic_filter_callbacks(app, id_prefix: str = "topic-filter"):
    """
    Enregistre les callbacks pour le composant de filtrage par cluster.
    
    Args:
        app: Application Dash
        id_prefix: Préfixe pour les IDs des éléments du composant
    """
    # Callback pour remplir le dropdown des fichiers de clusters
    @app.callback(
        Output(f"{id_prefix}-cluster-file-dropdown", "options"),
        Input(f"{id_prefix}-cluster-file-dropdown", "id")  # Déclenché au chargement
    )
    def populate_cluster_files_dropdown(_):
        """
        Remplit le dropdown des fichiers de clusters disponibles.
        """
        return get_cluster_files()
    
    # Callback pour remplir le dropdown des IDs de clusters en fonction du fichier sélectionné
    @app.callback(
        Output(f"{id_prefix}-cluster-id-dropdown", "options"),
        Input(f"{id_prefix}-cluster-file-dropdown", "value")
    )
    def populate_cluster_ids_dropdown(cluster_file):
        """
        Remplit le dropdown des IDs de clusters disponibles en fonction du fichier sélectionné.
        """
        if not cluster_file:
            return []
        
        return get_cluster_ids(cluster_file)
    
    # Callback pour afficher les filtres actifs
    @app.callback(
        Output(f"{id_prefix}-active-filters", "children"),
        Input(f"{id_prefix}-apply-button", "n_clicks"),
        State(f"{id_prefix}-cluster-file-dropdown", "value"),
        State(f"{id_prefix}-cluster-id-dropdown", "value"),
        prevent_initial_call=True
    )
    def update_active_filters(n_clicks, cluster_file, cluster_id):
        """
        Met à jour l'affichage des filtres actifs.
        """
        if not n_clicks:
            return "Aucun filtre actif"
        
        active_filters = []
        
        if cluster_file and cluster_id:
            # Récupérer le nom du fichier
            file_name = pathlib.Path(cluster_file).name
            active_filters.append(f"Cluster: {cluster_id} (dans {file_name})")
        
        if active_filters:
            return html.Div([
                html.P("Filtres actifs:"),
                html.Ul([html.Li(filter_text) for filter_text in active_filters])
            ])
        else:
            return "Aucun filtre actif"

def get_filter_parameters(id_prefix: str = "topic-filter"):
    """
    Retourne les paramètres de filtrage pour les utiliser dans d'autres callbacks.
    
    Args:
        id_prefix: Préfixe pour les IDs des éléments du composant
        
    Returns:
        Dictionnaire des paramètres de filtrage
    """
    return {
        "cluster_file": f"{id_prefix}-cluster-file-dropdown",
        "cluster_id": f"{id_prefix}-cluster-id-dropdown",
        "apply_button": f"{id_prefix}-apply-button"
    }

def get_filter_states(id_prefix: str = "topic-filter"):
    """
    Retourne les états de filtrage pour les utiliser dans d'autres callbacks.
    
    Args:
        id_prefix: Préfixe pour les IDs des éléments du composant
        
    Returns:
        Dictionnaire des états de filtrage
    """
    params = get_filter_parameters(id_prefix)
    return {key: State(value, "value") for key, value in params.items() if key != "apply_button"}

def are_filters_active(id_prefix: str = "topic-filter", ctx=None):
    """
    Vérifie si des filtres sont actifs.
    
    Args:
        id_prefix: Préfixe pour les IDs des éléments du composant
        ctx: Contexte du callback
        
    Returns:
        True si des filtres sont actifs, False sinon
    """
    if ctx is None:
        from dash import callback_context
        ctx = callback_context
    
    # Vérifier si le bouton d'application des filtres a été cliqué
    if not ctx.triggered:
        return False
    
    trigger_id = ctx.triggered[0]['prop_id'].split('.')[0]
    params = get_filter_parameters(id_prefix)
    
    if trigger_id == params["apply_button"]:
        # Récupérer les valeurs des états
        states = ctx.states
        
        # Vérifier si des filtres sont actifs
        cluster_file = states.get(f"{params['cluster_file']}.value")
        cluster_id = states.get(f"{params['cluster_id']}.value")
        
        return cluster_file is not None and cluster_id is not None and cluster_id != "all"
    
    return False
