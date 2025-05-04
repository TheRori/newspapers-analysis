"""
Composant de filtrage par topic/cluster pour l'application Dash.
Ce composant permet de filtrer les analyses par topic ou cluster.
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

def get_topic_results():
    """
    Récupère les fichiers de résultats de topic modeling disponibles.
    
    Returns:
        Liste d'options pour le dropdown
    """
    project_root = pathlib.Path(__file__).resolve().parents[2]
    results_dir = project_root / 'data' / 'results' / 'advanced_topic'
    
    if not results_dir.exists():
        return []
    
    # Récupérer tous les fichiers d'analyse avancée
    result_files = list(results_dir.glob('advanced_topic_analysis*.json'))
    
    # Trier par date de modification (plus récent en premier)
    result_files.sort(key=lambda x: os.path.getmtime(x), reverse=True)
    
    # Formater pour le dropdown
    options = [
        {'label': f"{f.stem} ({pd.to_datetime(os.path.getmtime(f), unit='s').strftime('%Y-%m-%d %H:%M')})", 
         'value': str(f)}
        for f in result_files
    ]
    
    # Ajouter le fichier par défaut (sans date)
    default_file = results_dir / 'advanced_topic_analysis.json'
    if default_file.exists() and default_file not in result_files:
        options.insert(0, {
            'label': f"Analyse par défaut ({pd.to_datetime(os.path.getmtime(default_file), unit='s').strftime('%Y-%m-%d %H:%M')})",
            'value': str(default_file)
        })
    
    return options

def get_cluster_results():
    """
    Récupère les fichiers de résultats de clustering disponibles.
    
    Returns:
        Liste d'options pour le dropdown
    """
    project_root = pathlib.Path(__file__).resolve().parents[2]
    results_dir = project_root / 'data' / 'results' / 'clusters'
    
    if not results_dir.exists():
        return []
    
    # Récupérer tous les fichiers de clustering
    result_files = list(results_dir.glob('doc_clusters_*.json'))
    
    # Trier par date de modification (plus récent en premier)
    result_files.sort(key=lambda x: os.path.getmtime(x), reverse=True)
    
    # Formater pour le dropdown
    options = [
        {'label': f"{f.stem} ({pd.to_datetime(os.path.getmtime(f), unit='s').strftime('%Y-%m-%d %H:%M')})", 
         'value': str(f)}
        for f in result_files
    ]
    
    return options

def load_topic_data(file_path: str) -> Dict[str, Any]:
    """
    Charge les données de topic modeling à partir d'un fichier.
    
    Args:
        file_path: Chemin vers le fichier de résultats
        
    Returns:
        Données de topic modeling
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data
    except Exception as e:
        print(f"Erreur lors du chargement des données de topic: {e}")
        return {}

def load_cluster_data(file_path: str) -> Dict[str, Any]:
    """
    Charge les données de clustering à partir d'un fichier.
    
    Args:
        file_path: Chemin vers le fichier de résultats
        
    Returns:
        Données de clustering
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data
    except Exception as e:
        print(f"Erreur lors du chargement des données de clustering: {e}")
        return {}

def extract_topic_options(topic_data: Dict[str, Any]) -> List[Dict[str, str]]:
    """
    Extrait les options de topics à partir des données de topic modeling.
    
    Args:
        topic_data: Données de topic modeling
        
    Returns:
        Liste d'options pour le dropdown
    """
    options = []
    
    # Récupérer les noms de topics s'ils existent
    topic_names = topic_data.get('topic_names_llm', {})
    
    # Récupérer les mots-clés par topic
    top_words = topic_data.get('weighted_words', {})
    
    # Récupérer le nombre d'articles par topic
    topic_counts = topic_data.get('topic_article_counts', {})
    
    # Créer les options
    for topic_id in sorted(top_words.keys()):
        # Récupérer le nom du topic s'il existe
        topic_name = topic_names.get(f"topic_{topic_id}", f"Topic {topic_id}")
        
        # Récupérer les mots-clés
        words = [word for word, _ in top_words.get(topic_id, [])][:5]
        
        # Récupérer le nombre d'articles
        count = topic_counts.get(topic_id, 0)
        
        # Créer l'option
        option = {
            'label': f"{topic_name} ({count} articles) - {', '.join(words)}",
            'value': topic_id
        }
        
        options.append(option)
    
    return options

def extract_cluster_options(cluster_data: Dict[str, Any]) -> List[Dict[str, str]]:
    """
    Extrait les options de clusters à partir des données de clustering.
    
    Args:
        cluster_data: Données de clustering
        
    Returns:
        Liste d'options pour le dropdown
    """
    options = []
    
    if not cluster_data or 'labels' not in cluster_data:
        return []
    
    # Compter les documents par cluster
    labels = cluster_data.get('labels', [])
    cluster_counts = {}
    for label in labels:
        label_str = str(label)
        if label_str not in cluster_counts:
            cluster_counts[label_str] = 0
        cluster_counts[label_str] += 1
    
    # Créer les options
    for cluster_id, count in sorted(cluster_counts.items(), key=lambda x: int(x[0])):
        # Créer l'option
        option = {
            'label': f"Cluster {cluster_id} ({count} articles)",
            'value': cluster_id
        }
        
        options.append(option)
    
    return options

def get_topic_filter_component(id_prefix: str = "topic-filter"):
    """
    Crée un composant de filtrage par topic/cluster.
    
    Args:
        id_prefix: Préfixe pour les IDs des éléments du composant
        
    Returns:
        Composant Dash
    """
    # Récupérer les fichiers de résultats disponibles
    topic_results = get_topic_results()
    cluster_results = get_cluster_results()
    
    # Créer le composant
    component = dbc.Card([
        dbc.CardHeader(html.H5("Filtrage par Topic / Cluster", className="mb-0")),
        dbc.CardBody([
            # Sélection du fichier de résultats
            dbc.Row([
                dbc.Col([
                    dbc.Label("Fichier de résultats de topic modeling:"),
                    dcc.Dropdown(
                        id=f"{id_prefix}-results-dropdown",
                        options=topic_results,
                        value=topic_results[0]['value'] if topic_results else None,
                        placeholder="Sélectionnez un fichier de résultats",
                        className="mb-3"
                    )
                ], width=12)
            ]),
            
            # Sélection du fichier de clustering
            dbc.Row([
                dbc.Col([
                    dbc.Label("Fichier de résultats de clustering:"),
                    dcc.Dropdown(
                        id=f"{id_prefix}-cluster-results-dropdown",
                        options=cluster_results,
                        value=cluster_results[0]['value'] if cluster_results else None,
                        placeholder="Sélectionnez un fichier de résultats de clustering",
                        className="mb-3"
                    )
                ], width=12)
            ]),
            
            # Filtres de topic/cluster
            dbc.Row([
                # Filtre par topic
                dbc.Col([
                    dbc.Label("Filtrer par topic:"),
                    dcc.Dropdown(
                        id=f"{id_prefix}-topic-dropdown",
                        options=[],
                        value=None,
                        placeholder="Sélectionnez un topic",
                        className="mb-3"
                    )
                ], width=6),
                
                # Filtre par cluster
                dbc.Col([
                    dbc.Label("Filtrer par cluster:"),
                    dcc.Dropdown(
                        id=f"{id_prefix}-cluster-dropdown",
                        options=[],
                        value=None,
                        placeholder="Sélectionnez un cluster",
                        className="mb-3"
                    )
                ], width=6)
            ]),
            
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
    Enregistre les callbacks pour le composant de filtrage par topic/cluster.
    
    Args:
        app: Application Dash
        id_prefix: Préfixe pour les IDs des éléments du composant
    """
    # Callback pour charger les options de topics
    @app.callback(
        Output(f"{id_prefix}-topic-dropdown", "options"),
        Input(f"{id_prefix}-results-dropdown", "value")
    )
    def update_topic_options(results_file):
        if not results_file:
            return []
        
        # Charger les données de topic modeling
        topic_data = load_topic_data(results_file)
        
        # Extraire les options de topics
        return extract_topic_options(topic_data)
    
    # Callback pour charger les options de clusters
    @app.callback(
        Output(f"{id_prefix}-cluster-dropdown", "options"),
        Input(f"{id_prefix}-cluster-results-dropdown", "value")
    )
    def update_cluster_options(cluster_file):
        if not cluster_file:
            return []
        
        # Charger les données de clustering
        cluster_data = load_cluster_data(cluster_file)
        
        # Extraire les options de clusters
        return extract_cluster_options(cluster_data)
    
    # Callback pour afficher les filtres actifs
    @app.callback(
        Output(f"{id_prefix}-active-filters", "children"),
        [
            Input(f"{id_prefix}-topic-dropdown", "value"),
            Input(f"{id_prefix}-cluster-dropdown", "value")
        ]
    )
    def update_active_filters(topic, cluster):
        active_filters = []
        
        if topic:
            active_filters.append(
                dbc.Badge(f"Topic: {topic}", color="primary", className="me-1")
            )
        
        if cluster:
            active_filters.append(
                dbc.Badge(f"Cluster: {cluster}", color="info", className="me-1")
            )
        
        if not active_filters:
            return html.Div("Aucun filtre actif", className="text-muted")
        
        return html.Div([
            html.Span("Filtres actifs: ", className="me-2"),
            *active_filters
        ])
    
    # Callback pour réinitialiser le filtre de topic quand on change de fichier
    @app.callback(
        Output(f"{id_prefix}-topic-dropdown", "value"),
        Input(f"{id_prefix}-results-dropdown", "value"),
        prevent_initial_call=True
    )
    def reset_topic_filter(results_file):
        return None
    
    # Callback pour réinitialiser le filtre de cluster quand on change de fichier
    @app.callback(
        Output(f"{id_prefix}-cluster-dropdown", "value"),
        Input(f"{id_prefix}-cluster-results-dropdown", "value"),
        prevent_initial_call=True
    )
    def reset_cluster_filter(cluster_file):
        return None

def get_filter_parameters(id_prefix: str = "topic-filter"):
    """
    Retourne les paramètres de filtrage pour les utiliser dans d'autres callbacks.
    
    Args:
        id_prefix: Préfixe pour les IDs des éléments du composant
        
    Returns:
        Dictionnaire des paramètres de filtrage
    """
    return {
        "topic_file": Input(f"{id_prefix}-results-dropdown", "value"),
        "topic": Input(f"{id_prefix}-topic-dropdown", "value"),
        "cluster_file": Input(f"{id_prefix}-cluster-results-dropdown", "value"),
        "cluster": Input(f"{id_prefix}-cluster-dropdown", "value"),
        "apply_button": Input(f"{id_prefix}-apply-button", "n_clicks")
    }

def get_filter_states(id_prefix: str = "topic-filter"):
    """
    Retourne les états de filtrage pour les utiliser dans d'autres callbacks.
    
    Args:
        id_prefix: Préfixe pour les IDs des éléments du composant
        
    Returns:
        Dictionnaire des états de filtrage
    """
    return {
        "topic_file": State(f"{id_prefix}-results-dropdown", "value"),
        "topic": State(f"{id_prefix}-topic-dropdown", "value"),
        "cluster_file": State(f"{id_prefix}-cluster-results-dropdown", "value"),
        "cluster": State(f"{id_prefix}-cluster-dropdown", "value")
    }

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
        ctx = callback_context
    
    # Vérifier si le bouton d'application des filtres a été cliqué
    if not ctx.triggered:
        return False
    
    trigger_id = ctx.triggered[0]["prop_id"].split(".")[0]
    
    if trigger_id == f"{id_prefix}-apply-button":
        return True
    elif trigger_id == f"{id_prefix}-reset-button":
        return False
    
    return False
