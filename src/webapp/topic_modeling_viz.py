"""
Topic Modeling Visualization Page for Dash app
"""

print("[topic_modeling_viz] Début de l'import du module")

import dash
print("[topic_modeling_viz] dash importé")
from dash import html, dcc, Input, Output, State, ctx, ALL, MATCH, no_update
print("[topic_modeling_viz] dash.html, dcc, Input, Output, State, ctx importés")
from src.webapp.topic_filter_component import get_topic_filter_component, register_topic_filter_callbacks
import dash_bootstrap_components as dbc
print("[topic_modeling_viz] dash_bootstrap_components importé")
import plotly.express as px
print("[topic_modeling_viz] plotly.express importé")
import plotly.graph_objects as go
print("[topic_modeling_viz] plotly.graph_objects importé")
import subprocess
print("[topic_modeling_viz] subprocess importé")
import pathlib
print("[topic_modeling_viz] pathlib importé")
import yaml
print("[topic_modeling_viz] yaml importé")
import pandas as pd
print("[topic_modeling_viz] pandas importé")
import json
print("[topic_modeling_viz] json importé")
import os
print("[topic_modeling_viz] os importé")
import sys
print("[topic_modeling_viz] sys importé")
import threading
print("[topic_modeling_viz] threading importé")
import re
print("[topic_modeling_viz] re importé")

print("[topic_modeling_viz] Début des définitions de fonctions")

# Helper to get config and paths
def get_config_and_paths():
    project_root = pathlib.Path(__file__).resolve().parents[2]
    config_path = project_root / 'config' / 'config.yaml'
    with open(config_path, encoding='utf-8') as f:
        config = yaml.safe_load(f)
    results_dir = project_root / config['data']['results_dir']
    advanced_topic_dir = results_dir / 'advanced_topic'
    return project_root, config, advanced_topic_dir

# Helper to get available topic modeling result files
def get_topic_modeling_results():
    print("Démarrage de get_topic_modeling_results()...")
    
    project_root, config, advanced_topic_dir = get_config_and_paths()
    
    if not advanced_topic_dir.exists():
        print(f"Répertoire {advanced_topic_dir} non trouvé")
        return []
    
    # Get all topic modeling result files
    print(f"Recherche des fichiers dans {advanced_topic_dir}...")
    result_files = list(advanced_topic_dir.glob('advanced_topic_analysis*.json'))
    print(f"Nombre de fichiers trouvés: {len(result_files)}")
    
    # Sort by modification time (newest first)
    result_files.sort(key=lambda x: os.stat(x).st_mtime, reverse=True)
    
    # Ajout du paramètre de cache-busting pour forcer le rechargement des fichiers
    import time as time_module
    cache_buster = int(time_module.time())
    
    # Format for dropdown
    options = []
    for file in result_files:
        # Extract date and time from filename if possible
        match = re.search(r'advanced_topic_analysis_?(\d+)?\.json', file.name)
        if match and match.group(1):
            # If there's a timestamp in the filename
            timestamp = match.group(1)
            # Try to format it nicely if it's a valid timestamp format
            try:
                from datetime import datetime
                date_str = datetime.strptime(timestamp, '%Y%m%d_%H%M%S').strftime('%d/%m/%Y %H:%M')
                label = f"Analyse du {date_str}"
            except ValueError:
                # If not a valid timestamp format, just use the raw string
                label = f"Analyse {timestamp}"
        else:
            # If no timestamp in filename, use last modified time
            from datetime import datetime
            mod_time = datetime.fromtimestamp(os.path.getmtime(file))
            label = f"Analyse du {mod_time.strftime('%d/%m/%Y %H:%M')}"
        
        # Add cache buster to force reload when needed
        value = f"{file}?cache={cache_buster}"
        options.append({"label": label, "value": value})
    
    # Add default option if no files found
    if not options:
        options = [{"label": "Aucun résultat disponible", "value": ""}]
    
    print(f"Options de fichiers de résultats: {len(options)}")
    return options

# Extract parser arguments from run_topic_modeling.py
def get_topic_modeling_args():
    import importlib.util
    import sys as _sys
    import os as _os
    import argparse as _argparse
    spec = importlib.util.spec_from_file_location(
        "run_topic_modeling", _os.path.join(_os.path.dirname(__file__), "..", "scripts", "run_topic_modeling.py")
    )
    run_topic_modeling = importlib.util.module_from_spec(spec)
    _sys.modules["run_topic_modeling"] = run_topic_modeling
    spec.loader.exec_module(run_topic_modeling)
    parser = run_topic_modeling.get_parser()
    parser_args = []
    for action in parser._actions:
        if action.dest == 'help':
            continue
        # Robust: Detect boolean flags via argparse action type
        is_bool = isinstance(action, (_argparse._StoreTrueAction, _argparse._StoreFalseAction))
        arg_type = 'bool' if is_bool else (getattr(action, "type", str).__name__ if hasattr(action, "type") and getattr(action, "type") is not None else "str")
        parser_args.append({
            "name": action.dest,
            "flags": action.option_strings,
            "help": action.help,
            "required": getattr(action, "required", False),
            "default": action.default,
            "type": arg_type,
            "choices": getattr(action, "choices", None)
        })
    return parser_args

# Helper to generate dash controls for each argument
from dash import html as _html

def get_topic_modeling_controls():
    parser_args = get_topic_modeling_args()
    controls = []
    controls.append(_html.Div(f"Nombre d'arguments trouvés: {len(parser_args)}", className="alert alert-info"))
    for arg in parser_args:
        label = arg['help'] or arg['name']
        input_id = f"arg-{arg['name']}"
        row = []
        row.append(dbc.Label(label, html_for=input_id, className="mb-1 fw-bold"))
        if arg['choices']:
            options = [{'label': str(c), 'value': c} for c in arg['choices']]
            if not arg['required']:
                options = [{'label': '-- Non spécifié --', 'value': ''}] + options
            row.append(dcc.Dropdown(
                id=input_id,
                options=options,
                value=str(arg['default']) if arg['default'] is not None else '',
                clearable=not arg['required'],
                className="mb-2"
            ))
        elif arg['type'] == 'int':
            # Set appropriate min/max values based on parameter name
            if arg['name'] == 'k_min':
                min_val = 2  # Allow k_min to be as low as 2
                max_val = 50  # Reasonable upper limit
            elif arg['name'] == 'k_max':
                min_val = 5
                max_val = 100
            elif arg['name'] == 'num_topics':
                min_val = 2
                max_val = 50
            else:
                # Default values for other integer parameters
                min_val = 0
                max_val = 100
                
            row.append(dcc.Input(
                id=input_id, 
                type="number", 
                value=arg['default'], 
                required=arg['required'], 
                className="mb-2", 
                min=min_val, 
                max=max_val
            ))
        elif arg['type'] == 'bool':
            row.append(dbc.Checkbox(id=input_id, value=bool(arg['default']), className="mb-2"))
        else:
            row.append(dcc.Input(id=input_id, type="text", value=arg['default'] if arg['default'] is not None else '', required=arg['required'], className="mb-2"))
        if arg['help']:
            row.append(_html.Div(arg['help'], className="form-text text-secondary mb-2"))
        controls.append(dbc.Row([dbc.Col(c) for c in row], className="mb-2"))
    return controls

# Layout for the topic modeling page
def get_topic_modeling_layout():
    return dbc.Container([
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader(_html.H3("Paramètres du Topic Modeling", className="mb-0")),
                    dbc.CardBody([
                        _html.P("Configurez les paramètres de l'analyse thématique ci-dessous, puis cliquez sur 'Lancer'.", className="text-muted mb-3"),
                        
                        # Fichier source personnalisé
                        _html.H5("Fichier source", className="mb-2"),
                        dbc.Row([
                            dbc.Col([
                                dbc.InputGroup([
                                    dbc.Input(
                                        id="arg-input-file",
                                        type="text",
                                        placeholder="Chemin vers le fichier JSON d'articles"
                                    ),
                                    dbc.Button("Parcourir", id="source-file-browse", color="secondary")
                                ]),
                                _html.Small("Laissez vide pour utiliser le fichier par défaut de la configuration.", className="text-muted")
                            ], width=12),
                        ], className="mb-3"),
                        
                        # Sélection de fichier de cache
                        _html.H5("Fichier de cache Spacy", className="mb-2"),
                        dbc.Row([
                            dbc.Col([
                                dbc.Select(
                                    id="cache-file-select",
                                    options=[{"label": "Aucun (utiliser le plus récent)", "value": ""}],
                                    value="",
                                    className="mb-2"
                                ),

                                _html.Small("Sélectionnez un fichier de cache Spacy existant pour accélérer le traitement.", className="text-muted d-block"),
                                _html.Div(id="cache-info-display", className="mt-2")
                            ], width=12),
                        ], className="mb-3"),
                        

                        
                        dbc.Form(get_topic_modeling_controls()),
                        
                        # Add topic filter component
                        _html.H5("Filtrage par cluster", className="mt-4 mb-3"),
                        get_topic_filter_component(id_prefix="topic-filter"),
                        
                        dbc.Button("Lancer le Topic Modeling", id="btn-run-topic-modeling", color="primary", n_clicks=0, className="mt-3 mb-2"),
                        _html.Div(id="topic-modeling-run-status", className="mb-3"),
                    ]),
                ], className="mb-4 shadow"),
            ], width=12)
        ]),
        # Sélecteur de fichiers de résultats
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader(_html.H4("Résultats de Topic Modeling", className="mb-0")),
                    dbc.CardBody([
                        _html.Div([
                            dbc.Label("Sélectionner un fichier de résultats:", className="fw-bold"),
                            dcc.Dropdown(
                                id="topic-modeling-results-dropdown",
                                options=get_topic_modeling_results(),
                                value=get_topic_modeling_results()[0]['value'] if get_topic_modeling_results() else None,
                                clearable=False,
                                className="mb-3"
                            ),
                        ]),
                    ])
                ], className="mb-4 shadow")
            ], width=12)
        ]),
        # Tabs for different visualizations
        dbc.Tabs([
            dbc.Tab([
                dbc.Row([
                    dbc.Col([
                        _html.H4("Statistiques avancées", className="mt-4 mb-3"),
                        dcc.Loading(
                            id="loading-advanced-topic-stats",
                            type="default",
                            children=_html.Div(id="advanced-topic-stats-content")
                        )
                    ], width=12)
                ])
            ], label="Statistiques", tab_id="stats-tab"),
            dbc.Tab([
                dbc.Row([
                    dbc.Col([
                        _html.H4("Explorateur d'articles", className="mt-4 mb-3"),
                        dcc.Loading(
                            id="loading-article-browser",
                            type="default",
                            children=_html.Div(id="article-browser-content")
                        )
                    ], width=12)
                ])
            ], label="Explorateur d'articles", tab_id="article-browser-tab"),
            dbc.Tab([
                dbc.Row([
                    dbc.Col([
                        _html.H4("Filtrage des publicités par topic", className="mt-4 mb-3"),
                        _html.P("Cet outil vous permet de détecter et filtrer les publicités d'un topic spécifique en utilisant un LLM.", className="text-muted"),
                        dbc.Card([
                            dbc.CardBody([
                                dbc.Row([
                                    dbc.Col([
                                        dbc.Label("Topic à analyser", html_for="ad-filter-topic-id"),
                                        dbc.InputGroup([
                                            dbc.Select(
                                                id="ad-filter-topic-id",
                                                options=[{"label": "Chargement des topics...", "value": ""}],  # Valeur initiale
                                                value=None,
                                                className="mb-2"
                                            ),
                                            dbc.Button("Rafraîchir", id="btn-refresh-topics", color="secondary", className="mb-2 ms-2")
                                        ]),
                                    ], width=6),
                                    dbc.Col([
                                        dbc.Label("Valeur minimale du topic", html_for="ad-filter-min-value"),
                                        dbc.Input(
                                            id="ad-filter-min-value",
                                            type="number",
                                            min=0.1,
                                            max=0.9,
                                            step=0.1,
                                            value=0.5,
                                            className="mb-2"
                                        ),
                                    ], width=6),
                                ]),
                                dbc.Row([
                                    dbc.Col([
                                        dbc.Label("Fichier de sortie", html_for="ad-filter-output-path"),
                                        dbc.Input(
                                            id="ad-filter-output-path",
                                            type="text",
                                            placeholder="Laissez vide pour générer automatiquement",
                                            className="mb-2"
                                        ),
                                    ], width=12),
                                ]),
                                dbc.Row([
                                    dbc.Col([
                                        dbc.Checkbox(
                                            id="ad-filter-dry-run",
                                            label="Mode test (ne pas écrire le fichier)",
                                            value=False,
                                            className="mb-3"
                                        ),
                                    ], width=12),
                                ]),
                                dbc.Button(
                                    "Lancer le filtrage des publicités",
                                    id="btn-run-ad-filter",
                                    color="primary",
                                    className="mt-2"
                                ),
                            ])
                        ], className="mb-4"),
                        _html.Div(id="ad-filter-status", className="mt-3"),
                        dcc.Loading(
                            id="loading-ad-filter-results",
                            type="default",
                            children=_html.Div(id="ad-filter-results")
                        )
                    ], width=12)
                ])
            ], label="Filtrage des publicités", tab_id="ad-filter-tab")
        ], id="topic-modeling-tabs", active_tab="stats-tab"),
        # Le Store pour l'état des filtres a été supprimé
    ], fluid=True)

# Callback registration
# Fonction pour obtenir les informations sur les fichiers de cache
def get_cache_info():
    """
    Récupère les informations sur les fichiers de cache Spacy existants.
    
    Returns:
        dict: Informations sur les fichiers de cache
    """
    project_root = pathlib.Path(__file__).resolve().parents[2]
    cache_dir = project_root / 'data' / 'cache'
    cache_files = list(cache_dir.glob("preprocessed_docs_*.pkl"))
    
    cache_info = {
        "count": len(cache_files),
        "files": []
    }
    
    for cache_file in cache_files:
        try:
            import pickle
            with open(cache_file, 'rb') as f:
                cache_data = pickle.load(f)
                
            # Extraire les informations du cache
            cache_key_data = cache_data.get('cache_key_data', {})
            articles_path = cache_key_data.get('articles_path', 'Inconnu')
            spacy_model = cache_key_data.get('spacy_model', 'Inconnu')
            allowed_pos = cache_key_data.get('allowed_pos', [])
            min_token_length = cache_key_data.get('min_token_length', 0)
            articles_count = cache_key_data.get('articles_count', 0)
            
            # Taille du fichier
            file_size_bytes = cache_file.stat().st_size
            file_size_mb = file_size_bytes / (1024 * 1024)
            
            # Date de création
            from datetime import datetime
            creation_time = datetime.fromtimestamp(cache_file.stat().st_mtime)
            
            cache_info["files"].append({
                "filename": cache_file.name,
                "articles_path": articles_path,
                "spacy_model": spacy_model,
                "allowed_pos": allowed_pos,
                "min_token_length": min_token_length,
                "articles_count": articles_count,
                "file_size_mb": file_size_mb,
                "creation_time": creation_time.strftime("%Y-%m-%d %H:%M:%S")
            })
        except Exception as e:
            print(f"Erreur lors de la lecture du fichier de cache {cache_file}: {e}")
            cache_info["files"].append({
                "filename": cache_file.name,
                "error": str(e),
                "file_size_mb": file_size_bytes / (1024 * 1024) if 'file_size_bytes' in locals() else 0
            })
    
    return cache_info

# Function to load and display the article browser with topic distribution
def load_article_browser(custom_doc_topic_path=None):
    """
    Loads the doc_topic_matrix.json file and creates an interactive table to browse articles
    with their topic distributions.
    
    Args:
        custom_doc_topic_path: Chemin personnalisé vers un fichier doc_topic_matrix.json
    
    Returns:
        dash components for the article browser
    """
    project_root = pathlib.Path(__file__).resolve().parents[2]
    config_path = project_root / 'config' / 'config.yaml'
    
    with open(config_path, encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    results_dir = project_root / config['data']['results_dir']
    
    # Utiliser le chemin personnalisé s'il est fourni, sinon utiliser le chemin par défaut
    if custom_doc_topic_path and os.path.exists(custom_doc_topic_path):
        doc_topic_matrix_path = custom_doc_topic_path
        print(f"Utilisation du fichier doc_topic_matrix personnalisé: {doc_topic_matrix_path}")
    else:
        doc_topic_matrix_path = results_dir / 'doc_topic_matrix.json'
        print(f"Utilisation du fichier doc_topic_matrix par défaut: {doc_topic_matrix_path}")
        
    articles_path = project_root / 'data' / 'processed' / 'articles.json'
    
    if not doc_topic_matrix_path.exists():
        return html.Div("Fichier doc_topic_matrix.json introuvable. Exécutez d'abord le topic modeling.", 
                       className="alert alert-warning")
    
    # Load doc_topic_matrix.json
    with open(doc_topic_matrix_path, 'r', encoding='utf-8') as f:
        doc_topic_data = json.load(f)
    
    # Check if the file has the expected structure
    if not isinstance(doc_topic_data, list) and 'doc_topic_matrix' in doc_topic_data:
        doc_topic_matrix = doc_topic_data['doc_topic_matrix']
    else:
        doc_topic_matrix = doc_topic_data
    
    # Load article information if available
    article_info = {}
    if articles_path.exists():
        try:
            with open(articles_path, 'r', encoding='utf-8') as f:
                articles = json.load(f)
            
            # Create a lookup dictionary for article information
            for article in articles:
                article_id = article.get('doc_id', article.get('id', ''))
                if article_id:
                    # Extract date and newspaper from article ID if available
                    date = ''
                    newspaper = ''
                    if isinstance(article_id, str) and 'article_' in article_id:
                        parts = article_id.split('_')
                        if len(parts) > 2:
                            date = parts[1]  # Extract date part
                        if len(parts) > 3:
                            newspaper = parts[2]  # Extract newspaper part
                    
                    article_info[str(article_id)] = {
                        'title': article.get('title', 'Sans titre'),
                        'date': article.get('date', date),
                        'newspaper': article.get('newspaper', newspaper),
                        'content': article.get('content', article.get('original_content', 'Contenu non disponible'))[:200] + '...'  # Preview
                    }
        except Exception as e:
            print(f"Erreur lors du chargement des articles: {e}")
    
    # Load topic names if available
    topic_names = {}
    advanced_topic_json = results_dir / 'advanced_topic' / 'advanced_topic_analysis.json'
    if advanced_topic_json.exists():
        try:
            with open(advanced_topic_json, encoding='utf-8') as f:
                stats = json.load(f)
            
            if stats.get('topic_names_llm'):
                # Can be string or dict
                if isinstance(stats['topic_names_llm'], dict):
                    topic_names = stats['topic_names_llm']
                else:
                    try:
                        import ast
                        topic_names = ast.literal_eval(stats['topic_names_llm'])
                    except Exception:
                        topic_names = {}
        except Exception as e:
            print(f"Erreur lors du chargement des noms de topics: {e}")
    
    # Prepare data for the table
    table_data = []
    for item in doc_topic_matrix:
        doc_id = item.get('doc_id', '')
        topic_distribution = item.get('topic_distribution', [])
        
        # Get article info if available
        info = article_info.get(str(doc_id), {})
        
        # Find dominant topic
        dominant_topic_idx = 0
        if topic_distribution:
            dominant_topic_idx = topic_distribution.index(max(topic_distribution))
        
        # Format topic distribution for display
        topic_dist_formatted = []
        for i, value in enumerate(topic_distribution):
            topic_name = topic_names.get(f'topic_{i}', f"Topic {i}")
            topic_dist_formatted.append({
                'topic_id': i,
                'topic_name': topic_name,
                'value': value
            })
        
        row = {
            'doc_id': doc_id,
            'title': info.get('title', 'Sans titre'),
            'date': info.get('date', ''),
            'newspaper': info.get('newspaper', ''),
            'content_preview': info.get('content', 'Contenu non disponible'),
            'dominant_topic': dominant_topic_idx,
            'dominant_topic_name': topic_names.get(f'topic_{dominant_topic_idx}', f"Topic {dominant_topic_idx}"),
            'dominant_topic_value': max(topic_distribution) if topic_distribution else 0,
            'topic_distribution': topic_distribution,
            'topic_dist_formatted': topic_dist_formatted
        }
        table_data.append(row)
    
    # Create dropdown for sorting options
    num_topics = len(table_data[0]['topic_distribution']) if table_data else 0
    sort_options = [{'label': 'ID du document', 'value': 'doc_id'}]
    sort_options.append({'label': 'Date', 'value': 'date'})
    sort_options.append({'label': 'Journal', 'value': 'newspaper'})
    sort_options.append({'label': 'Topic dominant', 'value': 'dominant_topic'})
    
    for i in range(num_topics):
        topic_name = topic_names.get(f'topic_{i}', f"Topic {i}")
        sort_options.append({'label': f'Valeur du {topic_name}', 'value': f'topic_{i}'})
    
    # Create the layout
    children = [
        dbc.Row([
            dbc.Col([
                html.H5("Trier les articles par:"),
                dcc.Dropdown(
                    id='article-sort-dropdown',
                    options=sort_options,
                    value='dominant_topic',
                    clearable=False
                ),
                dbc.Checkbox(
                    id='sort-descending-checkbox',
                    label="Ordre décroissant",
                    value=True,
                    className="mt-2 mb-3"
                )
            ], width=6),
            dbc.Col([
                html.H5("Filtrer par topic dominant:"),
                dcc.Dropdown(
                    id='dominant-topic-filter',
                    options=[{'label': 'Tous les topics', 'value': 'all'}] + [
                        {'label': topic_names.get(f'topic_{i}', f"Topic {i}"), 'value': i}
                        for i in range(num_topics)
                    ],
                    value='all',
                    clearable=False
                )
            ], width=6)
        ]),
        
        # Store the data
        dcc.Store(id='article-browser-data', data=table_data),
        
        # Table to display articles
        html.Div(id='article-browser-table-container', className="mt-4"),
        
        # Modal for viewing article details
        dbc.Modal([
            dbc.ModalHeader(dbc.ModalTitle("Détails de l'article")),
            dbc.ModalBody(id="article-detail-body"),
            dbc.ModalFooter(
                dbc.Button("Fermer", id="close-article-modal", className="ms-auto", n_clicks=0)
            ),
        ], id="article-detail-modal", size="lg", is_open=False),
    ]
    
    return html.Div(children)

def register_topic_modeling_callbacks(app):
    # Register the topic filter component callbacks
    register_topic_filter_callbacks(app, id_prefix="topic-filter")
    parser_args = get_topic_modeling_args()
    
    # Callback pour remplir la liste des topics disponibles pour le filtrage des publicités
    @app.callback(
        Output("ad-filter-topic-id", "options"),
        [Input("topic-modeling-tabs", "active_tab"),
         Input("btn-refresh-topics", "n_clicks"),
         Input("page-content", "children")],
        prevent_initial_call=False  # Permettre l'appel initial
    )
    def update_ad_filter_topic_options(active_tab, n_clicks, page_content):
        # Déterminer quel élément a déclenché le callback
        trigger = ctx.triggered_id if ctx.triggered else None
        
        # Si le déclencheur est le changement d'onglet et que ce n'est pas l'onglet de filtrage, ne rien faire
        if trigger == "topic-modeling-tabs" and active_tab != "ad-filter-tab":
            return dash.no_update
        
        # Récupérer les informations sur les topics
        project_root, config, advanced_topic_json = get_config_and_paths()
        results_dir = project_root / config['data']['results_dir']
        doc_topic_matrix_path = results_dir / 'doc_topic_matrix.json'
        
        # Vérifier si le fichier doc_topic_matrix.json existe
        if not doc_topic_matrix_path.exists():
            print("Fichier doc_topic_matrix.json introuvable. Exécutez d'abord le topic modeling.")
            return [{"label": "Aucun topic disponible", "value": ""}]
        
        # Initialiser les options avec une option par défaut
        topic_options = [{"label": "Sélectionnez un topic", "value": ""}]
        
        # Charger la matrice document-topic
        try:
            with open(doc_topic_matrix_path, 'r', encoding='utf-8') as f:
                doc_topic_data = json.load(f)
            
            # Vérifier la structure du fichier
            if not isinstance(doc_topic_data, list) and 'doc_topic_matrix' in doc_topic_data:
                doc_topic_matrix = doc_topic_data['doc_topic_matrix']
            else:
                doc_topic_matrix = doc_topic_data
            
            # Déterminer le nombre de topics à partir du premier document
            if doc_topic_matrix and isinstance(doc_topic_matrix, list) and len(doc_topic_matrix) > 0:
                first_doc = doc_topic_matrix[0]
                if 'topic_distribution' in first_doc:
                    num_topics = len(first_doc['topic_distribution'])
                    
                    # Essayer de charger les noms des topics depuis le fichier d'analyse avancée
                    topic_names = {}
                    if advanced_topic_json.exists():
                        try:
                            with open(advanced_topic_json, encoding='utf-8') as f:
                                stats = json.load(f)
                            
                            if stats.get('topic_names_llm'):
                                if isinstance(stats['topic_names_llm'], dict):
                                    topic_names = stats['topic_names_llm']
                                else:
                                    try:
                                        import ast
                                        topic_names = ast.literal_eval(stats['topic_names_llm'])
                                    except Exception:
                                        topic_names = {}
                        except Exception as e:
                            print(f"Erreur lors du chargement des noms de topics: {e}")
                    
                    # Essayer de charger les mots-clés des topics depuis le fichier de débogage des clusters
                    topic_keywords = {}
                    debug_clusters_path = results_dir / 'topic_clusters_debug.json'
                    if debug_clusters_path.exists():
                        try:
                            with open(debug_clusters_path, encoding='utf-8') as f:
                                debug_data = json.load(f)
                            
                            clusters = debug_data.get('clusters', [])
                            for i, cluster in enumerate(clusters):
                                if i < num_topics:  # S'assurer que nous n'avons pas plus de clusters que de topics
                                    keywords = cluster.get('keywords', [])
                                    count = cluster.get('count', 0)
                                    topic_keywords[i] = {
                                        'keywords': keywords,
                                        'count': count
                                    }
                        except Exception as e:
                            print(f"Erreur lors du chargement des clusters de débogage: {e}")
                    
                    # Créer les options pour le dropdown en combinant toutes les informations disponibles
                    for i in range(num_topics):
                        topic_name = topic_names.get(f'topic_{i}', f"Topic {i}")
                        
                        # Ajouter les mots-clés et le nombre d'articles si disponibles
                        if i in topic_keywords:
                            keywords = topic_keywords[i].get('keywords', [])
                            count = topic_keywords[i].get('count', 0)
                            keywords_str = ", ".join(keywords[:5]) if keywords else ""
                            
                            if keywords_str:
                                label = f"Topic {i} ({count} articles) - {keywords_str}"
                            else:
                                label = f"{topic_name} (Topic {i})"
                        else:
                            label = f"{topic_name} (Topic {i})"
                        
                        topic_options.append({
                            "label": label,
                            "value": str(i)
                        })
                    
                    print(f"Chargé {num_topics} topics pour le filtrage des publicités")
        except Exception as e:
            print(f"Erreur lors du chargement des topics: {e}")
            return [{"label": "Erreur lors du chargement des topics", "value": ""}]
        
        return topic_options
    
    # Callback pour rafraîchir la liste des fichiers de résultats
    @app.callback(
        Output("topic-modeling-results-dropdown", "options"),
        Input("btn-run-topic-modeling", "n_clicks"),
        prevent_initial_call=False
    )
    def refresh_results_files(n_clicks):
        return get_topic_modeling_results()
    
    # Callback pour activer/désactiver le sélecteur de cache Spacy en fonction du moteur sélectionné
    @app.callback(
        Output("cache-file-select", "disabled"),
        Output("cache-info-display", "children", allow_duplicate=True),
        Input("arg-engine", "value"),
        prevent_initial_call=True
    )
    def toggle_cache_selector(engine):
        if engine == "bertopic":
            return True, html.Div([
                html.P("Le cache Spacy est désactivé pour BERTopic", className="text-danger fw-bold"),
                html.P("BERTopic utilise directement le champ 'content' des articles sans prétraitement Spacy", className="text-muted")
            ])
        else:
            return False, html.Div("Sélectionnez un fichier de cache pour accélérer le traitement", className="text-muted")
    
    # Callback pour initialiser la liste des fichiers de cache
    @app.callback(
        Output("cache-file-select", "options", allow_duplicate=True),
        Output("cache-info-display", "children"),
        Input("page-content", "children"),
        prevent_initial_call=True
    )
    def refresh_cache_list(page_content):
        cache_info = get_cache_info()
        
        # Préparer les options pour le sélecteur de cache
        cache_options = [{"label": "Aucun (utiliser le plus récent)", "value": ""}]
        
        if cache_info["count"] == 0:
            return cache_options, html.Div("Aucun fichier de cache trouvé.", className="text-muted")
        
        # Ajouter les options de cache
        for cache_file in cache_info["files"]:
            if "error" not in cache_file:
                description = f"{cache_file['filename']} ({cache_file['articles_count']} articles, {cache_file['spacy_model']})"
                cache_options.append({"label": description, "value": cache_file["filename"]})
        
        # Créer un résumé des informations de cache
        cache_summary = html.Div([
            html.P(f"{cache_info['count']} fichiers de cache trouvés", className="text-info"),
        ])
        
        return cache_options, cache_summary
    
    # Callback pour le bouton de parcourir du fichier source
    @app.callback(
        Output("arg-input-file", "value"),
        Input("source-file-browse", "n_clicks"),
        State("arg-input-file", "value"),
        prevent_initial_call=True
    )
    def browse_source_file(n_clicks, current_value):
        if not n_clicks:
            return current_value
        
        # Obtenir le répertoire de départ pour la boîte de dialogue
        project_root, _, _ = get_config_and_paths()
        data_dir = project_root / "data" / "processed"
        
        # Utiliser une commande PowerShell pour afficher une boîte de dialogue de sélection de fichier
        try:
            cmd = [
                "powershell",
                "-Command",
                "Add-Type -AssemblyName System.Windows.Forms; " +
                "$openFileDialog = New-Object System.Windows.Forms.OpenFileDialog; " +
                "$openFileDialog.InitialDirectory = '" + str(data_dir).replace('\\', '\\\\') + "'; " +
                "$openFileDialog.Filter = 'Fichiers JSON (*.json)|*.json|Tous les fichiers (*.*)|*.*'; " +
                "$openFileDialog.ShowDialog() | Out-Null; " +
                "$openFileDialog.FileName"
            ]
            result = subprocess.run(cmd, capture_output=True, text=True)
            file_path = result.stdout.strip()
            
            if file_path and os.path.exists(file_path):
                return file_path
            return current_value
        except Exception as e:
            print(f"Erreur lors de l'ouverture de la boîte de dialogue: {e}")
            return current_value
    
    # Filtres supprimés
    filtered_parser_args = parser_args
    
    # Create input list for all arguments
    input_list = [Input(f"arg-{arg['name']}", "value") for arg in filtered_parser_args]
    
    # Add cluster filter inputs
    input_list.append(Input("topic-filter-cluster-file-dropdown", "value"))
    input_list.append(Input("topic-filter-cluster-id-dropdown", "value"))
    
    # Add input for source file
    input_list.append(Input("arg-input-file", "value"))
    
    # Add input for cache file
    input_list.append(Input("cache-file-select", "value"))
    
    # Add other inputs
    input_list += [Input("btn-run-topic-modeling", "n_clicks"), Input("page-content", "children")]
    
    # Ajouter l'input pour le dropdown de résultats
    input_list.append(Input("topic-modeling-results-dropdown", "value"))
    
    @app.callback(
        [Output("topic-modeling-run-status", "children"),
         Output("advanced-topic-stats-content", "children")],
        input_list,
        prevent_initial_call=True
    )
    def run_or_load_topic_modeling(*args):
        import json  # Import json at the function level to avoid variable shadowing
        ctx_trigger = ctx.triggered
        status = ""
        stats_content = None
        
        # Split args: filtered parser values, cluster values, source file, cache file, n_clicks, page_content, selected_results_file
        filtered_parser_values = args[:len(filtered_parser_args)]
        cluster_file = args[len(filtered_parser_args)]
        cluster_id = args[len(filtered_parser_args)+1]
        source_file = args[len(filtered_parser_args)+2]
        selected_cache = args[len(filtered_parser_args)+3]
        n_clicks = args[len(filtered_parser_args)+4]
        page_content = args[len(filtered_parser_args)+5]
        selected_results_file = args[len(filtered_parser_args)+6] if len(args) > len(filtered_parser_args)+6 else None
        
        print(f"Fichier de résultats sélectionné: {selected_results_file}")
        
        # Vérifier si le fichier de résultats sélectionné existe
        if selected_results_file and "?cache=" in selected_results_file:
            selected_results_file = selected_results_file.split("?cache=")[0]
        
        trigger_id = ctx_trigger[0]["prop_id"].split(".")[0] if ctx_trigger else None
        project_root, config, advanced_topic_json = get_config_and_paths()
        
        # Charger les données de clustering si disponibles
        cluster_data = None
        if cluster_file and cluster_id:
            try:
                from src.webapp.topic_filter_component import load_cluster_data
                cluster_data = load_cluster_data(cluster_file)
                print(f"Données de clustering chargées depuis {cluster_file}")
            except Exception as e:
                print(f"Erreur lors du chargement des données de clustering: {e}")
        
        if trigger_id == "btn-run-topic-modeling" and n_clicks:
            # Build argument list for filtered parser args
            arg_list = []
            for arg, val in zip(filtered_parser_args, filtered_parser_values):
                if arg['type'] == 'bool':
                    if val:
                        arg_list.append(f"--{arg['name'].replace('_','-')}")
                elif val is not None and val != "":
                    arg_list.append(f"--{arg['name'].replace('_','-')}")
                    arg_list.append(str(val))
            
            # Vérifier si un fichier source personnalisé est spécifié
            if source_file:
                # Comme l'argument --input-file a été supprimé du script, nous devons créer un fichier temporaire
                # qui sera utilisé comme source d'articles par défaut
                try:
                    # Copier le fichier source personnalisé vers le chemin par défaut attendu par le script
                    import shutil
                    temp_articles_path = project_root / 'data' / 'temp' / 'custom_source.json'
                    os.makedirs(os.path.dirname(temp_articles_path), exist_ok=True)
                    shutil.copy2(source_file, temp_articles_path)
                    print(f"Fichier source personnalisé copié vers {temp_articles_path}")
                    
                    # Modifier le chemin du fichier d'articles dans la configuration
                    os.environ['TOPIC_MODELING_SOURCE_FILE'] = str(temp_articles_path)
                    print(f"Variable d'environnement TOPIC_MODELING_SOURCE_FILE définie sur {temp_articles_path}")
                except Exception as e:
                    print(f"Erreur lors de la copie du fichier source personnalisé: {e}")
            
            # Ajouter le fichier de cache sélectionné si spécifié
            if selected_cache:
                # Créer un fichier de configuration pour indiquer quel cache utiliser
                cache_config_path = project_root / "config" / "cache_config.json"
                
                with open(cache_config_path, 'w', encoding='utf-8') as f:
                    json.dump({"selected_cache": selected_cache}, f, ensure_ascii=False, indent=2)
                    
                print(f"Cache sélectionné configuré: {selected_cache}")
            else:
                # Si aucun cache n'est sélectionné, supprimer le fichier de configuration s'il existe
                cache_config_path = project_root / "config" / "cache_config.json"
                if cache_config_path.exists():
                    os.remove(cache_config_path)
                    print("Configuration de cache supprimée, le cache par défaut sera utilisé s'il existe.")

            
            # Si nous avons des données de clustering et qu'aucun fichier source personnalisé n'est spécifié
            if cluster_data and cluster_id and not source_file:
                # Charger les articles
                from src.utils.filter_utils import filter_articles_by_cluster
                
                articles_path = project_root / 'data' / 'processed' / 'articles.json'
                with open(articles_path, 'r', encoding='utf-8') as f:
                    articles = json.load(f)
                
                # Filtrer les articles par cluster
                filtered_articles = filter_articles_by_cluster(articles, cluster_id, cluster_data)
                
                # Sauvegarder les articles filtrés dans un fichier temporaire
                temp_articles_path = project_root / 'data' / 'temp' / 'filtered_articles.json'
                os.makedirs(os.path.dirname(temp_articles_path), exist_ok=True)
                with open(temp_articles_path, 'w', encoding='utf-8') as f:
                    json.dump(filtered_articles, f, ensure_ascii=False, indent=2)
                
                # Comme l'argument --input-file a été supprimé du script, nous utilisons une variable d'environnement
                os.environ['TOPIC_MODELING_SOURCE_FILE'] = str(temp_articles_path)
                
                print(f"Articles filtrés par cluster {cluster_id}: {len(filtered_articles)} articles")

            script_path = project_root / 'src' / 'scripts' / 'run_topic_modeling.py'
            try:
                # Afficher la commande qui va être exécutée
                print(f"Exécution de la commande: {sys.executable} {script_path} {' '.join(arg_list)}")
                
                # Utiliser subprocess.run pour afficher les logs directement dans le terminal
                print("\n===== DÉBUT DE L'ANALYSE TOPIC MODELING =====\n")
                
                # Exécuter le processus avec stdout et stderr non redirigés
                # Cela permettra d'afficher les logs directement dans le terminal
                process = subprocess.Popen(
                    [sys.executable, str(script_path), *arg_list],
                    # Ne pas rediriger stdout et stderr pour qu'ils s'affichent directement
                )
                
                # Stocker les lignes pour l'interface web (vide pour l'instant)
                stdout_lines = []
                stderr_lines = []
                
                # Attendre la fin du processus
                return_code = process.wait()
                
                print("\n===== FIN DE L'ANALYSE TOPIC MODELING =====\n")
                
                # Vérifier si le processus s'est terminé avec succès
                if return_code != 0:
                    raise subprocess.CalledProcessError(return_code, [sys.executable, str(script_path), *arg_list])
                
                status = dbc.Alert("Topic modeling terminé avec succès !", color="success")
            except subprocess.CalledProcessError as e:
                print("===== [run_topic_modeling.py ERROR] =====")
                print(f"Erreur lors de l'exécution (code {e.returncode})")
                status = dbc.Alert(f"Erreur lors de l'exécution (code {e.returncode})", color="danger")
        
        # Utiliser le fichier de résultats sélectionné s'il existe
        if selected_results_file and os.path.exists(selected_results_file):
            print(f"Chargement des statistiques depuis le fichier sélectionné: {selected_results_file}")
            with open(selected_results_file, encoding='utf-8') as f:
                stats = json.load(f)
            stats_content = render_advanced_topic_stats(stats)
        else:
            # Si aucun fichier n'est sélectionné ou s'il n'existe pas, récupérer la liste des fichiers disponibles
            result_files = get_topic_modeling_results()
            if result_files and result_files[0]['value']:
                # Nettoyer le chemin du fichier (supprimer le paramètre de cache-busting)
                default_file = result_files[0]['value']
                if "?cache=" in default_file:
                    default_file = default_file.split("?cache=")[0]
                    
                if os.path.exists(default_file):
                    print(f"Chargement des statistiques depuis le fichier par défaut: {default_file}")
                    with open(default_file, encoding='utf-8') as f:
                        stats = json.load(f)
                    stats_content = render_advanced_topic_stats(stats)
                else:
                    stats_content = dbc.Alert(f"Fichier de statistiques avancées introuvable: {default_file}", color="warning")
            else:
                stats_content = dbc.Alert("Aucun fichier de résultats disponible.", color="warning")
        
        return status, stats_content
    
    # Callback pour exécuter le filtrage des publicités
    @app.callback(
        Output("ad-filter-status", "children"),
        Output("ad-filter-results", "children"),
        Input("btn-run-ad-filter", "n_clicks"),
        State("ad-filter-topic-id", "value"),
        State("ad-filter-min-value", "value"),
        State("ad-filter-output-path", "value"),
        State("ad-filter-dry-run", "value"),
        prevent_initial_call=True
    )
    def run_ad_filter(n_clicks, topic_id, min_value, output_path, dry_run):
        if not n_clicks or not topic_id:
            return dash.no_update, dash.no_update
        
        # Récupérer les chemins des fichiers nécessaires
        project_root, config, _ = get_config_and_paths()
        results_dir = project_root / config['data']['results_dir']
        doc_topic_matrix_path = results_dir / 'doc_topic_matrix.json'
        
        # Vérifier si un fichier source personnalisé est spécifié via une variable d'environnement
        custom_source_file = os.environ.get('TOPIC_MODELING_SOURCE_FILE')
        if custom_source_file and os.path.exists(custom_source_file):
            articles_path = custom_source_file
            print(f"Utilisation du fichier source personnalisé pour le filtrage: {articles_path}")
        else:
            # Utiliser le chemin par défaut
            articles_path = project_root / config['data']['processed_dir'] / 'articles.json'
        
        # Vérifier si les fichiers nécessaires existent
        if not doc_topic_matrix_path.exists():
            return dbc.Alert("Fichier doc_topic_matrix.json introuvable. Exécutez d'abord le topic modeling.", color="danger"), None
        
        if not os.path.exists(articles_path):
            return dbc.Alert(f"Fichier d'articles introuvable: {articles_path}", color="danger"), None
        
        # Préparer les arguments pour le script
        script_path = project_root / 'src' / 'scripts' / 'filter_ads_from_topic.py'
        
        args = [
            sys.executable,
            str(script_path),
            "--articles", str(articles_path),
            "--doc-topic-matrix", str(doc_topic_matrix_path),
            "--topic-id", str(topic_id),
            "--min-topic-value", str(min_value)
        ]
        
        if output_path:
            args.extend(["--output", output_path])
        
        if dry_run:
            args.append("--dry-run")
        
        # Afficher la commande qui va être exécutée
        cmd_str = " ".join(args)
        print(f"Exécution de la commande: {cmd_str}")
        
        try:
            # Exécuter le script
            print("\n===== DÉBUT DU FILTRAGE DES PUBLICITÉS =====\n")
            
            # Capturer la sortie pour l'afficher dans l'interface
            process = subprocess.run(args, check=True, text=True, capture_output=True)
            
            stdout = process.stdout
            stderr = process.stderr
            
            print("\n===== FIN DU FILTRAGE DES PUBLICITÉS =====\n")
            
            # Préparer l'affichage des résultats
            results_components = []
            
            # Essayer de charger le fichier de statistiques généré
            stats_path = None
            if not dry_run and not output_path:
                # Chercher le fichier de statistiques généré automatiquement
                stats_files = list(results_dir.glob(f"filter_stats_topic{topic_id}.json"))
                if stats_files:
                    stats_path = stats_files[0]
            elif not dry_run and output_path:
                # Construire le chemin du fichier de statistiques basé sur le chemin de sortie
                output_dir = os.path.dirname(output_path)
                stats_path = os.path.join(output_dir, f"filter_stats_topic{topic_id}.json")
                if not os.path.exists(stats_path):
                    stats_path = None
            
            # Afficher les statistiques si disponibles
            if stats_path and os.path.exists(stats_path):
                try:
                    with open(stats_path, 'r', encoding='utf-8') as f:
                        stats = json.load(f)
                    
                    # Créer un tableau de statistiques
                    stats_table = dbc.Table([
                        html.Thead(html.Tr([
                            html.Th("Statistique"),
                            html.Th("Valeur")
                        ])),
                        html.Tbody([
                            html.Tr([html.Td("Topic analysé"), html.Td(stats.get("topic_id", ""))]),
                            html.Tr([html.Td("Valeur minimale du topic"), html.Td(stats.get("min_topic_value", ""))]),
                            html.Tr([html.Td("Nombre total d'articles"), html.Td(stats.get("total_articles", ""))]),
                            html.Tr([html.Td("Articles dans le topic"), html.Td(stats.get("topic_articles", ""))]),
                            html.Tr([html.Td("Publicités détectées"), html.Td(stats.get("ads_detected", ""))]),
                            html.Tr([html.Td("Articles non-publicités"), html.Td(stats.get("non_ads", ""))]),
                            html.Tr([html.Td("Pourcentage de publicités"), html.Td(f"{stats.get('ads_percentage', '')}%")])
                        ])
                    ], bordered=True, hover=True, striped=True, className="mb-4")
                    
                    results_components.append(html.H5("Résultats du filtrage", className="mt-4 mb-3"))
                    results_components.append(stats_table)
                    
                    # Ajouter un lien vers le fichier filtré si disponible
                    if not dry_run and "output_path" in stats:
                        output_file = stats["output_path"]
                        results_components.append(html.Div([
                            html.P(["Fichier filtré: ", html.Code(output_file)]),
                            dbc.Button("Ouvrir le dossier", id="btn-open-filtered-folder", color="secondary", className="mt-2 mb-4"),
                            dcc.Store(id="filtered-file-path", data={"path": output_file})
                        ]))
                except Exception as e:
                    print(f"Erreur lors du chargement des statistiques: {e}")
            
            # Afficher la sortie du script
            results_components.append(html.H5("Logs d'exécution", className="mt-4 mb-3"))
            results_components.append(html.Pre(stdout, className="bg-light p-3 mb-4", style={"maxHeight": "300px", "overflow": "auto"}))
            
            if stderr.strip():
                results_components.append(html.H5("Erreurs", className="mt-4 mb-3"))
                results_components.append(html.Pre(stderr, className="bg-danger text-white p-3", style={"maxHeight": "200px", "overflow": "auto"}))
            
            return dbc.Alert("Filtrage des publicités terminé avec succès !", color="success"), html.Div(results_components)
        
        except subprocess.CalledProcessError as e:
            print(f"Erreur lors de l'exécution du script (code {e.returncode})")
            print(f"Sortie standard: {e.stdout}")
            print(f"Erreur standard: {e.stderr}")
            
            return dbc.Alert(f"Erreur lors de l'exécution du script (code {e.returncode})", color="danger"), html.Div([
                html.H5("Erreur d'exécution", className="mt-4 mb-3"),
                html.Pre(e.stderr, className="bg-danger text-white p-3", style={"maxHeight": "300px", "overflow": "auto"}),
                html.H5("Sortie standard", className="mt-4 mb-3"),
                html.Pre(e.stdout, className="bg-light p-3", style={"maxHeight": "300px", "overflow": "auto"})
            ])
        
        except Exception as e:
            print(f"Erreur lors de l'exécution du script: {str(e)}")
            return dbc.Alert(f"Erreur lors de l'exécution du script: {str(e)}", color="danger"), None
    
    # Callback pour charger l'explorateur d'articles quand on change d'onglet ou qu'on sélectionne un fichier de résultats
    @app.callback(
        Output("article-browser-content", "children"),
        Input("topic-modeling-tabs", "active_tab"),
        Input("topic-modeling-results-dropdown", "value"),
        prevent_initial_call=True
    )
    def update_article_browser_tab(active_tab, selected_file):
        if active_tab != "article-browser-tab":
            return None
            
        # Si un fichier de résultats est sélectionné, l'utiliser pour charger les données correspondantes
        if selected_file:
            # Nettoyer le chemin du fichier (supprimer le paramètre de cache-busting)
            if "?cache=" in selected_file:
                selected_file = selected_file.split("?cache=")[0]
                
            # Extraire l'UUID du fichier pour trouver le fichier doc_topic_matrix correspondant
            import re
            import os
            
            # Pattern pour extraire l'UUID du nom de fichier (supporte à la fois les formats avec timestamp et UUID)
            uuid_pattern = r'advanced_topic_analysis_([a-f0-9\-]+|\d+)\.json'
            match = re.search(uuid_pattern, os.path.basename(selected_file))
            
            if match and match.group(1):
                # Extraire l'identifiant (UUID ou timestamp)
                file_id = match.group(1)
                project_root, config, _ = get_config_and_paths()
                results_dir = project_root / config['data']['results_dir']
                doc_topic_matrix_path = results_dir / f'doc_topic_matrix_{file_id}.json'
                
                print(f"Recherche du fichier doc_topic_matrix correspondant: {doc_topic_matrix_path}")
                
                if os.path.exists(doc_topic_matrix_path):
                    print(f"Utilisation du fichier doc_topic_matrix spécifique: {doc_topic_matrix_path}")
                    return load_article_browser(doc_topic_matrix_path)
                else:
                    print(f"Fichier doc_topic_matrix spécifique non trouvé, utilisation du fichier par défaut")
        
        # Si aucun fichier spécifique n'est trouvé, utiliser le fichier par défaut
        return load_article_browser()
    
    # Callback pour mettre à jour la table des articles en fonction des options de tri et de filtrage
    @app.callback(
        Output("article-browser-table-container", "children"),
        Input("article-sort-dropdown", "value"),
        Input("sort-descending-checkbox", "value"),
        Input("dominant-topic-filter", "value"),
        State("article-browser-data", "data"),
        prevent_initial_call=True
    )
    def update_article_table(sort_by, sort_desc, topic_filter, table_data):
        if not table_data:
            return html.Div("Aucune donnée disponible", className="alert alert-warning")
        
        # Filtrer par topic dominant si nécessaire
        if topic_filter != 'all':
            filtered_data = [row for row in table_data if row['dominant_topic'] == int(topic_filter)]
        else:
            filtered_data = table_data
        
        # Trier les données
        if sort_by.startswith('topic_'):
            # Extraire l'index du topic pour le tri
            topic_idx = int(sort_by.split('_')[1])
            sorted_data = sorted(filtered_data, 
                                key=lambda x: x['topic_distribution'][topic_idx], 
                                reverse=sort_desc)
        else:
            # Tri par d'autres colonnes
            sorted_data = sorted(filtered_data, 
                                key=lambda x: x.get(sort_by, ''), 
                                reverse=sort_desc)
        
        # Limiter à 1000 articles pour éviter de surcharger l'interface
        if len(sorted_data) > 1000:
            sorted_data = sorted_data[:1000]
            warning = html.Div("Affichage limité aux 1000 premiers articles", 
                            className="alert alert-warning mb-3")
        else:
            warning = None
        
        # Créer le tableau
        table_header = [
            html.Thead(html.Tr([
                html.Th("ID Article"),
                html.Th("Titre"),
                html.Th("Date"),
                html.Th("Journal"),
                html.Th("Topic dominant"),
                html.Th("Distribution des topics")
            ]))
        ]
        
        rows = []
        for article in sorted_data:
            # Créer une représentation visuelle de la distribution des topics
            topic_bars = []
            for i, value in enumerate(article['topic_distribution']):
                # Calculer la largeur de la barre en fonction de la valeur (max 100%)
                width = min(value * 100, 100)
                topic_bars.append(
                    html.Div(
                        html.Div(
                            f"{value:.3f}",
                            className="text-white text-center small",
                            style={"overflow": "hidden", "textOverflow": "ellipsis"}
                        ),
                        className=f"bg-primary",
                        style={
                            "width": f"{width}%",
                            "minWidth": "30px" if width > 5 else "0",
                            "height": "20px",
                            "display": "inline-block",
                            "marginRight": "2px"
                        },
                        title=f"Topic {i}: {value:.3f}"
                    )
                )
            
            # Créer la ligne du tableau
            row = html.Tr([
                html.Td(article['doc_id']),
                html.Td(
                    html.A(
                        article['title'] or "Sans titre",
                        id={"type": "article-title-link", "index": article['doc_id']},
                        href="#",
                        className="text-primary"
                    )
                ),
                html.Td(article['date']),
                html.Td(article['newspaper']),
                html.Td(f"{article['dominant_topic_name']} ({article['dominant_topic_value']:.3f})"),
                html.Td(html.Div(topic_bars, style={"whiteSpace": "nowrap"}))
            ])
            rows.append(row)
        
        table_body = [html.Tbody(rows)]
        table = dbc.Table(table_header + table_body, striped=True, bordered=True, hover=True, responsive=True)
        
        # Retourner le tableau avec éventuellement un avertissement
        if warning:
            return html.Div([warning, table])
        else:
            return table
    
    # Callback pour afficher les détails d'un article dans un modal
    @app.callback(
        Output("article-detail-modal", "is_open"),
        Output("article-detail-body", "children"),
        Input({"type": "article-title-link", "index": ALL}, "n_clicks"),
        State({"type": "article-title-link", "index": ALL}, "id"),
        prevent_initial_call=True
    )
    def show_article_details(n_clicks, ids):
        # Vérifier si un lien a été cliqué
        if not any(n_clicks) or not ctx.triggered:
            return False, None
        
        # Trouver l'article qui a été cliqué
        clicked_index = ctx.triggered_id["index"]
        
        # Charger les articles depuis le fichier JSON
        project_root = pathlib.Path(__file__).resolve().parents[2]
        articles_path = project_root / 'data' / 'processed' / 'articles.json'
        
        if not articles_path.exists():
            return True, html.Div("Fichier d'articles introuvable", className="alert alert-danger")
        
        try:
            with open(articles_path, 'r', encoding='utf-8') as f:
                articles = json.load(f)
            
            # Rechercher l'article par son ID
            article = None
            for a in articles:
                # Comparer les ID sous forme de chaînes pour gérer les cas où l'un est un entier
                if str(a.get('id', '')) == str(clicked_index) or str(a.get('doc_id', '')) == str(clicked_index):
                    article = a
                    break
            
            if not article:
                return True, html.Div(f"Article avec ID {clicked_index} introuvable", className="alert alert-warning")
            
            # Préparer le contenu du modal
            content = [
                html.H4(article.get('title', 'Sans titre')),
                html.Hr(),
                dbc.Row([
                    dbc.Col([
                        html.Strong("Date: "),
                        html.Span(article.get('date', 'Non spécifiée'))
                    ], width=4),
                    dbc.Col([
                        html.Strong("Journal: "),
                        html.Span(article.get('newspaper', 'Non spécifié'))
                    ], width=4),
                    dbc.Col([
                        html.Strong("Canton: "),
                        html.Span(article.get('canton', 'Non spécifié'))
                    ], width=4)
                ], className="mb-3"),
                html.H5("Contenu:"),
                html.Div(
                    html.Pre(
                        article.get('content', article.get('original_content', 'Contenu non disponible')),
                        style={"whiteSpace": "pre-wrap", "maxHeight": "400px", "overflowY": "auto"}
                    ),
                    className="border p-3 bg-light"
                )
            ]
            
            return True, content
            
        except Exception as e:
            print(f"Erreur lors du chargement des articles: {e}")
            return True, html.Div(f"Erreur lors du chargement des articles: {str(e)}", className="alert alert-danger")
    
    # Callback pour fermer le modal des détails d'article
    @app.callback(
        Output("article-detail-modal", "is_open", allow_duplicate=True),
        Input("close-article-modal", "n_clicks"),
        prevent_initial_call=True
    )
    def close_article_modal(n_clicks):
        if n_clicks:
            return False
        return dash.no_update
    
    # Les callbacks pour les filtres ont été supprimés

# Helper to render advanced stats (adapt to your JSON structure)
def render_advanced_topic_stats(stats):
    import plotly.express as px
    import plotly.graph_objects as go
    import pandas as pd
    from dash import html
    import traceback  # Pour le débogage
    children = []
    # 1. Coherence Score
    if 'coherence_score' in stats:
        score = stats['coherence_score']
        if score is not None:
            children.append(dbc.Alert(f"Score de cohérence : {score:.3f}", color="info", className="mb-3"))
        else:
            children.append(dbc.Alert("Score de cohérence : N/A", color="info", className="mb-3"))
    # 2. Récupérer les noms LLM s'ils existent
    topic_names_llm = None
    if stats.get('topic_names_llm'):
        # Peut être string ou dict
        if isinstance(stats['topic_names_llm'], dict):
            topic_names_llm = stats['topic_names_llm']
        else:
            try:
                import ast
                topic_names_llm = ast.literal_eval(stats['topic_names_llm'])
            except Exception:
                topic_names_llm = None
    # 2. Répartition des topics
    if 'topic_distribution' in stats:
        dist = stats['topic_distribution']
        topics = [str(i) for i in range(len(dist))]
        if topic_names_llm:
            topics = [topic_names_llm.get(f'topic_{i}', f"Topic {i}") for i in range(len(dist))]
        df_dist = pd.DataFrame({
            'Topic': topics,
            'Proportion': dist
        })
        children.append(dcc.Graph(
            figure=px.bar(df_dist, x='Topic', y='Proportion', title='Distribution des topics (proportion)', text_auto='.2f')
        ))
    # Vérifier si topic_article_counts existe
    if 'topic_article_counts' in stats:
        counts = stats['topic_article_counts']
        topics = list(counts.keys())
        if topic_names_llm:
            topics = [topic_names_llm.get(f'topic_{i}', f"Topic {i}") for i in range(len(topics))]
        df_counts = pd.DataFrame({
            'Topic': topics,
            'Articles': list(counts.values())
        })
        children.append(dcc.Graph(
            figure=px.bar(df_counts, x='Topic', y='Articles', title="Nombre d'articles par topic", text_auto=True)
        ))
    # Alternative: utiliser doc_topic_distribution pour calculer le nombre d'articles par topic
    elif 'doc_topic_distribution' in stats:
        try:
            # Récupérer la distribution doc-topic
            doc_topic_dist = stats['doc_topic_distribution']
            # Compter le nombre d'articles par topic dominant
            topic_counts = {}
            for doc_id, topic_dist in doc_topic_dist.items():
                # Trouver le topic dominant pour ce document
                dominant_topic = max(range(len(topic_dist)), key=lambda i: topic_dist[i])
                # Incrémenter le compteur pour ce topic
                topic_counts[str(dominant_topic)] = topic_counts.get(str(dominant_topic), 0) + 1
            
            # Créer un DataFrame pour la visualisation
            topic_ids = sorted([int(t) for t in topic_counts.keys()])
            topics = []
            counts = []
            for topic_id in topic_ids:
                topic_key = str(topic_id)
                if topic_names_llm:
                    topic_label = topic_names_llm.get(f'topic_{topic_id}', f"Topic {topic_id}")
                else:
                    topic_label = f"Topic {topic_id}"
                topics.append(topic_label)
                counts.append(topic_counts.get(topic_key, 0))
            
            df_counts = pd.DataFrame({
                'Topic': topics,
                'Articles': counts
            })
            children.append(dcc.Graph(
                figure=px.bar(df_counts, x='Topic', y='Articles', title="Nombre d'articles par topic", text_auto=True)
            ))
        except Exception as e:
            # En cas d'erreur, ajouter un message d'erreur au lieu de faire planter l'application
            children.append(html.Div(f"Erreur lors de la génération du graphique 'Nombre d'articles par topic': {str(e)}", 
                                    className="alert alert-danger"))
    # Autre alternative: utiliser la distribution des topics si disponible
    elif 'topic_distribution' in stats and len(dist) > 0:
        try:
            # Estimer le nombre d'articles par topic à partir de la distribution
            # Supposons que nous connaissons le nombre total d'articles
            total_docs = stats.get('total_docs', 1000)  # Valeur par défaut si non disponible
            
            topics = [str(i) for i in range(len(dist))]
            if topic_names_llm:
                topics = [topic_names_llm.get(f'topic_{i}', f"Topic {i}") for i in range(len(dist))]
            
            # Calculer le nombre d'articles par topic en fonction de la distribution
            article_counts = [int(p * total_docs) for p in dist]
            
            df_counts = pd.DataFrame({
                'Topic': topics,
                'Articles': article_counts
            })
            children.append(dcc.Graph(
                figure=px.bar(df_counts, x='Topic', y='Articles', title="Nombre d'articles par topic (estimé)", text_auto=True)
            ))
        except Exception as e:
            children.append(html.Div(f"Erreur lors de l'estimation du nombre d'articles par topic: {str(e)}", 
                                    className="alert alert-danger"))
    # 3. Top mots par topic
    if 'weighted_words' in stats:
        children.append(html.H5("Top mots par topic", className="mt-4"))
        for topic, words in stats['weighted_words'].items():
            words_df = pd.DataFrame(words, columns=['Mot', 'Poids'])
            topic_label = topic_names_llm.get(f'topic_{topic}', f"Topic {topic}") if topic_names_llm else f"Topic {topic}"
            fig = px.bar(words_df, x='Poids', y='Mot', orientation='h', title=topic_label, text_auto='.3f')
            fig.update_layout(height=350, margin=dict(l=80, r=20, t=40, b=40))
            children.append(dcc.Graph(figure=fig))
    # 4. Documents représentatifs
    if 'representative_docs' in stats:
        children.append(html.H5("Documents représentatifs par topic", className="mt-4"))
        for topic, doc_ids in stats['representative_docs'].items():
            topic_label = topic_names_llm.get(f'topic_{topic}', f"Topic {topic}") if topic_names_llm else f"Topic {topic}"
            children.append(html.P(f"{topic_label} : {', '.join(str(i) for i in doc_ids)}", className="mb-2"))
    # 5. Noms LLM
    if stats.get('llm_name'):
        children.append(html.P(f"LLM utilisé : {stats['llm_name']}", className="text-muted"))
    if stats.get('topic_names_llm'):
        children.append(html.P(f"Noms de topics LLM : {stats['topic_names_llm']}", className="text-muted"))
    return html.Div(children)

# To be called in app.py: from src.webapp.topic_modeling_viz import register_topic_modeling_callbacks, get_topic_modeling_layout
