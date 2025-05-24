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
import ast


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
                        _html.H4("Nommage des topics avec LLM", className="mt-4 mb-3"),
                        _html.P("Cet outil vous permet de générer automatiquement des noms et des résumés pour vos topics en utilisant un LLM.", className="text-muted"),
                        dbc.Card([
                            dbc.CardBody([
                                dbc.Row([
                                    dbc.Col([
                                        dbc.Label("Méthode de génération", html_for="topic-naming-method"),
                                        dbc.Select(
                                            id="topic-naming-method",
                                            options=[
                                                {"label": "Utiliser les articles représentatifs", "value": "articles"},
                                                {"label": "Utiliser les mots-clés", "value": "keywords"}
                                            ],
                                            value="articles",
                                            className="mb-2"
                                        ),
                                    ], width=6),
                                    dbc.Col([
                                        dbc.Label("Nombre d'articles par topic", html_for="topic-naming-num-articles"),
                                        dbc.Input(
                                            id="topic-naming-num-articles",
                                            type="number",
                                            min=1,
                                            max=20,
                                            step=1,
                                            value=10,
                                            className="mb-2"
                                        ),
                                    ], width=6),
                                ]),
                                dbc.Row([
                                    dbc.Col([
                                        dbc.Label("Seuil de probabilité", html_for="topic-naming-threshold"),
                                        dbc.Input(
                                            id="topic-naming-threshold",
                                            type="number",
                                            min=0.1,
                                            max=0.9,
                                            step=0.1,
                                            value=0.5,
                                            className="mb-2"
                                        ),
                                    ], width=6),
                                    dbc.Col([
                                        dbc.Label("Fichier de sortie", html_for="topic-naming-output-path"),
                                        dbc.Input(
                                            id="topic-naming-output-path",
                                            type="text",
                                            placeholder="Laissez vide pour générer automatiquement",
                                            className="mb-2"
                                        ),
                                    ], width=6),
                                ]),
                                dbc.Button(
                                    "Générer les noms des topics",
                                    id="btn-run-topic-naming",
                                    color="primary",
                                    className="mt-2"
                                ),
                            ])
                        ], className="mb-4"),
                        _html.Div(id="topic-naming-status", className="mt-3"),
                        dcc.Loading(
                            id="loading-topic-naming-results",
                            type="default",
                            children=_html.Div(id="topic-naming-results")
                        )
                    ], width=12)
                ])
            ], label="Nommage des topics", tab_id="topic-naming-tab"),
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

# Helper to render the results of the topic naming script.
def render_topic_naming_results(topic_names):
    """
    Renders the topic naming results in a user-friendly format.
    
    Args:
        topic_names (dict): A dictionary containing the generated topic names and summaries.
    
    Returns:
        A Dash component to display the results.
    """
    if not topic_names:
        return html.Div("Aucun nom de topic n'a été généré.", className="alert alert-info")
    
    # Can be a dict or a string representation of a dict
    if isinstance(topic_names, str):
        try:
            topic_names = ast.literal_eval(topic_names)
        except (ValueError, SyntaxError) as e:
            return dbc.Alert(f"Erreur lors de la lecture des noms de topics: {e}", color="danger")

    accordion_items = []
    for topic_id, data in topic_names.items():
        title = "Titre non disponible"
        summary = "Résumé non disponible"
        
        if isinstance(data, dict):
            title = data.get('title', title)
            summary = data.get('summary', summary)
        elif isinstance(data, (list, tuple)) and len(data) > 0:
            title = data[0]
            if len(data) > 1:
                summary = data[1]

        item = dbc.AccordionItem(
            [
                html.P(summary, className="mb-0")
            ],
            title=f"{topic_id.replace('_', ' ').title()}: {title}",
        )
        accordion_items.append(item)
    
    children = [
        html.H5("Noms et résumés des topics générés", className="mt-4 mb-3"),
        dbc.Accordion(
            accordion_items,
            start_collapsed=True,
            always_open=True,
            className="mb-4"
        )
    ]
    
    # Add a button to apply the names to visualizations and a store for the names
    children.append(
        html.Div([
            dbc.Button(
                "Appliquer ces noms aux visualisations",
                id="btn-apply-topic-names",
                color="success",
                className="mb-3"
            ),
            dcc.Store(id="topic-names-store", data=topic_names),
            html.P("Note: Les noms générés seront automatiquement appliqués aux visualisations et à l'explorateur d'articles.", 
                  className="text-muted fst-italic small mt-2")
        ])
    )
    
    return html.Div(children)

def register_topic_modeling_callbacks(app):
    # Register the topic filter component callbacks
    register_topic_filter_callbacks(app, id_prefix="topic-filter")
    parser_args = get_topic_modeling_args()
    
    # Callback pour le bouton de nommage des topics
    @app.callback(
        Output("topic-naming-status", "children"),
        Output("loading-topic-naming-results", "children"),
        Output("topic-modeling-results-dropdown", "options", allow_duplicate=True),
        Input("btn-run-topic-naming", "n_clicks"),
        State("topic-naming-method", "value"),
        State("topic-naming-num-articles", "value"),
        State("topic-naming-threshold", "value"),
        State("topic-naming-output-path", "value"),
        prevent_initial_call=True
    )
    def run_topic_naming(n_clicks, method, num_articles, threshold, output_path):
        if not n_clicks:
            return dash.no_update, dash.no_update, dash.no_update
        
        # Récupérer les chemins nécessaires
        project_root, config, advanced_topic_dir = get_config_and_paths()
        results_dir = project_root / config['data']['results_dir']
        
        # Vérifier si les fichiers nécessaires existent
        doc_topic_matrix_path = results_dir / 'doc_topic_matrix.json'
        articles_path = project_root / config['data']['processed'] / 'articles.json'
        advanced_topic_path = advanced_topic_dir / 'advanced_topic_analysis.json'
        
        if not doc_topic_matrix_path.exists():
            return dbc.Alert("Fichier doc_topic_matrix.json introuvable. Exécutez d'abord le topic modeling.", color="danger"), dash.no_update, dash.no_update
        
        if not articles_path.exists():
            return dbc.Alert("Fichier d'articles introuvable.", color="danger"), dash.no_update, dash.no_update
        
        # Préparer le chemin de sortie
        if not output_path:
            from datetime import datetime
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = str(results_dir / f"topic_names_llm_{timestamp}.json")
        
        # Construire la commande
        script_path = project_root / "src" / "scripts" / "run_topic_naming.py"
        
        # Utiliser sys.executable pour s'assurer d'utiliser le bon interpréteur Python
        python_executable = sys.executable
        
        # Construire les arguments de la commande
        cmd_args = [
            python_executable,
            str(script_path),
            "--source-file", str(articles_path),
            "--doc-topic-matrix", str(doc_topic_matrix_path),
            "--method", method,
            "--output-file", output_path,
            "--num-articles", str(num_articles),
            "--threshold", str(threshold),
            "--config", str(project_root / "config" / "config.yaml")
        ]
        
        # Si la méthode est keywords, ajouter le chemin vers le fichier de mots-clés
        if method == "keywords" and advanced_topic_path.exists():
            cmd_args.extend(["--top-words-file", str(advanced_topic_path)])
        
        # Exécuter la commande
        try:
            # Créer un message de statut
            status = dbc.Alert(
                [
                    html.P("Lancement du script de nommage des topics...", className="mb-0"),
                    html.P(f"Méthode: {method}, Nombre d'articles: {num_articles}, Seuil: {threshold}", className="mb-0 small"),
                    html.P(f"Fichier de sortie: {output_path}", className="mb-0 small")
                ],
                color="info"
            )
            
            # Exécuter le script en arrière-plan
            process = subprocess.Popen(
                cmd_args,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,
                universal_newlines=True
            )
            
            # Attendre que le processus se termine
            stdout, stderr = process.communicate()
            
            # Vérifier si le processus s'est terminé avec succès
            if process.returncode == 0:
                # Charger les résultats
                try:
                    with open(output_path, 'r', encoding='utf-8') as f:
                        topic_names = json.load(f)
                    
                    # Mettre à jour le fichier d'analyse avancée avec les noms de topics
                    if advanced_topic_path.exists():
                        try:
                            with open(advanced_topic_path, 'r', encoding='utf-8') as f:
                                advanced_stats = json.load(f)
                            
                            # Ajouter les noms de topics au fichier d'analyse avancée
                            advanced_stats['topic_names_llm'] = topic_names
                            
                            with open(advanced_topic_path, 'w', encoding='utf-8') as f:
                                json.dump(advanced_stats, f, ensure_ascii=False, indent=2)
                                
                            print(f"Noms de topics ajoutés au fichier d'analyse avancée: {advanced_topic_path}")
                        except Exception as e:
                            print(f"Erreur lors de la mise à jour du fichier d'analyse avancée: {e}")
                    
                    # Afficher les résultats
                    results = render_topic_naming_results(topic_names)
                    return dbc.Alert("Nommage des topics terminé avec succès!", color="success"), results, get_topic_modeling_results()
                    
                except Exception as e:
                    return dbc.Alert(f"Erreur lors du chargement des résultats: {str(e)}", color="danger"), no_update, no_update
            else:
                # Afficher l'erreur
                return dbc.Alert(f"Erreur lors de l'exécution du script: {stderr}", color="danger"), no_update, no_update
                
        except Exception as e:
            return dbc.Alert(f"Erreur: {str(e)}", color="danger"), no_update, no_update
    
    # Callback pour remplir la liste des topics disponibles pour le filtrage des publicités
    @app.callback(
        Output("ad-filter-topic-id", "options"),
        [Input("topic-modeling-tabs", "active_tab"),
         Input("btn-refresh-topics", "n_clicks")],
        prevent_initial_call=True
    )
    def update_ad_filter_topic_options(active_tab, n_clicks):
        # Déterminer quel élément a déclenché le callback
        trigger = ctx.triggered_id if ctx.triggered else None
        
        # Only run if the ad filter tab is active or the refresh button is clicked
        if active_tab != "ad-filter-tab" and trigger != "btn-refresh-topics":
            return dash.no_update
        
        # Récupérer les informations sur les topics
        project_root, config, advanced_topic_dir = get_config_and_paths()
        results_dir = project_root / config['data']['results_dir']
        advanced_topic_path = advanced_topic_dir / 'advanced_topic_analysis.json'

        # Vérifier si le fichier d'analyse existe
        if not advanced_topic_path.exists():
            print("Fichier advanced_topic_analysis.json introuvable.")
            return [{"label": "Aucun topic disponible", "value": ""}]
        
        topic_options = [{"label": "Sélectionnez un topic", "value": ""}]
        
        try:
            with open(advanced_topic_path, 'r', encoding='utf-8') as f:
                stats = json.load(f)

            num_topics = len(stats.get('topic_distribution', []))
            if num_topics == 0:
                return [{"label": "Aucun topic disponible", "value": ""}]

            topic_names = {}
            if 'topic_names_llm' in stats:
                llm_names = stats['topic_names_llm']
                if isinstance(llm_names, str):
                    llm_names = ast.literal_eval(llm_names)
                if isinstance(llm_names, dict):
                    for topic_id, data in llm_names.items():
                        topic_num = int(re.search(r'\d+', topic_id).group())
                        title = data.get('title', f"Topic {topic_num}") if isinstance(data, dict) else data[0]
                        topic_names[topic_num] = title
            
            topic_keywords = {}
            if 'weighted_words' in stats:
                for topic_id, words_data in stats['weighted_words'].items():
                    topic_num = int(topic_id)
                    keywords = [word[0] for word in words_data[:5]]
                    topic_keywords[topic_num] = ", ".join(keywords)

            for i in range(num_topics):
                name = topic_names.get(i, f"Topic {i}")
                keywords_str = topic_keywords.get(i, "")
                label = f"{name}"
                if keywords_str:
                    label += f" - ({keywords_str})"
                
                topic_options.append({
                    "label": label,
                    "value": str(i)
                })

            print(f"Chargé {num_topics} topics pour le filtrage des publicités")
            return topic_options

        except Exception as e:
            print(f"Erreur lors du chargement des options de topic pour le filtre pub: {e}")
            return [{"label": "Erreur de chargement", "value": ""}]


# Helper to render advanced stats (adapt to your JSON structure)
def render_advanced_topic_stats_from_json(json_file_path):
    if not json_file_path or not os.path.exists(json_file_path):
        return html.Div("Fichier de résultats non trouvé.", className="alert alert-warning")
    try:
        with open(json_file_path, encoding='utf-8') as f:
            stats = json.load(f)
    except Exception as e:
        return html.Div(f"Erreur de lecture du JSON : {e}", className="alert alert-danger")
    
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
                topic_names_llm = ast.literal_eval(stats['topic_names_llm'])
            except Exception:
                topic_names_llm = None

    def get_topic_label(topic_id, default_prefix="Topic"):
        if topic_names_llm:
            # Check if data is a dict with title or a list/tuple
            topic_data = topic_names_llm.get(f'topic_{topic_id}')
            if isinstance(topic_data, dict):
                return topic_data.get('title', f"{default_prefix} {topic_id}")
            elif isinstance(topic_data, (list, tuple)) and topic_data:
                return topic_data[0]
        return f"{default_prefix} {topic_id}"

    # 2. Répartition des topics
    if 'topic_distribution' in stats:
        dist = stats['topic_distribution']
        topics = [get_topic_label(i) for i in range(len(dist))]
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
        topic_ids = sorted([int(k) for k in counts.keys()])
        topics = [get_topic_label(i) for i in topic_ids]
        articles = [counts[str(i)] for i in topic_ids]

        df_counts = pd.DataFrame({
            'Topic': topics,
            'Articles': articles
        })
        children.append(dcc.Graph(
            figure=px.bar(df_counts, x='Topic', y='Articles', title="Nombre d'articles par topic", text_auto=True)
        ))
    # Alternative: utiliser doc_topic_distribution pour calculer le nombre d'articles par topic
    elif 'doc_topic_distribution' in stats:
        try:
            doc_topic_dist = stats['doc_topic_distribution']
            topic_counts = {}
            for doc_id, topic_dist_list in doc_topic_dist.items():
                if not topic_dist_list: continue
                dominant_topic = max(range(len(topic_dist_list)), key=lambda i: topic_dist_list[i])
                topic_counts[dominant_topic] = topic_counts.get(dominant_topic, 0) + 1
            
            if topic_counts:
                topic_ids = sorted(topic_counts.keys())
                topics = [get_topic_label(i) for i in topic_ids]
                counts = [topic_counts[i] for i in topic_ids]
                
                df_counts = pd.DataFrame({'Topic': topics, 'Articles': counts})
                children.append(dcc.Graph(
                    figure=px.bar(df_counts, x='Topic', y='Articles', title="Nombre d'articles par topic (dominant)", text_auto=True)
                ))
        except Exception as e:
            children.append(html.Div(f"Erreur lors de la génération du graphique 'Nombre d'articles par topic': {str(e)}", 
                                    className="alert alert-danger"))

    # 3. Top mots par topic
    if 'weighted_words' in stats:
        children.append(html.H5("Top mots par topic", className="mt-4"))
        for topic, words in stats['weighted_words'].items():
            words_df = pd.DataFrame(words, columns=['Mot', 'Poids'])
            topic_label = get_topic_label(topic)
            fig = px.bar(words_df.head(10), x='Poids', y='Mot', orientation='h', title=topic_label, text_auto='.3f')
            fig.update_layout(height=350, margin=dict(l=120, r=20, t=40, b=40), yaxis={'categoryorder':'total ascending'})
            children.append(dcc.Graph(figure=fig))
            
    # 4. Documents représentatifs
    if 'representative_docs' in stats:
        children.append(html.H5("Documents représentatifs par topic", className="mt-4"))
        for topic, doc_ids in stats['representative_docs'].items():
            topic_label = get_topic_label(topic)
            children.append(html.P(f"{topic_label} : {', '.join(str(i) for i in doc_ids)}", className="mb-2"))
            
    # 5. Noms LLM
    if stats.get('llm_name'):
        children.append(html.P(f"LLM utilisé : {stats['llm_name']}", className="text-muted"))
    if topic_names_llm:
         children.append(render_topic_naming_results(topic_names_llm))

    return html.Div(children)


# To be called in app.py: from src.webapp.topic_modeling_viz import register_topic_modeling_callbacks, get_topic_modeling_layout