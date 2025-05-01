"""
Entity Recognition Visualization Page for Dash app
"""

from dash import html, dcc, Input, Output, State, ctx
import dash_bootstrap_components as dbc
import plotly.express as px
import plotly.graph_objects as go
import subprocess
import importlib.util
import sys
import os
import inspect
import argparse
import glob
import yaml
import pathlib
import pandas as pd
import numpy as np
import json

# Extract parser arguments from run_entity_recognition.py
def get_entity_recognition_args():
    spec = importlib.util.spec_from_file_location("run_entity_recognition", os.path.join(os.path.dirname(__file__), "..", "scripts", "run_entity_recognition.py"))
    run_entity_recognition = importlib.util.module_from_spec(spec)
    sys.modules["run_entity_recognition"] = run_entity_recognition
    spec.loader.exec_module(run_entity_recognition)
    parser = run_entity_recognition.argparse.ArgumentParser(description="Reconnaissance d'entités nommées")
    run_entity_recognition.main.__globals__["parser"] = parser
    # Parse only the arguments, not the main logic
    parser_args = [
        {"name": action.dest,
         "flags": action.option_strings,
         "help": action.help,
         "required": getattr(action, "required", False),
         "default": action.default,
         "type": getattr(action, "type", str).__name__ if hasattr(action, "type") else "str",
         "choices": getattr(action, "choices", None)
        }
        for action in parser._actions if action.dest != 'help'
    ]
    return parser_args

# Helper to get available entity recognition result files
def get_entity_results():
    project_root = pathlib.Path(__file__).resolve().parents[2]
    config_path = project_root / 'config' / 'config.yaml'
    with open(config_path, encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    results_dir = project_root / config['data']['results_dir'] / 'entity_recognition'
    
    if not results_dir.exists():
        return []
    
    # Get all entity summary files
    summary_files = list(results_dir.glob('entity_summary*.json'))
    
    # Sort by modification time (newest first)
    summary_files.sort(key=lambda x: os.path.getmtime(x), reverse=True)
    
    # Format for dropdown
    options = [
        {'label': f"{f.stem} ({pd.to_datetime(os.path.getmtime(f), unit='s').strftime('%Y-%m-%d %H:%M')})", 
         'value': str(f)}
        for f in summary_files
    ]
    
    return options

# Layout for the entity recognition page
def get_entity_recognition_layout():
    # Get available result files
    entity_results = get_entity_results()
    
    # Get parser arguments for the run form
    parser_args = get_entity_recognition_args()
    
    # Create form fields based on parser arguments
    form_fields = []
    for arg in parser_args:
        if arg['name'] in ['versioned', 'no_versioned', 'use_cache']:
            # Create a checkbox for boolean flags
            form_fields.append(
                dbc.Col([
                    dbc.Checkbox(
                        id=f"entity-{arg['name']}-input",
                        label=arg['help'],
                        value=arg['default'] if arg['default'] is not None else False
                    ),
                    html.Br()
                ], width=12)
            )
        elif arg['choices'] is not None:
            # Create a dropdown for choices
            form_fields.append(
                dbc.Col([
                    dbc.Label(arg['help']),
                    dcc.Dropdown(
                        id=f"entity-{arg['name']}-input",
                        options=[{'label': choice, 'value': choice} for choice in arg['choices']],
                        value=arg['default'],
                        clearable=True
                    ),
                    html.Br()
                ], width=6)
            )
        else:
            # Create a text input for other arguments
            form_fields.append(
                dbc.Col([
                    dbc.Label(arg['help']),
                    dbc.Input(
                        id=f"entity-{arg['name']}-input",
                        type="text" if arg['type'] == 'str' else "number",
                        placeholder=f"Default: {arg['default']}" if arg['default'] is not None else "",
                        value=""
                    ),
                    html.Br()
                ], width=6)
            )
    
    # Create the layout
    layout = html.Div([
        dbc.Row([
            dbc.Col([
                html.H2("Reconnaissance d'Entités Nommées"),
                html.P("Extraire et analyser les entités nommées des articles de presse."),
                html.Hr()
            ], width=12)
        ]),
        
        # Tabs for Run and Results
        dbc.Tabs([
            # Tab for running entity recognition
            dbc.Tab(label="Lancer une analyse", children=[
                html.Div([
                    html.H4("Paramètres d'analyse"),
                    html.P("Configurez les paramètres pour la reconnaissance d'entités."),
                    dbc.Form(dbc.Row(form_fields)),
                    html.Br(),
                    dbc.Button("Lancer l'analyse", id="run-entity-button", color="primary"),
                    html.Br(),
                    html.Div(id="entity-run-output")
                ], className="mt-3")
            ]),
            
            # Tab for viewing results
            dbc.Tab(label="Résultats", children=[
                html.Div([
                    html.H4("Visualisation des résultats"),
                    html.P("Sélectionnez un fichier de résultats pour visualiser les entités reconnues."),
                    dbc.Row([
                        dbc.Col([
                            dbc.Label("Fichier de résultats:"),
                            dcc.Dropdown(
                                id="entity-results-dropdown",
                                options=entity_results,
                                value=entity_results[0]['value'] if entity_results else None,
                                placeholder="Sélectionnez un fichier de résultats"
                            )
                        ], width=6)
                    ]),
                    html.Br(),
                    
                    # Results container
                    html.Div(id="entity-results-container", children=[
                        # This will be populated by the callback
                    ])
                ], className="mt-3")
            ])
        ])
    ])
    
    return layout

# Function to create entity visualizations
def create_entity_visualizations(summary_file_path):
    try:
        with open(summary_file_path, 'r', encoding='utf-8') as f:
            summary_data = json.load(f)
        
        # Get the corresponding articles file
        articles_file_path = summary_file_path.replace('entity_summary', 'articles_with_entities')
        with open(articles_file_path, 'r', encoding='utf-8') as f:
            articles_data = json.load(f)
        
        # Extract entity data
        summary = summary_data['summary']
        entity_frequencies = summary_data['entity_frequencies']
        
        # Create visualizations
        visualizations = []
        
        # 1. Entity distribution by type pie chart
        entity_types = list(summary['total_by_type'].keys())
        entity_counts = list(summary['total_by_type'].values())
        
        entity_distribution = go.Figure(data=[
            go.Pie(
                labels=entity_types,
                values=entity_counts,
                hole=0.4
            )
        ])
        entity_distribution.update_layout(
            title="Distribution des entités par type",
            height=400
        )
        
        # 2. Summary statistics card
        summary_card = dbc.Card(
            dbc.CardBody([
                html.H4("Statistiques d'entités", className="card-title"),
                html.P(f"Nombre d'articles analysés: {len(articles_data)}"),
                html.P(f"Nombre total d'entités: {sum(entity_counts)}"),
                html.P(f"Moyenne d'entités par article: {summary['avg_entities_per_doc']:.2f}"),
                html.P(f"Types d'entités: {', '.join(entity_types)}"),
            ])
        )
        
        # 3. Metadata card
        metadata_card = dbc.Card(
            dbc.CardBody([
                html.H4("Métadonnées", className="card-title"),
                html.P(f"ID d'exécution: {summary_data['run_id']}"),
                html.P(f"Horodatage: {summary_data['timestamp']}"),
                html.P(f"Durée: {summary_data['duration_seconds']:.2f} secondes"),
                html.P(f"Modèle: {summary_data['model']}"),
            ])
        )
        
        # Add cards to the container
        visualizations.append(
            dbc.Row([
                dbc.Col(summary_card, width=6),
                dbc.Col(metadata_card, width=6)
            ])
        )
        
        visualizations.append(html.Br())
        
        # Add entity distribution chart
        visualizations.append(
            dbc.Row([
                dbc.Col(dcc.Graph(figure=entity_distribution), width=12)
            ])
        )
        
        # 4. Entity frequency tabs for each entity type
        entity_tabs = []
        for entity_type in entity_frequencies:
            if entity_frequencies[entity_type]:
                # Create bar chart for top entities
                entities = list(entity_frequencies[entity_type].keys())[:20]  # Top 20
                counts = list(entity_frequencies[entity_type].values())[:20]
                
                fig = px.bar(
                    x=entities, 
                    y=counts,
                    labels={"x": "Entité", "y": "Fréquence"},
                    title=f"Top 20 entités de type {entity_type}"
                )
                fig.update_layout(height=400)
                
                # Create table for top entities
                rows = []
                for i, (entity, count) in enumerate(sorted(entity_frequencies[entity_type].items(), 
                                                         key=lambda x: x[1], reverse=True)[:50]):
                    rows.append(
                        html.Tr([
                            html.Td(i+1),
                            html.Td(entity),
                            html.Td(count)
                        ])
                    )
                
                entity_table = dbc.Table(
                    [
                        html.Thead(
                            html.Tr([
                                html.Th("#"),
                                html.Th("Entité"),
                                html.Th("Fréquence")
                            ])
                        ),
                        html.Tbody(rows)
                    ],
                    bordered=True,
                    hover=True,
                    responsive=True,
                    striped=True,
                    className="mt-3"
                )
                
                # Create tab for this entity type
                entity_tabs.append(
                    dbc.Tab(
                        label=entity_type,
                        children=[
                            dbc.Row([
                                dbc.Col(dcc.Graph(figure=fig), width=12)
                            ]),
                            dbc.Row([
                                dbc.Col([
                                    html.H4(f"Top 50 entités de type {entity_type}"),
                                    entity_table
                                ], width=12)
                            ])
                        ]
                    )
                )
        
        # Add entity frequency tabs
        if entity_tabs:
            visualizations.append(html.H4("Fréquence des entités par type", className="mt-4"))
            visualizations.append(dbc.Tabs(entity_tabs))
        
        # 5. Articles with most entities
        # Sort articles by entity count
        for article in articles_data:
            article['entity_count'] = len(article.get('entities', []))
        
        sorted_articles = sorted(articles_data, key=lambda x: x['entity_count'], reverse=True)[:10]
        
        # Create table
        article_rows = []
        for i, article in enumerate(sorted_articles):
            title = article.get('title', 'Sans titre')
            count = article['entity_count']
            newspaper = article.get('newspaper', 'Inconnu')
            date = article.get('date', 'Inconnue')
            article_id = article.get('id', '') or article.get('base_id', '')
            
            article_rows.append(
                html.Tr([
                    html.Td(i+1),
                    html.Td(html.A(title, id=f"entity-article-{i}", className="article-link")),
                    html.Td(count),
                    html.Td(newspaper),
                    html.Td(date),
                    html.Td(article_id, style={"display": "none"})
                ])
            )
        
        article_table = dbc.Table(
            [
                html.Thead(
                    html.Tr([
                        html.Th("#"),
                        html.Th("Titre"),
                        html.Th("Nombre d'entités"),
                        html.Th("Journal"),
                        html.Th("Date"),
                        html.Th("ID", style={"display": "none"})
                    ])
                ),
                html.Tbody(article_rows)
            ],
            bordered=True,
            hover=True,
            responsive=True,
            striped=True,
            className="mt-3"
        )
        
        visualizations.append(
            dbc.Row([
                dbc.Col([
                    html.H4("Articles avec le plus d'entités", className="mt-4"),
                    article_table
                ], width=12)
            ])
        )
        
        # Add modal for article display
        article_modal = dbc.Modal(
            [
                dbc.ModalHeader(dbc.ModalTitle("Contenu de l'article"), close_button=True),
                dbc.ModalBody(id="entity-article-modal-body"),
                dbc.ModalFooter(
                    dbc.Button("Fermer", id="entity-close-article-modal", className="ms-auto")
                ),
            ],
            id="entity-article-modal",
            size="xl",
        )
        
        visualizations.append(article_modal)
        
        # Store the article data in a hidden div for modal display
        visualizations.append(
            dcc.Store(id="entity-articles-data", data=articles_data)
        )
        
        return visualizations
    
    except Exception as e:
        return [html.Div([
            html.H4("Erreur lors du chargement des résultats"),
            html.P(f"Erreur: {str(e)}")
        ])]

# Callback registration (to be called from app.py)
def register_entity_recognition_callbacks(app):
    # Callback to run entity recognition
    @app.callback(
        Output("entity-run-output", "children"),
        Input("run-entity-button", "n_clicks"),
        [State(f"entity-{arg['name']}-input", "value") for arg in get_entity_recognition_args()],
        prevent_initial_call=True
    )
    def run_entity_recognition(n_clicks, *args):
        if n_clicks is None:
            return ""
        
        # Get argument names
        arg_names = [arg['name'] for arg in get_entity_recognition_args()]
        
        # Build command
        cmd = ["python", "-m", "src.scripts.run_entity_recognition"]
        
        # Add arguments
        for name, value in zip(arg_names, args):
            if value is None or value == "":
                continue
            
            # Handle boolean flags differently
            if name in ['versioned', 'use_cache']:
                if value:
                    cmd.append(f"--{name}")
                else:
                    if name == 'versioned':
                        cmd.append("--no-versioned")
            else:
                cmd.append(f"--{name}")
                cmd.append(str(value))
        
        # Run the command
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            return html.Div([
                html.H5("Reconnaissance d'entités terminée avec succès"),
                html.P("Sortie de la commande:"),
                html.Pre(result.stdout, style={"max-height": "300px", "overflow": "auto"}),
                dbc.Alert("Analyse terminée. Allez à l'onglet 'Résultats' pour visualiser les résultats.", color="success")
            ])
        except subprocess.CalledProcessError as e:
            return html.Div([
                html.H5("Erreur lors de la reconnaissance d'entités"),
                html.P("Erreur:"),
                html.Pre(e.stderr, style={"max-height": "300px", "overflow": "auto"}),
                dbc.Alert("Erreur lors de l'analyse. Vérifiez les paramètres et réessayez.", color="danger")
            ])
    
    # Callback to update results dropdown
    @app.callback(
        Output("entity-results-dropdown", "options"),
        Input("entity-run-output", "children")
    )
    def update_results_dropdown(run_output):
        return get_entity_results()
    
    # Callback to display results
    @app.callback(
        Output("entity-results-container", "children"),
        Input("entity-results-dropdown", "value")
    )
    def display_results(file_path):
        if not file_path:
            return html.Div("Aucun fichier de résultats sélectionné.")
        
        return create_entity_visualizations(file_path)
    
    # Callback to open article modal
    @app.callback(
        [Output("entity-article-modal", "is_open"),
         Output("entity-article-modal-body", "children")],
        [Input(f"entity-article-{i}", "n_clicks") for i in range(10)] + 
        [Input("entity-close-article-modal", "n_clicks")],
        [State("entity-article-modal", "is_open"),
         State("entity-articles-data", "data")]
    )
    def toggle_article_modal(*args):
        n_clicks_list = args[:-2]  # All n_clicks values
        is_open = args[-2]         # Current modal state
        articles_data = args[-1]   # Articles data
        
        if not ctx.triggered_id:
            return is_open, ""
        
        if ctx.triggered_id == "entity-close-article-modal":
            return False, ""
        
        if not articles_data:
            return is_open, "Données d'articles non disponibles."
        
        # Determine which article was clicked
        article_idx = None
        
        if ctx.triggered_id.startswith("entity-article-"):
            article_idx = int(ctx.triggered_id.split("-")[-1])
        
        if article_idx is None:
            return is_open, ""
        
        # Sort articles by entity count
        for article in articles_data:
            article['entity_count'] = len(article.get('entities', []))
        
        sorted_articles = sorted(articles_data, key=lambda x: x['entity_count'], reverse=True)[:10]
        article = sorted_articles[article_idx]
        
        # Create modal content
        modal_content = [
            html.H4(article.get('title', 'Sans titre')),
            html.P(f"Journal: {article.get('newspaper', 'Inconnu')}"),
            html.P(f"Date: {article.get('date', 'Inconnue')}"),
            html.P(f"ID: {article.get('id', '') or article.get('base_id', '')}"),
            html.Hr(),
            html.H5(f"Entités reconnues ({len(article.get('entities', []))})"),
        ]
        
        # Group entities by type
        entities_by_type = {}
        for entity in article.get('entities', []):
            entity_type = entity.get('label')
            if entity_type not in entities_by_type:
                entities_by_type[entity_type] = []
            entities_by_type[entity_type].append(entity.get('text'))
        
        # Create entity lists by type
        for entity_type, entities in sorted(entities_by_type.items()):
            # Count occurrences
            entity_counts = {}
            for entity in entities:
                if entity not in entity_counts:
                    entity_counts[entity] = 0
                entity_counts[entity] += 1
            
            # Create list items
            entity_items = []
            for entity, count in sorted(entity_counts.items(), key=lambda x: x[1], reverse=True):
                entity_items.append(html.Li(f"{entity} ({count})"))
            
            modal_content.append(html.H6(f"{entity_type} ({len(entities)})"))
            modal_content.append(html.Ul(entity_items))
        
        modal_content.extend([
            html.Hr(),
            html.H5("Contenu de l'article:"),
            html.Div(
                article.get('text', 'Contenu non disponible'),
                style={"max-height": "400px", "overflow": "auto", "white-space": "pre-wrap"}
            )
        ])
        
        return True, modal_content

# To be called in app.py: from src.webapp.entity_recognition_viz import register_entity_recognition_callbacks
