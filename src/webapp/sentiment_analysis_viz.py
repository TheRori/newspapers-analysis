"""
Sentiment Analysis Visualization Page for Dash app
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

# Extract parser arguments from run_sentiment_analysis.py
def get_sentiment_analysis_args():
    spec = importlib.util.spec_from_file_location("run_sentiment_analysis", os.path.join(os.path.dirname(__file__), "..", "scripts", "run_sentiment_analysis.py"))
    run_sentiment_analysis = importlib.util.module_from_spec(spec)
    sys.modules["run_sentiment_analysis"] = run_sentiment_analysis
    spec.loader.exec_module(run_sentiment_analysis)
    parser = run_sentiment_analysis.argparse.ArgumentParser(description="Analyse de sentiment d'un corpus")
    run_sentiment_analysis.main.__globals__["parser"] = parser
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

# Helper to get available sentiment analysis result files
def get_sentiment_results():
    project_root = pathlib.Path(__file__).resolve().parents[2]
    config_path = project_root / 'config' / 'config.yaml'
    with open(config_path, encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    results_dir = project_root / config['data']['results_dir'] / 'sentiment_analysis'
    
    if not results_dir.exists():
        return []
    
    # Get all sentiment summary files
    summary_files = list(results_dir.glob('sentiment_summary*.json'))
    
    # Sort by modification time (newest first)
    summary_files.sort(key=lambda x: os.path.getmtime(x), reverse=True)
    
    # Format for dropdown
    options = [
        {'label': f"{f.stem} ({pd.to_datetime(os.path.getmtime(f), unit='s').strftime('%Y-%m-%d %H:%M')})", 
         'value': str(f)}
        for f in summary_files
    ]
    
    return options

# Layout for the sentiment analysis page
def get_sentiment_analysis_layout():
    # Get available result files
    sentiment_results = get_sentiment_results()
    
    # Get parser arguments for the run form
    parser_args = get_sentiment_analysis_args()
    
    # Create form fields based on parser arguments
    form_fields = []
    for arg in parser_args:
        if arg['name'] in ['versioned', 'no_versioned', 'use_cache']:
            # Create a checkbox for boolean flags
            form_fields.append(
                dbc.Col([
                    dbc.Checkbox(
                        id=f"sentiment-{arg['name']}-input",
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
                        id=f"sentiment-{arg['name']}-input",
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
                        id=f"sentiment-{arg['name']}-input",
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
                html.H2("Analyse de Sentiment"),
                html.P("Analyser le sentiment des articles de presse et visualiser les résultats."),
                html.Hr()
            ], width=12)
        ]),
        
        # Tabs for Run and Results
        dbc.Tabs([
            # Tab for running sentiment analysis
            dbc.Tab(label="Lancer une analyse", children=[
                html.Div([
                    html.H4("Paramètres d'analyse"),
                    html.P("Configurez les paramètres pour l'analyse de sentiment."),
                    dbc.Form(dbc.Row(form_fields)),
                    html.Br(),
                    dbc.Button("Lancer l'analyse", id="run-sentiment-button", color="primary"),
                    html.Br(),
                    html.Div(id="sentiment-run-output")
                ], className="mt-3")
            ]),
            
            # Tab for viewing results
            dbc.Tab(label="Résultats", children=[
                html.Div([
                    html.H4("Visualisation des résultats"),
                    html.P("Sélectionnez un fichier de résultats pour visualiser l'analyse de sentiment."),
                    dbc.Row([
                        dbc.Col([
                            dbc.Label("Fichier de résultats:"),
                            dcc.Dropdown(
                                id="sentiment-results-dropdown",
                                options=sentiment_results,
                                value=sentiment_results[0]['value'] if sentiment_results else None,
                                placeholder="Sélectionnez un fichier de résultats"
                            )
                        ], width=6)
                    ]),
                    html.Br(),
                    
                    # Results container
                    html.Div(id="sentiment-results-container", children=[
                        # This will be populated by the callback
                    ])
                ], className="mt-3")
            ])
        ])
    ])
    
    return layout

# Function to create sentiment visualizations
def create_sentiment_visualizations(summary_file_path):
    try:
        with open(summary_file_path, 'r', encoding='utf-8') as f:
            summary_data = json.load(f)
        
        # Get the corresponding articles file
        articles_file_path = summary_file_path.replace('sentiment_summary', 'articles_with_sentiment')
        with open(articles_file_path, 'r', encoding='utf-8') as f:
            articles_data = json.load(f)
        
        # Extract sentiment data
        summary = summary_data['summary']
        
        # Create visualizations
        visualizations = []
        
        # 1. Sentiment distribution pie chart
        sentiment_distribution = go.Figure(data=[
            go.Pie(
                labels=['Positif', 'Neutre', 'Négatif'],
                values=[summary['positive_count'], summary['neutral_count'], summary['negative_count']],
                hole=0.4,
                marker_colors=['#2ca02c', '#d3d3d3', '#d62728']
            )
        ])
        sentiment_distribution.update_layout(
            title="Distribution du sentiment",
            height=400
        )
        
        # 2. Sentiment histogram (compound scores)
        compound_scores = [article['sentiment']['compound'] for article in articles_data]
        sentiment_histogram = px.histogram(
            x=compound_scores,
            nbins=50,
            labels={'x': 'Score de sentiment (compound)'},
            title="Distribution des scores de sentiment",
            color_discrete_sequence=['#1f77b4']
        )
        sentiment_histogram.update_layout(height=400)
        
        # 3. Summary statistics card
        summary_card = dbc.Card(
            dbc.CardBody([
                html.H4("Statistiques de sentiment", className="card-title"),
                html.P(f"Nombre d'articles analysés: {len(articles_data)}"),
                html.P(f"Score moyen (compound): {summary['mean_compound']:.4f}"),
                html.P(f"Score médian (compound): {summary['median_compound']:.4f}"),
                html.P(f"Écart-type (compound): {summary['std_compound']:.4f}"),
                html.P(f"Min/Max (compound): {summary['min_compound']:.4f} / {summary['max_compound']:.4f}"),
                html.P(f"Articles positifs: {summary['positive_count']} ({summary['positive_percentage']:.1f}%)"),
                html.P(f"Articles neutres: {summary['neutral_count']} ({summary['neutral_percentage']:.1f}%)"),
                html.P(f"Articles négatifs: {summary['negative_count']} ({summary['negative_percentage']:.1f}%)"),
            ])
        )
        
        # 4. Metadata card
        metadata_card = dbc.Card(
            dbc.CardBody([
                html.H4("Métadonnées", className="card-title"),
                html.P(f"ID d'exécution: {summary_data['run_id']}"),
                html.P(f"Horodatage: {summary_data['timestamp']}"),
                html.P(f"Durée: {summary_data['duration_seconds']:.2f} secondes"),
                html.P(f"Modèle: {summary_data['model']}"),
                html.P(f"Modèle transformers: {summary_data['transformer_model'] or 'N/A'}"),
            ])
        )
        
        # Add visualizations to the container
        visualizations.append(
            dbc.Row([
                dbc.Col(summary_card, width=6),
                dbc.Col(metadata_card, width=6)
            ])
        )
        
        visualizations.append(html.Br())
        
        visualizations.append(
            dbc.Row([
                dbc.Col(dcc.Graph(figure=sentiment_distribution), width=6),
                dbc.Col(dcc.Graph(figure=sentiment_histogram), width=6)
            ])
        )
        
        # 5. Table of most positive and negative articles
        # Sort articles by compound score
        sorted_articles = sorted(articles_data, key=lambda x: x['sentiment']['compound'])
        most_negative = sorted_articles[:5]
        most_positive = sorted_articles[-5:][::-1]
        
        # Create tables
        negative_rows = []
        for i, article in enumerate(most_negative):
            title = article.get('title', 'Sans titre')
            score = article['sentiment']['compound']
            newspaper = article.get('newspaper', 'Inconnu')
            date = article.get('date', 'Inconnue')
            article_id = article.get('id', '') or article.get('base_id', '')
            
            negative_rows.append(
                html.Tr([
                    html.Td(i+1),
                    html.Td(html.A(title, id=f"negative-article-{i}", className="article-link")),
                    html.Td(f"{score:.4f}"),
                    html.Td(newspaper),
                    html.Td(date),
                    html.Td(article_id, style={"display": "none"})
                ])
            )
        
        positive_rows = []
        for i, article in enumerate(most_positive):
            title = article.get('title', 'Sans titre')
            score = article['sentiment']['compound']
            newspaper = article.get('newspaper', 'Inconnu')
            date = article.get('date', 'Inconnue')
            article_id = article.get('id', '') or article.get('base_id', '')
            
            positive_rows.append(
                html.Tr([
                    html.Td(i+1),
                    html.Td(html.A(title, id=f"positive-article-{i}", className="article-link")),
                    html.Td(f"{score:.4f}"),
                    html.Td(newspaper),
                    html.Td(date),
                    html.Td(article_id, style={"display": "none"})
                ])
            )
        
        negative_table = dbc.Table(
            [
                html.Thead(
                    html.Tr([
                        html.Th("#"),
                        html.Th("Titre"),
                        html.Th("Score"),
                        html.Th("Journal"),
                        html.Th("Date"),
                        html.Th("ID", style={"display": "none"})
                    ])
                ),
                html.Tbody(negative_rows)
            ],
            bordered=True,
            hover=True,
            responsive=True,
            striped=True,
            className="mt-3"
        )
        
        positive_table = dbc.Table(
            [
                html.Thead(
                    html.Tr([
                        html.Th("#"),
                        html.Th("Titre"),
                        html.Th("Score"),
                        html.Th("Journal"),
                        html.Th("Date"),
                        html.Th("ID", style={"display": "none"})
                    ])
                ),
                html.Tbody(positive_rows)
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
                    html.H4("Articles les plus négatifs"),
                    negative_table
                ], width=6),
                dbc.Col([
                    html.H4("Articles les plus positifs"),
                    positive_table
                ], width=6)
            ])
        )
        
        # Add modal for article display
        article_modal = dbc.Modal(
            [
                dbc.ModalHeader(dbc.ModalTitle("Contenu de l'article"), close_button=True),
                dbc.ModalBody(id="sentiment-article-modal-body"),
                dbc.ModalFooter(
                    dbc.Button("Fermer", id="sentiment-close-article-modal", className="ms-auto")
                ),
            ],
            id="sentiment-article-modal",
            size="xl",
        )
        
        visualizations.append(article_modal)
        
        # Store the article data in a hidden div for modal display
        visualizations.append(
            dcc.Store(id="sentiment-articles-data", data=articles_data)
        )
        
        return visualizations
    
    except Exception as e:
        return [html.Div([
            html.H4("Erreur lors du chargement des résultats"),
            html.P(f"Erreur: {str(e)}")
        ])]

# Callback registration (to be called from app.py)
def register_sentiment_analysis_callbacks(app):
    # Callback to run sentiment analysis
    @app.callback(
        Output("sentiment-run-output", "children"),
        Input("run-sentiment-button", "n_clicks"),
        [State(f"sentiment-{arg['name']}-input", "value") for arg in get_sentiment_analysis_args()],
        prevent_initial_call=True
    )
    def run_sentiment_analysis(n_clicks, *args):
        if n_clicks is None:
            return ""
        
        # Get argument names
        arg_names = [arg['name'] for arg in get_sentiment_analysis_args()]
        
        # Build command
        cmd = ["python", "-m", "src.scripts.run_sentiment_analysis"]
        
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
                html.H5("Analyse de sentiment terminée avec succès"),
                html.P("Sortie de la commande:"),
                html.Pre(result.stdout, style={"max-height": "300px", "overflow": "auto"}),
                dbc.Alert("Analyse terminée. Allez à l'onglet 'Résultats' pour visualiser les résultats.", color="success")
            ])
        except subprocess.CalledProcessError as e:
            return html.Div([
                html.H5("Erreur lors de l'analyse de sentiment"),
                html.P("Erreur:"),
                html.Pre(e.stderr, style={"max-height": "300px", "overflow": "auto"}),
                dbc.Alert("Erreur lors de l'analyse. Vérifiez les paramètres et réessayez.", color="danger")
            ])
    
    # Callback to update results dropdown
    @app.callback(
        Output("sentiment-results-dropdown", "options"),
        Input("sentiment-run-output", "children")
    )
    def update_results_dropdown(run_output):
        return get_sentiment_results()
    
    # Callback to display results
    @app.callback(
        Output("sentiment-results-container", "children"),
        Input("sentiment-results-dropdown", "value")
    )
    def display_results(file_path):
        if not file_path:
            return html.Div("Aucun fichier de résultats sélectionné.")
        
        return create_sentiment_visualizations(file_path)
    
    # Callback to open article modal
    @app.callback(
        [Output("sentiment-article-modal", "is_open"),
         Output("sentiment-article-modal-body", "children")],
        [Input("positive-article-0", "n_clicks"),
         Input("positive-article-1", "n_clicks"),
         Input("positive-article-2", "n_clicks"),
         Input("positive-article-3", "n_clicks"),
         Input("positive-article-4", "n_clicks"),
         Input("negative-article-0", "n_clicks"),
         Input("negative-article-1", "n_clicks"),
         Input("negative-article-2", "n_clicks"),
         Input("negative-article-3", "n_clicks"),
         Input("negative-article-4", "n_clicks"),
         Input("sentiment-close-article-modal", "n_clicks")],
        [State("sentiment-article-modal", "is_open"),
         State("sentiment-articles-data", "data")]
    )
    def toggle_article_modal(p0, p1, p2, p3, p4, n0, n1, n2, n3, n4, close, is_open, articles_data):
        if not ctx.triggered_id:
            return is_open, ""
        
        if ctx.triggered_id == "sentiment-close-article-modal":
            return False, ""
        
        if not articles_data:
            return is_open, "Données d'articles non disponibles."
        
        # Determine which article was clicked
        article_idx = None
        is_positive = None
        
        if ctx.triggered_id.startswith("positive-article-"):
            is_positive = True
            article_idx = int(ctx.triggered_id.split("-")[-1])
        elif ctx.triggered_id.startswith("negative-article-"):
            is_positive = False
            article_idx = int(ctx.triggered_id.split("-")[-1])
        
        if article_idx is None or is_positive is None:
            return is_open, ""
        
        # Sort articles by compound score
        sorted_articles = sorted(articles_data, key=lambda x: x['sentiment']['compound'])
        
        if is_positive:
            # Get from the end of the list (most positive)
            article = sorted_articles[-5:][::-1][article_idx]
        else:
            # Get from the beginning of the list (most negative)
            article = sorted_articles[:5][article_idx]
        
        # Create modal content
        modal_content = [
            html.H4(article.get('title', 'Sans titre')),
            html.P(f"Journal: {article.get('newspaper', 'Inconnu')}"),
            html.P(f"Date: {article.get('date', 'Inconnue')}"),
            html.P(f"ID: {article.get('id', '') or article.get('base_id', '')}"),
            html.Hr(),
            html.H5("Scores de sentiment:"),
            html.P(f"Positif: {article['sentiment'].get('positive', 0):.4f}"),
            html.P(f"Négatif: {article['sentiment'].get('negative', 0):.4f}"),
            html.P(f"Neutre: {article['sentiment'].get('neutral', 0):.4f}"),
            html.P(f"Compound: {article['sentiment'].get('compound', 0):.4f}"),
            html.Hr(),
            html.H5("Contenu de l'article:"),
            html.Div(
                article.get('text', 'Contenu non disponible'),
                style={"max-height": "400px", "overflow": "auto", "white-space": "pre-wrap"}
            )
        ]
        
        return True, modal_content

# To be called in app.py: from src.webapp.sentiment_analysis_viz import register_sentiment_analysis_callbacks
