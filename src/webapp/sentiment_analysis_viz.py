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
import time  # Ajout du module time pour mesurer les performances

from src.webapp.topic_filter_component import (
    get_topic_filter_component, 
    register_topic_filter_callbacks, 
    get_filter_parameters,
    get_filter_states,
    are_filters_active
)

# Extract parser arguments from run_sentiment_analysis.py
def get_sentiment_analysis_args():
    start_time = time.time()  # Début du chronométrage
    print("Démarrage de get_sentiment_analysis_args()...")
    
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
    
    end_time = time.time()  # Fin du chronométrage
    execution_time = end_time - start_time
    print(f"Fin de get_sentiment_analysis_args() - Temps d'exécution: {execution_time:.4f} secondes")
    
    return parser_args

# Helper to get available sentiment analysis result files
def get_sentiment_results():
    start_time = time.time()  # Début du chronométrage
    print("Démarrage de get_sentiment_results()...")
    
    project_root = pathlib.Path(__file__).resolve().parents[2]
    config_path = project_root / 'config' / 'config.yaml'
    with open(config_path, encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    results_dir = project_root / config['data']['results_dir'] / 'sentiment_analysis'
    
    if not results_dir.exists():
        print(f"Répertoire {results_dir} non trouvé - Temps d'exécution: {time.time() - start_time:.4f} secondes")
        return []
    
    # Get all sentiment summary files
    print(f"Recherche des fichiers dans {results_dir}...")
    summary_files = list(results_dir.glob('sentiment_summary*.json'))
    print(f"Nombre de fichiers trouvés: {len(summary_files)}")
    
    # Sort by modification time (newest first)
    summary_files.sort(key=lambda x: os.path.getmtime(x), reverse=True)
    
    # Format for dropdown
    options = [
        {'label': f"{f.stem} ({pd.to_datetime(os.path.getmtime(f), unit='s').strftime('%Y-%m-%d %H:%M')})", 
         'value': str(f)}
        for f in summary_files
    ]
    
    end_time = time.time()  # Fin du chronométrage
    execution_time = end_time - start_time
    print(f"Fin de get_sentiment_results() - Temps d'exécution: {execution_time:.4f} secondes")
    
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
                    
                    # Ajout du composant de filtrage par topic/cluster
                    dbc.Row([
                        dbc.Col([
                            dbc.Accordion([
                                dbc.AccordionItem(
                                    get_topic_filter_component(id_prefix="sentiment-topic-filter"),
                                    title="Filtrage par Topic/Cluster"
                                )
                            ], start_collapsed=True, id="sentiment-filter-accordion")
                        ], width=12)
                    ]),
                    html.Br(),
                    
                    # Results container
                    html.Div(id="sentiment-results-container", children=[
                        # This will be populated by the callback
                    ])
                ], className="mt-3")
            ]),
            
            # Nouvelle tab pour les analyses filtrées
            dbc.Tab(label="Analyses Filtrées", children=[
                html.Div([
                    html.H4("Analyses de sentiment filtrées par topic/cluster"),
                    html.P("Lancez une analyse de sentiment filtrée par topic ou cluster."),
                    
                    # Composant de filtrage
                    get_topic_filter_component(id_prefix="sentiment-filtered-analysis"),
                    html.Br(),
                    
                    # Bouton pour lancer l'analyse filtrée
                    dbc.Button(
                        "Lancer l'analyse filtrée",
                        id="run-filtered-sentiment-button",
                        color="primary",
                        className="mb-3"
                    ),
                    html.Br(),
                    
                    # Conteneur pour les résultats filtrés
                    html.Div(id="filtered-sentiment-results-container")
                ], className="mt-3")
            ])
        ])
    ])
    
    return layout

# Function to create sentiment visualizations
def create_sentiment_visualizations(summary_file_path, is_filtered=False):
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
            title="Distribution du sentiment" if not is_filtered else "Distribution du sentiment (articles filtrés)",
            height=400
        )
        
        # 2. Sentiment histogram (compound scores)
        compound_scores = [article['sentiment']['compound'] for article in articles_data]
        sentiment_histogram = px.histogram(
            x=compound_scores,
            nbins=50,
            labels={'x': 'Score de sentiment (compound)'},
            title="Distribution des scores de sentiment" if not is_filtered else "Distribution des scores de sentiment (articles filtrés)",
            color_discrete_sequence=['#1f77b4']
        )
        sentiment_histogram.update_layout(height=400)
        
        # 3. Summary statistics card
        summary_card = dbc.Card(
            dbc.CardBody([
                html.H4("Statistiques de sentiment" if not is_filtered else "Statistiques de sentiment (articles filtrés)", className="card-title"),
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

# Function to create filtered sentiment visualizations
def create_filtered_sentiment_visualizations(summary_file_path, topic_results_path, topic_id, cluster_id, exclude_topic_id, exclude_cluster_id):
    """
    Crée des visualisations de sentiment filtrées par topic/cluster.
    
    Args:
        summary_file_path: Chemin vers le fichier de résultats de sentiment
        topic_results_path: Chemin vers le fichier de résultats de topic modeling
        topic_id: ID du topic à inclure
        cluster_id: ID du cluster à inclure
        exclude_topic_id: ID du topic à exclure
        exclude_cluster_id: ID du cluster à exclure
        
    Returns:
        Composants Dash pour les visualisations
    """
    try:
        # Charger les données de sentiment
        with open(summary_file_path, 'r', encoding='utf-8') as f:
            summary_data = json.load(f)
        
        # Charger les données de topic
        with open(topic_results_path, 'r', encoding='utf-8') as f:
            topic_data = json.load(f)
        
        # Charger les articles avec sentiment
        articles_file_path = summary_file_path.replace('sentiment_summary', 'articles_with_sentiment')
        with open(articles_file_path, 'r', encoding='utf-8') as f:
            articles_data = json.load(f)
        
        # Filtrer les articles par topic/cluster
        filtered_articles = []
        
        # Récupérer les informations de topic/cluster
        doc_topics = topic_data.get('doc_topics', {})
        doc_clusters = topic_data.get('clusters', {})
        
        for article in articles_data:
            article_id = article.get('id', article.get('doc_id', ''))
            
            # Vérifier si l'article doit être inclus
            include_article = True
            
            # Filtre par topic
            if topic_id is not None and article_id in doc_topics:
                if doc_topics[article_id].get('dominant_topic') != int(topic_id):
                    include_article = False
            
            # Filtre par cluster
            if cluster_id is not None and article_id in doc_clusters:
                if str(doc_clusters[article_id]) != str(cluster_id):
                    include_article = False
            
            # Exclusion par topic
            if exclude_topic_id is not None and article_id in doc_topics:
                if doc_topics[article_id].get('dominant_topic') == int(exclude_topic_id):
                    include_article = False
            
            # Exclusion par cluster
            if exclude_cluster_id is not None and article_id in doc_clusters:
                if str(doc_clusters[article_id]) == str(exclude_cluster_id):
                    include_article = False
            
            if include_article:
                filtered_articles.append(article)
        
        # Calculer les statistiques de sentiment pour les articles filtrés
        if not filtered_articles:
            return html.Div([
                html.P("Aucun article ne correspond aux filtres sélectionnés.", className="text-warning")
            ])
        
        # Extraire les scores de sentiment
        compound_scores = [article['sentiment']['compound'] for article in filtered_articles]
        
        # Calculer les statistiques
        filtered_summary = {
            'mean_compound': np.mean(compound_scores),
            'median_compound': np.median(compound_scores),
            'std_compound': np.std(compound_scores),
            'min_compound': np.min(compound_scores),
            'max_compound': np.max(compound_scores),
            'positive_count': sum(1 for score in compound_scores if score > 0.05),
            'neutral_count': sum(1 for score in compound_scores if -0.05 <= score <= 0.05),
            'negative_count': sum(1 for score in compound_scores if score < -0.05),
        }
        
        # Calculer les pourcentages
        total_count = len(compound_scores)
        filtered_summary['positive_percentage'] = filtered_summary['positive_count'] / total_count * 100
        filtered_summary['neutral_percentage'] = filtered_summary['neutral_count'] / total_count * 100
        filtered_summary['negative_percentage'] = filtered_summary['negative_count'] / total_count * 100
        
        # Créer les visualisations
        visualizations = []
        
        # Ajouter un message indiquant que les résultats sont filtrés
        filter_description = []
        if topic_id is not None:
            topic_names = topic_data.get('topic_names_llm', {})
            topic_name = topic_names.get(f"topic_{topic_id}", f"Topic {topic_id}")
            filter_description.append(f"Topic: {topic_name}")
        
        if cluster_id is not None:
            cluster_names = topic_data.get('cluster_names', {})
            cluster_name = cluster_names.get(str(cluster_id), f"Cluster {cluster_id}")
            filter_description.append(f"Cluster: {cluster_name}")
        
        if exclude_topic_id is not None:
            topic_names = topic_data.get('topic_names_llm', {})
            topic_name = topic_names.get(f"topic_{exclude_topic_id}", f"Topic {exclude_topic_id}")
            filter_description.append(f"Exclure Topic: {topic_name}")
        
        if exclude_cluster_id is not None:
            cluster_names = topic_data.get('cluster_names', {})
            cluster_name = cluster_names.get(str(exclude_cluster_id), f"Cluster {exclude_cluster_id}")
            filter_description.append(f"Exclure Cluster: {cluster_name}")
        
        visualizations.append(html.Div([
            html.H5("Résultats filtrés"),
            html.P(f"Filtres appliqués: {', '.join(filter_description)}"),
            html.P(f"Nombre d'articles filtrés: {len(filtered_articles)} sur {len(articles_data)} ({len(filtered_articles)/len(articles_data)*100:.1f}%)")
        ], className="alert alert-info"))
        
        # 1. Sentiment distribution pie chart
        sentiment_distribution = go.Figure(data=[
            go.Pie(
                labels=['Positif', 'Neutre', 'Négatif'],
                values=[filtered_summary['positive_count'], filtered_summary['neutral_count'], filtered_summary['negative_count']],
                hole=0.4,
                marker_colors=['#2ca02c', '#d3d3d3', '#d62728']
            )
        ])
        sentiment_distribution.update_layout(
            title="Distribution du sentiment (articles filtrés)",
            height=400
        )
        visualizations.append(dcc.Graph(figure=sentiment_distribution))
        
        # 2. Sentiment histogram
        sentiment_histogram = go.Figure()
        sentiment_histogram.add_trace(go.Histogram(
            x=compound_scores,
            nbinsx=20,
            marker_color='#1f77b4'
        ))
        sentiment_histogram.update_layout(
            title="Distribution des scores de sentiment (articles filtrés)",
            xaxis_title="Score de sentiment (compound)",
            yaxis_title="Nombre d'articles",
            height=400
        )
        visualizations.append(dcc.Graph(figure=sentiment_histogram))
        
        # 3. Summary statistics card
        summary_card = dbc.Card([
            dbc.CardHeader(html.H5("Statistiques de sentiment (articles filtrés)")),
            dbc.CardBody([
                html.P(f"Score moyen: {filtered_summary['mean_compound']:.3f}"),
                html.P(f"Score médian: {filtered_summary['median_compound']:.3f}"),
                html.P(f"Écart-type: {filtered_summary['std_compound']:.3f}"),
                html.P(f"Score minimum: {filtered_summary['min_compound']:.3f}"),
                html.P(f"Score maximum: {filtered_summary['max_compound']:.3f}"),
                html.P(f"Articles positifs: {filtered_summary['positive_count']} ({filtered_summary['positive_percentage']:.1f}%)"),
                html.P(f"Articles neutres: {filtered_summary['neutral_count']} ({filtered_summary['neutral_percentage']:.1f}%)"),
                html.P(f"Articles négatifs: {filtered_summary['negative_count']} ({filtered_summary['negative_percentage']:.1f}%)")
            ])
        ])
        visualizations.append(summary_card)
        
        return visualizations
    
    except Exception as e:
        return html.Div([
            html.P(f"Erreur lors de la création des visualisations filtrées: {str(e)}", className="text-danger")
        ])

# Callback registration (to be called from app.py)
def register_sentiment_analysis_callbacks(app):
    # Register callbacks for topic filter component
    register_topic_filter_callbacks(app, id_prefix="sentiment-topic-filter")
    register_topic_filter_callbacks(app, id_prefix="sentiment-filtered-analysis")
    
    # Callback to run sentiment analysis
    @app.callback(
        Output("sentiment-run-output", "children"),
        Input("run-sentiment-button", "n_clicks"),
        [State(f"sentiment-{arg['name']}-input", "value") for arg in get_sentiment_analysis_args()],
        prevent_initial_call=True
    )
    def run_sentiment_analysis(n_clicks, *args):
        if not n_clicks:
            return ""
        
        # Get argument names
        arg_names = [arg['name'] for arg in get_sentiment_analysis_args()]
        
        # Create command
        cmd = [sys.executable, os.path.join(os.path.dirname(__file__), "..", "scripts", "run_sentiment_analysis.py")]
        
        # Add arguments
        for arg_name, arg_value in zip(arg_names, args):
            if arg_value is not None and arg_value != "":
                if isinstance(arg_value, bool):
                    if arg_value:
                        cmd.append(f"--{arg_name}")
                else:
                    cmd.append(f"--{arg_name}")
                    cmd.append(str(arg_value))
        
        # Run command
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            return html.Div([
                html.P("Analyse de sentiment terminée avec succès !"),
                html.Pre(result.stdout, style={"maxHeight": "300px", "overflow": "auto", "backgroundColor": "#f0f0f0", "padding": "10px"})
            ])
        except subprocess.CalledProcessError as e:
            return html.Div([
                html.P("Erreur lors de l'analyse de sentiment:", className="text-danger"),
                html.Pre(e.stderr, style={"maxHeight": "300px", "overflow": "auto", "backgroundColor": "#f0f0f0", "padding": "10px"})
            ])
    
    # Callback to display sentiment analysis results
    @app.callback(
        Output("sentiment-results-container", "children"),
        [Input("sentiment-results-dropdown", "value"),
         Input("sentiment-topic-filter-apply-button", "n_clicks")],
        [State("sentiment-topic-filter-topic-dropdown", "value"),
         State("sentiment-topic-filter-cluster-dropdown", "value"),
         State("sentiment-topic-filter-exclude-topic-dropdown", "value"),
         State("sentiment-topic-filter-exclude-cluster-dropdown", "value"),
         State("sentiment-topic-filter-results-dropdown", "value")]
    )
    def display_sentiment_results(results_file, apply_filters, topic_id, cluster_id, exclude_topic_id, exclude_cluster_id, topic_results_file):
        ctx = dash.callback_context
        if not results_file:
            return html.P("Sélectionnez un fichier de résultats pour afficher les visualisations.")
        
        # Vérifier si les filtres sont actifs
        filters_active = are_filters_active("sentiment-topic-filter", ctx)
        
        if filters_active and topic_results_file:
            # Appliquer les filtres
            return create_filtered_sentiment_visualizations(
                results_file, 
                topic_results_file, 
                topic_id, 
                cluster_id, 
                exclude_topic_id, 
                exclude_cluster_id
            )
        else:
            # Afficher les résultats sans filtre
            return create_sentiment_visualizations(results_file)
    
    # Callback pour lancer une analyse de sentiment filtrée
    @app.callback(
        Output("filtered-sentiment-results-container", "children"),
        Input("run-filtered-sentiment-button", "n_clicks"),
        [State("sentiment-filtered-analysis-topic-dropdown", "value"),
         State("sentiment-filtered-analysis-cluster-dropdown", "value"),
         State("sentiment-filtered-analysis-exclude-topic-dropdown", "value"),
         State("sentiment-filtered-analysis-exclude-cluster-dropdown", "value"),
         State("sentiment-filtered-analysis-results-dropdown", "value")],
        prevent_initial_call=True
    )
    def run_filtered_sentiment_analysis(n_clicks, topic_id, cluster_id, exclude_topic_id, exclude_cluster_id, topic_results_file):
        if not n_clicks or not topic_results_file:
            return html.P("Veuillez sélectionner un fichier de résultats de topic modeling et configurer les filtres.")
        
        # Créer la commande
        cmd = [
            sys.executable, 
            os.path.join(os.path.dirname(__file__), "..", "scripts", "run_filtered_analysis.py"),
            "--analysis-type", "sentiment"
        ]
        
        # Ajouter les filtres
        if topic_id is not None:
            cmd.extend(["--topic-id", str(topic_id)])
        
        if cluster_id is not None:
            cmd.extend(["--cluster-id", str(cluster_id)])
        
        if exclude_topic_id is not None:
            cmd.extend(["--exclude-topic-id", str(exclude_topic_id)])
        
        if exclude_cluster_id is not None:
            cmd.extend(["--exclude-cluster-id", str(exclude_cluster_id)])
        
        # Ajouter le fichier de résultats de topic modeling
        cmd.extend(["--topic-results", str(topic_results_file)])
        
        # Exécuter la commande
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            
            # Extraire le chemin du fichier de résultats généré
            output_path = None
            for line in result.stdout.splitlines():
                if "Results saved to:" in line:
                    output_path = line.split("Results saved to:")[1].strip()
                    break
            
            if output_path and os.path.exists(output_path):
                return html.Div([
                    html.P("Analyse de sentiment filtrée terminée avec succès !"),
                    html.Pre(result.stdout, style={"maxHeight": "200px", "overflow": "auto", "backgroundColor": "#f0f0f0", "padding": "10px"}),
                    html.Hr(),
                    html.H5("Résultats de l'analyse filtrée :"),
                    *create_sentiment_visualizations(output_path, is_filtered=True)
                ])
            else:
                return html.Div([
                    html.P("Analyse terminée, mais impossible de trouver le fichier de résultats."),
                    html.Pre(result.stdout, style={"maxHeight": "300px", "overflow": "auto", "backgroundColor": "#f0f0f0", "padding": "10px"})
                ])
        except subprocess.CalledProcessError as e:
            return html.Div([
                html.P("Erreur lors de l'analyse de sentiment filtrée:", className="text-danger"),
                html.Pre(e.stderr, style={"maxHeight": "300px", "overflow": "auto", "backgroundColor": "#f0f0f0", "padding": "10px"})
            ])

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
