"""
Sentiment Analysis Visualization Page for Dash app
"""

from dash import html, dcc, Input, Output, State, ctx, ALL, no_update
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

# Import pour la sauvegarde des analyses
from src.utils.export_utils import save_analysis

from src.webapp.topic_filter_component import (
    get_topic_filter_component, 
    register_topic_filter_callbacks, 
    get_filter_parameters,
    get_filter_states,
    are_filters_active
)

# Import du composant d'exportation
from src.webapp.export_component import create_export_button, create_export_modal, create_feedback_toast, register_export_callbacks

# Import du module d'affichage des articles
from src.webapp.article_display_utils import (
    create_articles_modal,
    create_full_article_modal,
    register_articles_modal_callback,
    register_full_article_modal_callback
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
    # Utiliser os.stat au lieu de os.path.getmtime pour forcer la mise à jour des horodatages
    summary_files.sort(key=lambda x: os.stat(x).st_mtime, reverse=True)
    
    # Ajout du paramètre de cache-busting pour forcer le rechargement des fichiers
    import time as time_module
    current_time = int(time_module.time())  # Timestamp actuel pour éviter les problèmes de cache
    
    # Format for dropdown
    options = [
        {'label': f"{f.stem} ({pd.to_datetime(os.stat(f).st_mtime, unit='s').strftime('%Y-%m-%d %H:%M')})", 
         'value': f"{str(f)}?t={current_time}"  # Ajouter un paramètre pour éviter le cache
        }
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
        dbc.Container([
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader(html.H3("Analyse de sentiment", className="mb-0")),
                        dbc.CardBody([
                            html.P("Configurez les paramètres de l'analyse de sentiment ci-dessous, puis cliquez sur 'Lancer'.", className="text-muted mb-3"),
                            
                            # Add topic filter component
                            get_topic_filter_component(id_prefix="sentiment-run-filter"),
                            
                            # Champ de sélection du fichier source
                            html.Div([
                                dbc.Label("Fichier source (JSON)"),
                                dbc.InputGroup([
                                    dbc.Input(id="sentiment-source-file-input", type="text", placeholder="Chemin vers le fichier JSON d'articles"),
                                    dbc.Button("Parcourir", id="sentiment-source-file-browse", color="secondary")
                                ]),
                                html.Small("Laissez vide pour utiliser le fichier par défaut de la configuration.", className="text-muted"),
                                html.Br()
                            ], className="mb-3"),
                            
                            # Form fields
                            html.Div(form_fields, className="mb-3"),
                            
                            # Run button
                            dbc.Button("Lancer l'analyse", id="run-sentiment-button", color="primary", className="mb-3"),
                            
                            # Output
                            html.Div(id="sentiment-run-output", className="mt-3")
                        ])
                    ], className="mb-4")
                ], width=12)
            ]),
            
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader(html.H3("Résultats de l'analyse de sentiment", className="mb-0")),
                        dbc.CardBody([
                            # Results dropdown
                            dbc.Row([
                                dbc.Col([
                                    dbc.Label("Résultats disponibles"),
                                    dcc.Dropdown(
                                        id="sentiment-results-dropdown",
                                        options=sentiment_results,
                                        value=sentiment_results[0]['value'] if sentiment_results else None,
                                        placeholder="Sélectionnez un fichier de résultats"
                                    )
                                ], width=10),
                                dbc.Col([
                                    html.Div([
                                        create_export_button("sentiment_analysis", "sentiment-export-button")
                                    ], className="d-flex justify-content-end")
                                ], width=2)
                            ], className="mb-3"),
                            
                            # Add export modal and feedback toast
                            create_export_modal("sentiment_analysis", "sentiment-export-modal"),
                            create_feedback_toast("sentiment-export-feedback"),
                            
                            html.Br(),
                            
                            # Le composant de filtrage par cluster a été supprimé ici car il est utilisé uniquement en amont
                            
                            # Results container
                            html.Div(id="sentiment-results-container", children=[
                                # This will be populated by the callback
                            ])
                        ], className="mt-3")
                    ])
                ], width=12)
            ])
        ])
    ])
    
    # Add article modal (ancien modal)
    # Ajouter les nouveaux modals pour l'affichage des articles lors d'un clic sur les graphiques
    layout.children.append(create_articles_modal(id_prefix="sentiment"))
    layout.children.append(create_full_article_modal(id_prefix="sentiment"))
    
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
            height=400,
            clickmode='event+select'  # Activer les événements de clic
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
        sentiment_histogram.update_layout(
            height=400,
            clickmode='event+select'  # Activer les événements de clic
        )
        
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
                dbc.Col(dcc.Graph(
                    id={'type': 'sentiment-graph', 'subtype': 'sentiment-pie'},
                    figure=sentiment_distribution
                ), width=6),
                dbc.Col(dcc.Graph(
                    id={'type': 'sentiment-graph', 'subtype': 'sentiment-histogram'},
                    figure=sentiment_histogram
                ), width=6)
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
                    html.Td(html.A(title, id={'type': 'sentiment-graph', 'subtype': 'sentiment-top-negative', 'index': i}, className="article-link")),
                    html.Td(f"{score:.4f}"),
                    html.Td(newspaper),
                    html.Td(date),
                    html.Td(article_id, style={"display": "none"})
                ], id={'type': 'sentiment-article-row', 'sentiment': 'negative', 'index': i}, **{'data-article-id': article_id})
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
                    html.Td(html.A(title, id={'type': 'sentiment-graph', 'subtype': 'sentiment-top-positive', 'index': i}, className="article-link")),
                    html.Td(f"{score:.4f}"),
                    html.Td(newspaper),
                    html.Td(date),
                    html.Td(article_id, style={"display": "none"})
                ], id={'type': 'sentiment-article-row', 'sentiment': 'positive', 'index': i}, **{'data-article-id': article_id})
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
def create_filtered_sentiment_visualizations(summary_file_path, cluster_file, cluster_id):
    """
    Crée des visualisations de sentiment filtrées par cluster.
    
    Args:
        summary_file_path: Chemin vers le fichier de résultats de sentiment
        cluster_file: Chemin vers le fichier de clusters
        cluster_id: ID du cluster à inclure
        
    Returns:
        Composants Dash pour les visualisations
    """
    try:
        # Charger les données de sentiment
        with open(summary_file_path, 'r', encoding='utf-8') as f:
            summary_data = json.load(f)
        
        # Charger les données de cluster
        with open(cluster_file, 'r', encoding='utf-8') as f:
            cluster_data = json.load(f)
        
        # Charger les articles avec sentiment
        articles_file_path = summary_file_path.replace('sentiment_summary', 'articles_with_sentiment')
        with open(articles_file_path, 'r', encoding='utf-8') as f:
            articles_data = json.load(f)
        
        # Filtrer les articles par cluster
        filtered_articles = []
        
        # Récupérer les IDs d'articles du cluster sélectionné
        cluster_article_ids = set()
        
        # Vérifier le format du fichier de cluster
        if "doc_ids" in cluster_data and "labels" in cluster_data:
            # Format avec doc_ids et labels
            doc_ids = cluster_data.get("doc_ids", [])
            labels = cluster_data.get("labels", [])
            
            # Convertir cluster_id en entier
            try:
                cluster_id_int = int(cluster_id)
                
                # Filtrer les articles par label
                if len(doc_ids) == len(labels):
                    for i, label in enumerate(labels):
                        if label == cluster_id_int:
                            cluster_article_ids.add(doc_ids[i])
            except ValueError:
                print(f"Erreur: L'ID de cluster {cluster_id} n'est pas un entier valide")
        elif "clusters" in cluster_data:
            # Format avec une liste de clusters
            clusters = cluster_data.get("clusters", [])
            for cluster in clusters:
                if str(cluster.get("id", "")) == str(cluster_id):
                    cluster_article_ids.update(cluster.get("articles", []))
        else:
            # Format inconnu, essayer de détecter les clusters
            print(f"Format de fichier de cluster inconnu: {list(cluster_data.keys())}")
            
            # Si le fichier contient des clés numériques, supposer que ce sont des clusters
            if str(cluster_id) in cluster_data:
                articles = cluster_data.get(str(cluster_id), [])
                if isinstance(articles, list):
                    cluster_article_ids.update(articles)
        
        # Filtrer les articles
        for article in articles_data:
            article_id = article.get('id', article.get('doc_id', ''))
            
            # Vérifier si l'article est dans le cluster sélectionné
            if article_id in cluster_article_ids:
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

# Fonction pour extraire les données du clic sur un graphique de sentiment
def extract_sentiment_click_data(point, prop_id):
    try:
        # Extraire l'ID du graphique (sous forme de dictionnaire)
        graph_id = json.loads(prop_id.split('.')[0])
        subtype = graph_id.get('subtype', '')
        
        print(f"Clic sur graphique: {subtype}")
        print(f"Point data: {point}")
        
        filter_type = None
        filter_value = None
        term = None
        
        # Extraire les données en fonction du type de graphique
        if subtype == 'sentiment-by-newspaper':
            # Pour le graphique de sentiment par journal
            filter_type = 'newspaper'
            filter_value = point.get('label') or point.get('x')
            term = filter_value  # Simplified term for better filtering
        elif subtype == 'sentiment-by-date':
            # Pour le graphique de sentiment par date
            filter_type = 'date'
            filter_value = point.get('x')
            term = filter_value  # Simplified term for better filtering
        elif subtype == 'sentiment-by-topic':
            # Pour le graphique de sentiment par sujet
            filter_type = 'topic'
            filter_value = point.get('label') or point.get('x')
            term = filter_value  # Simplified term for better filtering
        elif subtype == 'sentiment-distribution':
            # Pour le graphique de distribution des sentiments (histogramme)
            filter_type = 'sentiment_score'
            score = point.get('x')
            
            # Déterminer la catégorie de sentiment en fonction du score
            if score <= -0.05:
                sentiment_category = "négatif"
            elif score >= 0.05:
                sentiment_category = "positif"
            else:
                sentiment_category = "neutre"
                
            filter_value = score
            # No term needed for sentiment filtering as it's handled by the filter_type
        elif subtype == 'sentiment-pie' or subtype == 'sentiment-histogram':
            # Pour le graphique en camembert des sentiments
            if 'label' in point:
                # Cas du camembert
                filter_type = 'sentiment_category'
                filter_value = point['label'].lower()  # positif, neutre, négatif
                # No term needed for sentiment filtering as it's handled by the filter_type
            else:
                # Cas de l'histogramme
                filter_type = 'sentiment_score'
                score = point.get('x')
                
                # Déterminer la catégorie de sentiment en fonction du score
                if score <= -0.05:
                    sentiment_category = "négatif"
                elif score >= 0.05:
                    sentiment_category = "positif"
                else:
                    sentiment_category = "neutre"
                    
                filter_value = score
                # No term needed for sentiment filtering as it's handled by the filter_type
        elif subtype == 'sentiment-top-positive' or subtype == 'sentiment-top-negative':
            # Pour les graphiques des articles les plus positifs/négatifs
            # Récupérer l'ID de l'article depuis les attributs de la ligne du tableau
            try:
                row_id = graph_id.get('index')
                # Trouver l'article correspondant dans la liste des articles
                filter_type = 'sentiment_category'
                if subtype == 'sentiment-top-positive':
                    filter_value = 'positif'
                else:
                    filter_value = 'négatif'
                # No term needed for sentiment filtering as it's handled by the filter_type
            except Exception as e:
                print(f"Erreur lors de l'extraction de l'ID d'article: {str(e)}")
                filter_type = None
                filter_value = None
                term = None
        
        print(f"Filtres extraits: type={filter_type}, value={filter_value}, term={term}")
        return filter_type, filter_value, term
    except Exception as e:
        print(f"Erreur lors de l'extraction des données de clic: {str(e)}")
        return None, None, None

# Callback registration (to be called from app.py)
def register_sentiment_analysis_callbacks(app):
    # Functions for export
    def get_sentiment_source_data():
        """Obtient les données source pour l'exportation."""
        import logging
        logger = logging.getLogger(__name__)
        
        # Log all available states for debugging
        logger.info("États disponibles dans le contexte:")
        for key, value in ctx.states.items():
            logger.info(f"  {key}: {value}")
        
        # Récupérer le fichier de résultats sélectionné
        results_file = ctx.states.get("sentiment-results-dropdown.value")
        logger.info(f"Fichier de résultats récupéré: {results_file}")
        
        # Si aucun fichier n'est sélectionné, essayer de récupérer le dernier fichier utilisé
        if not results_file:
            # Chercher dans les options du dropdown
            dropdown_options = ctx.states.get("sentiment-results-dropdown.options", [])
            logger.info(f"Options du dropdown: {dropdown_options}")
            
            if dropdown_options and len(dropdown_options) > 0:
                # Prendre le premier fichier disponible
                try:
                    results_file = dropdown_options[0].get('value')
                    logger.info(f"Utilisation du premier fichier disponible: {results_file}")
                except Exception as e:
                    logger.error(f"Erreur lors de la récupération du premier fichier: {str(e)}")
        
        # Extraire le chemin du fichier du paramètre de cache-busting
        if results_file and '?' in results_file:
            results_file = results_file.split('?')[0]
            logger.info(f"Fichier de résultats après nettoyage: {results_file}")
        
        # Si toujours aucun fichier, chercher directement dans le répertoire des résultats
        if not results_file:
            import os
            from pathlib import Path
            from src.utils.config_loader import load_config
            
            try:
                # Charger la configuration
                project_root = Path(__file__).parent.parent.parent
                config_path = os.path.join(project_root, "config", "config.yaml")
                config = load_config(config_path)
                
                # Construire le chemin vers le répertoire des résultats d'analyse de sentiment
                results_dir = os.path.join(config['data']['results_dir'], 'sentiment_analysis')
                logger.info(f"Recherche dans le répertoire: {results_dir}")
                
                # Trouver le fichier sentiment_summary le plus récent
                sentiment_files = [f for f in os.listdir(results_dir) if f.startswith('sentiment_summary_')]
                if sentiment_files:
                    sentiment_files.sort(key=lambda x: os.path.getmtime(os.path.join(results_dir, x)), reverse=True)
                    results_file = os.path.join(results_dir, sentiment_files[0])
                    logger.info(f"Fichier de sentiment le plus récent trouvé: {results_file}")
            except Exception as e:
                logger.error(f"Erreur lors de la recherche de fichiers de sentiment: {str(e)}")
        
        source_data = {
            "results_file": results_file
        }
        logger.info(f"Source data final: {source_data}")
        
        
        # Si un fichier de résultats est sélectionné, ajouter des métadonnées
        if results_file:
            try:
                with open(results_file, 'r', encoding='utf-8') as f:
                    summary_data = json.load(f)
                
                # Ajouter des métadonnées de base
                source_data.update({
                    "model": summary_data.get("model", "vader"),
                    "transformer_model": summary_data.get("transformer_model"),
                    "num_articles": summary_data.get("num_articles", 0),
                    "filter_settings": summary_data.get("filter_settings", {}),
                    "run_id": summary_data.get("run_id"),
                    "timestamp": summary_data.get("timestamp")
                })
                
                # Ajouter des statistiques de sentiment
                summary = summary_data.get("summary", {})
                if summary:
                    source_data.update({
                        "mean_compound": summary.get("mean_compound"),
                        "positive_count": summary.get("positive_count"),
                        "neutral_count": summary.get("neutral_count"),
                        "negative_count": summary.get("negative_count"),
                        "positive_percentage": summary.get("positive_percentage"),
                        "neutral_percentage": summary.get("neutral_percentage"),
                        "negative_percentage": summary.get("negative_percentage")
                    })
            except Exception as e:
                print(f"Erreur lors de la récupération des données source : {str(e)}")
        
        return source_data
    
    def get_sentiment_figure():
        """Obtient la figure pour l'exportation."""
        # Récupérer le fichier de résultats
        results_file = ctx.states.get("sentiment-results-dropdown.value")
        
        # Extraire le chemin du fichier du paramètre de cache-busting
        if results_file and '?' in results_file:
            results_file = results_file.split('?')[0]
        
        if not results_file:
            return {}
        
        try:
            # Charger les données de sentiment
            with open(results_file, 'r', encoding='utf-8') as f:
                summary_data = json.load(f)
            
            # Extraire les données de sentiment
            summary = summary_data.get("summary", {})
            
            if not summary:
                return {}
            
            # Créer un graphique en camembert pour la distribution des sentiments
            fig = go.Figure(data=[
                go.Pie(
                    labels=['Positif', 'Neutre', 'Négatif'],
                    values=[
                        summary.get("positive_count", 0),
                        summary.get("neutral_count", 0),
                        summary.get("negative_count", 0)
                    ],
                    hole=0.4,
                    marker_colors=['#2ca02c', '#d3d3d3', '#d62728']
                )
            ])
            
            fig.update_layout(
                title="Distribution du sentiment",
                annotations=[{
                    "text": f"Score moyen: {summary.get('mean_compound', 0):.2f}",
                    "showarrow": False,
                    "font_size": 14,
                    "x": 0.5,
                    "y": 0.5
                }]
            )
            
            return fig.to_dict()
        
        except Exception as e:
            print(f"Erreur lors de la création de la figure: {str(e)}")
            return {}
    
    # Register export callbacks
    register_export_callbacks(
        app,
        analysis_type="sentiment_analysis",
        get_source_data_function=get_sentiment_source_data,
        get_figure_function=get_sentiment_figure,
        button_id="sentiment-export-button",
        modal_id="sentiment-export-modal",
        toast_id="sentiment-export-feedback"
    )
    # Register callbacks for topic filter component (uniquement pour l'analyse)
    register_topic_filter_callbacks(app, id_prefix="sentiment-run-filter")
    
    # Enregistrer les callbacks pour l'affichage des articles lors d'un clic sur les graphiques
    register_articles_modal_callback(
        app,
        graph_id_pattern={'type': 'sentiment-graph', 'subtype': ALL},
        id_prefix="sentiment",
        data_extraction_func=extract_sentiment_click_data
    )
    
    # Enregistrer le callback pour l'affichage de l'article complet
    register_full_article_modal_callback(app, id_prefix="sentiment")
    
    # Callback to run sentiment analysis
    @app.callback(
        Output("sentiment-run-output", "children"),
        Output("sentiment-results-dropdown", "options"),
        Output("sentiment-results-dropdown", "value"),
        Input("run-sentiment-button", "n_clicks"),
        [State(f"sentiment-{arg['name']}-input", "value") for arg in get_sentiment_analysis_args()] +
        [State("sentiment-run-filter-cluster-file-dropdown", "value"),
         State("sentiment-run-filter-cluster-id-dropdown", "value"),
         State("sentiment-source-file-input", "value"),
         State("sentiment-results-dropdown", "options")],
        prevent_initial_call=True
    )
    def run_sentiment_analysis(n_clicks, *args):
        if not n_clicks:
            current_options = args[-1]
            return "", current_options, None
        
        # Get argument names
        arg_names = [arg['name'] for arg in get_sentiment_analysis_args()]
        
        # Extract cluster filter parameters, source file and current options
        cluster_file = args[-4]
        cluster_id = args[-3]
        source_file = args[-2]
        current_options = args[-1]
        args = args[:-4]  # Remove cluster filter parameters, source file and current options from args
        
        # Get absolute path to the script
        script_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "scripts", "run_sentiment_analysis.py"))
        
        # Get project root directory
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
        
        # Create command
        cmd = [sys.executable, script_path]
        
        # Add arguments
        for arg_name, arg_value in zip(arg_names, args):
            if arg_value is not None and arg_value != "":
                if isinstance(arg_value, bool):
                    if arg_value:
                        cmd.append(f"--{arg_name}")
                else:
                    cmd.append(f"--{arg_name}")
                    cmd.append(str(arg_value))
        
        # Add cluster filter parameters if provided
        if cluster_file and cluster_id:
            cmd.extend(["--cluster-file", str(cluster_file)])
            cmd.extend(["--cluster-id", str(cluster_id)])
            
        # Add source file parameter if provided
        if source_file:
            cmd.extend(["--source-file", str(source_file)])
        
        # Run command
        try:
            # Exécuter la commande avec le répertoire racine du projet comme répertoire de travail
            result = subprocess.run(cmd, capture_output=True, text=True, check=True, cwd=project_root)
            # Récupérer la liste mise à jour des résultats disponibles
            updated_results = get_sentiment_results()
            
            # Trouver le fichier de résultat le plus récent
            selected_value = updated_results[0]['value'] if updated_results else None
            
            return html.Div([
                html.P("Analyse de sentiment terminée avec succès !"),
                html.Pre(result.stdout, style={"maxHeight": "300px", "overflow": "auto", "backgroundColor": "#f0f0f0", "padding": "10px"})
            ]), updated_results, selected_value
        except subprocess.CalledProcessError as e:
            return html.Div([
                html.P("Erreur lors de l'analyse de sentiment:", className="text-danger"),
                html.Pre(e.stderr, style={"maxHeight": "300px", "overflow": "auto", "backgroundColor": "#f0f0f0", "padding": "10px"})
            ]), current_options, None
    
    # Callback to display sentiment analysis results
    @app.callback(
        Output("sentiment-results-container", "children"),
        Input("sentiment-results-dropdown", "value")
    )
    def display_sentiment_results(results_file):
        # Extraire le chemin du fichier du paramètre de cache-busting
        if results_file and '?' in results_file:
            results_file = results_file.split('?')[0]

        if not results_file:
            return html.P("Sélectionnez un fichier de résultats pour afficher les visualisations.")

        # Afficher les résultats sans filtre
        return create_sentiment_visualizations(results_file)

    # Le callback pour lancer une analyse de sentiment filtrée a été supprimé car nous utilisons maintenant le filtrage par cluster directement

    # Le callback toggle_article_modal a été supprimé car nous utilisons maintenant les fonctions de article_display_utils.py
    
    # Callback pour préremplir le titre et la description dans la modal d'exportation
    @app.callback(
        Output("sentiment-export-title-input", "value"),
        Output("sentiment-export-description-input", "value"),
        Input("sentiment-export-modal", "is_open"),
        State("sentiment-results-dropdown", "value"),
        prevent_initial_call=True
    )
    def prefill_export_modal(is_open, results_file):
        if not is_open or not results_file:
            return "", ""
        
        # Extraire le chemin du fichier du paramètre de cache-busting
        if '?' in results_file:
            results_file = results_file.split('?')[0]
        
        # Extraire le nom du fichier pour le titre par défaut
        file_name = os.path.basename(results_file)
        title = f"Analyse de sentiment - {file_name.replace('sentiment_summary_', '').replace('.json', '')}"
        
        # Description par défaut
        description = "Analyse de sentiment des articles de presse"
        
        return title, description
    
    # Callback pour le bouton de parcourir du fichier source
    @app.callback(
        Output("sentiment-source-file-input", "value"),
        Input("sentiment-source-file-browse", "n_clicks"),
        State("sentiment-source-file-input", "value"),
        prevent_initial_call=True
    )
    def browse_source_file(n_clicks, current_value):
        if not n_clicks:
            return current_value
        
        # Obtenir le répertoire de départ pour la boîte de dialogue
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
        data_dir = os.path.join(project_root, "data", "processed")
        
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

# To be called in app.py: from src.webapp.sentiment_analysis_viz import register_sentiment_analysis_callbacks
