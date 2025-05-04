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

from src.webapp.topic_filter_component import (
    get_topic_filter_component, 
    register_topic_filter_callbacks, 
    get_filter_parameters,
    get_filter_states,
    are_filters_active
)

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
                html.P("Analyser les entités nommées dans les articles de presse et visualiser les résultats."),
                html.Hr()
            ], width=12)
        ]),
        
        # Tabs for Run and Results
        dbc.Tabs([
            # Tab for running entity recognition
            dbc.Tab(label="Lancer une analyse", children=[
                html.Div([
                    html.H4("Paramètres d'analyse"),
                    html.P("Configurez les paramètres pour la reconnaissance d'entités nommées."),
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
                    html.P("Sélectionnez un fichier de résultats pour visualiser l'analyse d'entités nommées."),
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
                    
                    # Ajout du composant de filtrage par topic/cluster
                    dbc.Row([
                        dbc.Col([
                            dbc.Accordion([
                                dbc.AccordionItem(
                                    get_topic_filter_component(id_prefix="entity-topic-filter"),
                                    title="Filtrage par Topic/Cluster"
                                )
                            ], start_collapsed=True, id="entity-filter-accordion")
                        ], width=12)
                    ]),
                    html.Br(),
                    
                    # Results container
                    html.Div(id="entity-results-container", children=[
                        # This will be populated by the callback
                    ])
                ], className="mt-3")
            ]),
            
            # Nouvelle tab pour les analyses filtrées
            dbc.Tab(label="Analyses Filtrées", children=[
                html.Div([
                    html.H4("Analyses d'entités nommées filtrées par topic/cluster"),
                    html.P("Lancez une analyse d'entités nommées filtrée par topic ou cluster."),
                    
                    # Composant de filtrage
                    get_topic_filter_component(id_prefix="entity-filtered-analysis"),
                    html.Br(),
                    
                    # Bouton pour lancer l'analyse filtrée
                    dbc.Button(
                        "Lancer l'analyse filtrée",
                        id="run-filtered-entity-button",
                        color="primary",
                        className="mb-3"
                    ),
                    html.Br(),
                    
                    # Conteneur pour les résultats filtrés
                    html.Div(id="filtered-entity-results-container")
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

# Function to create filtered entity visualizations
def create_filtered_entity_visualizations(summary_file_path, topic_results_path, topic_id, cluster_id, exclude_topic_id, exclude_cluster_id):
    """
    Crée des visualisations d'entités nommées filtrées par topic/cluster.
    
    Args:
        summary_file_path: Chemin vers le fichier de résultats d'entités nommées
        topic_results_path: Chemin vers le fichier de résultats de topic modeling
        topic_id: ID du topic à inclure
        cluster_id: ID du cluster à inclure
        exclude_topic_id: ID du topic à exclure
        exclude_cluster_id: ID du cluster à exclure
        
    Returns:
        Composants Dash pour les visualisations
    """
    try:
        # Charger les données d'entités nommées
        with open(summary_file_path, 'r', encoding='utf-8') as f:
            summary_data = json.load(f)
        
        # Charger les données de topic
        with open(topic_results_path, 'r', encoding='utf-8') as f:
            topic_data = json.load(f)
        
        # Charger les articles avec entités
        articles_file_path = summary_file_path.replace('entity_summary', 'articles_with_entities')
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
        
        # Vérifier s'il y a des articles filtrés
        if not filtered_articles:
            return html.Div([
                html.P("Aucun article ne correspond aux filtres sélectionnés.", className="text-warning")
            ])
        
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
        
        # Extraire toutes les entités des articles filtrés
        all_entities = []
        for article in filtered_articles:
            if 'entities' in article:
                all_entities.extend(article['entities'])
        
        # Calculer les statistiques d'entités pour les articles filtrés
        entity_counts = {}
        entity_types = {}
        
        for entity in all_entities:
            entity_text = entity.get('text', '')
            entity_type = entity.get('type', '')
            
            if entity_text and entity_type:
                entity_key = f"{entity_text}|{entity_type}"
                entity_counts[entity_key] = entity_counts.get(entity_key, 0) + 1
                entity_types[entity_type] = entity_types.get(entity_type, 0) + 1
        
        # Créer un DataFrame pour les visualisations
        entity_df = pd.DataFrame([
            {'entity': key.split('|')[0], 'type': key.split('|')[1], 'count': count}
            for key, count in entity_counts.items()
        ])
        
        # Créer les visualisations
        if entity_df.empty:
            visualizations.append(html.P("Aucune entité trouvée dans les articles filtrés."))
            return visualizations
        
        # 1. Distribution des types d'entités
        entity_type_fig = px.pie(
            pd.DataFrame({'type': list(entity_types.keys()), 'count': list(entity_types.values())}),
            names='type',
            values='count',
            title="Distribution des types d'entités (articles filtrés)",
            hole=0.4
        )
        visualizations.append(dcc.Graph(figure=entity_type_fig))
        
        # 2. Top entités par type
        for entity_type in sorted(entity_df['type'].unique()):
            type_df = entity_df[entity_df['type'] == entity_type].sort_values('count', ascending=False).head(10)
            
            if not type_df.empty:
                type_fig = px.bar(
                    type_df,
                    x='entity',
                    y='count',
                    title=f"Top 10 entités de type {entity_type} (articles filtrés)",
                    labels={'entity': 'Entité', 'count': 'Nombre d\'occurrences'}
                )
                type_fig.update_layout(xaxis_tickangle=-45)
                visualizations.append(dcc.Graph(figure=type_fig))
        
        return visualizations
    
    except Exception as e:
        return html.Div([
            html.P(f"Erreur lors de la création des visualisations filtrées: {str(e)}", className="text-danger")
        ])

# Callback registration (to be called from app.py)
def register_entity_recognition_callbacks(app):
    # Register callbacks for topic filter component
    register_topic_filter_callbacks(app, id_prefix="entity-topic-filter")
    register_topic_filter_callbacks(app, id_prefix="entity-filtered-analysis")
    
    # Callback to run entity recognition
    @app.callback(
        Output("entity-run-output", "children"),
        Input("run-entity-button", "n_clicks"),
        [State(f"entity-{arg['name']}-input", "value") for arg in get_entity_recognition_args()],
        prevent_initial_call=True
    )
    def run_entity_recognition(n_clicks, *args):
        if not n_clicks:
            return ""
        
        # Get argument names
        arg_names = [arg['name'] for arg in get_entity_recognition_args()]
        
        # Create command
        cmd = [sys.executable, os.path.join(os.path.dirname(__file__), "..", "scripts", "run_entity_recognition.py")]
        
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
                html.P("Analyse d'entités nommées terminée avec succès !"),
                html.Pre(result.stdout, style={"maxHeight": "300px", "overflow": "auto", "backgroundColor": "#f0f0f0", "padding": "10px"})
            ])
        except subprocess.CalledProcessError as e:
            return html.Div([
                html.P("Erreur lors de l'analyse d'entités nommées:", className="text-danger"),
                html.Pre(e.stderr, style={"maxHeight": "300px", "overflow": "auto", "backgroundColor": "#f0f0f0", "padding": "10px"})
            ])
    
    # Callback to display entity recognition results
    @app.callback(
        Output("entity-results-container", "children"),
        [Input("entity-results-dropdown", "value"),
         Input("entity-topic-filter-apply-button", "n_clicks")],
        [State("entity-topic-filter-topic-dropdown", "value"),
         State("entity-topic-filter-cluster-dropdown", "value"),
         State("entity-topic-filter-exclude-topic-dropdown", "value"),
         State("entity-topic-filter-exclude-cluster-dropdown", "value"),
         State("entity-topic-filter-results-dropdown", "value")]
    )
    def display_entity_results(results_file, apply_filters, topic_id, cluster_id, exclude_topic_id, exclude_cluster_id, topic_results_file):
        ctx_trigger = ctx.triggered_id
        if not results_file:
            return html.P("Sélectionnez un fichier de résultats pour afficher les visualisations.")
        
        # Vérifier si les filtres sont actifs
        filters_active = are_filters_active("entity-topic-filter", ctx)
        
        if filters_active and topic_results_file:
            # Appliquer les filtres
            return create_filtered_entity_visualizations(
                results_file, 
                topic_results_file, 
                topic_id, 
                cluster_id, 
                exclude_topic_id, 
                exclude_cluster_id
            )
        else:
            # Afficher les résultats sans filtre
            return create_entity_visualizations(results_file)
    
    # Callback pour lancer une analyse d'entités nommées filtrée
    @app.callback(
        Output("filtered-entity-results-container", "children"),
        Input("run-filtered-entity-button", "n_clicks"),
        [State("entity-filtered-analysis-topic-dropdown", "value"),
         State("entity-filtered-analysis-cluster-dropdown", "value"),
         State("entity-filtered-analysis-exclude-topic-dropdown", "value"),
         State("entity-filtered-analysis-exclude-cluster-dropdown", "value"),
         State("entity-filtered-analysis-results-dropdown", "value")],
        prevent_initial_call=True
    )
    def run_filtered_entity_analysis(n_clicks, topic_id, cluster_id, exclude_topic_id, exclude_cluster_id, topic_results_file):
        if not n_clicks or not topic_results_file:
            return html.P("Veuillez sélectionner un fichier de résultats de topic modeling et configurer les filtres.")
        
        # Créer la commande
        cmd = [
            sys.executable, 
            os.path.join(os.path.dirname(__file__), "..", "scripts", "run_filtered_analysis.py"),
            "--analysis-type", "entity"
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
                    html.P("Analyse d'entités nommées filtrée terminée avec succès !"),
                    html.Pre(result.stdout, style={"maxHeight": "200px", "overflow": "auto", "backgroundColor": "#f0f0f0", "padding": "10px"}),
                    html.Hr(),
                    html.H5("Résultats de l'analyse filtrée :"),
                    *create_entity_visualizations(output_path, is_filtered=True)
                ])
            else:
                return html.Div([
                    html.P("Analyse terminée, mais impossible de trouver le fichier de résultats."),
                    html.Pre(result.stdout, style={"maxHeight": "300px", "overflow": "auto", "backgroundColor": "#f0f0f0", "padding": "10px"})
                ])
        except subprocess.CalledProcessError as e:
            return html.Div([
                html.P("Erreur lors de l'analyse d'entités nommées filtrée:", className="text-danger"),
                html.Pre(e.stderr, style={"maxHeight": "300px", "overflow": "auto", "backgroundColor": "#f0f0f0", "padding": "10px"})
            ])

# To be called in app.py: from src.webapp.entity_recognition_viz import register_entity_recognition_callbacks
