"""
Entity Recognition Visualization Page for Dash app
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
    # Utiliser os.stat au lieu de os.path.getmtime pour forcer la mise à jour des horodatages
    import time as time_module
    current_time = int(time_module.time())  # Timestamp actuel pour éviter les problèmes de cache
    summary_files.sort(key=lambda x: os.stat(x).st_mtime, reverse=True)
    
    # Format for dropdown
    options = [
        {'label': f"{f.stem} ({pd.to_datetime(os.stat(f).st_mtime, unit='s').strftime('%Y-%m-%d %H:%M')})", 
         'value': f"{str(f)}?t={current_time}"}  # Ajouter un paramètre pour éviter le cache
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
                    
                    # Champ de sélection du fichier source
                    html.Div([
                        dbc.Label("Fichier source (JSON)"),
                        dbc.InputGroup([
                            dbc.Input(id="entity-source-file-input", type="text", placeholder="Chemin vers le fichier JSON d'articles"),
                            dbc.Button("Parcourir", id="entity-source-file-browse", color="secondary")
                        ]),
                        html.Small("Laissez vide pour utiliser le fichier par défaut de la configuration.", className="text-muted"),
                        html.Br()
                    ], className="mb-3"),
                    
                    dbc.Form(dbc.Row(form_fields)),
                    html.Br(),
                    
                    # Ajout du composant de filtrage par cluster
                    html.H5("Filtrage par Cluster (optionnel)"),
                    html.P("Sélectionnez un fichier de cluster et un ID de cluster pour filtrer les articles."),
                    get_topic_filter_component(id_prefix="entity-run-filter"),
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
                        ], width=10),
                        dbc.Col([
                            html.Div([
                                create_export_button("entity_recognition", "entity-export-button")
                            ], className="d-flex justify-content-end")
                        ], width=2)
                    ]),
                    
                    # Add export modal and feedback toast
                    create_export_modal("entity_recognition", "entity-export-modal"),
                    create_feedback_toast("entity-export-feedback"),
                    html.Br(),
                    
                    # Le composant de filtrage par cluster a été supprimé ici car il est utilisé uniquement en amont
                    
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
            height=400,
            clickmode='event+select'
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
                dbc.Col(dcc.Graph(
                    id={'type': 'entity-graph', 'subtype': 'entity-distribution'},
                    figure=entity_distribution
                ), width=12)
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
                                dbc.Col(dcc.Graph(
                                    id={'type': 'entity-graph', 'subtype': f'entity-frequency-{entity_type}'},
                                    figure=fig
                                ), width=12)
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
        
        # Ajouter les modals pour l'affichage des articles
        visualizations.append(create_articles_modal(id_prefix="entity"))
        visualizations.append(create_full_article_modal(id_prefix="entity"))
        
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
def create_filtered_entity_visualizations(summary_file_path, cluster_file, cluster_id):
    """
    Crée des visualisations d'entités nommées filtrées par cluster.
    
    Args:
        summary_file_path: Chemin vers le fichier de résultats d'entités nommées
        cluster_file: Chemin vers le fichier de clusters
        cluster_id: ID du cluster à inclure
        
    Returns:
        Composants Dash pour les visualisations
    """
    try:
        # Charger les données d'entités nommées
        with open(summary_file_path, 'r', encoding='utf-8') as f:
            summary_data = json.load(f)
        
        # Charger les données de cluster
        with open(cluster_file, 'r', encoding='utf-8') as f:
            cluster_data = json.load(f)
        
        # Charger les articles avec entités
        articles_file_path = summary_file_path.replace('entity_summary', 'articles_with_entities')
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
    # Functions for export
    def get_entity_source_data():
        """Obtient les données source pour l'exportation."""
        import logging
        logger = logging.getLogger(__name__)
        
        # Log all available states for debugging
        logger.info("États disponibles dans le contexte:")
        for key, value in ctx.states.items():
            logger.info(f"  {key}: {value}")
        
        # Récupérer le fichier de résultats sélectionné
        results_file = ctx.states.get("entity-results-dropdown.value")
        logger.info(f"Fichier de résultats récupéré: {results_file}")
        
        # Si aucun fichier n'est sélectionné, essayer de récupérer le dernier fichier utilisé
        if not results_file:
            # Chercher dans les options du dropdown
            dropdown_options = ctx.states.get("entity-results-dropdown.options", [])
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
                
                # Construire le chemin vers le répertoire des résultats d'analyse d'entités
                results_dir = os.path.join(config['data']['results_dir'], 'entity_recognition')
                logger.info(f"Recherche dans le répertoire: {results_dir}")
                
                # Trouver le fichier entity_summary le plus récent
                entity_files = [f for f in os.listdir(results_dir) if f.startswith('entity_summary_')]
                if entity_files:
                    entity_files.sort(key=lambda x: os.path.getmtime(os.path.join(results_dir, x)), reverse=True)
                    results_file = os.path.join(results_dir, entity_files[0])
                    logger.info(f"Fichier d'entités le plus récent trouvé: {results_file}")
            except Exception as e:
                logger.error(f"Erreur lors de la recherche de fichiers d'entités: {str(e)}")
        
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
                    "model": summary_data.get("model", "spacy"),
                    "num_articles": summary_data.get("num_articles", 0),
                    "filter_settings": summary_data.get("filter_settings", {}),
                    "run_id": summary_data.get("run_id"),
                    "timestamp": summary_data.get("timestamp")
                })
                
                # Ajouter des statistiques d'entités
                summary = summary_data.get("summary", {})
                if summary:
                    source_data.update({
                        "total_entities": summary.get("total_entities", 0),
                        "unique_entities": summary.get("unique_entities", 0),
                        "entity_types": list(summary.get("total_by_type", {}).keys())
                    })
            except Exception as e:
                print(f"Erreur lors de la récupération des données source : {str(e)}")
        
        return source_data
    
    def get_entity_figure():
        """Obtient la figure pour l'exportation."""
        # Récupérer le fichier de résultats
        results_file = ctx.states.get("entity-results-dropdown.value")
        
        # Extraire le chemin du fichier du paramètre de cache-busting
        if results_file and '?' in results_file:
            results_file = results_file.split('?')[0]
        
        if not results_file:
            return {}
        
        try:
            # Charger les données d'entités
            with open(results_file, 'r', encoding='utf-8') as f:
                summary_data = json.load(f)
            
            # Extraire les données d'entités
            summary = summary_data.get("summary", {})
            
            if not summary:
                return {}
            
            # Créer un graphique en camembert pour la distribution des entités par type
            entity_types = list(summary.get("total_by_type", {}).keys())
            entity_counts = list(summary.get("total_by_type", {}).values())
            
            fig = go.Figure(data=[
                go.Pie(
                    labels=entity_types,
                    values=entity_counts,
                    hole=0.4
                )
            ])
            
            fig.update_layout(
                title="Distribution des entités par type",
                annotations=[{
                    "text": f"Total: {summary.get('total_entities', 0)}",
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
        analysis_type="entity_recognition",
        get_source_data_function=get_entity_source_data,
        get_figure_function=get_entity_figure,
        button_id="entity-export-button",
        modal_id="entity-export-modal",
        toast_id="entity-export-feedback"
    )
    
    # Register callbacks for topic filter component (uniquement pour l'analyse)
    register_topic_filter_callbacks(app, id_prefix="entity-run-filter")
    
    # Enregistrer les callbacks pour l'affichage des articles lors d'un clic sur les graphiques
    register_articles_modal_callback(
        app,
        graph_id_pattern={'type': 'entity-graph', 'subtype': ALL},
        id_prefix="entity",
        data_extraction_func=extract_entity_click_data
    )
    
    # Enregistrer le callback pour l'affichage de l'article complet
    register_full_article_modal_callback(app, id_prefix="entity")
    
    # Callback pour le bouton de parcourir du fichier source
    @app.callback(
        Output("entity-source-file-input", "value"),
        Input("entity-source-file-browse", "n_clicks"),
        State("entity-source-file-input", "value"),
        prevent_initial_call=True
    )
    def browse_source_file(n_clicks, current_value):
        if not n_clicks:
            return current_value
        
        # Obtenir le répertoire de départ pour la boîte de dialogue
        project_root = pathlib.Path(__file__).resolve().parents[2]
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
    
    # Callback pour préremplir le titre et la description dans la modal d'exportation
    @app.callback(
        Output("entity-export-title-input", "value"),
        Output("entity-export-description-input", "value"),
        Input("entity-export-modal", "is_open"),
        State("entity-results-dropdown", "value"),
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
        title = f"Analyse d'entités nommées - {file_name.replace('entity_summary_', '').replace('.json', '')}"
        
        # Description par défaut
        description = "Analyse d'entités nommées des articles de presse"
        
        return title, description
    
    # Callback to run entity recognition
    @app.callback(
        Output("entity-run-output", "children"),
        Output("entity-results-dropdown", "options"),
        Output("entity-results-dropdown", "value"),
        Input("run-entity-button", "n_clicks"),
        [State(f"entity-{arg['name']}-input", "value") for arg in get_entity_recognition_args()] +
        [State("entity-run-filter-cluster-file-dropdown", "value"),
         State("entity-run-filter-cluster-id-dropdown", "value"),
         State("entity-source-file-input", "value"),
         State("entity-results-dropdown", "options")],
        prevent_initial_call=True
    )
    def run_entity_recognition(n_clicks, *args):
        if not n_clicks:
            current_options = args[-1]
            return "", current_options, None
        
        # Get argument names
        arg_names = [arg['name'] for arg in get_entity_recognition_args()]
        
        # Extract cluster filter parameters, source file and current options
        cluster_file = args[-4]
        cluster_id = args[-3]
        source_file = args[-2]
        current_options = args[-1]
        args = args[:-4]  # Remove cluster filter parameters, source file and current options from args
        
        # Get absolute path to the script
        script_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "scripts", "run_entity_recognition.py"))
        
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
            updated_results = get_entity_results()
            
            # Trouver le fichier de résultat le plus récent
            selected_value = updated_results[0]['value'] if updated_results else None
            
            return html.Div([
                html.P("Analyse d'entités nommées terminée avec succès !"),
                html.Pre(result.stdout, style={"maxHeight": "300px", "overflow": "auto", "backgroundColor": "#f0f0f0", "padding": "10px"})
            ]), updated_results, selected_value
        except subprocess.CalledProcessError as e:
            return html.Div([
                html.P("Erreur lors de l'analyse d'entités nommées:", className="text-danger"),
                html.Pre(e.stderr, style={"maxHeight": "300px", "overflow": "auto", "backgroundColor": "#f0f0f0", "padding": "10px"})
            ]), current_options, None
    
    # Callback to display entity recognition results
    @app.callback(
        Output("entity-results-container", "children"),
        Input("entity-results-dropdown", "value")
    )
    def display_entity_results(results_file):
        # Extraire le chemin du fichier du paramètre de cache-busting
        if results_file and '?' in results_file:
            results_file = results_file.split('?')[0]

        if not results_file:
            return html.P("Sélectionnez un fichier de résultats pour afficher les visualisations.")
        
        # Afficher les résultats sans filtre
        return create_entity_visualizations(results_file)
    
    # Fonction pour extraire les données du clic sur un graphique d'entités nommées
def extract_entity_click_data(point, prop_id):
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
        if subtype == 'entity-distribution':
            # Pour le graphique de distribution des entités par type
            filter_type = 'entity_type'
            filter_value = point.get('label')
            term = filter_value  # Simplified term for better filtering
        elif subtype.startswith('entity-frequency-'):
            # Pour les graphiques de fréquence d'entités par type
            entity_type = subtype.replace('entity-frequency-', '')
            filter_type = 'entity'
            filter_value = point.get('x')
            term = filter_value  # Simplified term for better filtering
        elif subtype == 'entity-top-articles':
            # Pour le tableau des articles avec le plus d'entités
            try:
                row_id = graph_id.get('index')
                # Trouver l'article correspondant dans la liste des articles
                filter_type = 'article_id'
                filter_value = row_id
                # No term needed for article filtering as it's handled by the filter_type
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

# Le callback pour lancer une analyse d'entités nommées filtrée a été supprimé car nous utilisons maintenant le filtrage par cluster directement


# To be called in app.py: from src.webapp.entity_recognition_viz import register_entity_recognition_callbacks, get_entity_recognition_layout
