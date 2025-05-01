"""
Integrated Analysis Visualization Page for Dash app
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

# Extract parser arguments from run_integrated_analysis.py
def get_integrated_analysis_args():
    spec = importlib.util.spec_from_file_location("run_integrated_analysis", os.path.join(os.path.dirname(__file__), "..", "scripts", "run_integrated_analysis.py"))
    run_integrated_analysis = importlib.util.module_from_spec(spec)
    sys.modules["run_integrated_analysis"] = run_integrated_analysis
    spec.loader.exec_module(run_integrated_analysis)
    parser = run_integrated_analysis.argparse.ArgumentParser(description="Analyse intégrée (clusters, sentiment, entités)")
    run_integrated_analysis.main.__globals__["parser"] = parser
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

# Helper to get available cluster files
def get_cluster_files():
    project_root = pathlib.Path(__file__).resolve().parents[2]
    config_path = project_root / 'config' / 'config.yaml'
    with open(config_path, encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    results_dir = project_root / config['data']['results_dir']
    
    # Look for cluster files in both the main results dir and clusters subdirectory
    cluster_files = list(results_dir.glob('*clusters*.json'))
    clusters_dir = results_dir / 'clusters'
    if clusters_dir.exists():
        cluster_files.extend(list(clusters_dir.glob('*clusters*.json')))
    
    # Sort by modification time (newest first)
    cluster_files.sort(key=lambda x: os.path.getmtime(x), reverse=True)
    
    # Format for dropdown
    options = [
        {'label': f"{f.stem} ({pd.to_datetime(os.path.getmtime(f), unit='s').strftime('%Y-%m-%d %H:%M')})", 
         'value': str(f)}
        for f in cluster_files
    ]
    
    return options

# Helper to get available integrated analysis result files
def get_integrated_results():
    project_root = pathlib.Path(__file__).resolve().parents[2]
    config_path = project_root / 'config' / 'config.yaml'
    with open(config_path, encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    results_dir = project_root / config['data']['results_dir'] / 'integrated_analysis'
    
    if not results_dir.exists():
        return []
    
    # Get all integrated analysis files
    result_files = list(results_dir.glob('integrated_analysis_*.json'))
    
    # Sort by modification time (newest first)
    result_files.sort(key=lambda x: os.path.getmtime(x), reverse=True)
    
    # Format for dropdown
    options = [
        {'label': f"{f.stem} ({pd.to_datetime(os.path.getmtime(f), unit='s').strftime('%Y-%m-%d %H:%M')})", 
         'value': str(f)}
        for f in result_files
    ]
    
    return options

# Layout for the integrated analysis page
def get_integrated_analysis_layout():
    # Get available cluster files and result files
    cluster_files = get_cluster_files()
    integrated_results = get_integrated_results()
    
    # Get parser arguments for the run form
    parser_args = get_integrated_analysis_args()
    
    # Create form fields based on parser arguments
    form_fields = []
    for arg in parser_args:
        if arg['name'] == 'cluster_file':
            # Create a dropdown for cluster file selection
            form_fields.append(
                dbc.Col([
                    dbc.Label(arg['help']),
                    dcc.Dropdown(
                        id=f"integrated-{arg['name']}-input",
                        options=cluster_files,
                        value=cluster_files[0]['value'] if cluster_files else None,
                        placeholder="Sélectionnez un fichier de clusters"
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
                        id=f"integrated-{arg['name']}-input",
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
                        id=f"integrated-{arg['name']}-input",
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
                html.H2("Analyse Intégrée des Clusters"),
                html.P("Analyser les clusters de topics avec sentiment et entités nommées."),
                html.Hr()
            ], width=12)
        ]),
        
        # Tabs for Run and Results
        dbc.Tabs([
            # Tab for running integrated analysis
            dbc.Tab(label="Lancer une analyse", children=[
                html.Div([
                    html.H4("Paramètres d'analyse"),
                    html.P("Configurez les paramètres pour l'analyse intégrée."),
                    dbc.Form(dbc.Row(form_fields)),
                    html.Br(),
                    dbc.Button("Lancer l'analyse", id="run-integrated-button", color="primary"),
                    html.Br(),
                    html.Div(id="integrated-run-output")
                ], className="mt-3")
            ]),
            
            # Tab for viewing results
            dbc.Tab(label="Résultats", children=[
                html.Div([
                    html.H4("Visualisation des résultats"),
                    html.P("Sélectionnez un fichier de résultats pour visualiser l'analyse intégrée."),
                    dbc.Row([
                        dbc.Col([
                            dbc.Label("Fichier de résultats:"),
                            dcc.Dropdown(
                                id="integrated-results-dropdown",
                                options=integrated_results,
                                value=integrated_results[0]['value'] if integrated_results else None,
                                placeholder="Sélectionnez un fichier de résultats"
                            )
                        ], width=6)
                    ]),
                    html.Br(),
                    
                    # Results container
                    html.Div(id="integrated-results-container", children=[
                        # This will be populated by the callback
                    ])
                ], className="mt-3")
            ])
        ])
    ])
    
    return layout

# Function to create integrated analysis visualizations
def create_integrated_visualizations(results_file_path):
    try:
        with open(results_file_path, 'r', encoding='utf-8') as f:
            results_data = json.load(f)
        
        # Extract data
        cluster_summaries = results_data['cluster_summaries']
        cluster_analysis = results_data['cluster_analysis']
        
        # Create visualizations
        visualizations = []
        
        # 1. Metadata card
        metadata_card = dbc.Card(
            dbc.CardBody([
                html.H4("Métadonnées", className="card-title"),
                html.P(f"Horodatage: {results_data['timestamp']}"),
                html.P(f"Durée: {results_data['duration_seconds']:.2f} secondes"),
                html.P(f"Fichier de clusters: {results_data['cluster_file']}"),
                html.P(f"Modèle de sentiment: {results_data['sentiment_model']}"),
                html.P(f"Modèle NER: {results_data['ner_model']}"),
                html.P(f"Types d'entités: {', '.join(results_data['entity_types'])}"),
            ])
        )
        
        # 2. Summary statistics card
        summary_card = dbc.Card(
            dbc.CardBody([
                html.H4("Statistiques globales", className="card-title"),
                html.P(f"Nombre de clusters: {results_data['num_clusters']}"),
                html.P(f"Nombre total d'articles: {results_data['total_articles']}"),
            ])
        )
        
        # Add cards to the container
        visualizations.append(
            dbc.Row([
                dbc.Col(metadata_card, width=6),
                dbc.Col(summary_card, width=6)
            ])
        )
        
        visualizations.append(html.Br())
        
        # 3. Cluster size comparison
        if cluster_summaries:
            # Create dataframe for cluster comparison
            df = pd.DataFrame(cluster_summaries)
            
            # Cluster size chart
            cluster_size_fig = px.bar(
                df, 
                x='cluster_id', 
                y='num_articles',
                title="Taille des clusters (nombre d'articles)",
                labels={"cluster_id": "Cluster", "num_articles": "Nombre d'articles"},
                color='avg_sentiment',
                color_continuous_scale='RdBu',
                color_continuous_midpoint=0
            )
            cluster_size_fig.update_layout(height=400)
            
            # Sentiment distribution by cluster
            sentiment_fig = go.Figure()
            
            for i, row in df.iterrows():
                sentiment_fig.add_trace(go.Bar(
                    name=f"Cluster {row['cluster_id']}",
                    x=['Positif', 'Neutre', 'Négatif'],
                    y=[row['positive_percentage'], row['neutral_percentage'], row['negative_percentage']],
                    text=[f"{row['positive_percentage']:.1f}%", f"{row['neutral_percentage']:.1f}%", f"{row['negative_percentage']:.1f}%"],
                    textposition='auto'
                ))
            
            sentiment_fig.update_layout(
                title="Distribution des sentiments par cluster",
                xaxis_title="Sentiment",
                yaxis_title="Pourcentage",
                barmode='group',
                height=400
            )
            
            visualizations.append(
                dbc.Row([
                    dbc.Col(dcc.Graph(figure=cluster_size_fig), width=6),
                    dbc.Col(dcc.Graph(figure=sentiment_fig), width=6)
                ])
            )
        
        # 4. Cluster tabs for detailed analysis
        cluster_tabs = []
        
        for cluster_id, data in sorted(cluster_analysis.items(), key=lambda x: int(x[0]) if x[0].isdigit() else x[0]):
            # Create entity distribution pie chart
            entity_types = list(data['entities']['total_by_type'].keys())
            entity_counts = list(data['entities']['total_by_type'].values())
            
            entity_fig = go.Figure(data=[
                go.Pie(
                    labels=entity_types,
                    values=entity_counts,
                    hole=0.4
                )
            ])
            entity_fig.update_layout(
                title="Distribution des entités par type",
                height=400
            )
            
            # Create sentiment distribution pie chart
            sentiment_fig = go.Figure(data=[
                go.Pie(
                    labels=['Positif', 'Neutre', 'Négatif'],
                    values=[
                        data['sentiment']['positive_count'], 
                        data['sentiment']['neutral_count'], 
                        data['sentiment']['negative_count']
                    ],
                    hole=0.4,
                    marker_colors=['#2ca02c', '#d3d3d3', '#d62728']
                )
            ])
            sentiment_fig.update_layout(
                title="Distribution du sentiment",
                height=400
            )
            
            # Create newspaper distribution bar chart
            newspapers = list(data['newspaper_distribution'].keys())
            newspaper_counts = list(data['newspaper_distribution'].values())
            
            newspaper_fig = px.bar(
                x=newspapers, 
                y=newspaper_counts,
                title="Distribution par journal",
                labels={"x": "Journal", "y": "Nombre d'articles"}
            )
            newspaper_fig.update_layout(height=400)
            
            # Create top entities tables by type
            entity_tables = []
            
            for entity_type, entities in sorted(data['top_entities'].items()):
                if entities:
                    rows = []
                    for i, (entity, count) in enumerate(sorted(entities.items(), key=lambda x: x[1], reverse=True)[:10]):
                        rows.append(
                            html.Tr([
                                html.Td(i+1),
                                html.Td(entity),
                                html.Td(count)
                            ])
                        )
                    
                    entity_tables.append(
                        dbc.Col([
                            html.H5(f"Top 10 entités de type {entity_type}"),
                            dbc.Table(
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
                        ], width=6)
                    )
            
            # Create article table
            article_rows = []
            for i, article in enumerate(sorted(data['articles'], key=lambda x: x['sentiment'].get('compound', 0), reverse=True)[:10]):
                title = article.get('title', 'Sans titre')
                sentiment = article['sentiment'].get('compound', 0)
                newspaper = article.get('newspaper', 'Inconnu')
                date = article.get('date', 'Inconnue')
                
                article_rows.append(
                    html.Tr([
                        html.Td(i+1),
                        html.Td(title),
                        html.Td(f"{sentiment:.4f}"),
                        html.Td(newspaper),
                        html.Td(date),
                        html.Td(sum(article.get('entity_counts', {}).values()))
                    ])
                )
            
            article_table = dbc.Table(
                [
                    html.Thead(
                        html.Tr([
                            html.Th("#"),
                            html.Th("Titre"),
                            html.Th("Sentiment"),
                            html.Th("Journal"),
                            html.Th("Date"),
                            html.Th("Entités")
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
            
            # Create tab for this cluster
            cluster_tabs.append(
                dbc.Tab(
                    label=f"Cluster {cluster_id}",
                    children=[
                        html.H4(f"Cluster {cluster_id} - {data['num_articles']} articles", className="mt-3"),
                        html.P(f"Plage de dates: {data['date_range'][0]} à {data['date_range'][1]}"),
                        html.P(f"Longueur moyenne des articles: {data['avg_article_length']:.0f} mots"),
                        
                        # Charts
                        dbc.Row([
                            dbc.Col(dcc.Graph(figure=sentiment_fig), width=6),
                            dbc.Col(dcc.Graph(figure=entity_fig), width=6)
                        ]),
                        
                        dbc.Row([
                            dbc.Col(dcc.Graph(figure=newspaper_fig), width=12)
                        ]),
                        
                        # Entity tables
                        html.H4("Top entités par type", className="mt-4"),
                        dbc.Row(entity_tables),
                        
                        # Article table
                        html.H4("Top 10 articles par sentiment positif", className="mt-4"),
                        article_table
                    ]
                )
            )
        
        # Add cluster tabs
        if cluster_tabs:
            visualizations.append(html.H4("Analyse détaillée par cluster", className="mt-4"))
            visualizations.append(dbc.Tabs(cluster_tabs))
        
        return visualizations
    
    except Exception as e:
        return [html.Div([
            html.H4("Erreur lors du chargement des résultats"),
            html.P(f"Erreur: {str(e)}")
        ])]

# Callback registration (to be called from app.py)
def register_integrated_analysis_callbacks(app):
    # Callback to run integrated analysis
    @app.callback(
        Output("integrated-run-output", "children"),
        Input("run-integrated-button", "n_clicks"),
        [State(f"integrated-{arg['name']}-input", "value") for arg in get_integrated_analysis_args()],
        prevent_initial_call=True
    )
    def run_integrated_analysis(n_clicks, *args):
        if n_clicks is None:
            return ""
        
        # Get argument names
        arg_names = [arg['name'] for arg in get_integrated_analysis_args()]
        
        # Build command
        cmd = ["python", "-m", "src.scripts.run_integrated_analysis"]
        
        # Add arguments
        for name, value in zip(arg_names, args):
            if value is None or value == "":
                continue
            
            cmd.append(f"--{name}")
            cmd.append(str(value))
        
        # Run the command
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            return html.Div([
                html.H5("Analyse intégrée terminée avec succès"),
                html.P("Sortie de la commande:"),
                html.Pre(result.stdout, style={"max-height": "300px", "overflow": "auto"}),
                dbc.Alert("Analyse terminée. Allez à l'onglet 'Résultats' pour visualiser les résultats.", color="success")
            ])
        except subprocess.CalledProcessError as e:
            return html.Div([
                html.H5("Erreur lors de l'analyse intégrée"),
                html.P("Erreur:"),
                html.Pre(e.stderr, style={"max-height": "300px", "overflow": "auto"}),
                dbc.Alert("Erreur lors de l'analyse. Vérifiez les paramètres et réessayez.", color="danger")
            ])
    
    # Callback to update results dropdown
    @app.callback(
        Output("integrated-results-dropdown", "options"),
        Input("integrated-run-output", "children")
    )
    def update_results_dropdown(run_output):
        return get_integrated_results()
    
    # Callback to display results
    @app.callback(
        Output("integrated-results-container", "children"),
        Input("integrated-results-dropdown", "value")
    )
    def display_results(file_path):
        if not file_path:
            return html.Div("Aucun fichier de résultats sélectionné.")
        
        return create_integrated_visualizations(file_path)

# To be called in app.py: from src.webapp.integrated_analysis_viz import register_integrated_analysis_callbacks
