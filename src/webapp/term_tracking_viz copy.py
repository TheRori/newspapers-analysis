"""
Term Tracking Visualization Page for Dash app
"""

from dash import html, dcc, Input, Output, State, ctx, dash_table, ALL
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
from typing import Dict, List, Any, Optional

# Add the project root to the path to allow imports from other modules
project_root = pathlib.Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from src.utils.config_loader import load_config
from src.analysis.term_tracking import count_term_occurrences, count_terms_by_year, count_terms_by_newspaper
from src.webapp.export_component import create_export_button, create_export_modal, create_feedback_toast, register_export_callbacks

# Extract parser arguments from run_term_tracking.py
def get_term_tracking_args():
    spec = importlib.util.spec_from_file_location("run_term_tracking", os.path.join(os.path.dirname(__file__), "..", "scripts", "run_term_tracking.py"))
    run_term_tracking = importlib.util.module_from_spec(spec)
    sys.modules["run_term_tracking"] = run_term_tracking
    spec.loader.exec_module(run_term_tracking)
    parser = run_term_tracking.argparse.ArgumentParser(description="Analyse de suivi des termes dans un corpus d'articles")
    run_term_tracking.main.__globals__["parser"] = parser
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

# Helper to get available term files
def get_term_files():
    project_root = pathlib.Path(__file__).resolve().parents[2]
    
    # Look for term files in examples directory and data/processed
    term_files = []
    
    # Check examples directory
    examples_dir = project_root / 'examples'
    if examples_dir.exists():
        print(f"Searching for JSON files in {examples_dir}")
        example_files = list(examples_dir.glob('*.json'))
        print(f"Found {len(example_files)} JSON files in examples directory: {[f.name for f in example_files]}")
        term_files.extend(example_files)
    else:
        print(f"Examples directory not found: {examples_dir}")
    
    # Check processed directory
    config_path = project_root / 'config' / 'config.yaml'
    with open(config_path, encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    processed_dir = project_root / config['data']['processed_dir']
    if processed_dir.exists():
        print(f"Searching for JSON files in {processed_dir}")
        processed_files = list(processed_dir.glob('*.json'))
        print(f"Found {len(processed_files)} JSON files in processed directory: {[f.name for f in processed_files]}")
        term_files.extend(processed_files)
    else:
        print(f"Processed directory not found: {processed_dir}")
    
    # If no term files found, create a default example
    if not term_files and examples_dir.exists():
        print("No JSON files found, creating a default example")
        default_terms = {
            "exemple": ["terme1", "terme2", "terme3"]
        }
        default_path = examples_dir / "default_terms.json"
        with open(default_path, 'w', encoding='utf-8') as f:
            json.dump(default_terms, f, ensure_ascii=False, indent=2)
        term_files.append(default_path)
    
    # Sort by modification time (newest first)
    term_files.sort(key=lambda x: os.path.getmtime(x), reverse=True)
    
    # Format for dropdown
    options = [
        {'label': f"{f.stem} ({pd.to_datetime(os.path.getmtime(f), unit='s').strftime('%Y-%m-%d %H:%M')})", 
         'value': str(f)}
        for f in term_files
    ]
    
    print(f"Final term file options: {options}")
    return options

# Helper to get available term tracking result files
def get_term_tracking_results():
    project_root = pathlib.Path(__file__).resolve().parents[2]
    config_path = project_root / 'config' / 'config.yaml'
    with open(config_path, encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    results_dir = project_root / config['data']['results_dir'] / 'term_tracking'
    
    if not results_dir.exists():
        return []
    
    # Get all term tracking result files
    result_files = list(results_dir.glob('*.csv'))
    
    # Sort by modification time (newest first)
    result_files.sort(key=lambda x: os.path.getmtime(x), reverse=True)
    
    # Format for dropdown
    options = [
        {'label': f"{f.stem} ({pd.to_datetime(os.path.getmtime(f), unit='s').strftime('%Y-%m-%d %H:%M')})", 
         'value': str(f)}
        for f in result_files
    ]
    
    return options

# Helper to get available semantic drift result files
def get_semantic_drift_results():
    project_root = pathlib.Path(__file__).resolve().parents[2]
    config_path = project_root / 'config' / 'config.yaml'
    with open(config_path, encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    results_dir = project_root / config['data']['results_dir'] / 'term_tracking'
    
    if not results_dir.exists():
        return []
    
    # Get all semantic drift result files
    result_files = list(results_dir.glob('*semantic_drift*.csv'))
    
    # Sort by modification time (newest first)
    result_files.sort(key=lambda x: os.path.getmtime(x), reverse=True)
    
    # Format for dropdown
    options = [
        {'label': f"{f.stem} ({pd.to_datetime(os.path.getmtime(f), unit='s').strftime('%Y-%m-%d %H:%M')})", 
         'value': str(f)}
        for f in result_files
    ]
    
    return options

# Helper to get available similar terms result files
def get_similar_terms_results():
    """
    Get available similar terms result files.
    
    Returns:
        List of dictionaries with label and value for each result file
    """
    try:
        project_root = pathlib.Path(__file__).resolve().parents[2]
        config_path = project_root / 'config' / 'config.yaml'
        with open(config_path, encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        results_dir = project_root / config['data']['results_dir'] / 'term_tracking'
        
        print(f"Recherche de fichiers de termes similaires dans: {results_dir}")
        
        if not results_dir.exists():
            print(f"Le répertoire {results_dir} n'existe pas")
            return []
        
        # Get all CSV files in the directory
        all_csv_files = list(results_dir.glob('*.csv'))
        print(f"Tous les fichiers CSV trouvés: {[f.name for f in all_csv_files]}")
        
        # Get all similar terms result files
        result_files = list(results_dir.glob('*similar_terms*.csv'))
        print(f"Fichiers de termes similaires trouvés: {[f.name for f in result_files]}")
        
        # Si aucun fichier n'est trouvé avec le motif, vérifier si le fichier spécifique existe
        if not result_files:
            specific_file = results_dir / 'similar_terms_term_tracking_results.csv'
            if specific_file.exists():
                print(f"Fichier spécifique trouvé: {specific_file.name}")
                result_files = [specific_file]
        
        # Sort by modification time (newest first)
        result_files.sort(key=lambda x: os.path.getmtime(x), reverse=True)
        
        # Format for dropdown
        options = [
            {'label': f"{f.stem} ({pd.to_datetime(os.path.getmtime(f), unit='s').strftime('%Y-%m-%d %H:%M')})", 
             'value': str(f)}
            for f in result_files
        ]
        
        print(f"Options de dropdown générées: {options}")
        
        return options
    
    except Exception as e:
        print(f"Error getting similar terms results: {e}")
        return []

# Layout for the term tracking page
def get_term_tracking_layout():
    # Get available term files and result files
    term_files = get_term_files()
    term_tracking_results = get_term_tracking_results()
    semantic_drift_results = get_semantic_drift_results()
    similar_terms_results = get_similar_terms_results()
    
    # Get parser arguments for the run form
    parser_args = get_term_tracking_args()
    
    # Create the layout
    layout = html.Div([
        html.H2("Suivi des Termes"),
        html.P("Analysez la présence et l'évolution de termes spécifiques dans le corpus."),
        
        dbc.Tabs([
            # Tab for running analysis
            dbc.Tab(label="Lancer une analyse", children=[
                html.Div([
                    html.H4("Paramètres d'analyse"),
                    html.P("Configurez les paramètres pour l'analyse de suivi des termes."),
                    
                    # Afficher explicitement le sélecteur de fichier de termes en premier
                    html.Div([
                        html.H5("Fichier de termes"),
                        dbc.Row([
                            dbc.Col([
                                dcc.Dropdown(
                                    id="term-tracking-term-file-input",
                                    options=term_files,
                                    value=term_files[0]['value'] if term_files else None,
                                    placeholder="Sélectionnez un fichier de termes"
                                )
                            ], width=12),
                        ], className="mb-3"),
                        
                        html.H5("Type d'analyse"),
                        dbc.Row([
                            dbc.Col([
                                dbc.Checkbox(
                                    id="term-tracking-by-year-input",
                                    label="Agréger par année",
                                    value=False
                                )
                            ], width=4),
                            dbc.Col([
                                dbc.Checkbox(
                                    id="term-tracking-by-newspaper-input",
                                    label="Agréger par journal",
                                    value=False
                                )
                            ], width=4),
                            dbc.Col([
                                dbc.Checkbox(
                                    id="term-tracking-semantic-drift-input",
                                    label="Analyse de drift sémantique",
                                    value=False
                                )
                            ], width=4),
                        ], className="mb-3"),
                        
                        # Options pour l'analyse de drift sémantique (conditionnelles)
                        html.Div([
                            html.H5("Options d'analyse sémantique", className="mt-3"),
                            dbc.Row([
                                dbc.Col([
                                    dbc.Label("Type de période"),
                                    dbc.RadioItems(
                                        id="term-tracking-period-type-input",
                                        options=[
                                            {"label": "Année", "value": "year"},
                                            {"label": "Décennie", "value": "decade"},
                                            {"label": "Personnalisé", "value": "custom"}
                                        ],
                                        value="decade",
                                        inline=True
                                    )
                                ], width=6),
                                dbc.Col([
                                    dbc.Label("Périodes personnalisées (format JSON)"),
                                    dbc.Textarea(
                                        id="term-tracking-custom-periods-input",
                                        placeholder="[[1800, 1850], [1851, 1900], [1901, 1950], [1951, 2000]]",
                                        rows=2
                                    )
                                ], width=6),
                            ], className="mb-3"),
                            dbc.Row([
                                dbc.Col([
                                    dbc.Label("Taille des vecteurs"),
                                    dbc.Input(
                                        id="term-tracking-vector-size-input",
                                        type="number",
                                        value=100
                                    )
                                ], width=4),
                                dbc.Col([
                                    dbc.Label("Taille de fenêtre"),
                                    dbc.Input(
                                        id="term-tracking-window-input",
                                        type="number",
                                        value=5
                                    )
                                ], width=4),
                                dbc.Col([
                                    dbc.Label("Nombre min. d'occurrences"),
                                    dbc.Input(
                                        id="term-tracking-min-count-input",
                                        type="number",
                                        value=5
                                    )
                                ], width=4),
                            ], className="mb-3"),
                        ], id="semantic-drift-options", style={"display": "none"}),
                        
                        html.H5("Autres options", className="mt-3"),
                        dbc.Row([
                            dbc.Col([
                                dbc.Label("Nom du fichier de sortie"),
                                dbc.Input(
                                    id="term-tracking-output-input",
                                    type="text",
                                    placeholder="Nom du fichier de sortie",
                                    value="term_tracking_results.csv"
                                )
                            ], width=6),
                            dbc.Col([
                                dbc.Label("Limite d'articles (0 = pas de limite)"),
                                dbc.Input(
                                    id="term-tracking-limit-input",
                                    type="number",
                                    placeholder="Limite d'articles",
                                    value=0
                                )
                            ], width=6),
                        ], className="mb-3"),
                        
                        dbc.Row([
                            dbc.Col([
                                dbc.Button(
                                    "Lancer l'analyse",
                                    id="run-term-tracking-button",
                                    color="primary",
                                    className="mt-3"
                                ),
                                html.Div(id="term-tracking-run-output", className="mt-3")
                            ], width=12)
                        ])
                    ])
                ])
            ]),
            
            # Tab for viewing results
            dbc.Tab(label="Visualiser les résultats", children=[
                html.Div([
                    html.H4("Résultats de suivi des termes"),
                    html.P("Visualisez les résultats d'analyses précédentes."),
                    
                    dbc.Row([
                        dbc.Col([
                            dbc.Label("Fichier de résultats"),
                            dcc.Dropdown(
                                id="term-tracking-results-file",
                                options=term_tracking_results,
                                value=term_tracking_results[0]['value'] if term_tracking_results else None,
                                placeholder="Sélectionnez un fichier de résultats"
                            )
                        ], width=8),
                        dbc.Col([
                            dbc.Label("Type de visualisation"),
                            dcc.Dropdown(
                                id="term-tracking-viz-type",
                                options=[
                                    {"label": "Graphique à barres", "value": "bar"},
                                    {"label": "Graphique linéaire", "value": "line"},
                                    {"label": "Carte de chaleur", "value": "heatmap"},
                                    {"label": "Tableau", "value": "table"},
                                    {"label": "Radar", "value": "radar"},
                                    {"label": "Calendrier", "value": "calendar"}
                                ],
                                value="bar"
                            )
                        ], width=4),
                    ], className="mb-3"),
                    
                    html.Div(id="term-tracking-visualizations")
                ])
            ]),
            
            # Tab for semantic drift results
            dbc.Tab(label="Drift Sémantique", children=[
                html.Div([
                    html.H4("Analyse de Drift Sémantique"),
                    html.P("Visualisez l'évolution sémantique des termes dans le temps."),
                    
                    dbc.Row([
                        dbc.Col([
                            dbc.Label("Fichier de résultats"),
                            dcc.Dropdown(
                                id="semantic-drift-results-file",
                                options=semantic_drift_results,
                                value=semantic_drift_results[0]['value'] if semantic_drift_results else None,
                                placeholder="Sélectionnez un fichier de résultats"
                            )
                        ], width=8),
                        dbc.Col([
                            dbc.Label("Type de visualisation"),
                            dcc.Dropdown(
                                id="semantic-drift-viz-type",
                                options=[
                                    {"label": "Évolution temporelle", "value": "line"},
                                    {"label": "Carte de chaleur", "value": "heatmap"},
                                    {"label": "Tableau", "value": "table"},
                                    {"label": "Comparaison termes", "value": "comparison"}
                                ],
                                value="line"
                            )
                        ], width=4),
                    ], className="mb-3"),
                    
                    html.Div(id="semantic-drift-visualizations")
                ])
            ]),
            
            # Tab for similar terms results
            dbc.Tab(label="Termes Similaires", children=[
                html.Div([
                    html.H4("Analyse de Termes Similaires"),
                    html.P("Visualisez les termes similaires dans le corpus."),
                    
                    dbc.Row([
                        dbc.Col([
                            dbc.Label("Fichier de résultats"),
                            dcc.Dropdown(
                                id="similar-terms-results-file",
                                options=similar_terms_results,
                                value=similar_terms_results[0]['value'] if similar_terms_results else None,
                                placeholder="Sélectionnez un fichier de résultats"
                            )
                        ], width=8),
                        dbc.Col([
                            dbc.Label("Type de visualisation"),
                            dcc.Dropdown(
                                id="similar-terms-viz-type",
                                options=[
                                    {"label": "Tableau", "value": "table"},
                                    {"label": "Carte de chaleur", "value": "heatmap"},
                                    {"label": "Réseau", "value": "network"}
                                ],
                                value="table"
                            )
                        ], width=4),
                    ], className="mb-3"),
                    
                    html.Div(id="similar-terms-visualizations")
                ])
            ]),
        ]),
        
        # Add modals for displaying articles
        create_articles_modal(),
        create_full_article_modal(),
        
        # Add export components
        create_export_modal(analysis_type="term_tracking"),
        create_feedback_toast()
    ])
    
    return layout

# Function to create term tracking visualizations
def create_term_tracking_visualizations(results_file, viz_type="bar"):
    """
    Create visualizations for term tracking results.
    
    Args:
        results_file: Path to the results file
        viz_type: Type of visualization (bar, line, heatmap, table)
        
    Returns:
        HTML div with visualizations
    """
    if not results_file or not os.path.exists(results_file):
        return html.Div([
            html.P("Aucun fichier de résultats sélectionné ou fichier introuvable.", className="text-danger")
        ])
    
    try:
        # Charger les données
        df = pd.read_csv(results_file)
        
        # Vérifier si le fichier est vide
        if df.empty:
            return html.Div([
                html.P("Le fichier de résultats est vide.", className="text-danger")
            ])
        
        # Déterminer le type de résultats
        key_column = df.columns[0]
        
        # Rename the key column based on the type of results
        if key_column == 'key':
            # Try to determine the type of key
            first_key = df['key'].iloc[0]
            if isinstance(first_key, str) and len(first_key) > 20:
                # Likely an article ID
                key_type = "Article ID"
                df = df.rename(columns={'key': 'Article ID'})
            elif isinstance(first_key, (int, float)) or (isinstance(first_key, str) and first_key.isdigit()):
                # Likely a year
                key_type = "Année"
                df = df.rename(columns={'key': 'Année'})
            else:
                # Likely a newspaper
                key_type = "Journal"
                df = df.rename(columns={'key': 'Journal'})
        else:
            key_type = key_column
        
        # Get term columns (all columns except the key column)
        term_columns = df.columns[1:].tolist()
        
        # Si nous avons des IDs d'articles, extraire la date et le journal
        if key_type == "Article ID" or (df.columns[0].startswith('article_')):
            # Renommer la première colonne si nécessaire
            if df.columns[0].startswith('article_'):
                df = df.rename(columns={df.columns[0]: 'Article ID'})
                key_type = "Article ID"
            
            # Extraire la date et le journal des IDs d'articles
            # Format attendu: article_YYYY-MM-DD_journal_XXXX_source
            df['Date'] = df['Article ID'].str.extract(r'article_(\d{4}-\d{2}-\d{2})_')
            df['Journal'] = df['Article ID'].str.extract(r'article_\d{4}-\d{2}-\d{2}_([^_]+)_')
            
            # Convertir la date en datetime
            df['Date'] = pd.to_datetime(df['Date'])
            df['Année'] = df['Date'].dt.year
            df['Mois'] = df['Date'].dt.month
            df['Jour'] = df['Date'].dt.day
            
            # Créer une colonne pour l'année-mois (format YYYY-MM)
            df['Année-Mois'] = df['Date'].dt.strftime('%Y-%m')
            
            # Agréger par période
            df_by_year = df.groupby('Année')[term_columns].sum().reset_index()
            df_by_month = df.groupby('Année-Mois')[term_columns].sum().reset_index()
            df_by_journal = df.groupby('Journal')[term_columns].sum().reset_index()
            
            # Calculer des statistiques supplémentaires
            total_occurrences = df[term_columns].sum().sum()
            occurrences_by_term = df[term_columns].sum().reset_index()
            occurrences_by_term.columns = ['Terme', 'Occurrences']
            occurrences_by_term = occurrences_by_term.sort_values('Occurrences', ascending=False)
            
            # Créer des visualisations basées sur le type demandé
            if viz_type == "bar":
                # Graphique à barres par année
                fig_year = px.bar(
                    df_by_year, 
                    x='Année', 
                    y=term_columns,
                    title=f"Fréquence des termes par année",
                    labels={'value': 'Fréquence', 'variable': 'Terme'},
                    barmode='group'
                )
                year_graph = dcc.Graph(
                    id={'type': 'term-tracking-graph', 'subtype': 'year-bar'},
                    figure=fig_year
                )
                
                # Graphique à barres par journal
                fig_journal = px.bar(
                    df_by_journal, 
                    x='Journal', 
                    y=term_columns,
                    title=f"Fréquence des termes par journal",
                    labels={'value': 'Fréquence', 'variable': 'Terme'},
                    barmode='group'
                )
                journal_graph = dcc.Graph(
                    id={'type': 'term-tracking-graph', 'subtype': 'journal-bar'},
                    figure=fig_journal
                )
                
                # Top 20 articles
                top_df = df.copy()
                top_df['Total'] = top_df[term_columns].sum(axis=1)
                top_df = top_df.nlargest(20, 'Total')
                top_df = top_df.drop(columns=['Total'])
                
                fig_top = px.bar(
                    top_df.melt(id_vars=['Article ID'], value_vars=term_columns),
                    x='Article ID',
                    y='value',
                    color='variable',
                    title=f"Top 20 articles par fréquence totale de termes",
                    labels={'value': 'Fréquence', 'variable': 'Terme'}
                )
                top_graph = dcc.Graph(
                    id={'type': 'term-tracking-graph', 'subtype': 'article-bar'},
                    figure=fig_top
                )
                
                # Répartition des termes (camembert)
                fig_pie = px.pie(
                    occurrences_by_term,
                    values='Occurrences',
                    names='Terme',
                    title="Répartition des occurrences par terme"
                )
                pie_graph = dcc.Graph(
                    id={'type': 'term-tracking-graph', 'subtype': 'term-pie'},
                    figure=fig_pie
                )
                
                # Tableau récapitulatif
                summary_table = html.Div([
                    html.H4("Statistiques globales"),
                    html.P(f"Nombre total d'occurrences: {total_occurrences}"),
                    html.P(f"Nombre d'articles avec au moins un terme: {len(df)}"),
                    html.P(f"Nombre de journaux: {df['Journal'].nunique()}"),
                    html.P(f"Période couverte: {df['Date'].min().strftime('%d/%m/%Y')} - {df['Date'].max().strftime('%d/%m/%Y')}"),
                    
                    html.H5("Occurrences par terme"),
                    dash_table.DataTable(
                        data=occurrences_by_term.to_dict('records'),
                        columns=[{'name': col, 'id': col} for col in occurrences_by_term.columns],
                        style_table={'overflowX': 'auto'},
                        style_cell={'textAlign': 'left'},
                        style_header={'backgroundColor': 'rgb(230, 230, 230)', 'fontWeight': 'bold'}
                    )
                ])
                
                return html.Div([year_graph, journal_graph, top_graph, pie_graph, summary_table])
                
            elif viz_type == "line":
                # Évolution par année
                fig_year = px.line(
                    df_by_year, 
                    x='Année', 
                    y=term_columns,
                    title=f"Évolution des termes par année",
                    labels={'value': 'Fréquence', 'variable': 'Terme'},
                    markers=True
                )
                year_graph = dcc.Graph(
                    id={'type': 'term-tracking-graph', 'subtype': 'year-line'},
                    figure=fig_year
                )
                
                # Évolution par mois
                fig_month = px.line(
                    df_by_month, 
                    x='Année-Mois', 
                    y=term_columns,
                    title=f"Évolution des termes par mois",
                    labels={'value': 'Fréquence', 'variable': 'Terme'},
                    markers=True
                )
                fig_month.update_xaxes(tickangle=45)
                month_graph = dcc.Graph(
                    id={'type': 'term-tracking-graph', 'subtype': 'month-line'},
                    figure=fig_month
                )
                
                # Comparaison entre journaux (radar chart)
                radar_data = []
                for term in term_columns:
                    radar_data.append(go.Scatterpolar(
                        r=df_by_journal[term],
                        theta=df_by_journal['Journal'],
                        fill='toself',
                        name=term
                    ))
                
                radar_layout = go.Layout(
                    polar=dict(
                        radialaxis=dict(
                            visible=True,
                        )
                    ),
                    showlegend=True,
                    title="Comparaison des termes entre journaux"
                )
                
                radar_fig = go.Figure(data=radar_data, layout=radar_layout)
                radar_graph = dcc.Graph(
                    id={'type': 'term-tracking-graph', 'subtype': 'journal-radar'},
                    figure=radar_fig
                )
                
                # Tableau récapitulatif
                summary_table = html.Div([
                    html.H4("Statistiques globales"),
                    html.P(f"Nombre total d'occurrences: {total_occurrences}"),
                    html.P(f"Nombre d'articles avec au moins un terme: {len(df)}"),
                    html.P(f"Nombre de journaux: {df['Journal'].nunique()}"),
                    html.P(f"Période couverte: {df['Date'].min().strftime('%d/%m/%Y')} - {df['Date'].max().strftime('%d/%m/%Y')}"),
                    
                    html.H5("Occurrences par terme"),
                    dash_table.DataTable(
                        data=occurrences_by_term.to_dict('records'),
                        columns=[{'name': col, 'id': col} for col in occurrences_by_term.columns],
                        style_table={'overflowX': 'auto'},
                        style_cell={'textAlign': 'left'},
                        style_header={'backgroundColor': 'rgb(230, 230, 230)', 'fontWeight': 'bold'}
                    )
                ])
                
                return html.Div([year_graph, month_graph, radar_graph, summary_table])
                
            elif viz_type == "heatmap":
                # Heatmap par année et terme
                heatmap_year = px.imshow(
                    df_by_year.set_index('Année')[term_columns].T,
                    labels=dict(x="Année", y="Terme", color="Fréquence"),
                    title="Heatmap des termes par année",
                    color_continuous_scale="Viridis"
                )
                year_heatmap = dcc.Graph(
                    id={'type': 'term-tracking-graph', 'subtype': 'year-heatmap'},
                    figure=heatmap_year
                )
                
                # Heatmap par journal et terme
                heatmap_journal = px.imshow(
                    df_by_journal.set_index('Journal')[term_columns].T,
                    labels=dict(x="Journal", y="Terme", color="Fréquence"),
                    title="Heatmap des termes par journal",
                    color_continuous_scale="Viridis"
                )
                journal_heatmap = dcc.Graph(
                    id={'type': 'term-tracking-graph', 'subtype': 'journal-heatmap'},
                    figure=heatmap_journal
                )
                
                # Heatmap par mois (calendrier)
                # Créer un pivot pour avoir les années en colonnes et les mois en lignes
                df['Mois_Nom'] = df['Date'].dt.month_name()
                pivot_month = df.groupby(['Mois_Nom', 'Année'])[term_columns].sum().sum(axis=1).unstack()
                
                # Réordonner les mois
                month_order = ['January', 'February', 'March', 'April', 'May', 'June', 
                              'July', 'August', 'September', 'October', 'November', 'December']
                pivot_month = pivot_month.reindex(month_order)
                
                heatmap_calendar = px.imshow(
                    pivot_month,
                    labels=dict(x="Année", y="Mois", color="Fréquence"),
                    title="Calendrier des occurrences (tous termes confondus)",
                    color_continuous_scale="Viridis"
                )
                calendar_heatmap = dcc.Graph(
                    id={'type': 'term-tracking-graph', 'subtype': 'calendar-heatmap'},
                    figure=heatmap_calendar
                )
                
                # Tableau récapitulatif
                summary_table = html.Div([
                    html.H4("Statistiques globales"),
                    html.P(f"Nombre total d'occurrences: {total_occurrences}"),
                    html.P(f"Nombre d'articles avec au moins un terme: {len(df)}"),
                    html.P(f"Nombre de journaux: {df['Journal'].nunique()}"),
                    html.P(f"Période couverte: {df['Date'].min().strftime('%d/%m/%Y')} - {df['Date'].max().strftime('%d/%m/%Y')}"),
                    
                    html.H5("Occurrences par terme"),
                    dash_table.DataTable(
                        data=occurrences_by_term.to_dict('records'),
                        columns=[{'name': col, 'id': col} for col in occurrences_by_term.columns],
                        style_table={'overflowX': 'auto'},
                        style_cell={'textAlign': 'left'},
                        style_header={'backgroundColor': 'rgb(230, 230, 230)', 'fontWeight': 'bold'}
                    )
                ])
                
                return html.Div([year_heatmap, journal_heatmap, calendar_heatmap, summary_table])
                
            elif viz_type == "table":
                # Créer plusieurs tableaux
                tables = []
                
                # 1. Tableau des résultats par article
                tables.append(html.Div([
                    html.H4("Résultats par article"),
                    dash_table.DataTable(
                        id={'type': 'term-tracking-table', 'subtype': 'article-table'},
                        data=df.to_dict('records'),
                        columns=[{'name': col, 'id': col} for col in df.columns],
                        filter_action="native",
                        sort_action="native",
                        sort_mode="multi",
                        page_size=15,
                        style_table={'overflowX': 'auto'},
                        style_cell={'textAlign': 'left'},
                        style_header={'backgroundColor': 'rgb(230, 230, 230)', 'fontWeight': 'bold'}
                    )
                ]))
                
                # 2. Tableau des résultats par année
                tables.append(html.Div([
                    html.H4("Résultats par année"),
                    dash_table.DataTable(
                        id={'type': 'term-tracking-table', 'subtype': 'year-table'},
                        data=df_by_year.to_dict('records'),
                        columns=[{'name': col, 'id': col} for col in df_by_year.columns],
                        filter_action="native",
                        sort_action="native",
                        sort_mode="multi",
                        page_size=15,
                        style_table={'overflowX': 'auto'},
                        style_cell={'textAlign': 'left'},
                        style_header={'backgroundColor': 'rgb(230, 230, 230)', 'fontWeight': 'bold'}
                    )
                ]))
                
                # 3. Tableau des résultats par journal
                tables.append(html.Div([
                    html.H4("Résultats par journal"),
                    dash_table.DataTable(
                        id={'type': 'term-tracking-table', 'subtype': 'journal-table'},
                        data=df_by_journal.to_dict('records'),
                        columns=[{'name': col, 'id': col} for col in df_by_journal.columns],
                        filter_action="native",
                        sort_action="native",
                        sort_mode="multi",
                        page_size=15,
                        style_table={'overflowX': 'auto'},
                        style_cell={'textAlign': 'left'},
                        style_header={'backgroundColor': 'rgb(230, 230, 230)', 'fontWeight': 'bold'}
                    )
                ]))
                
                # 4. Tableau des résultats par mois
                tables.append(html.Div([
                    html.H4("Résultats par mois"),
                    dash_table.DataTable(
                        id={'type': 'term-tracking-table', 'subtype': 'month-table'},
                        data=df_by_month.to_dict('records'),
                        columns=[{'name': col, 'id': col} for col in df_by_month.columns],
                        filter_action="native",
                        sort_action="native",
                        sort_mode="multi",
                        page_size=15,
                        style_table={'overflowX': 'auto'},
                        style_cell={'textAlign': 'left'},
                        style_header={'backgroundColor': 'rgb(230, 230, 230)', 'fontWeight': 'bold'}
                    )
                ]))
                
                return html.Div(tables)
        
        # Cas standard (données déjà agrégées par année ou journal)
        else:
            # Create visualizations based on the type
            if viz_type == "bar":
                # Bar chart
                if key_type == "Année":
                    # For years, create a grouped bar chart
                    fig = px.bar(
                        df, 
                        x='Année', 
                        y=term_columns,
                        title=f"Fréquence des termes par année",
                        labels={'value': 'Fréquence', 'variable': 'Terme'},
                        barmode='group'
                    )
                    graph = dcc.Graph(
                        id={'type': 'term-tracking-graph', 'subtype': 'year-bar'},
                        figure=fig
                    )
                    
                elif key_type == "Journal":
                    # For newspapers, create a grouped bar chart
                    fig = px.bar(
                        df, 
                        x='Journal', 
                        y=term_columns,
                        title=f"Fréquence des termes par journal",
                        labels={'value': 'Fréquence', 'variable': 'Terme'},
                        barmode='group'
                    )
                    graph = dcc.Graph(
                        id={'type': 'term-tracking-graph', 'subtype': 'journal-bar'},
                        figure=fig
                    )
                    
                else:
                    # For articles, create a stacked bar chart of the top 20 articles
                    top_df = df.copy()
                    top_df['Total'] = top_df[term_columns].sum(axis=1)
                    top_df = top_df.nlargest(20, 'Total')
                    top_df = top_df.drop(columns=['Total'])
                    
                    fig = px.bar(
                        top_df.melt(id_vars=[key_type], value_vars=term_columns),
                        x=key_type,
                        y='value',
                        color='variable',
                        title=f"Top 20 articles par fréquence totale de termes",
                        labels={'value': 'Fréquence', 'variable': 'Terme'}
                    )
                    graph = dcc.Graph(
                        id={'type': 'term-tracking-graph', 'subtype': 'article-bar'},
                        figure=fig
                    )
                
                # Ajouter des informations supplémentaires
                info_section = html.Div([
                    html.Hr(),
                    html.P("Cliquez sur les éléments du graphique pour voir les articles correspondants.")
                ])
                
                return html.Div([graph, info_section])
                
            elif viz_type == "line":
                # Line chart (only makes sense for time series data)
                if key_type == "Année":
                    fig = px.line(
                        df, 
                        x='Année', 
                        y=term_columns,
                        title=f"Évolution des termes au fil du temps",
                        labels={'value': 'Fréquence', 'variable': 'Terme'},
                        markers=True
                    )
                    graph = dcc.Graph(
                        id={'type': 'term-tracking-graph', 'subtype': 'year-line'},
                        figure=fig
                    )
                    
                    # Ajouter des informations supplémentaires
                    info_section = html.Div([
                        html.Hr(),
                        html.P("Cliquez sur les points du graphique pour voir les articles correspondants.")
                    ])
                    
                    return html.Div([graph, info_section])
                    
                else:
                    return html.Div([
                        html.P("Le graphique en ligne n'est disponible que pour les données temporelles (par année).", className="text-warning")
                    ])
                    
            elif viz_type == "heatmap":
                # Heatmap
                if key_type in ["Année", "Journal"]:
                    # Transpose the data to have terms as rows and years/newspapers as columns
                    heatmap_df = df.set_index(df.columns[0])
                    
                    fig = px.imshow(
                        heatmap_df.T,
                        labels=dict(x=key_type, y="Terme", color="Fréquence"),
                        title="Heatmap des termes",
                        color_continuous_scale="Viridis"
                    )
                    graph = dcc.Graph(
                        id={'type': 'term-tracking-graph', 'subtype': f"{key_type.lower()}-heatmap"},
                        figure=fig
                    )
                    
                else:
                    # For articles, create a heatmap of the top 20 articles
                    top_df = df.copy()
                    top_df['Total'] = top_df[term_columns].sum(axis=1)
                    top_df = top_df.nlargest(20, 'Total')
                    top_df = top_df.drop(columns=['Total'])
                    
                    # Transpose for the heatmap
                    heatmap_df = top_df.set_index(key_type)[term_columns]
                    
                    fig = px.imshow(
                        heatmap_df,
                        title="Heatmap des termes pour les 20 articles les plus pertinents",
                        labels={'x': "Article", 'y': 'Terme', 'color': 'Fréquence'},
                        color_continuous_scale="Viridis"
                    )
                    graph = dcc.Graph(
                        id={'type': 'term-tracking-graph', 'subtype': 'article-heatmap'},
                        figure=fig
                    )
                
                # Ajouter des informations supplémentaires
                info_section = html.Div([
                    html.Hr(),
                    html.P("Cliquez sur les cellules de la heatmap pour voir les articles correspondants.")
                ])
                
                return html.Div([graph, info_section])
                
            else:
                # Table
                # Create a styled table
                table = dash_table.DataTable(
                    id={'type': 'term-tracking-table', 'subtype': 'main-table'},
                    columns=[{"name": col, "id": col} for col in df.columns],
                    data=df.to_dict('records'),
                    style_table={'overflowX': 'auto'},
                    style_cell={
                        'textAlign': 'left',
                        'padding': '8px',
                        'minWidth': '100px', 'width': '150px', 'maxWidth': '200px',
                        'whiteSpace': 'normal',
                        'height': 'auto',
                    },
                    style_header={
                        'backgroundColor': 'rgb(230, 230, 230)',
                        'fontWeight': 'bold'
                    },
                    style_data_conditional=[
                        {
                            'if': {'row_index': 'odd'},
                            'backgroundColor': 'rgb(248, 248, 248)'
                        }
                    ],
                    filter_action="native",
                    sort_action="native",
                    sort_mode="multi",
                    page_size=20
                )
                
                return html.Div([
                    table,
                    html.Hr(),
                    html.P("Utilisez les contrôles de filtrage et de tri du tableau pour explorer les données.")
                ])
            
    except Exception as e:
        return html.Div([
            html.P(f"Erreur lors de la création des visualisations : {str(e)}", className="text-danger")
        ])

# Function to create similar terms visualizations
def create_similar_terms_visualizations(results_file, viz_type="table"):
    """
    Create visualizations for similar terms analysis results.
    
    Args:
        results_file: Path to the results file
        viz_type: Type of visualization (table, heatmap, network)
        
    Returns:
        HTML div with visualizations
    """
    try:
        # Load results
        df = pd.read_csv(results_file)
        
        if df.empty:
            return html.Div("Aucun résultat à afficher.")
        
        # Check if we have the expected columns
        if not all(col in df.columns for col in ['term', 'period', 'rank', 'similar_word', 'similarity']):
            return html.Div("Format de fichier de résultats non reconnu. Attendu: 'term', 'period', 'rank', 'similar_word', 'similarity'.")
        
        # Get unique terms and periods
        terms = df['term'].unique()
        periods = sorted(df['period'].unique())
        
        # Create visualizations based on the selected type
        if viz_type == "table":
            # Create a table with the results
            table = dash_table.DataTable(
                id="similar-terms-table",
                columns=[{"name": col, "id": col} for col in df.columns],
                data=df.to_dict("records"),
                filter_action="native",
                sort_action="native",
                sort_mode="multi",
                page_size=20,
                style_table={"overflowX": "auto"},
                style_cell={
                    "textAlign": "left",
                    "padding": "10px",
                    "whiteSpace": "normal",
                    "height": "auto"
                },
                style_header={
                    "backgroundColor": "rgb(230, 230, 230)",
                    "fontWeight": "bold"
                },
                style_data_conditional=[
                    {
                        "if": {"row_index": "odd"},
                        "backgroundColor": "rgb(248, 248, 248)"
                    }
                ]
            )
            
            return html.Div([
                html.H5("Tableau des termes similaires"),
                html.P("Ce tableau présente les mots les plus proches vectoriellement pour chaque terme et période. Utilisez les filtres et le tri pour explorer les données."),
                table
            ])
            
        elif viz_type == "heatmap":
            # Create a pivot table for the heatmap
            # We'll use the top 5 similar words for each term and period
            top_words = df[df['rank'] <= 5].copy()
            
            # Create a composite column for the heatmap
            top_words['term_period'] = top_words['term'] + ' (' + top_words['period'] + ')'
            top_words['similar_rank'] = top_words['similar_word'] + ' (#' + top_words['rank'].astype(str) + ')'
            
            # Create pivot table
            pivot_df = top_words.pivot(index="term_period", columns="similar_rank", values="similarity")
            
            # Create heatmap
            fig = px.imshow(
                pivot_df,
                labels=dict(x="Mot similaire (rang)", y="Terme (période)", color="Similarité"),
                x=pivot_df.columns,
                y=pivot_df.index,
                color_continuous_scale="Viridis",
                title="Carte de chaleur des termes similaires"
            )
            
            # Improve layout
            fig.update_layout(
                xaxis_title="Mot similaire (rang)",
                yaxis_title="Terme (période)",
                coloraxis_colorbar=dict(title="Similarité")
            )
            
            return html.Div([
                html.H5("Carte de chaleur des termes similaires"),
                html.P("Cette carte de chaleur montre la similarité entre les termes analysés et leurs mots les plus proches vectoriellement."),
                dcc.Graph(figure=fig)
            ])
            
        elif viz_type == "network":
            # Create a network visualization showing the relationships between terms
            
            # Get unique periods
            periods = sorted(df['period'].unique())
            
            # Function to create network graph for a specific period
            def create_period_network(period):
                # Filter data for the selected period
                period_df = df[df['period'] == period]
                
                # Get unique terms for this period
                period_terms = period_df['term'].unique()
                
                # Filter to top 5 similar words for better visualization
                top_words = period_df[period_df['rank'] <= 5].copy()
                
                # Create network graph
                fig = go.Figure()
                
                # Calculate positions for main terms (in a circle)
                radius = 3
                term_positions = {}
                
                for i, term in enumerate(period_terms):
                    angle = 2 * np.pi * i / len(period_terms)
                    x = radius * np.cos(angle)
                    y = radius * np.sin(angle)
                    term_positions[term] = (x, y)
                    
                    # Add node for main term
                    fig.add_trace(go.Scatter(
                        x=[x],
                        y=[y],
                        mode="markers+text",
                        marker=dict(size=25, color="red"),
                        text=[term],
                        name=term,
                        textposition="middle center",
                        textfont=dict(color="white", size=12),
                        hoverinfo="text",
                        hovertext=f"<b>{term}</b>",
                        showlegend=False
                    ))
                
                # Add nodes and edges for similar words
                for term in period_terms:
                    term_x, term_y = term_positions[term]
                    term_similar = top_words[top_words['term'] == term]
                    
                    for _, row in term_similar.iterrows():
                        # Calculate position (in a circle around the main term)
                        angle = (row['rank'] - 1) * (2 * np.pi / 5)
                        distance = 1.5  # Distance from main term
                        x = term_x + distance * np.cos(angle)
                        y = term_y + distance * np.sin(angle)
                        
                        # Size and opacity based on similarity
                        node_size = 15 + (row['similarity'] * 10)
                        
                        # Add similar word node
                        fig.add_trace(go.Scatter(
                            x=[x],
                            y=[y],
                            mode="markers+text",
                            marker=dict(
                                size=node_size, 
                                color="blue",
                                opacity=0.7 + (row['similarity'] * 0.3)
                            ),
                            text=[row['similar_word']],
                            name=f"{row['similar_word']} ({row['similarity']:.2f})",
                            textposition="bottom center",
                            hoverinfo="text",
                            hovertext=f"<b>{row['similar_word']}</b><br>Similarité: {row['similarity']:.3f}",
                            showlegend=False
                        ))
                        
                        # Add edge with width based on similarity
                        fig.add_trace(go.Scatter(
                            x=[term_x, x],
                            y=[term_y, y],
                            mode="lines",
                            line=dict(
                                width=row['similarity'] * 5, 
                                color="rgba(100, 100, 100, 0.6)"
                            ),
                            hoverinfo="text",
                            hovertext=f"Similarité: {row['similarity']:.3f}",
                            showlegend=False
                        ))
                
                # Improve layout
                fig.update_layout(
                    title=f"Réseau de termes similaires - Période: {period}",
                    xaxis=dict(
                        showgrid=False, 
                        zeroline=False, 
                        showticklabels=False,
                        range=[-5, 5]
                    ),
                    yaxis=dict(
                        showgrid=False, 
                        zeroline=False, 
                        showticklabels=False,
                        range=[-5, 5],
                        scaleanchor="x",
                        scaleratio=1
                    ),
                    hovermode="closest",
                    plot_bgcolor="rgba(240, 240, 240, 0.8)"
                )
                
                return fig
            
            # Create initial graph with the most recent period
            initial_period = periods[-1]
            initial_graph = create_period_network(initial_period)
            
            # Create period selector
            period_selector = html.Div([
                html.Label("Sélectionner une période:"),
                dcc.Dropdown(
                    id="similar-terms-period-selector",
                    options=[{"label": p, "value": p} for p in periods],
                    value=initial_period,
                    clearable=False,
                    style={"width": "100%", "marginBottom": "15px"}
                )
            ])
            
            # Return the layout with the period selector and graph
            return html.Div([
                html.H5("Réseau de termes similaires"),
                html.P("Ce graphique montre les relations entre les termes analysés et leurs mots les plus proches vectoriellement. Les termes principaux sont en rouge, et les mots similaires en bleu. Cliquez sur un terme pour voir les articles correspondants."),
                period_selector,
                dcc.Graph(
                    id="similar-terms-network-graph",
                    figure=initial_graph,
                    config={"displayModeBar": True, "scrollZoom": True}
                ),
                # Ajouter un div pour afficher les articles
                html.Div(id="similar-terms-articles-container", style={"marginTop": "20px"})
            ])
        
        else:
            return html.Div([
                html.H5("Type de visualisation non pris en charge"),
                html.P(f"Le type de visualisation '{viz_type}' n'est pas pris en charge pour les termes similaires.")
            ])
            
    except Exception as e:
        return html.Div([
            html.P(f"Erreur lors de la création des visualisations : {str(e)}", className="text-danger")
        ])

# Function to create semantic drift visualizations
def create_semantic_drift_visualizations(results_file, viz_type="line"):
    """
    Create visualizations for semantic drift analysis results.
    
    Args:
        results_file: Path to the results file
        viz_type: Type of visualization (line, heatmap, table, comparison)
        
    Returns:
        HTML div with visualizations
    """
    try:
        # Load results
        df = pd.read_csv(results_file)
        
        if df.empty:
            return html.Div("Aucun résultat à afficher.")
        
        # Check if we have the expected columns
        if not all(col in df.columns for col in ['term', 'period', 'semantic_distance']):
            return html.Div("Format de fichier de résultats non reconnu. Attendu: 'term', 'period', 'semantic_distance'.")
        
        # Get unique terms and periods
        terms = df['term'].unique()
        periods = sorted(df['period'].unique())
        
        # Create visualizations based on the selected type
        if viz_type == "line":
            # Create a line chart showing semantic drift over time for each term
            fig = px.line(
                df, 
                x="period", 
                y="semantic_distance", 
                color="term",
                title="Évolution du drift sémantique dans le temps",
                labels={"period": "Période", "semantic_distance": "Distance sémantique", "term": "Terme"},
                markers=True
            )
            
            # Add horizontal line at y=0 for reference
            fig.add_shape(
                type="line",
                x0=periods[0],
                y0=0,
                x1=periods[-1],
                y1=0,
                line=dict(color="gray", width=1, dash="dash")
            )
            
            # Improve layout
            fig.update_layout(
                xaxis_title="Période",
                yaxis_title="Distance sémantique",
                legend_title="Terme",
                hovermode="closest"
            )
            
            return html.Div([
                html.H5("Évolution du drift sémantique dans le temps"),
                html.P("Ce graphique montre comment le sens des termes évolue au fil du temps. Une distance plus élevée indique un changement sémantique plus important."),
                dcc.Graph(figure=fig),
                html.Hr(),
                html.Div([
                    html.H5("Statistiques globales"),
                    html.Ul([
                        html.Li(f"Nombre de termes analysés: {len(terms)}"),
                        html.Li(f"Périodes couvertes: {', '.join(periods)}"),
                        html.Li(f"Distance sémantique moyenne: {df['semantic_distance'].mean():.4f}"),
                        html.Li(f"Distance sémantique maximale: {df['semantic_distance'].max():.4f} (terme: {df.loc[df['semantic_distance'].idxmax(), 'term']}, période: {df.loc[df['semantic_distance'].idxmax(), 'period']})")
                    ])
                ])
            ])
            
        elif viz_type == "heatmap":
            # Create a pivot table for the heatmap
            pivot_df = df.pivot(index="term", columns="period", values="semantic_distance")
            
            # Create heatmap
            fig = px.imshow(
                pivot_df,
                labels=dict(x="Période", y="Terme", color="Distance sémantique"),
                x=pivot_df.columns,
                y=pivot_df.index,
                color_continuous_scale="RdBu_r",
                title="Carte de chaleur du drift sémantique"
            )
            
            # Improve layout
            fig.update_layout(
                xaxis_title="Période",
                yaxis_title="Terme",
                coloraxis_colorbar=dict(title="Distance sémantique")
            )
            
            return html.Div([
                html.H5("Carte de chaleur du drift sémantique"),
                html.P("Cette carte de chaleur montre l'intensité du drift sémantique pour chaque terme et période. Les couleurs plus intenses indiquent un changement sémantique plus important."),
                dcc.Graph(figure=fig)
            ])
            
        elif viz_type == "table":
            # Create a table with the results
            table = dash_table.DataTable(
                id="semantic-drift-table",
                columns=[{"name": col, "id": col} for col in df.columns],
                data=df.to_dict("records"),
                filter_action="native",
                sort_action="native",
                sort_mode="multi",
                page_size=20,
                style_table={"overflowX": "auto"},
                style_cell={
                    "textAlign": "left",
                    "padding": "10px",
                    "whiteSpace": "normal",
                    "height": "auto"
                },
                style_header={
                    "backgroundColor": "rgb(230, 230, 230)",
                    "fontWeight": "bold"
                },
                style_data_conditional=[
                    {
                        "if": {"row_index": "odd"},
                        "backgroundColor": "rgb(248, 248, 248)"
                    }
                ]
            )
            
            return html.Div([
                html.H5("Tableau des résultats de drift sémantique"),
                html.P("Ce tableau présente les distances sémantiques pour chaque terme et période. Utilisez les filtres et le tri pour explorer les données."),
                table
            ])
            
        elif viz_type == "comparison":
            # Create a bar chart comparing semantic drift across terms
            term_avg_drift = df.groupby("term")["semantic_distance"].mean().reset_index()
            term_avg_drift = term_avg_drift.sort_values("semantic_distance", ascending=False)
            
            fig = px.bar(
                term_avg_drift,
                x="term",
                y="semantic_distance",
                title="Comparaison du drift sémantique moyen par terme",
                labels={"term": "Terme", "semantic_distance": "Distance sémantique moyenne"}
            )
            
            # Improve layout
            fig.update_layout(
                xaxis_title="Terme",
                yaxis_title="Distance sémantique moyenne",
                xaxis_tickangle=-45
            )
            
            # Create a radar chart for terms with the most drift
            top_terms = term_avg_drift.head(5)["term"].tolist()
            radar_df = df[df["term"].isin(top_terms)]
            
            # Create a figure with subplots
            radar_fig = go.Figure()
            
            # Add a trace for each term
            for term in top_terms:
                term_data = radar_df[radar_df["term"] == term]
                radar_fig.add_trace(go.Scatterpolar(
                    r=term_data["semantic_distance"].tolist(),
                    theta=term_data["period"].tolist(),
                    fill="toself",
                    name=term
                ))
            
            radar_fig.update_layout(
                polar=dict(
                    radialaxis=dict(
                        visible=True,
                        range=[0, radar_df["semantic_distance"].max() * 1.1]
                    )
                ),
                title="Profil de drift sémantique des termes les plus variables",
                showlegend=True
            )
            
            return html.Div([
                html.Div([
                    html.H5("Comparaison du drift sémantique moyen par terme"),
                    html.P("Ce graphique compare la distance sémantique moyenne pour chaque terme, tous périodes confondues."),
                    dcc.Graph(figure=fig)
                ]),
                html.Hr(),
                html.Div([
                    html.H5("Profil de drift sémantique des termes les plus variables"),
                    html.P("Ce graphique radar montre le profil de drift sémantique pour les 5 termes ayant le plus grand changement sémantique moyen."),
                    dcc.Graph(figure=radar_fig)
                ])
            ])
        
        else:
            return html.Div("Type de visualisation non reconnu.")
            
    except Exception as e:
        return html.Div(f"Erreur lors de la création des visualisations: {str(e)}")

# Ajouter une fonction pour récupérer les articles correspondant à une sélection
def get_articles_by_filter(articles, filter_type, filter_value, term=None):
    """
    Récupère les articles correspondant à un filtre donné.
    
    Args:
        articles: Liste d'articles
        filter_type: Type de filtre (année, journal, terme)
        filter_value: Valeur du filtre
        term: Terme spécifique à rechercher (optionnel)
        
    Returns:
        Liste d'articles filtrés
    """
    filtered_articles = []
    
    for article in articles:
        # Extraire la date et le journal de l'ID de l'article
        article_id = article.get('id', article.get('base_id', ''))
        if not isinstance(article_id, str):
            article_id = str(article_id)
            
        # Vérifier si l'ID a le format attendu
        if not article_id.startswith('article_'):
            continue
            
        # Appliquer le filtre
        if filter_type == 'année' and str(filter_value) == article_id.split('_')[1].split('-')[0]:
            # Si un terme est spécifié, vérifier s'il est présent dans le texte
            if term:
                text = article.get('text', article.get('content', ''))
                if term.lower() in text.lower():
                    filtered_articles.append(article)
            else:
                filtered_articles.append(article)
                
        elif filter_type == 'journal' and filter_value.lower() == article_id.split('_')[2].lower():
            # Si un terme est spécifié, vérifier s'il est présent dans le texte
            if term:
                text = article.get('text', article.get('content', ''))
                if term.lower() in text.lower():
                    filtered_articles.append(article)
            else:
                filtered_articles.append(article)
                
        elif filter_type == 'terme' and term:
            text = article.get('text', article.get('content', ''))
            if text and term.lower() in text.lower():
                filtered_articles.append(article)
    
    return filtered_articles

# Ajouter le modal pour afficher les articles
def create_articles_modal():
    """
    Crée un modal pour afficher les détails des articles.
    """
    return dbc.Modal(
        [
            dbc.ModalHeader([
                html.H4("Articles correspondants", className="modal-title"),
                dbc.Button("×", id="term-tracking-close-articles-modal", className="close")
            ]),
            dbc.ModalBody(id="term-tracking-articles-modal-body"),
            dbc.ModalFooter([
                dbc.Button("Fermer", id="term-tracking-close-articles-modal-footer", className="ml-auto")
            ]),
        ],
        id="term-tracking-articles-modal",
        size="xl",  # Modal plus grande pour afficher plus de contenu
        scrollable=True,  # Permet de faire défiler le contenu
    )

# Ajouter un modal pour afficher l'article complet
def create_full_article_modal():
    """
    Crée un modal pour afficher l'article complet avec tous les détails.
    """
    return dbc.Modal(
        [
            dbc.ModalHeader([
                html.H4("Article complet", className="modal-title"),
                dbc.Button("×", id="term-tracking-close-full-article-modal", className="close")
            ]),
            dbc.ModalBody(id="term-tracking-full-article-modal-body"),
            dbc.ModalFooter([
                dbc.Button("Fermer", id="term-tracking-close-full-article-modal-footer", className="ml-auto")
            ]),
        ],
        id="term-tracking-full-article-modal",
        size="xl",  # Modal plus grande pour afficher plus de contenu
        scrollable=True,  # Permet de faire défiler le contenu
    )

# Callback registration (to be called from app.py)
def register_term_tracking_callbacks(app):
    """Register callbacks for the term tracking page."""
    
    # Callback pour afficher un feedback sur le fichier de termes sélectionné
    @app.callback(
        Output("term-file-feedback", "children"),
        Input("term-tracking-term_file-input", "value"),
        prevent_initial_call=False
    )
    def update_term_file_feedback(term_file):
        if not term_file:
            # Obtenir la liste des fichiers disponibles
            term_files = get_term_files()
            if term_files:
                # Si des fichiers sont disponibles, suggérer d'en sélectionner un
                return html.P("Veuillez confirmer la sélection du fichier en cliquant dessus dans la liste.", className="text-warning")
            else:
                # Si aucun fichier n'est disponible
                return html.P("Erreur : Aucun fichier de termes disponible. Veuillez en créer un dans l'onglet 'Créer un fichier de termes'.", className="text-danger")
        
        try:
            # Vérifier si le fichier existe
            if not os.path.exists(term_file):
                return html.P(f"Erreur : Le fichier {term_file} n'existe pas.", className="text-danger")
            
            # Charger le fichier pour vérifier son contenu
            with open(term_file, 'r', encoding='utf-8') as f:
                terms = json.load(f)
            
            # Vérifier la structure du fichier
            if isinstance(terms, dict):
                categories = list(terms.keys())
                total_terms = sum(len(terms_list) for terms_list in terms.values())
                return html.P(f"✅ Fichier valide avec {len(categories)} catégories et {total_terms} termes au total.", className="text-success")
            elif isinstance(terms, list):
                return html.P(f"✅ Fichier valide avec {len(terms)} termes.", className="text-success")
            else:
                return html.P("⚠️ Format de fichier non reconnu. Le fichier doit contenir un dictionnaire ou une liste.", className="text-warning")
                
        except json.JSONDecodeError:
            return html.P("Erreur : Le fichier n'est pas un JSON valide.", className="text-danger")
        except Exception as e:
            return html.P(f"Erreur lors de la lecture du fichier : {str(e)}", className="text-danger")
    
    # Callback to run term tracking analysis
    @app.callback(
        Output("term-tracking-run-output", "children"),
        Input("run-term-tracking-button", "n_clicks"),
        State("term-tracking-term-file-input", "value"),
        State("term-tracking-output-input", "value"),
        State("term-tracking-by-year-input", "value"),
        State("term-tracking-by-newspaper-input", "value"),
        State("term-tracking-limit-input", "value"),
        State("term-tracking-semantic-drift-input", "value"),
        State("term-tracking-period-type-input", "value"),
        State("term-tracking-custom-periods-input", "value"),
        State("term-tracking-vector-size-input", "value"),
        State("term-tracking-window-input", "value"),
        State("term-tracking-min-count-input", "value"),
        prevent_initial_call=True
    )
    def run_term_tracking_analysis(n_clicks, term_file, output, by_year, by_newspaper, limit, 
                                  semantic_drift, period_type, custom_periods, vector_size, window, min_count):
        if not n_clicks or not term_file:
            return html.Div("Veuillez sélectionner un fichier de termes.")
        
        global project_root
        
        # Build command
        cmd = [sys.executable, os.path.join(project_root, "src", "scripts", "run_term_tracking.py")]
        cmd.extend(["--term-file", term_file])
        cmd.extend(["--output", output])
        
        if by_year:
            cmd.append("--by-year")
        
        if by_newspaper:
            cmd.append("--by-newspaper")
        
        if limit and int(limit) > 0:
            cmd.extend(["--limit", str(limit)])
        
        if semantic_drift:
            cmd.append("--semantic-drift")
            cmd.extend(["--period-type", period_type])
            
            if period_type == "custom" and custom_periods:
                cmd.extend(["--custom-periods", custom_periods])
            
            if vector_size:
                cmd.extend(["--vector-size", str(vector_size)])
            
            if window:
                cmd.extend(["--window", str(window)])
            
            if min_count:
                cmd.extend(["--min-count", str(min_count)])
        
        # Run the command
        try:
            result = subprocess.run(
                cmd,
                cwd=str(project_root),
                capture_output=True,
                text=True,
                check=True
            )
            
            # Parse the output to find the path to the results file
            output_lines = result.stdout.split('\n')
            results_file = None
            for line in output_lines:
                if "Résultats exportés vers" in line:
                    results_file = line.split("Résultats exportés vers")[-1].strip()
                    break
            
            return html.Div([
                html.P("Analyse terminée avec succès !", className="text-success"),
                html.Pre(result.stdout),
                html.P("Vous pouvez maintenant visualiser les résultats dans l'onglet 'Résultats'.", className="mt-3")
            ])
        
        except subprocess.CalledProcessError as e:
            return html.Div([
                html.P("Erreur lors de l'exécution de l'analyse :", className="text-danger"),
                html.Pre(e.stderr)
            ])
    
    # Callback to update term tracking results visualization
    @app.callback(
        Output("term-tracking-visualizations", "children"),
        [Input("term-tracking-results-file", "value"),
         Input("term-tracking-viz-type", "value")],
        prevent_initial_call=True
    )
    def update_term_tracking_results(results_file, viz_type):
        if not results_file:
            return html.Div([
                html.P("Veuillez sélectionner un fichier de résultats.", className="text-info")
            ])
        
        return create_term_tracking_visualizations(results_file, viz_type)
    
    # Callback pour traiter les clics sur les graphiques et gérer l'ouverture/fermeture du modal
    @app.callback(
        Output("term-tracking-articles-modal-body", "children"),
        Output("term-tracking-articles-modal", "is_open"),
        [
            Input({'type': 'term-tracking-graph', 'subtype': ALL}, 'clickData'),
            Input("term-tracking-close-articles-modal", "n_clicks"),
            Input("term-tracking-close-articles-modal-footer", "n_clicks")
        ],
        State("term-tracking-results-file", "value"),
        prevent_initial_call=True
    )
    def handle_articles_modal(click_data_list, close_header_clicks, close_footer_clicks, results_file):
        """
        Ce callback est déclenché lorsqu'un utilisateur:
        1. Clique sur un graphique pour ouvrir le modal avec les articles correspondants
        2. Clique sur un bouton de fermeture pour fermer le modal
        """
        from dash import callback_context as ctx
        
        # Vérifier si le callback a été déclenché
        if not ctx.triggered:
            return "", False
        
        # Déterminer quel élément a déclenché le callback
        trigger = ctx.triggered[0]
        prop_id = trigger['prop_id']
        
        # Si un bouton de fermeture a été cliqué, fermer le modal
        if "term-tracking-close-articles-modal" in prop_id:
            return "", False
        
        # Si aucun fichier de résultats n'est sélectionné, ne rien faire
        if not results_file:
            return "", False
        
        # Récupérer les données du clic
        click_data = trigger['value']
        
        # Vérifier si le clic provient d'un graphique (pattern-matching)
        if 'term-tracking-graph' not in prop_id:
            return "", False
            
        # Vérification robuste des données de clic
        if click_data is None:
            return "", False
            
        if 'points' not in click_data or not click_data['points']:
            return "", False
        
        try:
            # Extraire les informations du clic
            point = click_data['points'][0]
            
            # Extraire l'ID du graphique (sous forme de dictionnaire)
            import json
            graph_id = json.loads(prop_id.split('.')[0])
            subtype = graph_id.get('subtype', '')
            
            # Déterminer le type de filtre et la valeur
            filter_type = None
            filter_value = None
            term = None
            
            # Analyser le sous-type du graphique pour déterminer le type de filtre
            if 'year' in subtype:
                filter_type = "année"
                filter_value = point.get('x')
                term = point.get('curveNumber')  # Indice de la courbe (terme)
            elif 'journal' in subtype:
                filter_type = "journal"
                filter_value = point.get('x')
                term = point.get('curveNumber')  # Indice de la courbe (terme)
            elif 'month' in subtype:
                # Extraire l'année du format YYYY-MM
                year_month = point.get('x')
                if year_month and '-' in year_month:
                    filter_type = "année"
                    filter_value = year_month.split('-')[0]
                    term = point.get('curveNumber')  # Indice de la courbe (terme)
            elif 'term' in subtype:
                filter_type = "terme"
                filter_value = point.get('label')
                term = filter_value
            
            if not filter_type or not filter_value:
                return html.P("Impossible de déterminer les critères de filtrage."), True
            
            # Charger les articles depuis le fichier JSON
            project_root = pathlib.Path(__file__).resolve().parents[2]
            config_path = os.path.join(project_root, "config", "config.yaml")
            config = load_config(config_path)
            
            articles_path = os.path.join(project_root, config['data']['processed_dir'], "articles.json")
            
            with open(articles_path, 'r', encoding='utf-8') as f:
                articles = json.load(f)
            
            # Charger le fichier de résultats pour obtenir le nom du terme
            df_results = pd.read_csv(results_file)
            term_columns = df_results.columns[1:].tolist()
            if term is not None and isinstance(term, int) and term < len(term_columns):
                term_name = term_columns[term]
            elif term is not None and isinstance(term, str) and term in term_columns:
                term_name = term
            else:
                term_name = None
            
            # Filtrer les articles
            filtered_articles = []
            for article in articles:
                article_id = article.get('id', article.get('base_id', ''))
                if not isinstance(article_id, str):
                    article_id = str(article_id)
                
                # Vérifier si l'ID a le format attendu
                if not article_id.startswith('article_'):
                    continue
                
                # Extraire la date et le journal des IDs d'articles
                # Format attendu: article_YYYY-MM-DD_journal_XXXX_source
                try:
                    parts = article_id.split('_')
                    if len(parts) >= 3:
                        date_part = parts[1]
                        journal_part = parts[2]
                        year_part = date_part.split('-')[0] if '-' in date_part else date_part
                    else:
                        continue
                except IndexError:
                    continue
                
                # Appliquer le filtre
                match = False
                if filter_type == "année" and str(filter_value) == year_part:
                    match = True
                elif filter_type == "journal" and filter_value.lower() == journal_part.lower():
                    match = True
                elif filter_type == "terme" and term_name:
                    text = article.get('text', article.get('content', ''))
                    if text and term_name.lower() in text.lower():
                        match = True
                
                # Si correspondance et terme spécifié, vérifier la présence du terme
                if match and term_name and filter_type != "terme":
                    text = article.get('text', article.get('content', ''))
                    if text and term_name.lower() in text.lower():
                        filtered_articles.append(article)
                elif match:
                    filtered_articles.append(article)
            
            # Limiter à 20 articles pour éviter de surcharger l'interface
            filtered_articles = filtered_articles[:20]
            
            # Créer le contenu du modal
            if filtered_articles:
                title = f"Articles de {filter_value}"
                if term_name and filter_type != "terme":
                    title += f" contenant le terme '{term_name}'"
                elif filter_type == "terme":
                    title = f"Articles contenant le terme '{term_name}'"
                
                articles_content = [
                    html.H4(title),
                    html.P(f"{len(filtered_articles)} articles trouvés" + (" (affichage limité aux 20 premiers)" if len(filtered_articles) == 20 else "")),
                    html.Hr()
                ]
                
                for i, article in enumerate(filtered_articles):
                    article_id = article.get('id', article.get('base_id', ''))
                    date = article_id.split('_')[1] if '_' in article_id else ""
                    journal = article_id.split('_')[2] if '_' in article_id and len(article_id.split('_')) > 2 else ""
                    text = article.get('text', article.get('content', ''))
                    title = article.get('title', f"Article {i+1}")
                    url = article.get('url', '')
                    
                    # Créer un lien vers l'article original si disponible
                    article_link = None
                    if url:
                        article_link = html.A("Voir l'article original", href=url, target="_blank", className="btn btn-sm btn-primary mt-2 mb-2")
                    
                    # Mettre en évidence le terme recherché
                    if term_name and text:
                        # Mettre en évidence le terme
                        text_parts = []
                        remaining = text
                        term_lower = term_name.lower()
                        while term_lower in remaining.lower():
                            pos = remaining.lower().find(term_lower)
                            text_parts.append(remaining[:pos])
                            term_instance = remaining[pos:pos+len(term_name)]
                            text_parts.append(html.Mark(term_instance, style={"background-color": "yellow"}))
                            remaining = remaining[pos+len(term_name):]
                        text_parts.append(remaining)
                        
                        # Créer le contenu de l'article
                        article_content = html.Div([
                            html.H5(title if title else f"Article {i+1}"),
                            html.P(f"Date: {date} | Journal: {journal} | ID: {article_id}"),
                            article_link if article_link else html.Div(),
                            html.Div(text_parts, className="article-text p-3 border rounded", style={"max-height": "300px", "overflow-y": "auto"}),
                            dbc.Button(
                                "Afficher l'article complet", 
                                id={'type': 'show-full-article', 'index': i},
                                color="link", 
                                className="mt-2"
                            ),
                            html.Hr()
                        ])
                    else:
                        # Sans mise en évidence
                        article_content = html.Div([
                            html.H5(title if title else f"Article {i+1}"),
                            html.P(f"Date: {date} | Journal: {journal} | ID: {article_id}"),
                            article_link if article_link else html.Div(),
                            html.Div(text, className="article-text p-3 border rounded", style={"max-height": "300px", "overflow-y": "auto"}),
                            dbc.Button(
                                "Afficher l'article complet", 
                                id={'type': 'show-full-article', 'index': i},
                                color="link", 
                                className="mt-2"
                            ),
                            html.Hr()
                        ])
                    
                    articles_content.append(article_content)
                
                return articles_content, True
            else:
                return html.P("Aucun article correspondant trouvé."), True
                
        except Exception as e:
            print(f"Erreur lors de l'affichage des articles : {str(e)}")
            return html.P(f"Erreur lors de la récupération des articles : {str(e)}"), True
    
    # Callback to add a new category in the term file creation form
    @app.callback(
        Output("term-categories-container", "children"),
        Input("add-category-button", "n_clicks"),
        State("term-categories-container", "children"),
        prevent_initial_call=True
    )
    def add_category(n_clicks, current_categories):
        if not n_clicks:
            return current_categories
        
        # Get the current number of categories
        num_categories = len(current_categories) // 2  # Each category has a row and a br
        
        # Add a new category
        new_category = html.Div([
            dbc.Row([
                dbc.Col([
                    dbc.Label(f"Catégorie {num_categories + 1}:"),
                    dbc.Input(
                        id=f"term-category-{num_categories + 1}",
                        type="text",
                        placeholder="économie",
                        value=""
                    )
                ], width=4),
                dbc.Col([
                    dbc.Label("Termes (séparés par des virgules):"),
                    dbc.Textarea(
                        id=f"term-list-{num_categories + 1}",
                        placeholder="économie, finance, marché",
                        value="",
                        style={"height": "100px"}
                    )
                ], width=8)
            ]),
            html.Br()
        ])
        
        return current_categories + [new_category]
    
    # Callback to create a new term file
    @app.callback(
        Output("create-term-file-output", "children"),
        Input("create-term-file-button", "n_clicks"),
        [State("new-term-file-name", "value"),
         State("term-categories-container", "children")],
        prevent_initial_call=True
    )
    def create_term_file(n_clicks, file_name, categories_container):
        if not n_clicks:
            return ""
        
        if not file_name:
            return html.P("Erreur : Vous devez spécifier un nom de fichier.", className="text-danger")
        
        # Get the number of categories
        num_categories = len(categories_container) // 2  # Each category has a row and a br
        
        # Get the values from the form
        term_dict = {}
        for i in range(1, num_categories + 1):
            category = ctx.inputs.get(f"term-category-{i}.value")
            terms_str = ctx.inputs.get(f"term-list-{i}.value")
            
            if category and terms_str:
                # Split the terms by comma and remove whitespace
                terms = [term.strip() for term in terms_str.split(",") if term.strip()]
                if terms:
                    term_dict[category] = terms
        
        if not term_dict:
            return html.P("Erreur : Vous devez spécifier au moins une catégorie avec des termes.", className="text-danger")
        
        # Save the term file
        project_root = pathlib.Path(__file__).resolve().parents[2]
        examples_dir = project_root / "examples"
        file_path = examples_dir / f"{file_name}.json"
        
        try:
            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(term_dict, f, ensure_ascii=False, indent=2)
            
            return html.Div([
                html.P(f"Fichier de termes créé avec succès : {file_path}", className="text-success"),
                html.P("Vous pouvez maintenant l'utiliser dans l'onglet 'Lancer une analyse'.")
            ])
        except Exception as e:
            return html.P(f"Erreur lors de la création du fichier : {str(e)}", className="text-danger")
    
    # Fonctions pour l'exportation
    def get_term_tracking_source_data():
        """Obtient les données source pour l'exportation."""
        results_file = ctx.states.get("term-tracking-results-file.value")
        viz_type = ctx.states.get("term-tracking-viz-type.value", "bar")
        
        source_data = {
            "results_file": results_file,
            "visualization_type": viz_type
        }
        
        # Si un fichier de résultats est sélectionné, ajouter des métadonnées
        if results_file:
            try:
                df = pd.read_csv(results_file)
                
                # Déterminer le type de résultats
                key_column = df.columns[0]
                
                # Rename the key column based on the type of results
                if key_column == 'key':
                    # Try to determine the type of key
                    first_key = df['key'].iloc[0]
                    if isinstance(first_key, str) and len(first_key) > 20:
                        # Likely an article ID
                        key_type = "Article ID"
                        df = df.rename(columns={'key': 'Article ID'})
                    elif isinstance(first_key, (int, float)) or (isinstance(first_key, str) and first_key.isdigit()):
                        # Likely a year
                        key_type = "Année"
                        df = df.rename(columns={'key': 'Année'})
                    else:
                        # Likely a newspaper
                        key_type = "Journal"
                        df = df.rename(columns={'key': 'Journal'})
                else:
                    key_type = key_column
                
                # Get term columns (all columns except the key column)
                term_columns = df.columns[1:].tolist()
                
                source_data.update({
                    "key_type": key_type,
                    "terms": term_columns,
                    "num_records": len(df)
                })
                
                # Ajouter des statistiques de base
                if len(df) > 0:
                    total_occurrences = df[term_columns].sum().sum()
                    most_frequent_term = df[term_columns].sum().idxmax()
                    
                    source_data.update({
                        "total_occurrences": int(total_occurrences),
                        "most_frequent_term": most_frequent_term
                    })
            except Exception as e:
                print(f"Erreur lors de la récupération des données source : {str(e)}")
        
        return source_data
    
    def get_term_tracking_figure():
        """Obtient la figure pour l'exportation."""
        # Récupérer les valeurs des composants
        results_file = ctx.states.get("term-tracking-results-file.value")
        viz_type = ctx.states.get("term-tracking-viz-type.value", "bar")
        
        if not results_file:
            return {}
        
        try:
            # Charger les données
            df = pd.read_csv(results_file)
            
            # Vérifier si le fichier est vide
            if df.empty:
                return {}
            
            # Déterminer le type de résultats
            key_column = df.columns[0]
            
            # Renommer la colonne clé
            if key_column == 'key':
                # Try to determine the type of key
                first_key = df['key'].iloc[0]
                if isinstance(first_key, str) and len(first_key) > 20:
                    # Likely an article ID
                    df = df.rename(columns={'key': 'Article ID'})
                    key_column = 'Article ID'
                elif isinstance(first_key, (int, float)) or (isinstance(first_key, str) and first_key.isdigit()):
                    # Likely a year
                    df = df.rename(columns={'key': 'Année'})
                    key_column = 'Année'
                else:
                    # Likely a newspaper
                    df = df.rename(columns={'key': 'Journal'})
                    key_column = 'Journal'
            
            # Obtenir les colonnes de termes
            term_columns = df.columns[1:].tolist()
            
            # Créer la visualisation appropriée
            if viz_type == "bar":
                # Bar chart
                if key_column == "Année":
                    # For years, create a grouped bar chart
                    fig = px.bar(
                        df, 
                        x='Année', 
                        y=term_columns,
                        title=f"Fréquence des termes par année",
                        labels={'value': 'Fréquence', 'variable': 'Terme'},
                        barmode='group'
                    )
                elif key_column == "Journal":
                    # For newspapers, create a grouped bar chart
                    fig = px.bar(
                        df, 
                        x='Journal', 
                        y=term_columns,
                        title=f"Fréquence des termes par journal",
                        labels={'value': 'Fréquence', 'variable': 'Terme'},
                        barmode='group'
                    )
                else:
                    # For articles, create a stacked bar chart of the top 20 articles
                    top_df = df.copy()
                    top_df['Total'] = top_df[term_columns].sum(axis=1)
                    top_df = top_df.nlargest(20, 'Total')
                    top_df = top_df.drop(columns=['Total'])
                    
                    fig = px.bar(
                        top_df.melt(id_vars=[key_column], value_vars=term_columns),
                        x=key_column,
                        y='value',
                        color='variable',
                        title=f"Top 20 articles par fréquence totale de termes",
                        labels={'value': 'Fréquence', 'variable': 'Terme'}
                    )
                
            elif viz_type == "line":
                # Line chart (only makes sense for time series data)
                if key_column == "Année":
                    fig = px.line(
                        df, 
                        x='Année', 
                        y=term_columns,
                        title=f"Évolution des termes au fil du temps",
                        labels={'value': 'Fréquence', 'variable': 'Terme'},
                        markers=True
                    )
                else:
                    return {}
                    
            elif viz_type == "heatmap":
                # Heatmap
                if len(term_columns) > 1:
                    # Transpose the dataframe for the heatmap
                    heatmap_df = df.set_index(df.columns[0])
                    
                    fig = px.imshow(
                        heatmap_df.T,
                        labels=dict(x=key_column, y="Terme", color="Fréquence"),
                        title="Heatmap des termes",
                        color_continuous_scale='Viridis'
                    )
                else:
                    return {}
            else:
                # Pour le tableau, pas de figure à exporter
                return {}
            
            return fig.to_dict()
            
        except Exception as e:
            print(f"Erreur lors de la création de la figure: {str(e)}")
            return {}
    
    # Register export callbacks
    register_export_callbacks(
        app,
        analysis_type="term_tracking",
        get_source_data_function=get_term_tracking_source_data,
        get_figure_function=get_term_tracking_figure,
        modal_id="term-tracking-export-modal",
        toast_id="term-tracking-export-toast"
    )

    # Callback pour gérer l'ouverture/fermeture du modal d'article complet
    @app.callback(
        Output("term-tracking-full-article-modal-body", "children"),
        Output("term-tracking-full-article-modal", "is_open"),
        [
            Input({'type': 'show-full-article', 'index': ALL}, 'n_clicks'),
            Input("term-tracking-close-full-article-modal", "n_clicks"),
            Input("term-tracking-close-full-article-modal-footer", "n_clicks")
        ],
        [
            State("term-tracking-results-file", "value"),
            State("term-tracking-articles-modal-body", "children")
        ],
        prevent_initial_call=True
    )
    def handle_full_article_modal(show_clicks, close_header_clicks, close_footer_clicks, results_file, articles_body):
        """
        Ce callback est déclenché lorsqu'un utilisateur:
        1. Clique sur le bouton "Afficher l'article complet" pour ouvrir le modal avec tous les détails
        2. Clique sur un bouton de fermeture pour fermer le modal
        """
        from dash import callback_context as ctx
        
        # Vérifier si le callback a été déclenché
        if not ctx.triggered:
            return "", False
        
        # Déterminer quel élément a déclenché le callback
        trigger = ctx.triggered[0]
        prop_id = trigger['prop_id']
        
        # Si un bouton de fermeture a été cliqué, fermer le modal
        if "term-tracking-close-full-article-modal" in prop_id:
            return "", False
        
        # Si aucun fichier de résultats n'est sélectionné, ne rien faire
        if not results_file or not articles_body:
            return "", False
        
        # Vérifier si le déclencheur est un bouton "Afficher l'article complet"
        if 'show-full-article' not in prop_id:
            return "", False
            
        # Vérifier si le bouton a été cliqué (n_clicks n'est pas None)
        if trigger['value'] is None:
            return "", False
            
        try:
            # Extraire l'index de l'article à afficher
            import json
            button_id = json.loads(prop_id.split('.')[0])
            article_index = button_id.get('index')
            
            # Charger les articles depuis le fichier JSON
            project_root = pathlib.Path(__file__).resolve().parents[2]
            config_path = os.path.join(project_root, "config", "config.yaml")
            config = load_config(config_path)
            
            articles_path = os.path.join(project_root, config['data']['processed_dir'], "articles.json")
            
            with open(articles_path, 'r', encoding='utf-8') as f:
                articles = json.load(f)
            
            # Trouver l'article correspondant à l'index
            # Nous devons extraire l'ID de l'article à partir du contenu du modal
            article_id = None
            
            # Parcourir les éléments du modal pour trouver l'ID de l'article
            for i, item in enumerate(articles_body):
                if isinstance(item, dict) and item.get('type') == 'Div' and item.get('props', {}).get('children'):
                    children = item.get('props', {}).get('children', [])
                    # Chercher l'élément qui contient l'ID de l'article
                    for child in children:
                        if isinstance(child, dict) and child.get('type') == 'P' and child.get('props', {}).get('children'):
                            text = child.get('props', {}).get('children')
                            if isinstance(text, str) and 'ID:' in text:
                                # Extraire l'ID de l'article
                                article_id = text.split('ID:')[-1].strip()
                                break
                        
                    # Si nous avons trouvé l'ID et que c'est l'article que nous cherchons
                    if article_id and i == article_index + 3:  # +3 pour tenir compte des éléments d'en-tête dans le modal
                        break
            
            if not article_id:
                return html.P("Impossible de trouver l'ID de l'article."), True
            
            # Trouver l'article correspondant à l'ID
            article = None
            for a in articles:
                if a.get('id') == article_id or a.get('base_id') == article_id:
                    article = a
                    break
            
            if not article:
                return html.P(f"Article non trouvé: {article_id}"), True
            
            # Créer le contenu du modal avec tous les détails de l'article
            article_content = []
            
            # Titre et métadonnées principales
            article_content.append(html.H3(article.get('title', 'Article sans titre'), className="mb-3"))
            
            # Carte d'identité de l'article
            metadata_items = []
            
            # Métadonnées principales
            if article.get('date'):
                metadata_items.append(html.Li([html.Strong("Date: "), article.get('date')]))
            if article.get('newspaper'):
                metadata_items.append(html.Li([html.Strong("Journal: "), article.get('newspaper')]))
            if article.get('canton'):
                metadata_items.append(html.Li([html.Strong("Canton: "), article.get('canton')]))
            if article.get('id'):
                metadata_items.append(html.Li([html.Strong("ID: "), article.get('id')]))
            if article.get('base_id'):
                metadata_items.append(html.Li([html.Strong("Base ID: "), article.get('base_id')]))
            if article.get('word_count'):
                metadata_items.append(html.Li([html.Strong("Nombre de mots: "), str(article.get('word_count'))]))
            
            # Lien vers l'article original
            if article.get('url'):
                metadata_items.append(html.Li([
                    html.Strong("URL: "), 
                    html.A("Voir l'article original", href=article.get('url'), target="_blank")
                ]))
            
            # Thèmes/topics
            if article.get('topics') and isinstance(article.get('topics'), list):
                topics_str = ", ".join(article.get('topics'))
                metadata_items.append(html.Li([html.Strong("Thèmes: "), topics_str]))
            
            # Méthode de correction
            if article.get('correction_method'):
                metadata_items.append(html.Li([html.Strong("Méthode de correction: "), article.get('correction_method')]))
            if article.get('spell_corrected') is not None:
                metadata_items.append(html.Li([html.Strong("Correction orthographique: "), "Oui" if article.get('spell_corrected') else "Non"]))
            
            # Autres métadonnées
            if article.get('created_at'):
                metadata_items.append(html.Li([html.Strong("Date de création: "), article.get('created_at')]))
            if article.get('raw_path'):
                metadata_items.append(html.Li([html.Strong("Chemin du fichier brut: "), article.get('raw_path')]))
            if article.get('versions') and isinstance(article.get('versions'), list):
                versions_str = ", ".join(article.get('versions'))
                metadata_items.append(html.Li([html.Strong("Versions: "), versions_str]))
            
            # Ajouter la liste des métadonnées
            article_content.append(html.Div([
                html.H4("Métadonnées", className="mt-4 mb-3"),
                html.Ul(metadata_items, className="list-unstyled")
            ], className="metadata-section p-3 border rounded bg-light"))
            
            # Contenu de l'article
            if article.get('content'):
                # Convertir les sauts de ligne en éléments html.Br()
                content_parts = []
                for part in article.get('content').split('\n'):
                    content_parts.append(part)
                    content_parts.append(html.Br())
                # Supprimer le dernier Br() si la liste n'est pas vide
                if content_parts:
                    content_parts.pop()
                
                article_content.append(html.Div([
                    html.H4("Contenu", className="mt-4 mb-3"),
                    html.Div(content_parts, className="p-3 border rounded")
                ]))
            
            # Contenu original (si disponible)
            if article.get('original_content'):
                # Convertir les sauts de ligne en éléments html.Br()
                original_content_parts = []
                for part in article.get('original_content').split('\n'):
                    original_content_parts.append(part)
                    original_content_parts.append(html.Br())
                # Supprimer le dernier Br() si la liste n'est pas vide
                if original_content_parts:
                    original_content_parts.pop()
                
                article_content.append(html.Div([
                    html.H4("Contenu original", className="mt-4 mb-3"),
                    html.Div(original_content_parts, className="p-3 border rounded bg-light")
                ]))
            
            # Autres champs JSON (pour être exhaustif)
            other_fields = {}
            for key, value in article.items():
                if key not in ['id', 'base_id', 'title', 'date', 'newspaper', 'canton', 'content', 'original_content', 
                              'url', 'topics', 'word_count', 'correction_method', 'spell_corrected', 'created_at', 
                              'raw_path', 'versions', '_id']:
                    other_fields[key] = value
            
            if other_fields:
                other_fields_items = []
                for key, value in other_fields.items():
                    if isinstance(value, (list, dict)):
                        value = json.dumps(value, indent=2, ensure_ascii=False)
                    other_fields_items.append(html.Li([html.Strong(f"{key}: "), str(value)]))
                
                article_content.append(html.Div([
                    html.H4("Autres informations", className="mt-4 mb-3"),
                    html.Ul(other_fields_items, className="list-unstyled")
                ], className="p-3 border rounded"))
            
            return article_content, True
            
        except Exception as e:
            print(f"Erreur lors de l'affichage de l'article complet : {str(e)}")
            return html.P(f"Erreur lors de la récupération de l'article complet : {str(e)}"), True

    @app.callback(
        Output("semantic-drift-options", "style"),
        Input("term-tracking-semantic-drift-input", "value")
    )
    def toggle_semantic_drift_options(semantic_drift_enabled):
        if semantic_drift_enabled:
            return {"display": "block"}
        else:
            return {"display": "none"}
    
    @app.callback(
        Output("semantic-drift-visualizations", "children"),
        Input("semantic-drift-results-file", "value"),
        Input("semantic-drift-viz-type", "value")
    )
    def update_semantic_drift_visualizations(results_file, viz_type):
        if not results_file or not viz_type:
            return html.Div("Veuillez sélectionner un fichier de résultats et un type de visualisation.")
        
        return create_semantic_drift_visualizations(results_file, viz_type)

    @app.callback(
        Output("similar-terms-visualizations", "children"),
        Input("similar-terms-results-file", "value"),
        Input("similar-terms-viz-type", "value")
    )
    def update_similar_terms_visualizations(results_file, viz_type):
        if not results_file or not viz_type:
            return html.Div("Veuillez sélectionner un fichier de résultats et un type de visualisation.")
        
        return create_similar_terms_visualizations(results_file, viz_type)

    # Callback pour mettre à jour le graphique en réseau lorsque l'utilisateur change de période
    @app.callback(
        Output("similar-terms-network-graph", "figure"),
        [Input("similar-terms-period-selector", "value")],
        [State("similar-terms-results-file", "value")]
    )
    def update_network_graph(selected_period, results_file):
        if not selected_period or not results_file:
            return go.Figure()
        
        try:
            # Charger les données
            df = pd.read_csv(results_file)
            
            # Créer la fonction pour générer le graphique en réseau
            def create_period_network(period):
                # Filtrer les données pour la période sélectionnée
                period_df = df[df['period'] == period]
                
                # Obtenir les termes uniques pour cette période
                period_terms = period_df['term'].unique()
                
                # Filtrer pour les 5 mots les plus similaires pour une meilleure visualisation
                top_words = period_df[period_df['rank'] <= 5].copy()
                
                # Créer le graphique en réseau
                fig = go.Figure()
                
                # Calculer les positions pour les termes principaux (dans un cercle)
                radius = 3
                term_positions = {}
                
                for i, term in enumerate(period_terms):
                    angle = 2 * np.pi * i / len(period_terms)
                    x = radius * np.cos(angle)
                    y = radius * np.sin(angle)
                    term_positions[term] = (x, y)
                    
                    # Ajouter un nœud pour le terme principal
                    fig.add_trace(go.Scatter(
                        x=[x],
                        y=[y],
                        mode="markers+text",
                        marker=dict(size=25, color="red"),
                        text=[term],
                        name=term,
                        textposition="middle center",
                        textfont=dict(color="white", size=12),
                        hoverinfo="text",
                        hovertext=f"<b>{term}</b>",
                        showlegend=False
                    ))
                
                # Ajouter des nœuds et des arêtes pour les mots similaires
                for term in period_terms:
                    term_x, term_y = term_positions[term]
                    term_similar = top_words[top_words['term'] == term]
                    
                    for _, row in term_similar.iterrows():
                        # Calculer la position (dans un cercle autour du terme principal)
                        angle = (row['rank'] - 1) * (2 * np.pi / 5)
                        distance = 1.5  # Distance du terme principal
                        x = term_x + distance * np.cos(angle)
                        y = term_y + distance * np.sin(angle)
                        
                        # Taille et opacité basées sur la similarité
                        node_size = 15 + (row['similarity'] * 10)
                        
                        # Ajouter un nœud pour le mot similaire
                        fig.add_trace(go.Scatter(
                            x=[x],
                            y=[y],
                            mode="markers+text",
                            marker=dict(
                                size=node_size, 
                                color="blue",
                                opacity=0.7 + (row['similarity'] * 0.3)
                            ),
                            text=[row['similar_word']],
                            name=f"{row['similar_word']} ({row['similarity']:.2f})",
                            textposition="bottom center",
                            hoverinfo="text",
                            hovertext=f"<b>{row['similar_word']}</b><br>Similarité: {row['similarity']:.3f}<br>Cliquez pour voir les articles",
                            showlegend=False
                        ))
                        
                        # Ajouter une arête avec une largeur basée sur la similarité
                        fig.add_trace(go.Scatter(
                            x=[term_x, x],
                            y=[term_y, y],
                            mode="lines",
                            line=dict(
                                width=row['similarity'] * 5, 
                                color="rgba(100, 100, 100, 0.6)"
                            ),
                            hoverinfo="text",
                            hovertext=f"Similarité: {row['similarity']:.3f}",
                            showlegend=False
                        ))
                
                # Améliorer la mise en page
                fig.update_layout(
                    title=f"Réseau de termes similaires - Période: {period}",
                    xaxis=dict(
                        showgrid=False, 
                        zeroline=False, 
                        showticklabels=False,
                        range=[-5, 5]
                    ),
                    yaxis=dict(
                        showgrid=False, 
                        zeroline=False, 
                        showticklabels=False,
                        range=[-5, 5],
                        scaleanchor="x",
                        scaleratio=1
                    ),
                    hovermode="closest",
                    plot_bgcolor="rgba(240, 240, 240, 0.8)",
                    clickmode="event+select"  # Activer les événements de clic
                )
                
                return fig
            
            # Générer le graphique pour la période sélectionnée
            return create_period_network(selected_period)
            
        except Exception as e:
            # En cas d'erreur, retourner un graphique vide avec un message d'erreur
            fig = go.Figure()
            fig.update_layout(
                title=f"Erreur: {str(e)}",
                annotations=[
                    dict(
                        text=f"Erreur lors de la génération du graphique: {str(e)}",
                        showarrow=False,
                        xref="paper",
                        yref="paper",
                        x=0.5,
                        y=0.5
                    )
                ]
            )
            return fig
            
    # Callback pour afficher les articles lorsqu'on clique sur un terme dans le graphique en réseau
    @app.callback(
        Output("similar-terms-articles-container", "children"),
        [Input("similar-terms-network-graph", "clickData")],
        [State("similar-terms-period-selector", "value")]
    )
    def display_similar_term_articles(click_data, period):
        if not click_data:
            return html.Div()
        
        try:
            # Extraire le terme à partir des données de clic
            term = click_data["points"][0]["text"]
            
            # Charger les articles
            from src.utils.config_loader import load_config
            project_root = pathlib.Path(__file__).resolve().parents[2]
            config_path = project_root / 'config' / 'config.yaml'
            config = load_config(config_path)
            
            articles_path = Path(config['data']['processed']) / "articles.json"
            
            with open(articles_path, 'r', encoding='utf-8') as f:
                articles = json.load(f)
            
            # Filtrer les articles par période
            filtered_articles = []
            for article in articles:
                article_id = str(article.get('id', article.get('base_id', '')))
                if not article_id:
                    continue
                
                # Extraire l'année de l'ID de l'article
                year_match = re.search(r'article_(\d{4})-\d{2}-\d{2}', article_id)
                if not year_match:
                    continue
                
                year = int(year_match.group(1))
                
                # Vérifier si l'année correspond à la période
                if period == "1960s" and 1960 <= year < 1970:
                    filtered_articles.append(article)
                elif period == "1970s" and 1970 <= year < 1980:
                    filtered_articles.append(article)
                elif period == "1980s" and 1980 <= year < 1990:
                    filtered_articles.append(article)
                elif period == "1990s" and 1990 <= year < 2000:
                    filtered_articles.append(article)
                elif period == str(year):
                    filtered_articles.append(article)
            
            # Rechercher le terme dans les articles
            articles_with_term = []
            for article in filtered_articles:
                text = article.get('text', article.get('content', ''))
                if not text:
                    continue
                
                # Vérifier si le terme est présent dans le texte
                if re.search(r'\b' + re.escape(term) + r'\b', text, re.IGNORECASE):
                    # Extraire un extrait de texte autour du terme
                    match = re.search(r'.{0,100}\b' + re.escape(term) + r'\b.{0,100}', text, re.IGNORECASE)
                    excerpt = match.group(0) if match else ""
                    
                    # Mettre en évidence le terme dans l'extrait
                    highlighted_excerpt = re.sub(
                        r'\b(' + re.escape(term) + r')\b', 
                        r'<span style="background-color: yellow; font-weight: bold;">\1</span>', 
                        excerpt, 
                        flags=re.IGNORECASE
                    )
                    
                    # Ajouter l'article à la liste
                    articles_with_term.append({
                        'id': article.get('id', article.get('base_id', '')),
                        'title': article.get('title', 'Sans titre'),
                        'date': article.get('date', 'Date inconnue'),
                        'journal': article.get('journal', 'Journal inconnu'),
                        'excerpt': highlighted_excerpt
                    })
            
            # Limiter à 20 articles maximum
            articles_with_term = articles_with_term[:20]
            
            if not articles_with_term:
                return html.Div([
                    html.H5(f"Articles contenant le terme '{term}' - Période: {period}"),
                    html.P("Aucun article trouvé pour ce terme dans cette période.")
                ])
            
            # Créer la liste d'articles
            article_list = []
            for i, article in enumerate(articles_with_term):
                article_list.append(html.Div([
                    html.H6(f"{i+1}. {article['title']}", style={"marginBottom": "5px"}),
                    html.P([
                        f"Date: {article['date']} | Journal: {article['journal']} | ",
                        html.A("Voir l'article complet", 
                               id={'type': 'similar-term-article-link', 'index': i},
                               href="#",
                               style={"color": "blue", "textDecoration": "underline", "cursor": "pointer"})
                    ], style={"fontSize": "0.9em", "marginBottom": "5px"}),
                    html.Div([
                        html.P("Extrait:", style={"fontWeight": "bold", "marginBottom": "5px"}),
                        html.Div(dangerously_set_inner_html={"__html": article['excerpt']},
                               style={"backgroundColor": "#f8f9fa", "padding": "10px", "borderRadius": "5px"})
                    ]),
                    html.Hr()
                ], style={"marginBottom": "15px"}))
            
            return html.Div([
                html.H5(f"Articles contenant le terme '{term}' - Période: {period}"),
                html.P(f"Nombre d'articles trouvés: {len(articles_with_term)}"),
                html.Div(article_list)
            ])
            
        except Exception as e:
            return html.Div([
                html.P(f"Erreur lors de la recherche d'articles: {str(e)}", className="text-danger")
            ])
