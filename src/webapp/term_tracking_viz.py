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

# Layout for the term tracking page
def get_term_tracking_layout():
    # Get available term files and result files
    term_files = get_term_files()
    term_tracking_results = get_term_tracking_results()
    
    # Get parser arguments for the run form
    parser_args = get_term_tracking_args()
    
    # Create form fields based on parser arguments
    form_fields = []
    for arg in parser_args:
        if arg['name'] == 'term_file':
            # Create a dropdown for term file selection
            form_fields.append(
                dbc.Col([
                    dbc.Label(arg['help']),
                    dcc.Dropdown(
                        id=f"term-tracking-{arg['name']}-input",
                        options=term_files,
                        value=term_files[0]['value'] if term_files else None,
                        placeholder=f"Sélectionnez un {arg['name']}"
                    )
                ], width=12)
            )
        elif arg['name'] in ['by_year', 'by_newspaper']:
            # Create a checkbox for boolean arguments
            form_fields.append(
                dbc.Col([
                    dbc.Checkbox(
                        id=f"term-tracking-{arg['name']}-input",
                        label=arg['help'],
                        value=arg['default']
                    )
                ], width=6)
            )
        else:
            # Create a standard input for other arguments
            form_fields.append(
                dbc.Col([
                    dbc.Label(arg['help']),
                    dbc.Input(
                        id=f"term-tracking-{arg['name']}-input",
                        type="number" if arg['type'] == 'int' else "text",
                        placeholder=f"Entrez {arg['name']}",
                        value=arg['default']
                    )
                ], width=6)
            )
    
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
                                dbc.Label("Sélectionnez un fichier de termes:"),
                                dcc.Dropdown(
                                    id="term-tracking-term_file-input",
                                    options=term_files,
                                    value=term_files[0]['value'] if term_files else None,
                                    placeholder="Sélectionnez un fichier de termes"
                                ),
                                html.Div(id="term-file-feedback", className="mt-2"),
                            ], width=12)
                        ]),
                        html.Hr(),
                    ]),
                    
                    # Autres paramètres du formulaire
                    html.H5("Options d'analyse"),
                    dbc.Form(dbc.Row([
                        field for field in form_fields if "term_file" not in str(field)
                    ])),
                    
                    html.Br(),
                    dbc.Button("Lancer l'analyse", id="run-term-tracking-button", color="primary"),
                    html.Br(),
                    html.Div(id="term-tracking-run-output")
                ], className="mt-3")
            ]),
            
            # Tab for viewing results
            dbc.Tab(label="Résultats", children=[
                html.Div([
                    html.H4("Visualisation des résultats"),
                    html.P("Sélectionnez un fichier de résultats pour visualiser l'analyse de suivi des termes."),
                    dbc.Row([
                        dbc.Col([
                            dbc.Label("Fichier de résultats:"),
                            dcc.Dropdown(
                                id="term-tracking-results-dropdown",
                                options=term_tracking_results,
                                value=term_tracking_results[0]['value'] if term_tracking_results else None,
                                placeholder="Sélectionnez un fichier de résultats"
                            )
                        ], width=6),
                        dbc.Col([
                            dbc.Label("Type de visualisation:"),
                            dcc.RadioItems(
                                id="term-tracking-viz-type",
                                options=[
                                    {"label": "Graphique à barres", "value": "bar"},
                                    {"label": "Graphique linéaire", "value": "line"},
                                    {"label": "Carte de chaleur", "value": "heatmap"},
                                    {"label": "Tableau", "value": "table"}
                                ],
                                value="bar",
                                inline=True
                            )
                        ], width=6)
                    ]),
                    html.Br(),
                    
                    # Bouton d'exportation
                    create_export_button(analysis_type="term_tracking", button_id="term-tracking-export-button"),
                    
                    # Results container
                    html.Div(id="term-tracking-results-container", children=[
                        # This will be populated by the callback
                    ])
                ], className="mt-3")
            ]),
            
            # Tab for creating a new term file
            dbc.Tab(label="Créer un fichier de termes", children=[
                html.Div([
                    html.H4("Créer un nouveau fichier de termes"),
                    html.P("Ajoutez des catégories et des termes pour créer un nouveau fichier de termes."),
                    
                    # Form for creating a new term file
                    dbc.Form([
                        dbc.Row([
                            dbc.Col([
                                dbc.Label("Nom du fichier:"),
                                dbc.Input(
                                    id="new-term-file-name",
                                    type="text",
                                    placeholder="mon_fichier_termes",
                                    value=""
                                )
                            ], width=6)
                        ]),
                        html.Br(),
                        
                        # Dynamic form for adding categories and terms
                        html.Div([
                            dbc.Row([
                                dbc.Col([
                                    dbc.Label("Catégorie 1:"),
                                    dbc.Input(
                                        id="term-category-1",
                                        type="text",
                                        placeholder="politique",
                                        value=""
                                    )
                                ], width=4),
                                dbc.Col([
                                    dbc.Label("Termes (séparés par des virgules):"),
                                    dbc.Textarea(
                                        id="term-list-1",
                                        placeholder="président, gouvernement, ministre",
                                        value="",
                                        style={"height": "100px"}
                                    )
                                ], width=8)
                            ]),
                            html.Br()
                        ], id="term-categories-container"),
                        
                        dbc.Button("Ajouter une catégorie", id="add-category-button", color="secondary", className="me-2"),
                        html.Br(),
                        html.Br(),
                        dbc.Button("Créer le fichier", id="create-term-file-button", color="primary"),
                        html.Br(),
                        html.Div(id="create-term-file-output")
                    ])
                ], className="mt-3")
            ])
        ])
    ])
    
    # Ajouter la modal d'exportation, le toast de feedback et la modal d'articles
    layout = html.Div([
        layout,
        create_export_modal(analysis_type="term_tracking", modal_id="term-tracking-export-modal"),
        create_feedback_toast(toast_id="term-tracking-export-toast"),
        create_articles_modal(),
        create_full_article_modal(),
        dcc.Store(id="term-tracking-last-result", data=None),  # Store pour le dernier fichier de résultats
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
            
        # Extraire la date et le journal
        try:
            date_part = article_id.split('_')[1]
            journal_part = article_id.split('_')[2]
            year_part = date_part.split('-')[0]
        except IndexError:
            continue
            
        # Appliquer le filtre
        if filter_type == 'année' and str(filter_value) == year_part:
            # Si un terme est spécifié, vérifier s'il est présent dans le texte
            if term:
                text = article.get('text', article.get('content', ''))
                if term.lower() in text.lower():
                    filtered_articles.append(article)
            else:
                filtered_articles.append(article)
                
        elif filter_type == 'journal' and filter_value.lower() == journal_part.lower():
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
        [Output("term-tracking-run-output", "children"),
         Output("term-tracking-last-result", "data")],  # Store pour le dernier fichier de résultats
        Input("run-term-tracking-button", "n_clicks"),
        [State("term-tracking-term_file-input", "value")] + 
        [State(f"term-tracking-{arg['name']}-input", "value") for arg in get_term_tracking_args() if arg['name'] != 'term_file'],
        prevent_initial_call=True
    )
    def run_term_tracking_analysis(n_clicks, term_file, *args):
        if not n_clicks:
            return "", None
        
        # Get the arguments
        arg_names = [arg['name'] for arg in get_term_tracking_args() if arg['name'] != 'term_file']
        arg_dict = dict(zip(arg_names, args))
        
        # Ajouter le fichier de termes aux arguments
        arg_dict['term_file'] = term_file
        
        # Check if term file is provided
        if not term_file:
            return html.Div([
                html.P("Erreur : Vous devez sélectionner un fichier de termes.", className="text-danger")
            ]), None
        
        # Vérifier si le fichier existe
        if not os.path.exists(term_file):
            return html.Div([
                html.P(f"Erreur : Le fichier {term_file} n'existe pas.", className="text-danger")
            ]), None
        
        # Build the command
        # Utiliser le même interpréteur Python que celui qui exécute l'application
        python_executable = sys.executable
        cmd = [python_executable, "-m", "src.scripts.run_term_tracking"]
        
        # Add arguments
        for name, value in arg_dict.items():
            if name in ['by_year', 'by_newspaper'] and value:
                cmd.append(f"--{name.replace('_', '-')}")
            elif value is not None and value != "":
                cmd.append(f"--{name.replace('_', '-')}")
                cmd.append(str(value))
        
        # Afficher la commande pour débogage
        print(f"Exécution de la commande avec l'interpréteur: {python_executable}")
        print(f"Commande complète: {' '.join(cmd)}")
        
        # Run the command
        try:
            project_root = pathlib.Path(__file__).resolve().parents[2]
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
            ]), results_file
            
        except subprocess.CalledProcessError as e:
            return html.Div([
                html.P("Erreur lors de l'exécution de l'analyse :", className="text-danger"),
                html.Pre(e.stderr)
            ]), None
    
    # Callback to update term tracking results dropdown
    @app.callback(
        Output("term-tracking-results-dropdown", "options"),
        Output("term-tracking-results-dropdown", "value"),
        Input("term-tracking-last-result", "data"),
        prevent_initial_call=True
    )
    def update_results_dropdown(last_result):
        # Get the updated list of result files
        result_options = get_term_tracking_results()
        
        # Set the value to the last result if available
        value = last_result if last_result and os.path.exists(last_result) else (result_options[0]['value'] if result_options else None)
        
        return result_options, value
    
    # Callback to update term tracking results visualization
    @app.callback(
        Output("term-tracking-results-container", "children"),
        [Input("term-tracking-results-dropdown", "value"),
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
        State("term-tracking-results-dropdown", "value"),
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
        results_file = ctx.states.get("term-tracking-results-dropdown.value")
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
        results_file = ctx.states.get("term-tracking-results-dropdown.value")
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
            State("term-tracking-results-dropdown", "value"),
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
