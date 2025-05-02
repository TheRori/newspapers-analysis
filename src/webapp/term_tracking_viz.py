"""
Term Tracking Visualization Page for Dash app
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
from typing import Dict, List, Any, Optional

# Add the project root to the path to allow imports from other modules
project_root = pathlib.Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from src.utils.config_loader import load_config
from src.analysis.term_tracking import count_term_occurrences, count_terms_by_year, count_terms_by_newspaper

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
        term_files.extend(list(examples_dir.glob('*terms*.json')))
    
    # Check processed directory
    config_path = project_root / 'config' / 'config.yaml'
    with open(config_path, encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    processed_dir = project_root / config['data']['processed_dir']
    if processed_dir.exists():
        term_files.extend(list(processed_dir.glob('*terms*.json')))
    
    # Sort by modification time (newest first)
    term_files.sort(key=lambda x: os.path.getmtime(x), reverse=True)
    
    # Format for dropdown
    options = [
        {'label': f"{f.stem} ({pd.to_datetime(os.path.getmtime(f), unit='s').strftime('%Y-%m-%d %H:%M')})", 
         'value': str(f)}
        for f in term_files
    ]
    
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
                        placeholder="Sélectionnez un fichier de termes"
                    ),
                    html.Br()
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
                        id=f"term-tracking-{arg['name']}-input",
                        type="text" if arg['type'] == 'str' else "number",
                        placeholder=f"Default: {arg['default']}" if arg['default'] is not None else "",
                        value=arg['default'] if arg['default'] is not None else ""
                    ),
                    html.Br()
                ], width=6)
            )
    
    # Create the layout
    layout = html.Div([
        dbc.Row([
            dbc.Col([
                html.H2("Suivi des Termes"),
                html.P("Analyser la fréquence et la distribution de termes spécifiques dans le corpus."),
                html.Hr()
            ], width=12)
        ]),
        
        # Tabs for Run and Results
        dbc.Tabs([
            # Tab for running term tracking analysis
            dbc.Tab(label="Lancer une analyse", children=[
                html.Div([
                    html.H4("Paramètres d'analyse"),
                    html.P("Configurez les paramètres pour l'analyse de suivi des termes."),
                    dbc.Form(dbc.Row(form_fields)),
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
    
    return layout

# Function to create term tracking visualizations
def create_term_tracking_visualizations(results_file_path, viz_type="bar"):
    """
    Create visualizations for term tracking results.
    
    Args:
        results_file_path: Path to the results file
        viz_type: Type of visualization (bar, line, heatmap, table)
        
    Returns:
        HTML div with visualizations
    """
    try:
        # Load the results file
        df = pd.read_csv(results_file_path)
        
        # Check if the file is empty
        if df.empty:
            return html.Div([
                html.H4("Aucun résultat à afficher"),
                html.P("Le fichier de résultats est vide.")
            ])
        
        # Determine the type of results based on the columns
        key_column = df.columns[0]
        
        # Rename the key column based on the type of results
        if key_column == 'key':
            # Try to determine the type of key
            first_key = df['key'].iloc[0]
            if isinstance(first_key, str) and len(first_key) > 20:
                # Likely an article ID
                key_type = "Article"
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
            
            graph = dcc.Graph(figure=fig)
            
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
                graph = dcc.Graph(figure=fig)
            else:
                graph = html.Div([
                    html.H4("Visualisation non adaptée"),
                    html.P("Le graphique linéaire n'est adapté qu'aux données temporelles.")
                ])
                
        elif viz_type == "heatmap":
            # Heatmap
            if len(term_columns) > 1:
                # Transpose the dataframe for the heatmap
                heatmap_df = df.set_index(df.columns[0])
                
                fig = px.imshow(
                    heatmap_df.T,
                    title=f"Carte de chaleur des fréquences de termes",
                    labels={'x': key_type, 'y': 'Terme', 'color': 'Fréquence'},
                    color_continuous_scale='Viridis'
                )
                graph = dcc.Graph(figure=fig)
            else:
                graph = html.Div([
                    html.H4("Données insuffisantes"),
                    html.P("Une carte de chaleur nécessite plusieurs termes.")
                ])
                
        elif viz_type == "table":
            # Table
            table = dbc.Table.from_dataframe(
                df, 
                striped=True, 
                bordered=True, 
                hover=True,
                responsive=True
            )
            graph = html.Div([
                html.H4(f"Tableau des fréquences de termes"),
                table
            ])
        
        # Create summary statistics
        total_occurrences = df[term_columns].sum().sum()
        most_frequent_term = df[term_columns].sum().idxmax()
        most_frequent_count = df[term_columns].sum().max()
        
        if key_type == "Année":
            year_with_most = df.loc[df[term_columns].sum(axis=1).idxmax(), 'Année']
            summary_text = f"Année avec le plus d'occurrences : {year_with_most}"
        elif key_type == "Journal":
            newspaper_with_most = df.loc[df[term_columns].sum(axis=1).idxmax(), 'Journal']
            summary_text = f"Journal avec le plus d'occurrences : {newspaper_with_most}"
        else:
            summary_text = f"Nombre d'articles contenant au moins un terme : {len(df)}"
        
        # Create the results container
        results_container = html.Div([
            html.H4("Résumé"),
            html.P([
                f"Nombre total d'occurrences : {total_occurrences}",
                html.Br(),
                f"Terme le plus fréquent : {most_frequent_term} ({most_frequent_count} occurrences)",
                html.Br(),
                summary_text
            ]),
            html.Hr(),
            html.H4("Visualisation"),
            graph
        ])
        
        return results_container
        
    except Exception as e:
        return html.Div([
            html.H4("Erreur lors du chargement des résultats"),
            html.P(f"Erreur : {str(e)}")
        ])

# Callback registration (to be called from app.py)
def register_term_tracking_callbacks(app):
    """Register callbacks for the term tracking page."""
    
    # Callback to run term tracking analysis
    @app.callback(
        Output("term-tracking-run-output", "children"),
        Input("run-term-tracking-button", "n_clicks"),
        [State(f"term-tracking-{arg['name']}-input", "value") for arg in get_term_tracking_args()],
        prevent_initial_call=True
    )
    def run_term_tracking_analysis(n_clicks, *args):
        if not n_clicks:
            return ""
        
        # Get the arguments
        arg_names = [arg['name'] for arg in get_term_tracking_args()]
        arg_dict = dict(zip(arg_names, args))
        
        # Check if term file is provided
        if not arg_dict.get('term_file'):
            return html.Div([
                html.P("Erreur : Vous devez sélectionner un fichier de termes.", className="text-danger")
            ])
        
        # Build the command
        cmd = ["python", os.path.join(os.path.dirname(__file__), "..", "scripts", "run_term_tracking.py")]
        
        # Add arguments
        for name, value in arg_dict.items():
            if name in ['by_year', 'by_newspaper'] and value:
                cmd.append(f"--{name.replace('_', '-')}")
            elif value is not None and value != "":
                cmd.append(f"--{name.replace('_', '-')}")
                cmd.append(str(value))
        
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
            ])
            
        except subprocess.CalledProcessError as e:
            return html.Div([
                html.P("Erreur lors de l'exécution de l'analyse :", className="text-danger"),
                html.Pre(e.stderr)
            ])
    
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
    def create_term_file(n_clicks, file_name, category_children):
        if not n_clicks:
            return ""
        
        if not file_name:
            return html.P("Erreur : Vous devez spécifier un nom de fichier.", className="text-danger")
        
        # Get the number of categories
        num_categories = len(category_children) // 2  # Each category has a row and a br
        
        # Collect category data
        term_dict = {}
        for i in range(1, num_categories + 1):
            category_id = f"term-category-{i}"
            terms_id = f"term-list-{i}"
            
            # Find the category and terms inputs in the children
            category_value = None
            terms_value = None
            
            for child in category_children:
                if isinstance(child, dict) and 'props' in child and 'children' in child['props']:
                    row = child['props']['children']
                    if isinstance(row, list) and len(row) >= 2:
                        for col in row:
                            if isinstance(col, dict) and 'props' in col and 'children' in col['props']:
                                col_children = col['props']['children']
                                if isinstance(col_children, list):
                                    for item in col_children:
                                        if isinstance(item, dict) and 'props' in item and 'id' in item['props']:
                                            if item['props']['id'] == category_id:
                                                category_value = item['props'].get('value', '')
                                            elif item['props']['id'] == terms_id:
                                                terms_value = item['props'].get('value', '')
            
            if category_value and terms_value:
                # Split terms by comma and strip whitespace
                terms_list = [term.strip() for term in terms_value.split(',') if term.strip()]
                if terms_list:
                    term_dict[category_value] = terms_list
        
        if not term_dict:
            return html.P("Erreur : Vous devez spécifier au moins une catégorie avec des termes.", className="text-danger")
        
        try:
            # Create the file in the examples directory
            project_root = pathlib.Path(__file__).resolve().parents[2]
            examples_dir = project_root / 'examples'
            examples_dir.mkdir(exist_ok=True)
            
            file_path = examples_dir / f"{file_name}.json"
            
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(term_dict, f, ensure_ascii=False, indent=2)
            
            return html.Div([
                html.P(f"Fichier de termes créé avec succès : {file_path}", className="text-success"),
                html.P("Vous pouvez maintenant l'utiliser pour l'analyse de suivi des termes.", className="mt-2")
            ])
            
        except Exception as e:
            return html.P(f"Erreur lors de la création du fichier : {str(e)}", className="text-danger")
