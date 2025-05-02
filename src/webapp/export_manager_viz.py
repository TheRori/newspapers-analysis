"""
Export Manager Visualization Page for Dash app
Module pour gérer les analyses sauvegardées et les collections thématiques.
"""

from dash import html, dcc, Input, Output, State, ctx, dash_table
import dash_bootstrap_components as dbc
import dash
import plotly.express as px
import plotly.graph_objects as go
import subprocess
import importlib.util
import sys
import os
import inspect
import glob
import yaml
import pathlib
import pandas as pd
import numpy as np
import json
from typing import Dict, List, Any, Optional
import datetime

# Add the project root to the path to allow imports from other modules
project_root = pathlib.Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from src.utils.config_loader import load_config
from src.utils.export_utils import (
    load_saved_analyses, 
    get_analysis_details, 
    load_figure, 
    get_collections,
    create_collection,
    delete_analysis,
    export_collection_for_mediation
)

# Chemin vers le fichier de configuration
config_path = str(project_root / "config" / "config.yaml")

# Layout for the export manager page
def get_export_manager_layout():
    # Get available collections
    collections = get_collections(config=load_config(config_path))
    collection_options = [
        {'label': f"{c['name']} ({c['analyses_count']} analyses)", 'value': c['name']}
        for c in collections
    ]
    
    # Add an option for analyses without collection
    collection_options.insert(0, {'label': 'Toutes les analyses', 'value': ''})
    
    # Get all saved analyses
    analyses = load_saved_analyses(config=load_config(config_path))
    
    # Create the layout
    layout = html.Div([
        dbc.Row([
            dbc.Col([
                html.H2("Gestionnaire d'Exportations"),
                html.P("Gérez vos analyses sauvegardées et collections thématiques pour la médiation."),
                html.Hr()
            ], width=12)
        ]),
        
        # Tabs for Collections and Analyses
        dbc.Tabs([
            # Tab for managing collections
            dbc.Tab(label="Collections", children=[
                html.Div([
                    html.H4("Collections Thématiques"),
                    html.P("Créez et gérez des collections d'analyses pour la médiation."),
                    
                    # Form for creating a new collection
                    dbc.Card([
                        dbc.CardHeader("Créer une nouvelle collection"),
                        dbc.CardBody([
                            dbc.Row([
                                dbc.Col([
                                    dbc.Label("Nom de la collection:"),
                                    dbc.Input(
                                        id="collection-name-input",
                                        type="text",
                                        placeholder="ma_collection",
                                        value=""
                                    )
                                ], width=6),
                                dbc.Col([
                                    dbc.Label("Description:"),
                                    dbc.Textarea(
                                        id="collection-description-input",
                                        placeholder="Description de la collection...",
                                        value="",
                                        style={"height": "100px"}
                                    )
                                ], width=6)
                            ]),
                            html.Br(),
                            dbc.Button("Créer la collection", id="create-collection-button", color="primary")
                        ])
                    ]),
                    html.Br(),
                    
                    # Display existing collections
                    html.H5("Collections existantes"),
                    html.Div(id="collections-container", children=[
                        # This will be populated by the callback
                        html.Div([
                            dbc.Table(
                                # Header
                                [html.Tr([
                                    html.Th("Nom"),
                                    html.Th("Description"),
                                    html.Th("Date de création"),
                                    html.Th("Analyses"),
                                    html.Th("Actions")
                                ])] +
                                # Body
                                [html.Tr([
                                    html.Td(c['name']),
                                    html.Td(c['description']),
                                    html.Td(datetime.datetime.fromisoformat(c['created_at']).strftime('%Y-%m-%d %H:%M')),
                                    html.Td(c['analyses_count']),
                                    html.Td([
                                        dbc.Button("Voir", id=f"view-collection-{c['name']}", color="info", size="sm", className="me-2"),
                                        dbc.Button("Exporter", id=f"export-collection-{c['name']}", color="success", size="sm")
                                    ])
                                ]) for c in collections],
                                bordered=True,
                                hover=True,
                                responsive=True,
                                striped=True
                            ) if collections else html.P("Aucune collection trouvée.")
                        ])
                    ])
                ], className="mt-3")
            ]),
            
            # Tab for viewing saved analyses
            dbc.Tab(label="Analyses Sauvegardées", children=[
                html.Div([
                    html.H4("Analyses Sauvegardées"),
                    html.P("Consultez et gérez vos analyses sauvegardées."),
                    
                    # Filter controls
                    dbc.Row([
                        dbc.Col([
                            dbc.Label("Filtrer par collection:"),
                            dcc.Dropdown(
                                id="filter-collection-dropdown",
                                options=collection_options,
                                value="",
                                placeholder="Toutes les collections"
                            )
                        ], width=6),
                        dbc.Col([
                            dbc.Label("Filtrer par type d'analyse:"),
                            dcc.Dropdown(
                                id="filter-analysis-type-dropdown",
                                options=[
                                    {'label': 'Tous les types', 'value': ''},
                                    {'label': 'Suivi des Termes', 'value': 'term_tracking'},
                                    {'label': 'Topic Modeling', 'value': 'topic_modeling'},
                                    {'label': 'Clustering', 'value': 'clustering'},
                                    {'label': 'Analyse de Sentiment', 'value': 'sentiment'},
                                    {'label': 'Entités Nommées', 'value': 'entity_recognition'},
                                    {'label': 'Analyse Intégrée', 'value': 'integrated_analysis'}
                                ],
                                value="",
                                placeholder="Tous les types"
                            )
                        ], width=6)
                    ]),
                    html.Br(),
                    
                    # Display analyses
                    html.Div(id="analyses-container", children=[
                        # This will be populated by the callback
                        html.Div([
                            dbc.Table(
                                # Header
                                [html.Tr([
                                    html.Th("Titre"),
                                    html.Th("Type d'analyse"),
                                    html.Th("Date de création"),
                                    html.Th("Collection"),
                                    html.Th("Actions")
                                ])] +
                                # Body
                                [html.Tr([
                                    html.Td(a['title']),
                                    html.Td(a['analysis_type']),
                                    html.Td(datetime.datetime.fromisoformat(a['created_at']).strftime('%Y-%m-%d %H:%M')),
                                    html.Td(a['collection'] or '-'),
                                    html.Td([
                                        dbc.Button("Voir", id=f"view-analysis-{a['id']}", color="info", size="sm", className="me-2"),
                                        dbc.Button("Supprimer", id=f"delete-analysis-{a['id']}", color="danger", size="sm")
                                    ])
                                ]) for a in analyses],
                                bordered=True,
                                hover=True,
                                responsive=True,
                                striped=True
                            ) if analyses else html.P("Aucune analyse sauvegardée trouvée.")
                        ])
                    ])
                ], className="mt-3")
            ])
        ]),
        
        # Modal for viewing analysis details
        dbc.Modal(
            [
                dbc.ModalHeader(dbc.ModalTitle("Détails de l'analyse"), close_button=True),
                dbc.ModalBody(id="analysis-details-modal-body"),
                dbc.ModalFooter(
                    dbc.Button("Fermer", id="close-analysis-modal", className="ms-auto", n_clicks=0)
                ),
            ],
            id="analysis-details-modal",
            size="xl",
            is_open=False,
        ),
        
        # Store for selected analysis ID
        dcc.Store(id="selected-analysis-id", data=""),
        
        # Store for operation feedback
        dcc.Store(id="operation-feedback", data=""),
        
        # Toast for feedback messages
        dbc.Toast(
            id="feedback-toast",
            header="Notification",
            is_open=False,
            dismissable=True,
            duration=4000,
            style={"position": "fixed", "top": 66, "right": 10, "width": 350},
        )
    ])
    
    return layout

# Function to create analysis details view
def create_analysis_details_view(analysis_id):
    """
    Create a detailed view of an analysis.
    
    Args:
        analysis_id: ID of the analysis to view
        
    Returns:
        HTML div with analysis details
    """
    try:
        # Get analysis details
        details = get_analysis_details(analysis_id, config=load_config(config_path))
        
        # Try to load the figure if available
        figure_html = None
        if "figure_html" in details and os.path.exists(details["figure_html"]):
            with open(details["figure_html"], 'r', encoding='utf-8') as f:
                figure_html = f.read()
        
        # Create metadata table
        metadata_rows = []
        for key, value in details.items():
            if key not in ["id", "figure_html", "figure_json", "source_data"]:
                if isinstance(value, dict) or isinstance(value, list):
                    value = json.dumps(value, ensure_ascii=False, indent=2)
                metadata_rows.append(html.Tr([html.Td(key), html.Td(str(value))]))
        
        # Create source data table
        source_data_rows = []
        for key, value in details.get("source_data", {}).items():
            if isinstance(value, dict) or isinstance(value, list):
                value = json.dumps(value, ensure_ascii=False, indent=2)
            source_data_rows.append(html.Tr([html.Td(key), html.Td(str(value))]))
        
        # Create the details view
        details_view = html.Div([
            html.H3(details["title"]),
            html.P(details["description"]),
            html.Hr(),
            
            # Metadata
            html.H5("Métadonnées"),
            dbc.Table(
                metadata_rows,
                bordered=True,
                hover=True,
                responsive=True,
                striped=True,
                size="sm"
            ),
            html.Hr(),
            
            # Source data
            html.H5("Données source"),
            dbc.Table(
                source_data_rows,
                bordered=True,
                hover=True,
                responsive=True,
                striped=True,
                size="sm"
            ),
            html.Hr(),
            
            # Figure
            html.H5("Visualisation"),
            html.Div([
                html.Iframe(
                    srcDoc=figure_html,
                    style={"width": "100%", "height": "600px", "border": "none"}
                )
            ]) if figure_html else html.P("Aucune visualisation disponible pour cette analyse.")
        ])
        
        return details_view
        
    except Exception as e:
        return html.Div([
            html.H4("Erreur lors du chargement des détails"),
            html.P(f"Erreur : {str(e)}")
        ])

# Callback registration (to be called from app.py)
def register_export_manager_callbacks(app):
    """Register callbacks for the export manager page."""
    
    # Callback to create a new collection
    @app.callback(
        Output("collections-container", "children"),
        Output("operation-feedback", "data"),
        Input("create-collection-button", "n_clicks"),
        State("collection-name-input", "value"),
        State("collection-description-input", "value"),
        prevent_initial_call=True
    )
    def create_new_collection(n_clicks, name, description):
        if not n_clicks:
            return dash.no_update, ""
        
        if not name:
            return dash.no_update, "Erreur : Vous devez spécifier un nom de collection."
        
        try:
            # Create the collection
            create_collection(name, description or "", config=load_config(config_path))
            
            # Refresh the collections list
            collections = get_collections(config=load_config(config_path))
            
            collections_table = dbc.Table(
                # Header
                [html.Tr([
                    html.Th("Nom"),
                    html.Th("Description"),
                    html.Th("Date de création"),
                    html.Th("Analyses"),
                    html.Th("Actions")
                ])] +
                # Body
                [html.Tr([
                    html.Td(c['name']),
                    html.Td(c['description']),
                    html.Td(datetime.datetime.fromisoformat(c['created_at']).strftime('%Y-%m-%d %H:%M')),
                    html.Td(c['analyses_count']),
                    html.Td([
                        dbc.Button("Voir", id=f"view-collection-{c['name']}", color="info", size="sm", className="me-2"),
                        dbc.Button("Exporter", id=f"export-collection-{c['name']}", color="success", size="sm")
                    ])
                ]) for c in collections],
                bordered=True,
                hover=True,
                responsive=True,
                striped=True
            ) if collections else html.P("Aucune collection trouvée.")
            
            return html.Div([collections_table]), "Collection créée avec succès !"
            
        except Exception as e:
            return dash.no_update, f"Erreur lors de la création de la collection : {str(e)}"
    
    # Callback to filter analyses
    @app.callback(
        Output("analyses-container", "children"),
        [Input("filter-collection-dropdown", "value"),
         Input("filter-analysis-type-dropdown", "value")],
        prevent_initial_call=True
    )
    def filter_analyses(collection, analysis_type):
        # Load analyses with filters
        analyses = load_saved_analyses(collection=collection if collection else None, 
                                      analysis_type=analysis_type if analysis_type else None, 
                                      config=load_config(config_path))
        
        analyses_table = dbc.Table(
            # Header
            [html.Tr([
                html.Th("Titre"),
                html.Th("Type d'analyse"),
                html.Th("Date de création"),
                html.Th("Collection"),
                html.Th("Actions")
            ])] +
            # Body
            [html.Tr([
                html.Td(a['title']),
                html.Td(a['analysis_type']),
                html.Td(datetime.datetime.fromisoformat(a['created_at']).strftime('%Y-%m-%d %H:%M')),
                html.Td(a['collection'] or '-'),
                html.Td([
                    dbc.Button("Voir", id={"type": "view-analysis-btn", "index": a['id']}, color="info", size="sm", className="me-2"),
                    dbc.Button("Supprimer", id={"type": "delete-analysis-btn", "index": a['id']}, color="danger", size="sm")
                ])
            ]) for a in analyses],
            bordered=True,
            hover=True,
            responsive=True,
            striped=True
        ) if analyses else html.P("Aucune analyse trouvée correspondant aux critères.")
        
        return html.Div([analyses_table])
    
    # Callback to view analysis details
    @app.callback(
        Output("analysis-details-modal", "is_open"),
        Output("analysis-details-modal-body", "children"),
        Output("selected-analysis-id", "data"),
        [Input({"type": "view-analysis-btn", "index": dash.dependencies.ALL}, "n_clicks")],
        [State("analysis-details-modal", "is_open"),
         State("selected-analysis-id", "data")],
        prevent_initial_call=True
    )
    def toggle_analysis_modal(n_clicks_list, is_open, selected_id):
        ctx_triggered = dash.callback_context.triggered
        
        if not ctx_triggered or not any(n_clicks_list):
            return is_open, dash.no_update, selected_id
        
        # Get the ID of the clicked button
        button_id = ctx_triggered[0]['prop_id'].split('.')[0]
        analysis_id = json.loads(button_id)['index']
        
        # If the modal is already open and showing the same analysis, close it
        if is_open and selected_id == analysis_id:
            return False, dash.no_update, ""
        
        # Otherwise, open the modal with the selected analysis
        details_view = create_analysis_details_view(analysis_id)
        return True, details_view, analysis_id
    
    # Callback to close the analysis modal
    @app.callback(
        Output("analysis-details-modal", "is_open", allow_duplicate=True),
        Input("close-analysis-modal", "n_clicks"),
        State("analysis-details-modal", "is_open"),
        prevent_initial_call=True
    )
    def close_analysis_modal(n_clicks, is_open):
        if n_clicks:
            return False
        return is_open
    
    # Callback to delete an analysis
    @app.callback(
        Output("analyses-container", "children", allow_duplicate=True),
        Output("operation-feedback", "data", allow_duplicate=True),
        [Input({"type": "delete-analysis-btn", "index": dash.dependencies.ALL}, "n_clicks")],
        [State("filter-collection-dropdown", "value"),
         State("filter-analysis-type-dropdown", "value")],
        prevent_initial_call=True
    )
    def delete_analysis_callback(n_clicks_list, collection, analysis_type):
        ctx_triggered = dash.callback_context.triggered
        
        if not ctx_triggered or not any(n_clicks_list):
            return dash.no_update, ""
        
        # Get the ID of the clicked button
        button_id = ctx_triggered[0]['prop_id'].split('.')[0]
        analysis_id = json.loads(button_id)['index']
        
        try:
            # Delete the analysis
            success = delete_analysis(analysis_id, config=load_config(config_path))
            
            if not success:
                return dash.no_update, "Erreur : Impossible de supprimer l'analyse."
            
            # Refresh the analyses list
            analyses = load_saved_analyses(collection=collection if collection else None, 
                                          analysis_type=analysis_type if analysis_type else None, 
                                          config=load_config(config_path))
            
            analyses_table = dbc.Table(
                # Header
                [html.Tr([
                    html.Th("Titre"),
                    html.Th("Type d'analyse"),
                    html.Th("Date de création"),
                    html.Th("Collection"),
                    html.Th("Actions")
                ])] +
                # Body
                [html.Tr([
                    html.Td(a['title']),
                    html.Td(a['analysis_type']),
                    html.Td(datetime.datetime.fromisoformat(a['created_at']).strftime('%Y-%m-%d %H:%M')),
                    html.Td(a['collection'] or '-'),
                    html.Td([
                        dbc.Button("Voir", id={"type": "view-analysis-btn", "index": a['id']}, color="info", size="sm", className="me-2"),
                        dbc.Button("Supprimer", id={"type": "delete-analysis-btn", "index": a['id']}, color="danger", size="sm")
                    ])
                ]) for a in analyses],
                bordered=True,
                hover=True,
                responsive=True,
                striped=True
            ) if analyses else html.P("Aucune analyse trouvée correspondant aux critères.")
            
            return html.Div([analyses_table]), "Analyse supprimée avec succès !"
            
        except Exception as e:
            return dash.no_update, f"Erreur lors de la suppression de l'analyse : {str(e)}"
    
    # Callback to show feedback toast
    @app.callback(
        Output("feedback-toast", "is_open"),
        Output("feedback-toast", "header"),
        Output("feedback-toast", "children"),
        Input("operation-feedback", "data"),
        prevent_initial_call=True
    )
    def show_feedback_toast(feedback):
        if not feedback:
            return False, "", ""
        
        if feedback.startswith("Erreur"):
            header = "Erreur"
            color = "danger"
        else:
            header = "Succès"
            color = "success"
        
        return True, header, html.Div(feedback, className=f"text-{color}")
