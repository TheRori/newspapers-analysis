"""
Export Component for Dash app
Composant réutilisable pour ajouter la fonctionnalité d'exportation aux pages de visualisation.
"""

from dash import html, dcc, Input, Output, State, ctx
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
import json
from typing import Dict, List, Any, Optional
import datetime
import sys
from pathlib import Path

# Add the project root to the path to allow imports from other modules
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from src.utils.config_loader import load_config
from src.utils.export_utils import save_analysis, get_collections

# Chemin vers le fichier de configuration
config_path = str(project_root / "config" / "config.yaml")

def create_export_button(analysis_type: str, button_id: str = "export-button"):
    """
    Crée un bouton d'exportation pour une page de visualisation.
    
    Args:
        analysis_type: Type d'analyse (term_tracking, topic_modeling, etc.)
        button_id: ID du bouton (optionnel)
        
    Returns:
        Bouton d'exportation
    """
    return dbc.Button(
        "📌 Ajouter à la médiation", 
        id=button_id, 
        color="success", 
        className="mt-2",
        style={"marginBottom": "10px"}
    )

def create_export_modal(analysis_type: str, modal_id: str = "export-modal"):
    """
    Crée une modal d'exportation pour une page de visualisation.
    
    Args:
        analysis_type: Type d'analyse (term_tracking, topic_modeling, etc.)
        modal_id: ID de la modal (optionnel)
        
    Returns:
        Modal d'exportation
    """
    # Get available collections
    collections = get_collections(config=load_config(config_path))
    collection_options = [
        {'label': f"{c['name']} ({c['analyses_count']} analyses)", 'value': c['name']}
        for c in collections
    ]
    
    # Add option for no collection
    collection_options.insert(0, {'label': 'Aucune collection', 'value': ''})
    
    return dbc.Modal(
        [
            dbc.ModalHeader(dbc.ModalTitle("Ajouter à la médiation"), close_button=True),
            dbc.ModalBody([
                html.P("Enregistrez cette analyse pour la réutiliser dans votre interface de médiation."),
                dbc.Form([
                    dbc.Row([
                        dbc.Col([
                            dbc.Label("Titre:"),
                            dbc.Input(
                                id=f"{modal_id}-title-input",
                                type="text",
                                placeholder="Titre de l'analyse",
                                value=""
                            )
                        ], width=12)
                    ]),
                    html.Br(),
                    dbc.Row([
                        dbc.Col([
                            dbc.Label("Description:"),
                            dbc.Textarea(
                                id=f"{modal_id}-description-input",
                                placeholder="Description détaillée de l'analyse...",
                                value="",
                                style={"height": "100px"}
                            )
                        ], width=12)
                    ]),
                    html.Br(),
                    dbc.Row([
                        dbc.Col([
                            dbc.Label("Collection:"),
                            dcc.Dropdown(
                                id=f"{modal_id}-collection-dropdown",
                                options=collection_options,
                                value="",
                                placeholder="Sélectionnez une collection"
                            )
                        ], width=12)
                    ]),
                    html.Br(),
                    dbc.Row([
                        dbc.Col([
                            dbc.Label("Créer une nouvelle collection:"),
                            dbc.Input(
                                id=f"{modal_id}-new-collection-input",
                                type="text",
                                placeholder="Nom de la nouvelle collection (optionnel)",
                                value=""
                            )
                        ], width=12)
                    ])
                ])
            ]),
            dbc.ModalFooter([
                dbc.Button("Annuler", id=f"{modal_id}-cancel-button", className="me-2"),
                dbc.Button("Enregistrer", id=f"{modal_id}-save-button", color="success")
            ]),
            # Store pour les données source
            dcc.Store(id=f"{modal_id}-source-data", data={}),
            # Store pour la figure
            dcc.Store(id=f"{modal_id}-figure-data", data={}),
            # Store pour le feedback
            dcc.Store(id=f"{modal_id}-feedback", data="")
        ],
        id=modal_id,
        size="lg",
        is_open=False,
    )

def create_feedback_toast(toast_id: str = "export-feedback-toast"):
    """
    Crée un toast pour afficher les messages de feedback.
    
    Args:
        toast_id: ID du toast (optionnel)
        
    Returns:
        Toast de feedback
    """
    return dbc.Toast(
        id=toast_id,
        header="Notification",
        is_open=False,
        dismissable=True,
        duration=4000,
        style={"position": "fixed", "top": 66, "right": 10, "width": 350},
    )

def register_export_callbacks(
    app,
    analysis_type: str,
    get_source_data_function,
    get_figure_function,
    button_id: str = "export-button",
    modal_id: str = "export-modal",
    toast_id: str = "export-feedback-toast"
):
    """
    Enregistre les callbacks pour l'exportation.
    
    Args:
        app: Application Dash
        analysis_type: Type d'analyse (term_tracking, topic_modeling, etc.)
        get_source_data_function: Fonction pour obtenir les données source
        get_figure_function: Fonction pour obtenir la figure
        button_id: ID du bouton d'exportation (optionnel)
        modal_id: ID de la modal d'exportation (optionnel)
        toast_id: ID du toast de feedback (optionnel)
    """
    # Callback pour ouvrir la modal d'exportation
    @app.callback(
        Output(modal_id, "is_open"),
        Output(f"{modal_id}-source-data", "data"),
        Output(f"{modal_id}-figure-data", "data"),
        Input(button_id, "n_clicks"),
        prevent_initial_call=True
    )
    def open_export_modal(n_clicks):
        if not n_clicks:
            return False, {}, {}
        
        # Obtenir les données source et la figure
        source_data = get_source_data_function()
        figure_data = get_figure_function()
        
        return True, source_data, figure_data
    
    # Callback pour fermer la modal d'exportation
    @app.callback(
        Output(modal_id, "is_open", allow_duplicate=True),
        Input(f"{modal_id}-cancel-button", "n_clicks"),
        State(modal_id, "is_open"),
        prevent_initial_call=True
    )
    def close_export_modal(n_clicks, is_open):
        if n_clicks:
            return False
        return is_open
    
    # Callback pour enregistrer l'analyse
    @app.callback(
        Output(modal_id, "is_open", allow_duplicate=True),
        Output(f"{modal_id}-feedback", "data"),
        Input(f"{modal_id}-save-button", "n_clicks"),
        [State(f"{modal_id}-title-input", "value"),
         State(f"{modal_id}-description-input", "value"),
         State(f"{modal_id}-collection-dropdown", "value"),
         State(f"{modal_id}-new-collection-input", "value"),
         State(f"{modal_id}-source-data", "data"),
         State(f"{modal_id}-figure-data", "data")],
        prevent_initial_call=True
    )
    def save_analysis_callback(n_clicks, title, description, collection, new_collection, source_data, figure_data):
        if not n_clicks:
            return False, ""
        
        if not title:
            return True, "Erreur : Vous devez spécifier un titre."
        
        try:
            # Si une nouvelle collection est spécifiée, l'utiliser
            if new_collection:
                from src.utils.export_utils import create_collection
                create_collection(new_collection, f"Collection créée depuis {analysis_type}", config=load_config(config_path))
                collection = new_collection
            
            # Convertir la figure JSON en objet Figure
            figure = None
            if figure_data:
                import plotly.io as pio
                figure = pio.from_json(json.dumps(figure_data))
            
            # Enregistrer l'analyse
            from src.utils.export_utils import save_analysis
            metadata = save_analysis(
                title=title,
                description=description,
                source_data=source_data,
                analysis_type=analysis_type,
                figure=figure,
                collection=collection,
                config=load_config(config_path),
                save_source_files=True
            )
            
            return False, "Analyse enregistrée avec succès !"
            
        except Exception as e:
            return True, f"Erreur lors de l'enregistrement : {str(e)}"
    
    # Callback pour afficher le toast de feedback
    @app.callback(
        Output(toast_id, "is_open"),
        Output(toast_id, "header"),
        Output(toast_id, "children"),
        Input(f"{modal_id}-feedback", "data"),
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
