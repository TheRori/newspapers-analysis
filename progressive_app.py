"""
Application Dash progressive pour identifier le module problématique.
"""

import os
import sys
from pathlib import Path

# Ajouter le répertoire racine du projet au chemin pour permettre les importations
project_root = Path(__file__).parent
sys.path.append(str(project_root))

import dash
from dash import html, dcc, ctx
import dash_bootstrap_components as dbc

# Charger la configuration
from src.utils.config_loader import load_config
config_path = str(project_root / "config" / "config.yaml")
config = load_config(config_path)

# Créer l'application Dash
app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.DARKLY],
    meta_tags=[{"name": "viewport", "content": "width=device-width, initial-scale=1"}],
    suppress_callback_exceptions=True,
)
app.title = "Newspaper Articles Analysis"
server = app.server

# Importer uniquement les modules lexical_analysis_viz, sentiment_analysis_viz et topic_modeling_viz
print("Importation du module lexical_analysis_viz...")
from src.webapp.lexical_analysis_viz import get_lexical_analysis_layout, register_lexical_analysis_callbacks
register_lexical_analysis_callbacks(app)
print("✓ Module lexical_analysis_viz importé et callbacks enregistrés")

print("Importation du module sentiment_analysis_viz...")
from src.webapp.sentiment_analysis_viz import get_sentiment_analysis_layout, register_sentiment_analysis_callbacks
register_sentiment_analysis_callbacks(app)
print("✓ Module sentiment_analysis_viz importé et callbacks enregistrés")

print("Importation du module topic_modeling_viz...")
from src.webapp.topic_modeling_viz import get_topic_modeling_layout, register_topic_modeling_callbacks
register_topic_modeling_callbacks(app)
print("✓ Module topic_modeling_viz importé et callbacks enregistrés")

print("Importation du module entity_recognition_viz...")
from src.webapp.entity_recognition_viz import get_entity_recognition_layout, register_entity_recognition_callbacks
register_entity_recognition_callbacks(app)
print("✓ Module entity_recognition_viz importé et callbacks enregistrés")

print("Importation du module export_manager_viz...")
from src.webapp.export_manager_viz import get_export_manager_layout, register_export_manager_callbacks
register_export_manager_callbacks(app)
print("✓ Module export_manager_viz importé et callbacks enregistrés")

print("Importation du module term_tracking_viz...")
from src.webapp.term_tracking_viz import get_term_tracking_layout, register_term_tracking_callbacks
register_term_tracking_callbacks(app)
print("✓ Module term_tracking_viz importé et callbacks enregistrés")

# Définir un layout avec plusieurs boutons
app.layout = dbc.Container([
    dbc.Row([
        dbc.Col([
            html.H1("Newspaper Articles Analysis Dashboard", className="text-center mb-4"),
            html.Div([
                dbc.Button("Lexical Analysis", id="btn-lexical", color="primary", className="me-2", n_clicks=0),
                dbc.Button("Sentiment Analysis", id="btn-sentiment", color="secondary", className="me-2", n_clicks=0),
                dbc.Button("Topic Modeling", id="btn-topic", color="info", className="me-2", n_clicks=0),
                dbc.Button("Entity Recognition", id="btn-entity", color="warning", className="me-2", n_clicks=0),
                dbc.Button("Term Tracking", id="btn-term-tracking", color="danger", className="me-2", n_clicks=0),
                dbc.Button("Export Manager", id="btn-export", color="success", className="me-2", n_clicks=0),
            ], className="text-center mb-4"),
        ], width=12)
    ], className="mt-4"),
    dbc.Row([
        dbc.Col(html.Div(id="page-content", className="p-4 rounded-3", style={"backgroundColor": "#23272b"}), width=12)
    ]),
    dbc.Row([
        dbc.Col(html.Footer("Newspaper Articles Analysis Dashboard - Created with Dash", className="text-center text-muted mb-4"), width=12)
    ]),
], fluid=True, className="px-4")

# Définir le callback principal multi-modules
@app.callback(
    dash.Output("page-content", "children"),
    [dash.Input("btn-lexical", "n_clicks"),
     dash.Input("btn-sentiment", "n_clicks"),
     dash.Input("btn-topic", "n_clicks"),
     dash.Input("btn-entity", "n_clicks"),
     dash.Input("btn-term-tracking", "n_clicks"),
     dash.Input("btn-export", "n_clicks")],
)
def display_page(btn_lexical, btn_sentiment, btn_topic, btn_entity, btn_term_tracking, btn_export):
    ctx = dash.callback_context
    print("ctx.triggered_id =", getattr(ctx, 'triggered_id', None))
    if not getattr(ctx, 'triggered_id', None):
        return get_lexical_analysis_layout()
    if ctx.triggered_id == "btn-sentiment":
        return get_sentiment_analysis_layout()
    elif ctx.triggered_id == "btn-topic":
        return get_topic_modeling_layout()
    elif ctx.triggered_id == "btn-entity":
        return get_entity_recognition_layout()
    elif ctx.triggered_id == "btn-term-tracking":
        return get_term_tracking_layout()
    elif ctx.triggered_id == "btn-export":
        return get_export_manager_layout()
    return get_lexical_analysis_layout()

# Lancer l'application
if __name__ == "__main__":
    print("Démarrage de l'application progressive multi-modules...")
    app.run(debug=False, port=8050)
    print("Application terminée.")
