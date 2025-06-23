"""
Lightweight shell for the Dash application that loads quickly.
"""
import dash
import dash_bootstrap_components as dbc
from dash import html, dcc

def create_app_shell():
    """Create a lightweight app shell that loads quickly."""
    app = dash.Dash(
        __name__,
        external_stylesheets=[dbc.themes.DARKLY],
        meta_tags=[{"name": "viewport", "content": "width=device-width, initial-scale=1"}],
        suppress_callback_exceptions=True,
    )
    app.title = "Newspaper Articles Analysis"
    
    # Simple loading layout
    app.layout = dbc.Container([
        dcc.Store(id="app-loading-store", storage_type="memory"),
        dbc.Row([
            dbc.Col([
                html.H1("Newspaper Articles Analysis Dashboard", className="text-center mb-4"),
                html.Hr(),
                html.Div([
                    dbc.Spinner(
                        html.Div(id="loading-content", children=[
                            html.H3("Chargement de l'application...", className="text-center"),
                            html.P("Préparation des données et des modèles d'analyse...", className="text-center"),
                        ]),
                        size="lg",
                        color="primary",
                        type="border",
                        fullscreen=False,
                    ),
                ], className="text-center py-5"),
            ], width=12)
        ], className="mt-4"),
    ], fluid=True, className="px-4")
    
    return app
