"""
Module to initialize the full app after the server has started.
"""
import os
import sys
import logging
from pathlib import Path
import pandas as pd
import plotly.io as pio

def initialize_app(app):
    """Initialize the full app with all components after server has started."""
    from dash import dcc, html, Input, Output, State, ctx
    import dash_bootstrap_components as dbc
    import plotly.express as px
    import plotly.graph_objects as go
    import numpy as np

    # Add the project root to the path to allow imports from other modules
    project_root = Path(__file__).parent.parent.parent
    sys.path.append(str(project_root))

    # Configuration globale pour les graphiques Plotly avec thème sombre
    pio.templates.default = "plotly_dark"
    plotly_config = {
        'layout': {
            'paper_bgcolor': '#222',
            'plot_bgcolor': '#222',
            'font': {'color': '#fff'},
            'xaxis': {'gridcolor': '#444', 'zerolinecolor': '#444'},
            'yaxis': {'gridcolor': '#444', 'zerolinecolor': '#444'}
        }
    }
    pio.templates['plotly_dark'].layout.update(plotly_config['layout'])

    # Import all the necessary modules
    from src.visualization.visualizer import Visualizer
    from src.utils.config_loader import load_config
    from src.webapp.lexical_analysis_viz import get_lexical_analysis_layout, register_lexical_analysis_callbacks
    from src.webapp.topic_modeling_viz import get_topic_modeling_layout, register_topic_modeling_callbacks
    from src.webapp.topic_clustering_viz import get_clustering_layout, get_clustering_args, register_clustering_callbacks
    from src.webapp.cluster_map_viz import get_cluster_map_layout, register_cluster_map_callbacks
    from src.webapp.sentiment_analysis_viz import get_sentiment_analysis_layout, register_sentiment_analysis_callbacks
    from src.webapp.entity_recognition_viz import get_entity_recognition_layout, register_entity_recognition_callbacks
    from src.webapp.integrated_analysis_viz import get_integrated_analysis_layout, register_integrated_analysis_callbacks
    from src.webapp.term_tracking_viz import get_term_tracking_layout, register_term_tracking_callbacks
    from src.webapp.export_manager_viz import get_export_manager_layout, register_export_manager_callbacks
    from src.webapp.source_manager_viz import get_source_manager_layout, register_source_manager_callbacks
    from src.webapp.article_library_viz import get_article_library_layout, register_article_library_callbacks
    from src.webapp.home_guide_viz import get_enhanced_home_layout
    from src.webapp.data_provider import DashDataProvider

    # Configuration de la journalisation
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger("webapp")

    # Load configuration
    logger.info("Chargement du fichier de configuration...")
    config_path = str(project_root / "config" / "config.yaml")
    config = load_config(config_path)
    logger.info(f"Configuration chargée : {config_path}")

    # --- Génération et chargement du CSV biblio enrichi ---
    provider = DashDataProvider()
    provider.export_biblio_csv()  # Met à jour/exporte le CSV à chaque démarrage
    biblio_csv_path = project_root / "data" / "biblio_enriched.csv"
    logger.info(f"Chargement du CSV biblio enrichi : {biblio_csv_path}")
    biblio_df = pd.read_csv(biblio_csv_path)

    # Register callbacks for lexical analysis form
    logger.info("Enregistrement des callbacks...")
    register_lexical_analysis_callbacks(app)
    logger.info("Callbacks lexical_analysis enregistrés.")
    register_topic_modeling_callbacks(app)
    logger.info("Callbacks topic_modeling enregistrés.")
    register_clustering_callbacks(app)
    logger.info("Callbacks clustering enregistrés.")
    register_cluster_map_callbacks(app)
    logger.info("Callbacks cluster_map enregistrés.")
    register_sentiment_analysis_callbacks(app)
    logger.info("Callbacks sentiment_analysis enregistrés.")
    register_entity_recognition_callbacks(app)
    logger.info("Callbacks entity_recognition enregistrés.")
    register_integrated_analysis_callbacks(app)
    logger.info("Callbacks integrated_analysis enregistrés.")
    register_term_tracking_callbacks(app)
    logger.info("Callbacks term_tracking enregistrés.")
    register_export_manager_callbacks(app)
    logger.info("Callbacks export_manager enregistrés.")
    register_source_manager_callbacks(app)
    logger.info("Callbacks source_manager enregistrés.")
    register_article_library_callbacks(app)

    # Define the app layout (replace loading screen)
    app.layout = dbc.Container([
        dcc.Store(id="global-analysis-state", storage_type="session"),
        dbc.Toast(
            id="global-analysis-notification",
            header="Analyse en cours",
            icon="info",
            is_open=False,
            duration=None,
            dismissable=False,
            style={
                "position": "fixed",
                "top": 80,
                "right": 30,
                "minWidth": 350,
                "zIndex": 2000,
                "backgroundColor": "#222",
                "color": "#fff",
            },
            children="L'analyse est en cours. Vous pouvez continuer à naviguer."
        ),
        dbc.Row([
            dbc.Col([
                html.H1("Newspaper Articles Analysis Dashboard", className="text-center mb-4"),
                html.Hr(),
                html.Div([
                    dbc.Button("🏠", id="btn-home", color="light", className="me-2", n_clicks=0, title="Accueil"),
                    dbc.Button("Commencer l'exploration", id="btn-home-explore", n_clicks=0, style={"display": "none"}),
                    dbc.Button("Bibliothèque d'articles", id="btn-article-library", color="dark", className="me-2", n_clicks=0),
                    dbc.Button("Gestion Source", id="btn-source-manager", color="primary", className="me-2", n_clicks=0),
                    dbc.Button("Topic Modeling", id="btn-topic", color="secondary", className="me-2", n_clicks=0),
                    dbc.Button("Clustering", id="btn-clustering", color="info", className="me-2", n_clicks=0),
                    dbc.Button("Analyse de Sentiment", id="btn-sentiment", color="warning", className="me-2", n_clicks=0),
                    dbc.Button("Entités Nommées", id="btn-entity", color="danger", className="me-2", n_clicks=0),
                    dbc.Button("Analyse Intégrée", id="btn-integrated", color="primary", className="me-2", n_clicks=0),
                    dbc.Button("Suivi des Termes", id="btn-term-tracking", color="secondary", className="me-2", n_clicks=0),
                    dbc.Button("Médiation 📌", id="btn-export-manager", color="success", n_clicks=0),
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

    # Register the main navigation and other callbacks as in your original app.py
    @app.callback(
        Output("page-content", "children"),
        Input("btn-home", "n_clicks"),
        Input("btn-article-library", "n_clicks"),
        Input("btn-source-manager", "n_clicks"),
        Input("btn-home-explore", "n_clicks"),
        Input("btn-topic", "n_clicks"),
        Input("btn-clustering", "n_clicks"),
        Input("btn-sentiment", "n_clicks"),
        Input("btn-entity", "n_clicks"),
        Input("btn-integrated", "n_clicks"),
        Input("btn-term-tracking", "n_clicks"),
        Input("btn-export-manager", "n_clicks")
    )
    def display_page(btn_home, btn_article_library, btn_source_manager, btn_home_explore, btn_topic, btn_clustering, btn_sentiment, btn_entity, btn_integrated, btn_term_tracking, btn_export_manager):
        ctx_msg = ctx.triggered_id
        if ctx_msg == "btn-home":
            return get_enhanced_home_layout()
        elif ctx_msg == "btn-article-library":
            return get_article_library_layout()
        elif ctx_msg == "btn-source-manager" or ctx_msg == "btn-home-explore":
            return get_source_manager_layout()
        elif ctx_msg == "btn-topic":
            return get_topic_modeling_layout()
        elif ctx_msg == "btn-clustering":
            return get_clustering_layout()
        elif ctx_msg == "btn-sentiment":
            return get_sentiment_analysis_layout()
        elif ctx_msg == "btn-entity":
            return get_entity_recognition_layout()
        elif ctx_msg == "btn-integrated":
            return get_integrated_analysis_layout()
        elif ctx_msg == "btn-term-tracking":
            return get_term_tracking_layout()
        elif ctx_msg == "btn-export-manager":
            return get_export_manager_layout()
        else:
            return get_enhanced_home_layout()

    # (Other callbacks from app.py can be added here as needed)
