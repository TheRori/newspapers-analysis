"""
Script de débogage pour identifier où l'application se bloque.
"""

import os
import sys
from pathlib import Path

# Ajouter le répertoire racine du projet au chemin pour permettre les importations
project_root = Path(__file__).parent
sys.path.append(str(project_root))

print("1. Importation des modules de base...")
import dash
from dash import dcc, html
import dash_bootstrap_components as dbc
import plotly.express as px
import pandas as pd
import numpy as np
print("   ✓ Modules de base importés avec succès")

print("2. Importation des modules du projet...")
try:
    from src.utils.config_loader import load_config
    print("   ✓ Module config_loader importé avec succès")
except Exception as e:
    print(f"   ✗ Erreur lors de l'importation de config_loader: {e}")

try:
    # Charger la configuration
    config_path = str(project_root / "config" / "config.yaml")
    config = load_config(config_path)
    print(f"   ✓ Configuration chargée avec succès depuis {config_path}")
except Exception as e:
    print(f"   ✗ Erreur lors du chargement de la configuration: {e}")

print("3. Importation des modules de visualisation...")
try:
    from src.webapp.lexical_analysis_viz import get_lexical_analysis_layout, register_lexical_analysis_callbacks
    print("   ✓ Module lexical_analysis_viz importé avec succès")
except Exception as e:
    print(f"   ✗ Erreur lors de l'importation de lexical_analysis_viz: {e}")

try:
    from src.webapp.topic_modeling_viz import get_topic_modeling_layout, register_topic_modeling_callbacks
    print("   ✓ Module topic_modeling_viz importé avec succès")
except Exception as e:
    print(f"   ✗ Erreur lors de l'importation de topic_modeling_viz: {e}")

try:
    from src.webapp.topic_clustering_viz import get_clustering_layout, get_clustering_args, register_clustering_callbacks
    print("   ✓ Module topic_clustering_viz importé avec succès")
except Exception as e:
    print(f"   ✗ Erreur lors de l'importation de topic_clustering_viz: {e}")

try:
    from src.webapp.cluster_map_viz import get_cluster_map_layout, register_cluster_map_callbacks
    print("   ✓ Module cluster_map_viz importé avec succès")
except Exception as e:
    print(f"   ✗ Erreur lors de l'importation de cluster_map_viz: {e}")

try:
    from src.webapp.sentiment_analysis_viz import get_sentiment_analysis_layout, register_sentiment_analysis_callbacks
    print("   ✓ Module sentiment_analysis_viz importé avec succès")
except Exception as e:
    print(f"   ✗ Erreur lors de l'importation de sentiment_analysis_viz: {e}")

try:
    from src.webapp.entity_recognition_viz import get_entity_recognition_layout, register_entity_recognition_callbacks
    print("   ✓ Module entity_recognition_viz importé avec succès")
except Exception as e:
    print(f"   ✗ Erreur lors de l'importation de entity_recognition_viz: {e}")

try:
    from src.webapp.integrated_analysis_viz import get_integrated_analysis_layout, register_integrated_analysis_callbacks
    print("   ✓ Module integrated_analysis_viz importé avec succès")
except Exception as e:
    print(f"   ✗ Erreur lors de l'importation de integrated_analysis_viz: {e}")

try:
    from src.webapp.term_tracking_viz import get_term_tracking_layout, register_term_tracking_callbacks
    print("   ✓ Module term_tracking_viz importé avec succès")
except Exception as e:
    print(f"   ✗ Erreur lors de l'importation de term_tracking_viz: {e}")

try:
    from src.webapp.export_manager_viz import get_export_manager_layout, register_export_manager_callbacks
    print("   ✓ Module export_manager_viz importé avec succès")
except Exception as e:
    print(f"   ✗ Erreur lors de l'importation de export_manager_viz: {e}")

print("4. Initialisation de l'application Dash...")
try:
    app = dash.Dash(
        __name__,
        external_stylesheets=[dbc.themes.DARKLY],
        meta_tags=[{"name": "viewport", "content": "width=device-width, initial-scale=1"}],
        suppress_callback_exceptions=True,
    )
    app.title = "Newspaper Articles Analysis"
    server = app.server  # Expose the server for deployment platforms
    print("   ✓ Application Dash initialisée avec succès")
except Exception as e:
    print(f"   ✗ Erreur lors de l'initialisation de l'application Dash: {e}")

print("5. Enregistrement des callbacks...")
try:
    register_lexical_analysis_callbacks(app)
    print("   ✓ Callbacks lexical_analysis enregistrés avec succès")
except Exception as e:
    print(f"   ✗ Erreur lors de l'enregistrement des callbacks lexical_analysis: {e}")

try:
    register_topic_modeling_callbacks(app)
    print("   ✓ Callbacks topic_modeling enregistrés avec succès")
except Exception as e:
    print(f"   ✗ Erreur lors de l'enregistrement des callbacks topic_modeling: {e}")

try:
    register_clustering_callbacks(app)
    print("   ✓ Callbacks clustering enregistrés avec succès")
except Exception as e:
    print(f"   ✗ Erreur lors de l'enregistrement des callbacks clustering: {e}")

try:
    register_cluster_map_callbacks(app)
    print("   ✓ Callbacks cluster_map enregistrés avec succès")
except Exception as e:
    print(f"   ✗ Erreur lors de l'enregistrement des callbacks cluster_map: {e}")

try:
    register_sentiment_analysis_callbacks(app)
    print("   ✓ Callbacks sentiment_analysis enregistrés avec succès")
except Exception as e:
    print(f"   ✗ Erreur lors de l'enregistrement des callbacks sentiment_analysis: {e}")

try:
    register_entity_recognition_callbacks(app)
    print("   ✓ Callbacks entity_recognition enregistrés avec succès")
except Exception as e:
    print(f"   ✗ Erreur lors de l'enregistrement des callbacks entity_recognition: {e}")

try:
    register_integrated_analysis_callbacks(app)
    print("   ✓ Callbacks integrated_analysis enregistrés avec succès")
except Exception as e:
    print(f"   ✗ Erreur lors de l'enregistrement des callbacks integrated_analysis: {e}")

try:
    register_term_tracking_callbacks(app)
    print("   ✓ Callbacks term_tracking enregistrés avec succès")
except Exception as e:
    print(f"   ✗ Erreur lors de l'enregistrement des callbacks term_tracking: {e}")

try:
    register_export_manager_callbacks(app)
    print("   ✓ Callbacks export_manager enregistrés avec succès")
except Exception as e:
    print(f"   ✗ Erreur lors de l'enregistrement des callbacks export_manager: {e}")

print("6. Définition du layout de l'application...")
try:
    app.layout = dbc.Container([
        dbc.Row([
            dbc.Col([
                html.H1("Newspaper Articles Analysis Dashboard", className="text-center mb-4"),
                html.Div([
                    dbc.Button("Lexical Analysis", id="btn-lexical", color="primary", className="me-2", n_clicks=0),
                    dbc.Button("Topic Modeling", id="btn-topic", color="secondary", className="me-2", n_clicks=0),
                    dbc.Button("Clustering", id="btn-clustering", color="info", className="me-2", n_clicks=0),
                    dbc.Button("Carte des Clusters", id="btn-cluster-map", color="success", className="me-2", n_clicks=0),
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
    print("   ✓ Layout de l'application défini avec succès")
except Exception as e:
    print(f"   ✗ Erreur lors de la définition du layout de l'application: {e}")

print("7. Définition du callback principal...")
try:
    @app.callback(
        dash.dependencies.Output("page-content", "children"),
        [dash.dependencies.Input("btn-lexical", "n_clicks"),
         dash.dependencies.Input("btn-topic", "n_clicks"),
         dash.dependencies.Input("btn-clustering", "n_clicks"),
         dash.dependencies.Input("btn-cluster-map", "n_clicks"),
         dash.dependencies.Input("btn-sentiment", "n_clicks"),
         dash.dependencies.Input("btn-entity", "n_clicks"),
         dash.dependencies.Input("btn-integrated", "n_clicks"),
         dash.dependencies.Input("btn-term-tracking", "n_clicks"),
         dash.dependencies.Input("btn-export-manager", "n_clicks")],
    )
    def display_page(btn_lexical, btn_topic, btn_clustering, btn_cluster_map, btn_sentiment, btn_entity, btn_integrated, btn_term_tracking, btn_export_manager):
        ctx = dash.callback_context
        if not ctx.triggered:
            return get_lexical_analysis_layout()
        button_id = ctx.triggered[0]["prop_id"].split(".")[0]
        if button_id == "btn-lexical":
            return get_lexical_analysis_layout()
        elif button_id == "btn-topic":
            return get_topic_modeling_layout()
        elif button_id == "btn-clustering":
            return get_clustering_layout()
        elif button_id == "btn-cluster-map":
            return get_cluster_map_layout()
        elif button_id == "btn-sentiment":
            return get_sentiment_analysis_layout()
        elif button_id == "btn-entity":
            return get_entity_recognition_layout()
        elif button_id == "btn-integrated":
            return get_integrated_analysis_layout()
        elif button_id == "btn-term-tracking":
            return get_term_tracking_layout()
        elif button_id == "btn-export-manager":
            return get_export_manager_layout()
        return get_lexical_analysis_layout()
    print("   ✓ Callback principal défini avec succès")
except Exception as e:
    print(f"   ✗ Erreur lors de la définition du callback principal: {e}")

print("8. Vérification des fichiers de données...")
try:
    processed_dir = Path(config['data']['processed_dir'])
    articles_path = processed_dir / "articles.json"
    if articles_path.exists():
        print(f"   ✓ Fichier d'articles trouvé: {articles_path}")
    else:
        print(f"   ✗ Fichier d'articles non trouvé: {articles_path}")
except Exception as e:
    print(f"   ✗ Erreur lors de la vérification des fichiers de données: {e}")

print("9. Vérification du modèle spaCy...")
try:
    import spacy
    model_name = config['analysis']['ner']['model']
    try:
        nlp = spacy.load(model_name)
        print(f"   ✓ Modèle spaCy '{model_name}' chargé avec succès")
    except OSError:
        print(f"   ✗ Modèle spaCy '{model_name}' non trouvé")
except Exception as e:
    print(f"   ✗ Erreur lors de la vérification du modèle spaCy: {e}")

print("10. Tout est prêt pour le lancement de l'application !")
print("    Pour lancer l'application, exécutez: app.run_server(debug=True, port=8050)")
