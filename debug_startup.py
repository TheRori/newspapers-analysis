"""
Script de débogage détaillé pour identifier où l'application se bloque au démarrage.
"""

import os
import sys
import time
import traceback
from pathlib import Path

# Ajouter le répertoire racine du projet au chemin pour permettre les importations
project_root = Path(__file__).parent
sys.path.append(str(project_root))

def log_step(step_num, description):
    """Affiche et enregistre une étape de débogage"""
    message = f"ÉTAPE {step_num}: {description}"
    separator = "=" * len(message)
    print(f"\n{separator}\n{message}\n{separator}")
    sys.stdout.flush()  # Forcer l'affichage immédiat
    return step_num + 1

step = 1
try:
    step = log_step(step, "Importation des modules de base")
    import dash
    from dash import dcc, html, Input, Output, State, ctx
    import dash_bootstrap_components as dbc
    import plotly.express as px
    import pandas as pd
    import numpy as np
    print("✓ Modules de base importés avec succès")

    step = log_step(step, "Importation des modules du projet")
    from src.utils.config_loader import load_config
    print("✓ Module config_loader importé avec succès")

    step = log_step(step, "Chargement de la configuration")
    config_path = str(project_root / "config" / "config.yaml")
    config = load_config(config_path)
    print(f"✓ Configuration chargée avec succès depuis {config_path}")

    step = log_step(step, "Importation des modules de visualisation")
    print("Importation de lexical_analysis_viz...")
    from src.webapp.lexical_analysis_viz import get_lexical_analysis_layout, register_lexical_analysis_callbacks
    print("✓ Module lexical_analysis_viz importé avec succès")
    
    print("Importation de topic_modeling_viz...")
    from src.webapp.topic_modeling_viz import get_topic_modeling_layout, register_topic_modeling_callbacks
    print("✓ Module topic_modeling_viz importé avec succès")
    
    print("Importation de topic_clustering_viz...")
    from src.webapp.topic_clustering_viz import get_clustering_layout, get_clustering_args, register_clustering_callbacks
    print("✓ Module topic_clustering_viz importé avec succès")
    
    print("Importation de cluster_map_viz...")
    from src.webapp.cluster_map_viz import get_cluster_map_layout, register_cluster_map_callbacks
    print("✓ Module cluster_map_viz importé avec succès")
    
    print("Importation de sentiment_analysis_viz...")
    from src.webapp.sentiment_analysis_viz import get_sentiment_analysis_layout, register_sentiment_analysis_callbacks
    print("✓ Module sentiment_analysis_viz importé avec succès")
    
    print("Importation de entity_recognition_viz...")
    from src.webapp.entity_recognition_viz import get_entity_recognition_layout, register_entity_recognition_callbacks
    print("✓ Module entity_recognition_viz importé avec succès")
    
    print("Importation de integrated_analysis_viz...")
    from src.webapp.integrated_analysis_viz import get_integrated_analysis_layout, register_integrated_analysis_callbacks
    print("✓ Module integrated_analysis_viz importé avec succès")
    
    print("Importation de term_tracking_viz...")
    from src.webapp.term_tracking_viz import get_term_tracking_layout, register_term_tracking_callbacks
    print("✓ Module term_tracking_viz importé avec succès")
    
    print("Importation de export_manager_viz...")
    from src.webapp.export_manager_viz import get_export_manager_layout, register_export_manager_callbacks
    print("✓ Module export_manager_viz importé avec succès")

    step = log_step(step, "Initialisation de l'application Dash")
    app = dash.Dash(
        __name__,
        external_stylesheets=[dbc.themes.DARKLY],
        meta_tags=[{"name": "viewport", "content": "width=device-width, initial-scale=1"}],
        suppress_callback_exceptions=True,
    )
    app.title = "Newspaper Articles Analysis"
    server = app.server  # Expose the server for deployment platforms
    print("✓ Application Dash initialisée avec succès")

    step = log_step(step, "Enregistrement des callbacks")
    print("Enregistrement des callbacks lexical_analysis...")
    register_lexical_analysis_callbacks(app)
    print("✓ Callbacks lexical_analysis enregistrés avec succès")
    
    print("Enregistrement des callbacks topic_modeling...")
    register_topic_modeling_callbacks(app)
    print("✓ Callbacks topic_modeling enregistrés avec succès")
    
    print("Enregistrement des callbacks clustering...")
    register_clustering_callbacks(app)
    print("✓ Callbacks clustering enregistrés avec succès")
    
    print("Enregistrement des callbacks cluster_map...")
    register_cluster_map_callbacks(app)
    print("✓ Callbacks cluster_map enregistrés avec succès")
    
    print("Enregistrement des callbacks sentiment_analysis...")
    register_sentiment_analysis_callbacks(app)
    print("✓ Callbacks sentiment_analysis enregistrés avec succès")
    
    print("Enregistrement des callbacks entity_recognition...")
    register_entity_recognition_callbacks(app)
    print("✓ Callbacks entity_recognition enregistrés avec succès")
    
    print("Enregistrement des callbacks integrated_analysis...")
    register_integrated_analysis_callbacks(app)
    print("✓ Callbacks integrated_analysis enregistrés avec succès")
    
    print("Enregistrement des callbacks term_tracking...")
    register_term_tracking_callbacks(app)
    print("✓ Callbacks term_tracking enregistrés avec succès")
    
    print("Enregistrement des callbacks export_manager...")
    register_export_manager_callbacks(app)
    print("✓ Callbacks export_manager enregistrés avec succès")

    step = log_step(step, "Définition du layout de l'application")
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
    print("✓ Layout de l'application défini avec succès")

    step = log_step(step, "Définition du callback principal")
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
        ctx_trigger = ctx.triggered
        if not ctx_trigger:
            return get_lexical_analysis_layout()
        button_id = ctx_trigger[0]["prop_id"].split(".")[0]
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
    print("✓ Callback principal défini avec succès")

    step = log_step(step, "Lancement de l'application")
    print("Tentative de lancement de l'application...")
    print("Si l'application se bloque ici, c'est probablement à cause d'un problème lors du démarrage du serveur.")
    print("Démarrage du serveur dans 3 secondes...")
    time.sleep(1)
    print("2...")
    time.sleep(1)
    print("1...")
    time.sleep(1)
    
    # Lancer l'application avec un timeout
    import threading
    import queue

    def run_app():
        try:
            app.run(debug=False, port=8050)  # Utiliser app.run au lieu de app.run_server
            q.put(None)  # Signaler que l'app a démarré normalement
        except Exception as e:
            q.put(e)  # Mettre l'exception dans la queue

    q = queue.Queue()
    app_thread = threading.Thread(target=run_app)
    app_thread.daemon = True  # Le thread s'arrêtera quand le programme principal s'arrête
    
    print("Démarrage du thread de l'application...")
    app_thread.start()
    
    # Attendre que l'application démarre ou qu'une erreur se produise
    try:
        error = q.get(timeout=10)  # Attendre 10 secondes max
        if error:
            print(f"ERREUR lors du démarrage de l'application: {error}")
            raise error
        else:
            print("Application démarrée avec succès!")
    except queue.Empty:
        print("L'application semble avoir démarré correctement (aucune erreur détectée).")
        print("Si vous ne pouvez pas y accéder, vérifiez s'il y a des erreurs dans la console.")

except Exception as e:
    print(f"\n❌ ERREUR à l'étape {step}:")
    print(f"Exception: {type(e).__name__}: {e}")
    print("\nTraceback complet:")
    traceback.print_exc()
    
print("\nFin du script de débogage.")
