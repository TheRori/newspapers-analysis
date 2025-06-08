import dash
from dash import html, dcc, Input, Output, State, dash_table
import dash_bootstrap_components as dbc
import pandas as pd
import os
import json
from pathlib import Path
from src.webapp.data_provider import DashDataProvider
from src.webapp.source_manager_viz import get_topic_matrix_files, get_clusters_files, get_processed_json_files, get_sentiment_files, get_entity_files

def get_article_library_layout():
    # Récupérer les fichiers de sentiment et d'entités disponibles
    sentiment_file_options = get_sentiment_files()
    entity_file_options = get_entity_files()
    # Par défaut, les plus récents
    default_sentiment = sentiment_file_options[0]['value'] if sentiment_file_options else None
    default_entity = entity_file_options[0]['value'] if entity_file_options else None
    """
    Crée le layout de la bibliothèque d'articles en se basant uniquement sur le CSV.
    """
    # Chemin vers le fichier CSV contenant les données enrichies
    project_root = Path(__file__).parent.parent.parent
    csv_path = project_root / "data" / "biblio_enriched.csv"
    
    # Charger les données depuis le fichier CSV
    try:
        df = pd.read_csv(csv_path, encoding="utf-8")
        print("DEBUG: Chargement réussi du CSV.")
        print("DEBUG: Colonnes disponibles:", df.columns.tolist())
    except FileNotFoundError:
        print(f"ERREUR: Le fichier {csv_path} n'a pas été trouvé.")
        # Créer un DataFrame vide avec les colonnes attendues pour éviter les erreurs
        df = pd.DataFrame(columns=[
            "id_article", "titre", "date", "journal", "nom_du_topic", 
            "score_topic", "cluster", "sentiment", "entities", "entities_org", "entities_loc"
        ])

    # Récupérer les fichiers sources disponibles pour la section réglages
    provider = DashDataProvider()
    available_sources = get_processed_json_files()
    topic_matrix_files = get_topic_matrix_files()
    clusters_file_options = get_clusters_files()
    
    # ---- Préparation des filtres dynamiques à partir du DataFrame ----

    # Filtre pour les topics
    topic_options = []
    if "nom_du_topic" in df.columns:
        unique_topics = df["nom_du_topic"].dropna().unique()
        topic_options = sorted([{"label": str(t), "value": t} for t in unique_topics], key=lambda x: x["label"])

    # Filtre pour les clusters
    cluster_options = []
    if "cluster" in df.columns:
        unique_clusters = df["cluster"].dropna().unique()
        cluster_options = sorted([{"label": str(c), "value": c} for c in unique_clusters], key=lambda x: str(x["label"]))

    # Filtre pour le sentiment
    sentiment_options = []
    if "sentiment" in df.columns:
        unique_sentiments = df["sentiment"].dropna().unique()
        sentiment_options = sorted([{"label": str(s), "value": s} for s in unique_sentiments], key=lambda x: x["label"])

    # ---- Définition des colonnes du tableau ----
    
    # Mapping pour des noms de colonnes plus lisibles
    display_names = {
        "id_article": "ID",
        "titre": "Titre",
        "date": "Date",
        "journal": "Journal",
        "nom_du_topic": "Topic",
        "score_topic": "Score Topic",
        "cluster": "Cluster",
        "sentiment": "Sentiment",
        "entities": "Toutes Entités",
        "entities_org": "Organisations",
        "entities_loc": "Lieux"
    }
    
    # Créer les colonnes pour la DataTable, en n'affichant que celles qui existent
    columns = []
    for col_id in display_names.keys():
        if col_id in df.columns:
            columns.append({"name": display_names[col_id], "id": col_id})

    # Conversion de la colonne 'entities' en chaîne de caractères pour l'affichage initial
    if "entities" in df.columns:
        df["entities"] = df["entities"].apply(lambda x: str(x) if pd.notna(x) else "")

    # ---- Layout de la page ----
    layout = dbc.Container([
        html.H2("Bibliothèque d'articles enrichis", className="mb-4"),
        
        # Section Réglages (simplifiée)
        dbc.Card([
            dbc.CardHeader(
                dbc.Button("⚙️ Réglages des sources", id="lib-settings-collapse-button", color="link")
            ),
            dbc.Collapse(
                dbc.CardBody([
                    html.H5("Fichiers sources", className="mb-3"),
                    dbc.Row([
                        dbc.Col([
                            html.Label("Fichier source principal", className="fw-bold"),
                            dcc.Dropdown(
                                id="lib-main-source-dropdown",
                                options=available_sources,
                                value=provider.get_current_source_path(),
                                clearable=False
                            )
                        ], width=12, className="mb-3")
                    ]),
                    dbc.Row([
                        dbc.Col([
                            html.Label("Fichier source des topics", className="fw-bold"),
                            dcc.Dropdown(
                                id="lib-topic-matrix-dropdown",
                                options=topic_matrix_files,
                                value=topic_matrix_files[0]["value"] if topic_matrix_files else None,
                                placeholder="Sélectionnez un fichier de matrice de topics",
                                clearable=False
                            ),
                            html.Small("Sélectionnez un fichier de matrice de topics pour mettre à jour les topics des articles", 
                                      className="text-muted")
                        ], width=12, className="mb-3")
                    ]),
                    dbc.Row([
                        dbc.Col([
                            html.Label("Fichier de clusters", className="fw-bold"),
                            dcc.Dropdown(
                                id="lib-clusters-dropdown",
                                options=clusters_file_options,
                                value=clusters_file_options[0]["value"] if clusters_file_options else None,
                                placeholder="Sélectionnez un fichier de clusters",
                                clearable=False
                            ),
                            html.Small("Sélectionnez un fichier de clusters pour mettre à jour les clusters des articles", 
                                      className="text-muted")
                        ], width=4),
                        dbc.Col([
                            html.Label("Fichier de sentiment", className="fw-bold"),
                            dcc.Dropdown(
                                id="lib-sentiment-dropdown",
                                options=sentiment_file_options,
                                value=default_sentiment,
                                clearable=False
                            ),
                            html.Small("Sélectionnez un fichier de sentiment pour mettre à jour les sentiments des articles", 
                                      className="text-muted")
                        ], width=4),
                        dbc.Col([
                            html.Label("Fichier d'entités", className="fw-bold"),
                            dcc.Dropdown(
                                id="lib-entity-dropdown",
                                options=entity_file_options,
                                value=default_entity,
                                clearable=False
                            ),
                            html.Small("Sélectionnez un fichier d'entités pour afficher les organisations et lieux", 
                                      className="text-muted")
                        ], width=4),
                    ]),
                    dbc.Button("Appliquer les changements", id="lib-apply-sources-button", color="primary", className="w-100"),
                    html.Div(id="lib-sources-feedback")
                ]),
                id="lib-settings-collapse",
                is_open=False,
            ),
        ], className="mb-4"),
        
        # Section des filtres
        dbc.Row([
            dbc.Col([html.Label("Filtrer par topic"), dcc.Dropdown(id="lib-topic-filter", options=topic_options, multi=True)], width=3),
            dbc.Col([html.Label("Filtrer par cluster"), dcc.Dropdown(id="lib-cluster-filter", options=cluster_options, multi=True)], width=3),
            dbc.Col([html.Label("Filtrer par sentiment"), dcc.Dropdown(id="lib-sentiment-filter", options=sentiment_options, multi=True)], width=3),
            dbc.Col([html.Label("Filtrer par entité"), dcc.Input(id="lib-entity-filter", type="text", placeholder="Rechercher une entité...", debounce=True)], width=3),
        ], className="mb-3"),
        
        # Tableau de données - Charger les données initiales ici
        dash_table.DataTable(
            id="lib-articles-table",
            columns=columns,
            data=df.to_dict("records"),  # Données initiales chargées ici
            filter_action="native",
            sort_action="native",
            page_size=20,
            style_table={"overflowX": "auto"},
            style_cell={"textAlign": "left", "whiteSpace": "normal", "padding": "5px"},
            style_header={"backgroundColor": "rgb(230, 230, 230)", "fontWeight": "bold"},
            style_data_conditional=[
                # Sentiment négatif (rouge)
                {
                    'if': {'filter_query': '{sentiment} < -0.05', 'column_id': 'sentiment'},
                    'color': 'white',
                    'backgroundColor': '#d9534f'
                },
                # Sentiment positif (vert)
                {
                    'if': {'filter_query': '{sentiment} > 0.05', 'column_id': 'sentiment'},
                    'color': 'white',
                    'backgroundColor': '#5cb85c'
                },
                # Sentiment neutre (gris/noir)
                {
                    'if': {'filter_query': '-0.05 <= {sentiment} && {sentiment} <= 0.05', 'column_id': 'sentiment'},
                    'color': 'black',
                    'backgroundColor': '#f7f7f7'
                },
            ],
        )
    ])
    return layout

def register_article_library_callbacks(app):
    provider = DashDataProvider()
    
    # Utiliser un pattern différent pour gérer les callbacks multiples sur le même output
    # Utiliser un seul callback avec ctx.triggered pour déterminer quelle action a déclenché le callback
    @app.callback(
        [Output("lib-articles-table", "data"),
         Output("lib-sources-feedback", "children")],
        [
            # Trigger pour l'initialisation
            Input("lib-articles-table", "id"),
            # Trigger pour les filtres
            Input("lib-topic-filter", "value"),
            Input("lib-cluster-filter", "value"),
            Input("lib-sentiment-filter", "value"),
            Input("lib-entity-filter", "value"),
            # Trigger pour la mise à jour des sources
            Input("lib-apply-sources-button", "n_clicks")
        ],
        [
            # États nécessaires pour la mise à jour des sources
            State("lib-main-source-dropdown", "value"),
            State("lib-topic-matrix-dropdown", "value"),
            State("lib-clusters-dropdown", "value"),
            State("lib-sentiment-dropdown", "value"),
            State("lib-entity-dropdown", "value"),
        ]
    )
    def update_table(table_id, topics, clusters, sentiments, entity, apply_clicks, main_source, topic_matrix_source, clusters_source, sentiment_source, entity_source):
        # Utiliser ctx.triggered pour déterminer quelle action a déclenché le callback
        ctx = dash.callback_context
        if not ctx.triggered:
            # Initialisation au chargement de la page
            trigger_id = "lib-articles-table.id"
        else:
            trigger_id = ctx.triggered[0]["prop_id"]
        
        # Charger le CSV (source de vérité)
        project_root = Path(__file__).parent.parent.parent
        csv_path = project_root / "data" / "biblio_enriched.csv"
        
        # Cas 1: Initialisation ou filtrage
        if "lib-articles-table.id" in trigger_id or any(f in trigger_id for f in ["topic-filter", "cluster-filter", "sentiment-filter", "entity-filter"]):
            try:
                df = pd.read_csv(csv_path, encoding="utf-8")
                
                # Appliquer les filtres si ce n'est pas l'initialisation
                if "filter" in trigger_id:
                    # Filtrer par topic
                    if topics and "nom_du_topic" in df.columns:
                        df = df[df["nom_du_topic"].isin(topics)]
                    
                    # Filtrer par cluster
                    if clusters and "cluster" in df.columns:
                        df = df[df["cluster"].astype(str).isin([str(c) for c in clusters])]
                    
                    # Filtrer par sentiment
                    if sentiments and "sentiment" in df.columns:
                        df = df[df["sentiment"].isin(sentiments)]
                    
                    # Filtrer par entité
                    if entity and entity.strip() and "entities" in df.columns:
                        search_term = entity.lower().strip()
                        df = df[df["entities"].fillna("").str.lower().str.contains(search_term)]
                
                # Formater les entités pour l'affichage
                if "entities" in df.columns:
                    df["entities"] = df["entities"].apply(lambda x: str(x) if pd.notna(x) else "")
                
                return df.to_dict("records"), dash.no_update
            except Exception as e:
                print(f"Erreur lors du chargement/filtrage du CSV: {e}")
                return [], dash.no_update
        
        # Cas 2: Mise à jour des sources
        elif "lib-apply-sources-button" in trigger_id:
            try:
                # Créer le dictionnaire de configuration
                sources_config = {
                    "main_source": main_source,
                    "topic_matrix_source": topic_matrix_source,
                    "clusters_source": clusters_source,
                    "sentiment_source": sentiment_source,
                    "entity_source": entity_source
                }
                
                # Sauvegarder la configuration
                config_dir = project_root / "config"
                config_file = config_dir / "custom_sources.json"
                
                if not config_dir.exists():
                    config_dir.mkdir(parents=True)
                    
                with open(config_file, "w", encoding="utf-8") as f:
                    json.dump(sources_config, f, indent=4)
                    
                # Mettre à jour le provider et régénérer le CSV
                provider.update_custom_sources(sources_config)
                print(f"Régénération du CSV avec la matrice de topics: {topic_matrix_source}, le fichier de clusters: {clusters_source}, le fichier de sentiment: {sentiment_source} et le fichier d'entités: {entity_source}")
                provider.export_biblio_csv()
                
                # Recharger le CSV mis à jour
                df = pd.read_csv(csv_path, encoding="utf-8")
                
                # Message de succès
                feedback = dbc.Alert(
                    "Les sources ont été mises à jour et le CSV a été régénéré avec succès.",
                    color="success",
                    duration=5000
                )
                
                return df.to_dict("records"), feedback
            except Exception as e:
                print(f"Erreur lors de la mise à jour des sources: {e}")
                error_msg = dbc.Alert(
                    f"Erreur lors de la mise à jour des sources: {str(e)}",
                    color="danger"
                )
                return dash.no_update, error_msg
        
        # Cas par défaut
        return dash.no_update, dash.no_update
    
    # Callback pour la section réglages (inchangé)
    @app.callback(
        Output("lib-settings-collapse", "is_open"),
        Input("lib-settings-collapse-button", "n_clicks"),
        State("lib-settings-collapse", "is_open"),
    )
    def toggle_settings_collapse(n, is_open):
        if n:
            return not is_open
        return is_open