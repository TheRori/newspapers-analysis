import dash
from dash import html, dcc, Input, Output, State, dash_table, ctx, ALL
import dash_bootstrap_components as dbc
import pandas as pd
import os
import json
import re
from pathlib import Path
from src.webapp.data_provider import DashDataProvider
from src.webapp.source_manager_viz import get_topic_matrix_files, get_clusters_files, get_processed_json_files, get_sentiment_files, get_entity_files
from src.webapp.article_display_utils import extract_excerpt, create_full_article_modal

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

    # Récupérer les valeurs uniques pour les filtres
    topic_options = []
    cluster_options = []
    sentiment_options = []
    journal_options = []
    
    if "nom_du_topic" in df.columns:
        unique_topics = df["nom_du_topic"].dropna().unique()
        topic_options = sorted([{"label": str(t), "value": t} for t in unique_topics], key=lambda x: x["label"])
    
    if "cluster" in df.columns:
        unique_clusters = df["cluster"].dropna().unique()
        cluster_options = sorted([{"label": str(c), "value": c} for c in unique_clusters], key=lambda x: x["label"])
    
    if "sentiment" in df.columns:
        unique_sentiments = df["sentiment"].dropna().unique()
        sentiment_options = sorted([{"label": str(s), "value": s} for s in unique_sentiments], key=lambda x: x["label"])
        
    if "journal" in df.columns:
        unique_journals = df["journal"].dropna().unique()
        journal_options = sorted([{"label": str(j), "value": j} for j in unique_journals], key=lambda x: x["label"])

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
        # Ne pas inclure les colonnes _full dans l'affichage
        if col_id in df.columns and not col_id.endswith("_full"):
            column_def = {"name": display_names[col_id], "id": col_id}
            columns.append(column_def)

    # Conversion de la colonne 'entities' en chaîne de caractères pour l'affichage initial
    if "entities" in df.columns:
        df["entities"] = df["entities"].apply(lambda x: str(x) if pd.notna(x) else "")

    # ---- Layout de la page ----
    layout = dbc.Container([
        html.H2("Bibliothèque d'articles enrichis", className="mb-4"),
        dbc.Alert([
            html.H4("Explorez votre corpus en détail", className="alert-heading"),
            html.P(
                "La bibliothèque d'articles est le point central pour explorer l'ensemble du corpus. "
                "Elle présente tous les articles dans un tableau interactif, enrichi avec les résultats des différentes analyses (topics, clusters, sentiment, entités). "
                "Vous pouvez trier, filtrer et rechercher des articles en fonction de multiples critères pour affiner votre exploration. "
                "En cliquant sur une cellule, vous pouvez visualiser le contenu complet d'un article. "
                "La section 'Réglages' vous permet de mettre à jour les informations en appliquant de nouvelles analyses."
            )
        ], color="info", className="mb-4"),
        
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
            dbc.Col([html.Label("Filtrer par topic"), dcc.Dropdown(id="lib-topic-filter", options=topic_options, multi=True, placeholder="Sélectionner un ou plusieurs topics")], width=3),
            dbc.Col([html.Label("Filtrer par cluster"), dcc.Dropdown(id="lib-cluster-filter", options=cluster_options, multi=True, placeholder="Sélectionner un ou plusieurs clusters")], width=3),
            dbc.Col([html.Label("Filtrer par sentiment"), dcc.Dropdown(id="lib-sentiment-filter", options=sentiment_options, multi=True, placeholder="Sélectionner un ou plusieurs sentiments")], width=3),
            dbc.Col([html.Label("Filtrer par journal"), dcc.Dropdown(id="lib-journal-filter", options=journal_options, multi=True, placeholder="Sélectionner un ou plusieurs journaux")], width=3),
        ], className="mb-3"),
        dbc.Row([
            dbc.Col([html.Label("Filtrer par entité"), dcc.Input(id="lib-entity-filter", type="text", placeholder="Rechercher une entité...", debounce=True)], width=12),
        ], className="mb-3"),
        
        
        # Tableau de données - Charger les données initiales ici
        dash_table.DataTable(
            id="lib-articles-table",
            # Exclure les colonnes _full du tableau
            columns=[col for col in columns if not col["id"].endswith("_full")],
            data=df.to_dict("records"),  # Données initiales
            filter_action="native",
            sort_action="native",
            page_size=20,
            # Configuration des tooltips
            tooltip_delay=0,
            tooltip_duration=None,  # Reste affiché tant que la souris est dessus
            cell_selectable=True,  # Permet de sélectionner les cellules pour voir le contenu complet
            style_table={"overflowX": "auto"},
            style_cell={"textAlign": "left", "whiteSpace": "normal", "padding": "5px"},
            style_header={"backgroundColor": "rgb(230, 230, 230)", "fontWeight": "bold"},
            style_cell_conditional=[
                # Limiter la largeur des colonnes d'entités
                {"if": {"column_id": "entities"}, "maxWidth": "200px", "textOverflow": "ellipsis", "overflow": "hidden"},
                {"if": {"column_id": "entities_org"}, "maxWidth": "150px", "textOverflow": "ellipsis", "overflow": "hidden"},
                {"if": {"column_id": "entities_loc"}, "maxWidth": "150px", "textOverflow": "ellipsis", "overflow": "hidden"},
            ],
            style_data_conditional=[
                # Sentiment très négatif (rouge foncé)
                {
                    'if': {
                        'column_id': 'sentiment',
                        'filter_query': '{sentiment} <= -0.5'
                    },
                    'backgroundColor': '#8b0000',
                    'color': 'white'
                },
                # Sentiment négatif (rouge)
                {
                    'if': {
                        'column_id': 'sentiment',
                        'filter_query': '{sentiment} > -0.5 && {sentiment} <= -0.25'
                    },
                    'backgroundColor': '#d9534f',
                    'color': 'white'
                },
                # Sentiment légèrement négatif (rouge clair)
                {
                    'if': {
                        'column_id': 'sentiment',
                        'filter_query': '{sentiment} > -0.25 && {sentiment} < -0.05'
                    },
                    'backgroundColor': '#ff6666',
                    'color': 'white'
                },
                # Sentiment neutre (gris/noir)
                {
                    'if': {
                        'column_id': 'sentiment',
                        'filter_query': '{sentiment} >= -0.05 && {sentiment} <= 0.05'
                    },
                    'backgroundColor': '#f7f7f7',
                    'color': 'black'
                },
                # Sentiment légèrement positif (vert clair)
                {
                    'if': {
                        'column_id': 'sentiment',
                        'filter_query': '{sentiment} > 0.05 && {sentiment} < 0.25'
                    },
                    'backgroundColor': '#8fbc8f',
                    'color': 'white'
                },
                # Sentiment positif (vert)
                {
                    'if': {
                        'column_id': 'sentiment',
                        'filter_query': '{sentiment} >= 0.25 && {sentiment} < 0.5'
                    },
                    'backgroundColor': '#5cb85c',
                    'color': 'white'
                },
                # Sentiment très positif (vert foncé)
                {
                    'if': {
                        'column_id': 'sentiment',
                        'filter_query': '{sentiment} >= 0.5'
                    },
                    'backgroundColor': '#006400',
                    'color': 'white'
                },
            ],
        )
    ])
    
    # Ajouter le modal pour afficher le contenu complet de l'article
    layout.children.append(create_full_article_modal(id_prefix="lib-"))
    
    return layout

def register_article_library_callbacks(app):
    provider = DashDataProvider()
    
    # Callback pour afficher le contenu complet de l'article dans un modal
    @app.callback(
        Output("lib-full-article-modal", "is_open"),
        Output("lib-full-article-modal-title", "children"),
        Output("lib-full-article-modal-body", "children"),
        Input("lib-articles-table", "active_cell"),
        State("lib-articles-table", "data"),
        State("lib-articles-table", "derived_virtual_data"),
        State("lib-articles-table", "derived_virtual_selected_rows"),
        prevent_initial_call=True
    )
    def show_article_content(active_cell, data, filtered_data, selected_rows):
        if active_cell is None:
            return False, "", ""
        
        try:
            # Utiliser les données filtrées/triées actuellement affichées
            row_index = active_cell["row"]
            
            # Si nous avons des données filtrées, les utiliser
            if filtered_data is not None:
                article_data = filtered_data[row_index]
            else:
                article_data = data[row_index]
            
            # Récupérer les informations de l'article
            article_id = article_data.get("id_article", "")
            title = article_data.get("titre", "Sans titre")
            date = article_data.get("date", "")
            journal = article_data.get("journal", "")
            
            print(f"DEBUG: Affichage de l'article {article_id}, titre: {title}")
            
            # Charger le contenu de l'article depuis le fichier JSON
            project_root = Path(__file__).parent.parent.parent
            config = load_config()
            articles_path = project_root / config['data']['processed_dir'] / "articles.json"
            
            try:
                with open(articles_path, 'r', encoding='utf-8') as f:
                    articles = json.load(f)
                
                # Rechercher l'article par ID
                article_content = ""
                for article in articles:
                    if article.get("id", "") == article_id or article.get("base_id", "") == article_id:
                        article_content = article.get("content") or article.get("original_content") or article.get("text", "")
                        break
                
                if not article_content:
                    return True, f"{title}", html.P("Contenu de l'article non disponible", className="text-danger")
                
                # Créer le contenu du modal
                modal_title = f"{title}"
                modal_body = html.Div([
                    html.H6(f"{date} - {journal}", className="text-muted mb-3"),
                    html.Hr(),
                    html.Div([
                        html.P(paragraph) for paragraph in article_content.split("\n") if paragraph.strip()
                    ]),
                    html.Hr(),
                    html.Small(f"ID de l'article: {article_id}", className="text-muted")
                ])
                
                return True, modal_title, modal_body
                
            except Exception as e:
                print(f"Erreur lors du chargement des articles: {e}")
                return True, "Erreur", html.P(f"Erreur lors du chargement des articles: {str(e)}", className="text-danger")
                
        except Exception as e:
            print(f"Erreur lors de l'affichage du contenu: {e}")
            return True, "Erreur", html.P(f"Erreur lors de l'affichage du contenu: {str(e)}", className="text-danger")
    
    # Fonction utilitaire pour charger la configuration
    def load_config():
        import yaml
        
        project_root = Path(__file__).resolve().parents[2]
        config_path = project_root / 'config' / 'config.yaml'
        
        with open(config_path, encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        return config
    
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
            Input("lib-journal-filter", "value"),
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
    def update_table(table_id, topics, clusters, sentiments, journals, entity, apply_clicks, main_source, topic_matrix_source, clusters_source, sentiment_source, entity_source):
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
        if "lib-articles-table.id" in trigger_id or any(f in trigger_id for f in ["topic-filter", "cluster-filter", "sentiment-filter", "journal-filter", "entity-filter"]):
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
                    
                    # Filtrer par journal (utiliser la colonne journal)
                    if journals and "journal" in df.columns:
                        df = df[df["journal"].isin(journals)]
                    
                    # Filtrer par entité
                    if entity and entity.strip() and "entities" in df.columns:
                        search_term = entity.lower().strip()
                        df = df[df["entities"].fillna("").str.lower().str.contains(search_term)]
                
                # La colonne contenu a été supprimée
                
                # Formater les entités pour l'affichage avec troncature
                if "entities" in df.columns:
                    df["entities"] = df["entities"].apply(lambda x: f"{str(x)[:50]}{'...' if len(str(x)) > 50 else ''}" if pd.notna(x) and str(x).strip() else "")
                
                if "entities_org" in df.columns:
                    df["entities_org"] = df["entities_org"].apply(lambda x: f"{str(x)[:40]}{'...' if len(str(x)) > 40 else ''}" if pd.notna(x) and str(x).strip() else "")
                
                if "entities_loc" in df.columns:
                    df["entities_loc"] = df["entities_loc"].apply(lambda x: f"{str(x)[:40]}{'...' if len(str(x)) > 40 else ''}" if pd.notna(x) and str(x).strip() else "")
                    
                # Charger les entités par défaut au chargement de l'application
                if "lib-articles-table.id" in trigger_id and "entity_source" in provider.custom_sources and provider.custom_sources["entity_source"]:
                    try:
                        entity_source = provider.custom_sources["entity_source"]
                        print(f"[ENTITIES] Chargement automatique des entités depuis : {entity_source}")
                        with open(entity_source, "r", encoding="utf-8") as f:
                            entities_json = json.load(f)
                        
                        entity_map = {}
                        # Traiter selon le format du fichier d'entités
                        if isinstance(entities_json, list):
                            # Format liste d'articles avec entités
                            for art in entities_json:
                                aid = art.get("id") or art.get("base_id")
                                if aid and "entities" in art and isinstance(art["entities"], list):
                                    entity_map[aid] = art["entities"]
                        elif isinstance(entities_json, dict) and "entities" in entities_json:
                            # Format {"entities": {"doc_id": [entities]}}  
                            entity_map = entities_json["entities"]
                        
                        # Appliquer les entités aux articles
                        for index, row in df.iterrows():
                            article_id = row.get("id") or row.get("base_id")
                            if article_id in entity_map:
                                entities = entity_map[article_id]
                                if isinstance(entities, list):
                                    # Créer une chaîne complète pour les entités
                                    entities_str = ", ".join(sorted(set(e.get('text', '') for e in entities if isinstance(e, dict))))
                                    
                                    # Extraire les entités par type
                                    entities_org = [e.get('text', '') for e in entities if isinstance(e, dict) and e.get('label') == 'ORG']
                                    entities_loc = [e.get('text', '') for e in entities if isinstance(e, dict) and e.get('label') == 'LOC']
                                    
                                    # Version tronquée pour l'affichage avec trois points
                                    df.at[index, "entities"] = f"{entities_str[:50]}{'...' if len(entities_str) > 50 else ''}"
                                    
                                    # Ajouter les entités par type si ces colonnes existent
                                    if "entities_org" in df.columns and entities_org:
                                        org_str = ", ".join(sorted(set(entities_org)))
                                        df.at[index, "entities_org"] = f"{org_str[:40]}{'...' if len(org_str) > 40 else ''}"
                                    
                                    if "entities_loc" in df.columns and entities_loc:
                                        loc_str = ", ".join(sorted(set(entities_loc)))
                                        df.at[index, "entities_loc"] = f"{loc_str[:40]}{'...' if len(loc_str) > 40 else ''}"

                    except Exception as e:
                        print(f"Erreur lors du chargement automatique des entités: {e}")
                
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