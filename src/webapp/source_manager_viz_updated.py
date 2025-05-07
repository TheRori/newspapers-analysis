"""
Module pour la gestion du fichier source commun à toutes les analyses.
Ce module permet de changer le chemin du fichier source, de filtrer les articles,
et d'appliquer d'autres opérations de prétraitement avant les analyses.
"""

import os
import sys
import json
import shutil
import pandas as pd
from pathlib import Path
from datetime import datetime
import dash
from dash import html, dcc, Input, Output, State, ctx, ALL, MATCH, no_update
import dash_bootstrap_components as dbc
import plotly.express as px
import plotly.graph_objects as go

# Add the project root to the path to allow imports from other modules
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from src.utils.config_loader import load_config
from src.webapp.article_display_utils import (
    get_articles_data, 
    filter_articles_by_criteria,
    create_articles_modal,
    register_articles_modal_callback
)
from src.webapp.journal_filter_utils import group_newspapers, filter_articles_by_journals

# Charger la configuration
config_path = str(project_root / "config" / "config.yaml")
config = load_config(config_path)

# --- Callbacks pour la gestion du fichier source ---
# Variables globales pour stocker les années et journaux disponibles
available_years = []
available_newspapers = []

def register_source_manager_callbacks(app):
    """
    Enregistre les callbacks Dash pour la gestion du fichier source.
    
    Args:
        app: L'application Dash
    """
    
    # Callback : Changer le chemin du fichier source
    @app.callback(
        Output("source-file-feedback", "children"),
        Output("source-file-path", "value"),
        Input("apply-source-file", "n_clicks"),
        State("source-file-path", "value")
    )
    def apply_new_source_file(n_clicks, new_path):
        if not n_clicks or not new_path:
            return dash.no_update, dash.no_update
        try:
            path = Path(new_path)
            if not path.exists():
                return dbc.Alert(f"Fichier non trouvé : {new_path}", color="danger"), new_path
            # Mettre à jour le fichier de config
            config = load_config(config_path)
            config['data']['processed_dir'] = str(path.parent)
            with open(config_path, 'w', encoding='utf-8') as f:
                import yaml
                yaml.safe_dump(config, f, allow_unicode=True)
            return dbc.Alert("Chemin du fichier source mis à jour avec succès !", color="success"), str(new_path)
        except Exception as e:
            return dbc.Alert(f"Erreur : {str(e)}", color="danger"), new_path
    
    # Callback : Mettre à jour la plage d'années disponibles
    @app.callback(
        Output("year-range-slider", "min"),
        Output("year-range-slider", "max"),
        Output("year-range-slider", "value"),
        Output("year-range-slider", "marks"),
        Input("source-file-path", "value")
    )
    def update_year_range(source_path):
        try:
            if not source_path or not os.path.exists(source_path):
                return 2000, 2023, [2000, 2023], {2000: '2000', 2023: '2023'}
            
            # Utiliser les années déjà détectées lors du chargement initial
            global available_years
            if not available_years:
                # Si les années n'ont pas été détectées, les charger maintenant
                print("Chargement des années depuis le fichier source...")
                with open(source_path, encoding='utf-8') as f:
                    articles = json.load(f)
                
                years = []
                for a in articles:
                    try:
                        if 'date' in a and a['date']:
                            year = int(a['date'][:4])
                            years.append(year)
                    except (ValueError, TypeError) as e:
                        print(f"Erreur lors de l'extraction de l'année: {e}")
                
                if years:
                    available_years = sorted(years)
                    min_year, max_year = min(available_years), max(available_years)
                else:
                    min_year, max_year = 2000, 2023
                    available_years = list(range(min_year, max_year + 1))
            else:
                min_year, max_year = min(available_years), max(available_years)
            
            # Créer les marques pour le slider
            marks = {}
            for year in range(min_year, max_year + 1):
                if (year - min_year) % max(1, (max_year - min_year) // 10) == 0:
                    marks[year] = str(year)
            
            return min_year, max_year, [min_year, max_year], marks
        except Exception as e:
            print(f"Erreur lors de la mise à jour de la plage d'années: {e}")
            return 2000, 2023, [2000, 2023], {2000: '2000', 2023: '2023'}
    
    # Callback : Peupler la checklist des journaux
    @app.callback(
        Output("journal-checklist", "options"),
        Output("journal-checklist", "value"),
        Input("source-file-path", "value"),
        Input("select-all-journals", "n_clicks"),
        Input("deselect-all-journals", "n_clicks")
    )
    def populate_journal_checklist(source_path, select_all_clicks, deselect_all_clicks):
        try:
            print(f"Tentative de chargement des journaux depuis: {source_path}")
            if not source_path or not os.path.exists(source_path):
                print("Chemin de fichier invalide ou inexistant")
                return [], []
                
            # Utiliser les journaux déjà détectés lors du chargement initial
            global available_newspapers
            if not available_newspapers:
                # Si les journaux n'ont pas été détectés, les charger maintenant
                print("Chargement des journaux depuis le fichier source...")
                with open(source_path, encoding='utf-8') as f:
                    articles = json.load(f)
                
                # Utiliser la fonction group_newspapers pour regrouper les journaux
                available_newspapers = group_newspapers(articles)
                print(f"Journaux détectés (après regroupement): {available_newspapers}")
            else:
                print(f"Utilisation des journaux déjà détectés: {available_newspapers}")
            
            options = [{"label": j, "value": j} for j in available_newspapers]
            
            # Déterminer si l'utilisateur a cliqué sur "Tout sélectionner" ou "Tout désélectionner"
            triggered_id = ctx.triggered_id if ctx.triggered else None
            if triggered_id == "select-all-journals":
                print("Sélection de tous les journaux")
                return options, available_newspapers
            elif triggered_id == "deselect-all-journals":
                print("Désélection de tous les journaux")
                return options, []
            else:
                # Par défaut, tout est sélectionné
                return options, available_newspapers
        except Exception as e:
            print(f"Erreur lors du chargement des journaux: {e}")
            return [], []
    
    # Callback : Prévisualiser les articles filtrés
    @app.callback(
        Output("filter-feedback", "children", allow_duplicate=True),
        Output("source-preview-modal", "is_open"),
        Output("source-preview-modal-body", "children"),
        Input("preview-filtered", "n_clicks"),
        State("year-range-slider", "value"),
        State("journal-checklist", "value"),
        State("keywords-textarea", "value"),
        State("keywords-mode", "value"),
        State("length-range-slider", "value"),
        State("cluster-file-dropdown", "value"),
        State("cluster-checklist", "value"),
        State("sample-size-input", "value"),
        State("sample-type", "value"),
        State("stratified-sampling", "value"),
        State("source-file-path", "value"),
        prevent_initial_call=True
    )
    def preview_filtered_articles(
        n_clicks, year_range, journals, keywords, keywords_mode, length_range, cluster_file, clusters,
        sample_size, sample_type, stratified, source_path
    ):
        if not n_clicks:
            return dash.no_update, False
        try:
            print("\n=== Début de la prévisualisation des filtres ===")
            print(f"Paramètres de filtrage:")
            print(f"  - Années: {year_range}")
            print(f"  - Journaux: {journals}")
            print(f"  - Mots-clés: {keywords}")
            print(f"  - Mode mots-clés: {keywords_mode}")
            print(f"  - Longueur: {length_range}")
            print(f"  - Fichier de clusters: {cluster_file}")
            print(f"  - Clusters: {clusters}")
            print(f"  - Taille de l'échantillon: {sample_size}")
            print(f"  - Type d'échantillonnage: {sample_type}")
            print(f"  - Échantillonnage stratifié: {stratified}")
            print(f"  - Fichier source: {source_path}")
            
            # Charger les articles
            print("Chargement des articles...")
            with open(source_path, encoding='utf-8') as f:
                articles = json.load(f)
            print(f"Nombre total d'articles chargés: {len(articles)}")
            
            # Filtrage par étape pour faciliter le débogage
            filtered = articles.copy()
            
            # Filtrage par journal
            if journals:
                print(f"Filtrage par journal: {journals}")
                # Utiliser la fonction filter_articles_by_journals pour filtrer les articles
                # en tenant compte des journaux avec numéros
                journal_filtered = filter_articles_by_journals(filtered, journals)
                print(f"Articles après filtrage par journal: {len(journal_filtered)}/{len(filtered)}")
                filtered = journal_filtered
            
            # Filtrage par année
            if year_range:
                print(f"Filtrage par année: {year_range}")
                year_filtered = []
                for a in filtered:
                    try:
                        if 'date' in a and a['date']:
                            year = int(a.get('date', '2000')[:4])
                            if year_range[0] <= year <= year_range[1]:
                                year_filtered.append(a)
                    except (ValueError, TypeError) as e:
                        print(f"Erreur lors de l'extraction de l'année pour l'article {a.get('id', 'inconnu')}: {e}")
                print(f"Articles après filtrage par année: {len(year_filtered)}/{len(filtered)}")
                filtered = year_filtered
            
            # Filtrage par mots-clés
            if keywords:
                print(f"Filtrage par mots-clés: {keywords}")
                keywords_list = [k.strip() for k in keywords.split('\n') if k.strip()]
                if keywords_list:
                    print(f"Liste de mots-clés: {keywords_list}")
                    keywords_filtered = []
                    for a in filtered:
                        text = a.get('content', '') or a.get('text', '') or a.get('cleaned_text', '')
                        if not text:
                            continue
                        
                        # Vérifier si les mots-clés sont présents selon le mode
                        if keywords_mode == "any":
                            if any(k.lower() in text.lower() for k in keywords_list):
                                keywords_filtered.append(a)
                        elif keywords_mode == "all":
                            if all(k.lower() in text.lower() for k in keywords_list):
                                keywords_filtered.append(a)
                        elif keywords_mode == "none":
                            if not any(k.lower() in text.lower() for k in keywords_list):
                                keywords_filtered.append(a)
                    
                    print(f"Articles après filtrage par mots-clés: {len(keywords_filtered)}/{len(filtered)}")
                    filtered = keywords_filtered
            
            # Filtrage par longueur
            if length_range:
                print(f"Filtrage par longueur: {length_range}")
                length_filtered = []
                for a in filtered:
                    # Essayer différents champs pour la longueur
                    length = a.get('word_count', 0)
                    if not length and 'content' in a:
                        length = len(a['content'].split())
                    if not length and 'text' in a:
                        length = len(a['text'].split())
                    
                    if length_range[0] <= length <= length_range[1]:
                        length_filtered.append(a)
                
                print(f"Articles après filtrage par longueur: {len(length_filtered)}/{len(filtered)}")
                filtered = length_filtered
            
            # TODO: Implémenter le filtrage par clusters
            
            # TODO: Implémenter l'échantillonnage
            
            print(f"Nombre final d'articles après tous les filtres: {len(filtered)}")
            print("=== Fin de la prévisualisation des filtres ===\n")
            
            if not filtered:
                return dbc.Alert("Aucun article ne correspond aux filtres.", color="warning"), False, []
            
            # Limiter à 20 articles pour l'affichage
            preview_articles = filtered[:20]
            
            # Créer les cartes d'articles pour le modal
            from src.webapp.article_display_utils import create_article_card
            article_cards = [
                html.H4(f"Articles correspondants ({len(filtered)} trouvés, affichage des 20 premiers)", className="mb-4"),
                html.Hr()
            ]
            
            # Ajouter les cartes d'articles
            for i, article in enumerate(preview_articles):
                article_cards.append(create_article_card(article, i))
            
            return dbc.Alert(f"{len(filtered)} articles correspondent aux filtres. Cliquez pour voir les 20 premiers.", color="info"), True, article_cards
        except Exception as e:
            import traceback
            print(f"Erreur lors du filtrage: {str(e)}")
            print(traceback.format_exc())
            return dbc.Alert(f"Erreur lors du filtrage : {str(e)}", color="danger"), False, []

    # Callback : Appliquer les filtres et sauvegarder un nouveau fichier source filtré
    @app.callback(
        Output("filter-feedback", "children", allow_duplicate=True),
        Input("apply-filters", "n_clicks"),
        State("year-range-slider", "value"),
        State("journal-checklist", "value"),
        State("keywords-textarea", "value"),
        State("keywords-mode", "value"),
        State("length-range-slider", "value"),
        State("cluster-file-dropdown", "value"),
        State("cluster-checklist", "value"),
        State("sample-size-input", "value"),
        State("sample-type", "value"),
        State("stratified-sampling", "value"),
        State("source-file-path", "value"),
        prevent_initial_call=True
    )
    def apply_filters(
        n_clicks, year_range, journals, keywords, keywords_mode, length_range, cluster_file, clusters,
        sample_size, sample_type, stratified, source_path
    ):
        if not n_clicks:
            return dash.no_update
        try:
            print("\n=== Début de l'application des filtres et sauvegarde ===")
            print(f"Paramètres de filtrage:")
            print(f"  - Années: {year_range}")
            print(f"  - Journaux: {journals}")
            print(f"  - Mots-clés: {keywords}")
            print(f"  - Mode mots-clés: {keywords_mode}")
            print(f"  - Longueur: {length_range}")
            print(f"  - Fichier de clusters: {cluster_file}")
            print(f"  - Clusters: {clusters}")
            print(f"  - Taille de l'échantillon: {sample_size}")
            print(f"  - Type d'échantillonnage: {sample_type}")
            print(f"  - Échantillonnage stratifié: {stratified}")
            print(f"  - Fichier source: {source_path}")
            
            # Charger les articles
            print("Chargement des articles...")
            with open(source_path, encoding='utf-8') as f:
                articles = json.load(f)
            print(f"Nombre total d'articles chargés: {len(articles)}")
            
            # Filtrage par étape pour faciliter le débogage
            filtered = articles.copy()
            
            # Filtrage par journal
            if journals:
                print(f"Filtrage par journal: {journals}")
                # Utiliser la fonction filter_articles_by_journals pour filtrer les articles
                # en tenant compte des journaux avec numéros
                journal_filtered = filter_articles_by_journals(filtered, journals)
                print(f"Articles après filtrage par journal: {len(journal_filtered)}/{len(filtered)}")
                filtered = journal_filtered
            
            # Filtrage par année
            if year_range:
                print(f"Filtrage par année: {year_range}")
                year_filtered = []
                for a in filtered:
                    try:
                        if 'date' in a and a['date']:
                            year = int(a.get('date', '2000')[:4])
                            if year_range[0] <= year <= year_range[1]:
                                year_filtered.append(a)
                    except (ValueError, TypeError) as e:
                        print(f"Erreur lors de l'extraction de l'année pour l'article {a.get('id', 'inconnu')}: {e}")
                print(f"Articles après filtrage par année: {len(year_filtered)}/{len(filtered)}")
                filtered = year_filtered
            
            # Filtrage par mots-clés
            if keywords:
                print(f"Filtrage par mots-clés: {keywords}")
                keywords_list = [k.strip() for k in keywords.split('\n') if k.strip()]
                if keywords_list:
                    print(f"Liste de mots-clés: {keywords_list}")
                    keywords_filtered = []
                    for a in filtered:
                        text = a.get('content', '') or a.get('text', '') or a.get('cleaned_text', '')
                        if not text:
                            continue
                        
                        # Vérifier si les mots-clés sont présents selon le mode
                        if keywords_mode == "any":
                            if any(k.lower() in text.lower() for k in keywords_list):
                                keywords_filtered.append(a)
                        elif keywords_mode == "all":
                            if all(k.lower() in text.lower() for k in keywords_list):
                                keywords_filtered.append(a)
                        elif keywords_mode == "none":
                            if not any(k.lower() in text.lower() for k in keywords_list):
                                keywords_filtered.append(a)
                    
                    print(f"Articles après filtrage par mots-clés: {len(keywords_filtered)}/{len(filtered)}")
                    filtered = keywords_filtered
            
            # Filtrage par longueur
            if length_range:
                print(f"Filtrage par longueur: {length_range}")
                length_filtered = []
                for a in filtered:
                    # Essayer différents champs pour la longueur
                    length = a.get('word_count', 0)
                    if not length and 'content' in a:
                        length = len(a['content'].split())
                    if not length and 'text' in a:
                        length = len(a['text'].split())
                    
                    if length_range[0] <= length <= length_range[1]:
                        length_filtered.append(a)
                
                print(f"Articles après filtrage par longueur: {len(length_filtered)}/{len(filtered)}")
                filtered = length_filtered
            
            # TODO: Implémenter le filtrage par clusters
            
            # Échantillonnage
            if sample_size and sample_size > 0 and sample_size < len(filtered):
                print(f"Échantillonnage: {sample_size} articles ({sample_type})")
                import random
                
                if sample_type == "random":
                    # Échantillonnage aléatoire simple
                    sampled = random.sample(filtered, sample_size)
                elif sample_type == "systematic":
                    # Échantillonnage systématique
                    step = len(filtered) // sample_size
                    sampled = [filtered[i] for i in range(0, len(filtered), step)][:sample_size]
                else:
                    # Par défaut, échantillonnage aléatoire
                    sampled = random.sample(filtered, sample_size)
                
                print(f"Articles après échantillonnage: {len(sampled)}/{len(filtered)}")
                filtered = sampled
            
            print(f"Nombre final d'articles après tous les filtres: {len(filtered)}")
            
            # Sauvegarder le résultat dans un nouveau fichier
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_dir = Path(source_path).parent
            output_file = output_dir / f"filtered_articles_{timestamp}.json"
            
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(filtered, f, ensure_ascii=False, indent=2)
            
            print(f"Fichier sauvegardé: {output_file}")
            print("=== Fin de l'application des filtres et sauvegarde ===\n")
            
            return dbc.Alert(
                [
                    f"{len(filtered)} articles ont été filtrés et sauvegardés dans ",
                    html.Code(str(output_file))
                ],
                color="success"
            )
        except Exception as e:
            import traceback
            print(f"Erreur lors de l'application des filtres: {str(e)}")
            print(traceback.format_exc())
            return dbc.Alert(f"Erreur lors de l'application des filtres : {str(e)}", color="danger")

def get_source_manager_layout():
    """
    Crée le layout pour la page de gestion du fichier source.
    
    Returns:
        Composant HTML pour l'onglet de gestion du fichier source
    """
    # Obtenir le chemin actuel du fichier source depuis la configuration
    current_source_path = str(project_root / config['data']['processed_dir'] / "articles.json")
    
    # Calculer le nombre d'articles dans le fichier source
    try:
        with open(current_source_path, 'r', encoding='utf-8') as f:
            articles = json.load(f)
        article_count = len(articles)
    except Exception as e:
        print(f"Erreur lors du chargement du fichier source: {e}")
        article_count = 0
    
    return html.Div([
        html.H2("Gestionnaire de Fichier Source", className="mb-4"),
        
        # Section 1: Fichier Source Actuel
        html.Div([
            html.H3("Fichier Source Actuel", className="mb-3"),
            html.P([
                "Chemin actuel: ",
                html.Code(id="current-source-path", children=current_source_path)
            ]),
            html.P([
                f"Nombre d'articles: ",
                html.Span(id="article-count", children=str(article_count))
            ]),
            
            # Formulaire pour changer le chemin
            dbc.InputGroup([
                dbc.InputGroupText("Nouveau chemin:"),
                dbc.Input(id="source-file-path", type="text", value=current_source_path),
                dbc.Button("Appliquer", id="apply-source-file", color="primary")
            ], className="mb-3"),
            
            # Feedback pour le changement de chemin
            html.Div(id="source-file-feedback")
        ], className="mb-5"),
        
        # Section 2: Filtrage des Articles
        html.Div([
            html.H3("Filtrage des Articles", className="mb-3"),
            
            # Filtres
            dbc.Row([
                # Colonne 1: Filtres de base
                dbc.Col([
                    # Filtre par année
                    html.Div([
                        html.H5("Année de publication", className="mb-2"),
                        dcc.RangeSlider(
                            id="year-range-slider",
                            min=2000,
                            max=2023,
                            value=[2000, 2023],
                            marks={i: str(i) for i in range(2000, 2024, 5)},
                            className="mb-4"
                        )
                    ], className="mb-4"),
                    
                    # Filtre par journal
                    html.Div([
                        html.H5("Journal", className="mb-2"),
                        html.Div([
                            dbc.Button("Tout sélectionner", id="select-all-journals", color="secondary", size="sm", className="me-2 mb-2"),
                            dbc.Button("Tout désélectionner", id="deselect-all-journals", color="secondary", size="sm", className="mb-2")
                        ]),
                        dbc.Checklist(
                            id="journal-checklist",
                            options=[],
                            value=[],
                            className="mb-4"
                        )
                    ], className="mb-4"),
                    
                    # Filtre par mots-clés
                    html.Div([
                        html.H5("Mots-clés", className="mb-2"),
                        dbc.Textarea(
                            id="keywords-textarea",
                            placeholder="Entrez des mots-clés (un par ligne)",
                            className="mb-2",
                            style={"height": "100px"}
                        ),
                        dbc.RadioItems(
                            id="keywords-mode",
                            options=[
                                {"label": "Au moins un mot-clé", "value": "any"},
                                {"label": "Tous les mots-clés", "value": "all"},
                                {"label": "Aucun des mots-clés", "value": "none"}
                            ],
                            value="any",
                            inline=True,
                            className="mb-4"
                        )
                    ], className="mb-4")
                ], width=6),
                
                # Colonne 2: Filtres avancés
                dbc.Col([
                    # Filtre par longueur
                    html.Div([
                        html.H5("Longueur (mots)", className="mb-2"),
                        dcc.RangeSlider(
                            id="length-range-slider",
                            min=0,
                            max=2000,
                            value=[0, 2000],
                            marks={i: str(i) for i in range(0, 2001, 500)},
                            className="mb-4"
                        )
                    ], className="mb-4"),
                    
                    # Filtre par cluster
                    html.Div([
                        html.H5("Clusters", className="mb-2"),
                        dbc.Select(
                            id="cluster-file-dropdown",
                            options=[
                                {"label": "Aucun", "value": ""}
                            ],
                            value="",
                            className="mb-2"
                        ),
                        dbc.Checklist(
                            id="cluster-checklist",
                            options=[],
                            value=[],
                            className="mb-4"
                        )
                    ], className="mb-4"),
                    
                    # Options d'échantillonnage
                    html.Div([
                        html.H5("Échantillonnage", className="mb-2"),
                        dbc.InputGroup([
                            dbc.InputGroupText("Taille:"),
                            dbc.Input(id="sample-size-input", type="number", value=0, min=0)
                        ], className="mb-2"),
                        dbc.RadioItems(
                            id="sample-type",
                            options=[
                                {"label": "Aléatoire", "value": "random"},
                                {"label": "Systématique", "value": "systematic"}
                            ],
                            value="random",
                            inline=True,
                            className="mb-2"
                        ),
                        dbc.Checkbox(
                            id="stratified-sampling",
                            label="Échantillonnage stratifié",
                            value=False,
                            className="mb-4"
                        )
                    ], className="mb-4")
                ], width=6)
            ]),
            
            # Boutons d'action
            dbc.Row([
                dbc.Col([
                    dbc.Button(
                        "Prévisualiser les articles filtrés",
                        id="preview-filtered",
                        color="info",
                        className="me-2"
                    ),
                    dbc.Button(
                        "Appliquer les filtres et sauvegarder",
                        id="apply-filters",
                        color="success"
                    )
                ])
            ], className="mb-3"),
            
            # Feedback pour le filtrage
            html.Div(id="filter-feedback")
        ]),
        
        # Modal pour prévisualiser les articles
        create_articles_modal(id_prefix="source-preview")
    ])
