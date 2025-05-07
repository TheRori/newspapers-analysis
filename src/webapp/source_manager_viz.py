"""
Module pour la gestion du fichier source commun à toutes les analyses.
Ce module permet de changer le chemin du fichier source, de filtrer les articles,
et d'appliquer d'autres opérations de prétraitement avant les analyses.
"""

import os
import sys
import json
import shutil
import pickle
import hashlib
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
from src.preprocessing.spacy_preprocessor import SpacyPreprocessor

# Charger la configuration
config_path = str(project_root / "config" / "config.yaml")
config = load_config(config_path)

# --- Callbacks pour la gestion du fichier source ---
# Variables globales pour stocker les années et journaux disponibles
available_years = []
available_newspapers = []

# Fonction utilitaire pour détecter les fichiers de clusters disponibles
def get_cluster_files():
    """
    Récupère les fichiers de clusters disponibles.
    
    Returns:
        Liste de dictionnaires avec label et value pour chaque fichier
    """
    project_root = Path(__file__).resolve().parents[2]
    
    # Chercher les fichiers de clusters dans le répertoire des résultats
    cluster_files = []
    
    # Vérifier le répertoire des résultats de clusters
    clusters_dir = project_root / 'data' / 'results' / 'clusters'
    if clusters_dir.exists():
        print(f"Recherche de fichiers de clusters dans {clusters_dir}")
        # Chercher les fichiers JSON qui contiennent "cluster" dans leur nom
        cluster_files_found = list(clusters_dir.glob('*cluster*.json'))
        print(f"Trouvé {len(cluster_files_found)} fichiers de clusters: {[f.name for f in cluster_files_found]}")
        cluster_files.extend(cluster_files_found)
    else:
        print(f"Répertoire de clusters non trouvé: {clusters_dir}")
    
    # Vérifier aussi le répertoire des résultats de topic modeling
    topic_dir = project_root / 'data' / 'results' / 'topic_modeling'
    if topic_dir.exists():
        print(f"Recherche de fichiers de clusters dans {topic_dir}")
        # Chercher les fichiers JSON qui contiennent "cluster" dans leur nom
        topic_cluster_files = list(topic_dir.glob('*cluster*.json'))
        print(f"Trouvé {len(topic_cluster_files)} fichiers de clusters: {[f.name for f in topic_cluster_files]}")
        cluster_files.extend(topic_cluster_files)
    else:
        print(f"Répertoire de topic modeling non trouvé: {topic_dir}")
    
    # Trier par date de modification (le plus récent en premier)
    cluster_files.sort(key=lambda x: os.path.getmtime(x), reverse=True)
    
    # Convertir en options pour le dropdown
    options = []
    for file_path in cluster_files:
        # Créer un label plus lisible
        label = f"{file_path.name} ({file_path.parent.name})"
        options.append({
            "label": label,
            "value": str(file_path)
        })
    
    return options

def get_cluster_ids(cluster_file):
    """
    Récupère les IDs de clusters disponibles dans un fichier de clusters.
    
    Args:
        cluster_file: Chemin vers le fichier de clusters
        
    Returns:
        Liste de dictionnaires avec label et value pour chaque ID de cluster
    """
    if not cluster_file:
        return []
    
    try:
        with open(cluster_file, 'r', encoding='utf-8') as f:
            cluster_data = json.load(f)
        
        options = []
        
        # Vérifier le format du fichier
        if "doc_ids" in cluster_data and "labels" in cluster_data and "n_clusters" in cluster_data:
            # Format avec doc_ids, labels et n_clusters
            n_clusters = cluster_data.get("n_clusters", 0)
            labels = cluster_data.get("labels", [])
            
            # Compter le nombre d'articles par cluster
            cluster_counts = {}
            for label in labels:
                if label not in cluster_counts:
                    cluster_counts[label] = 0
                cluster_counts[label] += 1
            
            # Créer les options pour chaque cluster
            for i in range(n_clusters):
                count = cluster_counts.get(i, 0)
                options.append({
                    "label": f"Cluster {i} ({count} articles)",
                    "value": str(i)
                })
        elif "clusters" in cluster_data:
            # Format avec une liste de clusters
            clusters = cluster_data.get("clusters", [])
            for cluster in clusters:
                cluster_id = cluster.get("id", "")
                articles = cluster.get("articles", [])
                options.append({
                    "label": f"Cluster {cluster_id} ({len(articles)} articles)",
                    "value": str(cluster_id)
                })
        else:
            # Format inconnu, essayer de détecter les clusters
            print(f"Format de fichier de clusters inconnu: {list(cluster_data.keys())}")
            
            # Si le fichier contient des clés numériques, supposer que ce sont des clusters
            numeric_keys = [k for k in cluster_data.keys() if str(k).isdigit()]
            if numeric_keys:
                for key in numeric_keys:
                    articles = cluster_data.get(key, [])
                    if isinstance(articles, list):
                        options.append({
                            "label": f"Cluster {key} ({len(articles)} articles)",
                            "value": str(key)
                        })
        
        return options
    
    except Exception as e:
        print(f"Erreur lors de la lecture du fichier de clusters: {e}")
        return []

# Fonction utilitaire pour détecter les années dans un fichier d'articles
def detect_years_from_file(file_path):
    """Détecte les années disponibles dans un fichier d'articles JSON."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            articles = json.load(f)
        
        years = []
        for a in articles:
            try:
                if 'date' in a and a['date']:
                    # Format attendu: 'YYYY-MM-DD' (ex: '1998-02-12')
                    date_str = a['date']
                    if '-' in date_str and len(date_str) >= 4:
                        year_str = date_str.split('-')[0]
                        if year_str.isdigit():
                            year = int(year_str)
                            if 1800 <= year <= 2100:  # Plage raisonnable
                                years.append(year)
            except (ValueError, TypeError, IndexError):
                pass
        
        return sorted(years) if years else []
    except Exception as e:
        print(f"Erreur lors de la détection des années: {e}")
        return []

def get_processed_json_files():
    """Récupère la liste des fichiers JSON dans le dossier data/processed.
    
    Returns:
        list: Liste de dictionnaires avec label et value pour chaque fichier JSON
    """
    processed_dir = project_root / 'data' / 'processed'
    json_files = list(processed_dir.glob('*.json'))
    
    # Trier par date de modification (le plus récent en premier)
    json_files.sort(key=lambda x: os.path.getmtime(x), reverse=True)
    
    # Convertir en options pour le dropdown
    options = []
    for file_path in json_files:
        # Créer un label plus lisible
        file_size = file_path.stat().st_size / (1024 * 1024)  # Taille en MB
        mod_time = datetime.fromtimestamp(file_path.stat().st_mtime).strftime("%Y-%m-%d %H:%M")
        label = f"{file_path.name} ({file_size:.1f} MB, {mod_time})"
        options.append({
            "label": label,
            "value": str(file_path)
        })
    
    return options

def get_cache_info():
    """Récupère les informations sur les fichiers de cache Spacy existants.
    
    Returns:
        dict: Informations sur les fichiers de cache
    """
    cache_dir = project_root / 'data' / 'cache'
    cache_files = list(cache_dir.glob("preprocessed_docs_*.pkl"))
    
    cache_info = {
        "count": len(cache_files),
        "files": []
    }
    
    for cache_file in cache_files:
        try:
            with open(cache_file, 'rb') as f:
                cache_data = pickle.load(f)
                
            # Extraire les informations du cache
            cache_key_data = cache_data.get('cache_key_data', {})
            articles_path = cache_key_data.get('articles_path', 'Inconnu')
            spacy_model = cache_key_data.get('spacy_model', 'Inconnu')
            allowed_pos = cache_key_data.get('allowed_pos', [])
            min_token_length = cache_key_data.get('min_token_length', 0)
            articles_count = cache_key_data.get('articles_count', 0)
            
            # Taille du fichier
            file_size_bytes = cache_file.stat().st_size
            file_size_mb = file_size_bytes / (1024 * 1024)
            
            # Date de création
            creation_time = datetime.fromtimestamp(cache_file.stat().st_mtime)
            
            cache_info["files"].append({
                "filename": cache_file.name,
                "articles_path": articles_path,
                "spacy_model": spacy_model,
                "allowed_pos": allowed_pos,
                "min_token_length": min_token_length,
                "articles_count": articles_count,
                "file_size_mb": file_size_mb,
                "creation_time": creation_time.strftime("%Y-%m-%d %H:%M:%S")
            })
        except Exception as e:
            print(f"Erreur lors de la lecture du fichier de cache {cache_file}: {e}")
            cache_info["files"].append({
                "filename": cache_file.name,
                "error": str(e),
                "file_size_mb": file_size_bytes / (1024 * 1024) if 'file_size_bytes' in locals() else 0
            })
    
    return cache_info

def register_source_manager_callbacks(app):
    """
    Enregistre les callbacks Dash pour la gestion du fichier source.
    
    Args:
        app: L'application Dash
    """
    
    # Callback : Stocker temporairement le chemin sélectionné dans le dropdown
    @app.callback(
        Output("source-file-path", "value"),
        Input("json-files-dropdown", "value")
    )
    def update_source_path_from_dropdown(selected_file):
        if not selected_file:
            return dash.no_update
        return selected_file
    
    # Callback : Rafraîchir la liste des fichiers JSON
    @app.callback(
        Output("json-files-dropdown", "options"),
        Input("refresh-json-files", "n_clicks")
    )
    def refresh_json_files(n_clicks):
        return get_processed_json_files()
    
    # Callback : Appliquer le chemin sélectionné dans le dropdown
    @app.callback(
        Output("source-file-feedback", "children"),
        Input("apply-dropdown-selection", "n_clicks"),
        State("source-file-path", "value")
    )
    def apply_selected_source_file(n_clicks, selected_path):
        if not n_clicks or not selected_path:
            return dash.no_update
        try:
            path = Path(selected_path)
            if not path.exists():
                return dbc.Alert(f"Fichier non trouvé : {selected_path}", color="danger")
            # Mettre à jour le fichier de config
            config = load_config(config_path)
            config['data']['processed_dir'] = str(path.parent)
            with open(config_path, 'w', encoding='utf-8') as f:
                import yaml
                yaml.safe_dump(config, f, allow_unicode=True)
            return dbc.Alert("Chemin du fichier source mis à jour avec succès !", color="success")
        except Exception as e:
            return dbc.Alert(f"Erreur : {str(e)}", color="danger")
    
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
            # Déclarer la variable globale au début de la fonction
            global available_years
            
            print("\n==== Début de update_year_range ====")
            print(f"Source path: {source_path}")
            print(f"Variables globales au début: available_years={available_years}")
            
            # FORCER les valeurs pour correspondre aux années 1998 vues dans les exemples
            # Valeurs par défaut en cas d'erreur
            default_min_year = 1998
            default_max_year = 2023
            
            if not source_path or not os.path.exists(source_path):
                return default_min_year, default_max_year, [default_min_year, default_max_year], {default_min_year: str(default_min_year), default_max_year: str(default_max_year)}
            
            # Utiliser la fonction utilitaire pour détecter les années
            years = detect_years_from_file(source_path)
            print(f"Années détectées par la fonction utilitaire: {years[:10] if len(years) > 10 else years}")
            
            # Déterminer les années minimales et maximales
            if years:
                # Mettre à jour la variable globale uniquement si nous avons trouvé des années
                if not available_years:  # Si pas déjà détecté dans get_source_manager_layout
                    available_years = sorted(years)
                
                # Utiliser les années disponibles (détectées soit ici, soit dans get_source_manager_layout)
                min_year, max_year = min(available_years), max(available_years)
                print(f"Années détectées dans update_year_range: min={min_year}, max={max_year}")
            else:
                # Vérifier si nous avons déjà des années détectées dans get_source_manager_layout
                if available_years:
                    min_year, max_year = min(available_years), max(available_years)
                    print(f"Utilisation des années détectées précédemment: min={min_year}, max={max_year}")
                else:
                    print("Aucune année valide trouvée dans les données, utilisation des valeurs par défaut")
                    min_year, max_year = default_min_year, default_max_year
                    available_years = list(range(min_year, max_year + 1))
            
            # S'assurer que min_year et max_year sont différents
            if min_year == max_year:
                if min_year > default_min_year:
                    min_year = min_year - 1
                else:
                    max_year = max_year + 1
            
            # Créer les marques pour le slider
            marks = {}
            step = max(1, (max_year - min_year) // 5)  # Limiter à environ 5-6 marques
            for year in range(min_year, max_year + 1, step):
                marks[year] = str(year)
            # S'assurer que l'année maximale est toujours affichée
            marks[max_year] = str(max_year)
            
            print(f"Slider configuré avec min={min_year}, max={max_year}, valeurs={[min_year, max_year]}")
            print(f"Marques du slider: {marks}")
            print("==== Fin de update_year_range ====\n")
            
            # Utiliser les années détectées pour le slider
            print(f"Slider configuré avec min={min_year}, max={max_year}, valeurs=[{min_year}, {max_year}]")
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
        Output("source-preview-articles-modal", "is_open"),
        Output("source-preview-articles-modal-body", "children"),
        Input("preview-filtered", "n_clicks"),
        State("year-range-slider", "value"),
        State("journal-checklist", "value"),
        State("keywords-textarea", "value"),
        State("keywords-mode", "value"),
        State("length-range-slider", "value"),
        State("cluster-file-dropdown", "value"),
        State("cluster-checklist", "value"),
        State("topics-checklist", "value"),
        State("sample-size-input", "value"),
        State("sample-type", "value"),
        State("stratified-sampling", "value"),
        State("source-file-path", "value"),
        prevent_initial_call=True
    )
    def preview_filtered_articles(
        n_clicks, year_range, journals, keywords, keywords_mode, length_range, cluster_file, clusters,
        topics, sample_size, sample_type, stratified, source_path
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
            
            # Filtrage par topics
            if topics:
                print(f"Filtrage par topics: {topics}")
                topics_filtered = []
                for a in filtered:
                    article_topics = a.get('topics', [])
                    if article_topics and any(topic in article_topics for topic in topics):
                        topics_filtered.append(a)
                
                print(f"Articles après filtrage par topics: {len(topics_filtered)}/{len(filtered)}")
                filtered = topics_filtered
            
            # Filtrage par clusters
            if cluster_file and clusters:
                print(f"Filtrage par clusters: {clusters} dans {cluster_file}")
                try:
                    # Charger le fichier de clusters
                    with open(cluster_file, 'r', encoding='utf-8') as f:
                        cluster_data = json.load(f)
                    
                    # Déterminer le format du fichier de clusters et extraire les articles correspondants
                    cluster_filtered = []
                    
                    # Format avec doc_ids, labels et n_clusters
                    if "doc_ids" in cluster_data and "labels" in cluster_data:
                        doc_ids = cluster_data.get("doc_ids", [])
                        labels = cluster_data.get("labels", [])
                        
                        # Créer un dictionnaire pour mapper les doc_ids aux labels
                        doc_id_to_label = {}
                        for i, doc_id in enumerate(doc_ids):
                            if i < len(labels):
                                doc_id_to_label[str(doc_id)] = labels[i]
                        
                        # Filtrer les articles par cluster
                        for article in filtered:
                            article_id = str(article.get('id', ''))
                            if article_id in doc_id_to_label and str(doc_id_to_label[article_id]) in clusters:
                                cluster_filtered.append(article)
                    
                    # Format avec une liste de clusters
                    elif "clusters" in cluster_data:
                        clusters_list = cluster_data.get("clusters", [])
                        
                        # Créer un ensemble d'IDs d'articles pour chaque cluster sélectionné
                        selected_article_ids = set()
                        for cluster_info in clusters_list:
                            cluster_id = str(cluster_info.get("id", ""))
                            if cluster_id in clusters:
                                articles_in_cluster = cluster_info.get("articles", [])
                                for article_id in articles_in_cluster:
                                    selected_article_ids.add(str(article_id))
                        
                        # Filtrer les articles par ID
                        for article in filtered:
                            article_id = str(article.get('id', ''))
                            if article_id in selected_article_ids:
                                cluster_filtered.append(article)
                    
                    # Format avec des clés numériques
                    else:
                        # Vérifier si le fichier contient des clés numériques
                        numeric_keys = [k for k in cluster_data.keys() if str(k).isdigit()]
                        if numeric_keys:
                            # Créer un ensemble d'IDs d'articles pour chaque cluster sélectionné
                            selected_article_ids = set()
                            for cluster_id in clusters:
                                if cluster_id in cluster_data:
                                    articles_in_cluster = cluster_data.get(cluster_id, [])
                                    for article_id in articles_in_cluster:
                                        selected_article_ids.add(str(article_id))
                            
                            # Filtrer les articles par ID
                            for article in filtered:
                                article_id = str(article.get('id', ''))
                                if article_id in selected_article_ids:
                                    cluster_filtered.append(article)
                    
                    print(f"Articles après filtrage par clusters: {len(cluster_filtered)}/{len(filtered)}")
                    filtered = cluster_filtered
                except Exception as e:
                    print(f"Erreur lors du filtrage par clusters: {e}")
            
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

    # Callback : Rafraîchir les informations du cache
    @app.callback(
        Output("cache-info", "children"),
        Input("refresh-cache-info", "n_clicks"),
        Input("preprocess-spacy", "n_clicks")
    )
    def refresh_cache_info(refresh_clicks, preprocess_clicks):
        cache_info = get_cache_info()
        
        if cache_info["count"] == 0:
            return html.Div([
                html.P("Aucun fichier de cache trouvé.", className="text-muted")
            ])
        
        cache_cards = []
        for cache_file in cache_info["files"]:
            if "error" in cache_file:
                # Fichier de cache corrompu ou illisible
                cache_cards.append(
                    dbc.Card([
                        dbc.CardHeader(cache_file["filename"], className="text-danger"),
                        dbc.CardBody([
                            html.P(f"Erreur: {cache_file['error']}", className="text-danger"),
                            html.P(f"Taille: {cache_file['file_size_mb']:.2f} MB")
                        ])
                    ], className="mb-2")
                )
            else:
                # Fichier de cache valide
                cache_cards.append(
                    dbc.Card([
                        dbc.CardHeader(cache_file["filename"]),
                        dbc.CardBody([
                            html.P([
                                html.Strong("Source: "), 
                                html.Span(Path(cache_file["articles_path"]).name)
                            ]),
                            html.P([
                                html.Strong("Modèle: "), 
                                html.Span(cache_file["spacy_model"])
                            ]),
                            html.P([
                                html.Strong("POS: "), 
                                html.Span(", ".join(cache_file["allowed_pos"]))
                            ]),
                            html.P([
                                html.Strong("Longueur min: "), 
                                html.Span(str(cache_file["min_token_length"]))
                            ]),
                            html.P([
                                html.Strong("Articles: "), 
                                html.Span(str(cache_file["articles_count"]))
                            ]),
                            html.P([
                                html.Strong("Taille: "), 
                                html.Span(f"{cache_file['file_size_mb']:.2f} MB")
                            ]),
                            html.P([
                                html.Strong("Créé le: "), 
                                html.Span(cache_file["creation_time"])
                            ])
                        ])
                    ], className="mb-2")
                )
                
                # Cette partie a été supprimée car nous n'utilisons plus cache_options
        
        return html.Div([
            html.H6(f"{cache_info['count']} fichiers de cache trouvés"),
            html.Div(cache_cards, style={"maxHeight": "400px", "overflowY": "auto"})
        ])
        


    
    # Callback : Remplir le dropdown des fichiers de clusters
    @app.callback(
        Output("cluster-file-dropdown", "options"),
        Input("refresh-clusters", "n_clicks")
    )
    def populate_cluster_files_dropdown(n_clicks):
        """
        Remplit le dropdown des fichiers de clusters disponibles.
        """
        # Ajouter l'option "Aucun" en premier
        options = [{"label": "Aucun", "value": ""}]
        options.extend(get_cluster_files())
        return options
    
    # Callback : Remplir la checklist des clusters en fonction du fichier sélectionné
    @app.callback(
        Output("cluster-checklist", "options"),
        Output("cluster-checklist", "value"),
        Input("cluster-file-dropdown", "value"),
        Input("select-all-clusters", "n_clicks"),
        Input("deselect-all-clusters", "n_clicks"),
        State("cluster-checklist", "value")
    )
    def populate_cluster_checklist(cluster_file, select_all_clicks, deselect_all_clicks, current_clusters):
        """
        Remplit la checklist des clusters disponibles en fonction du fichier sélectionné.
        """
        triggered_id = ctx.triggered_id if ctx.triggered else None
        
        if triggered_id == "select-all-clusters" and current_clusters is not None:
            # Sélectionner tous les clusters (utiliser les options actuelles)
            return dash.no_update, [option["value"] for option in ctx.states.get("cluster-checklist.options", [])]
        
        if triggered_id == "deselect-all-clusters":
            # Désélectionner tous les clusters
            return dash.no_update, []
        
        if not cluster_file:
            return [], []
        
        try:
            # Récupérer les IDs de clusters disponibles
            options = get_cluster_ids(cluster_file)
            
            # Si aucun cluster n'est actuellement sélectionné, sélectionner tous les clusters par défaut
            if not current_clusters:
                return options, [option["value"] for option in options]
            else:
                return options, current_clusters
        
        except Exception as e:
            print(f"Erreur lors de la détection des clusters: {e}")
            return [], []
    
    # Callback : Détecter les topics disponibles
    @app.callback(
        Output("topics-checklist", "options"),
        Output("topics-checklist", "value"),
        Input("detect-topics", "n_clicks"),
        Input("select-all-topics", "n_clicks"),
        Input("deselect-all-topics", "n_clicks"),
        State("source-file-path", "value"),
        State("topics-checklist", "value"),
        prevent_initial_call=True
    )
    def update_topics_checklist(detect_clicks, select_all_clicks, deselect_all_clicks, source_path, current_topics):
        triggered_id = ctx.triggered_id if ctx.triggered else None
        
        if triggered_id == "select-all-topics" and current_topics is not None:
            # Sélectionner tous les topics (utiliser les options actuelles)
            return dash.no_update, [option["value"] for option in dash.callback_context.states["topics-checklist.options"]]
        
        if triggered_id == "deselect-all-topics":
            # Désélectionner tous les topics
            return dash.no_update, []
        
        if not source_path or not os.path.exists(source_path):
            return [], []
        
        try:
            # Charger les articles
            with open(source_path, 'r', encoding='utf-8') as f:
                articles = json.load(f)
            
            # Extraire tous les topics uniques
            all_topics = set()
            for article in articles:
                if "topics" in article and isinstance(article["topics"], list):
                    for topic in article["topics"]:
                        all_topics.add(topic)
            
            # Trier les topics par ordre alphabétique
            sorted_topics = sorted(all_topics)
            
            # Créer les options pour la checklist
            options = [{"label": topic, "value": topic} for topic in sorted_topics]
            
            # Si aucun topic n'est actuellement sélectionné, sélectionner tous les topics par défaut
            if not current_topics:
                return options, sorted_topics
            else:
                return options, current_topics
        
        except Exception as e:
            print(f"Erreur lors de la détection des topics: {e}")
            return [], []
    
    # Callback : Prétraiter avec Spacy et mettre en cache
    @app.callback(
        Output("spacy-preprocessing-feedback", "children"),
        Input("preprocess-spacy", "n_clicks"),
        State("source-file-path", "value"),
        State("spacy-model-select", "value"),
        State("pos-tags-checklist", "value"),
        State("min-token-length", "value"),
        State("use-multiprocessing", "value"),
        State("num-processes", "value"),
        State("year-range-slider", "value"),
        State("journal-checklist", "value"),
        State("keywords-textarea", "value"),
        State("keywords-mode", "value"),
        State("length-range-slider", "value"),
        State("topics-checklist", "value"),
        prevent_initial_call=True
    )
    def preprocess_with_spacy_and_cache(
        n_clicks, source_path, spacy_model, pos_tags, min_token_length, 
        use_multiprocessing, num_processes, year_range, journals, keywords, 
        keywords_mode, length_range, topics
    ):
        if not n_clicks:
            return dash.no_update
        
        try:
            # Vérifier que le fichier source existe
            if not source_path or not os.path.exists(source_path):
                return dbc.Alert("Fichier source introuvable.", color="danger")
            
            # Charger les articles
            with open(source_path, 'r', encoding='utf-8') as f:
                articles = json.load(f)
            
            # Appliquer les filtres
            filtered = articles.copy()
            
            # Filtrage par journal
            if journals:
                journal_filtered = filter_articles_by_journals(filtered, journals)
                filtered = journal_filtered
            
            # Filtrage par année
            if year_range:
                year_filtered = []
                for a in filtered:
                    try:
                        if 'date' in a and a['date']:
                            year = int(a.get('date', '2000')[:4])
                            if year_range[0] <= year <= year_range[1]:
                                year_filtered.append(a)
                    except (ValueError, TypeError):
                        pass
                filtered = year_filtered
            
            # Filtrage par mots-clés
            if keywords:
                keywords_list = [k.strip() for k in keywords.split('\n') if k.strip()]
                if keywords_list:
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
                    
                    filtered = keywords_filtered
            
            # Filtrage par longueur
            if length_range:
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
                
                filtered = length_filtered
            
            # Filtrage par topics
            if topics:
                topics_filtered = []
                for a in filtered:
                    article_topics = a.get('topics', [])
                    if any(topic in article_topics for topic in topics):
                        topics_filtered.append(a)
                
                filtered = topics_filtered
            
            # Vérifier qu'il reste des articles après filtrage
            if not filtered:
                return dbc.Alert("Aucun article ne correspond aux filtres. Prétraitement impossible.", color="warning")
            
            # Créer un identifiant de cache basé sur les paramètres
            cache_key_data = {
                'articles_path': source_path,
                'spacy_model': spacy_model,
                'allowed_pos': pos_tags,
                'min_token_length': min_token_length,
                'articles_count': len(filtered),
                'articles_last_modified': os.path.getmtime(source_path),
                'filtered_count': len(filtered)
            }
            
            # Créer un hash de la clé de cache
            cache_key = hashlib.md5(json.dumps(cache_key_data, sort_keys=True).encode()).hexdigest()
            cache_dir = project_root / 'data' / 'cache'
            os.makedirs(cache_dir, exist_ok=True)
            cache_file = cache_dir / f"preprocessed_docs_{cache_key}.pkl"
            
            # Vérifier si le cache existe déjà
            if cache_file.exists():
                return dbc.Alert([
                    "Un cache avec ces paramètres existe déjà : ",
                    html.Code(str(cache_file.name)),
                    ". Vous pouvez l'utiliser directement avec l'option --use-cache dans run_topic_modeling.py."
                ], color="info")
            
            # Initialiser le préprocesseur Spacy
            preprocessing_config = {
                'spacy_model': spacy_model,
                'allowed_pos': pos_tags,
                'min_token_length': min_token_length
            }
            
            spacy_preprocessor = SpacyPreprocessor(preprocessing_config)
            
            # Prétraiter les articles
            texts = []
            tokenized_texts = []
            
            # Créer un feedback pour l'utilisateur
            feedback = dbc.Alert([
                html.P("Prétraitement Spacy en cours..."),
                dbc.Progress(id="spacy-progress", value=0, striped=True, animated=True)
            ], color="info")
            
            # Extraire le texte des articles et prétraiter
            for i, doc in enumerate(filtered):
                if 'cleaned_text' in doc:
                    text = doc['cleaned_text']
                elif 'content' in doc:
                    text = doc['content']
                elif 'text' in doc:
                    text = doc['text']
                else:
                    text = ""
                
                texts.append(text)
                # Tokeniser avec Spacy pour une meilleure modélisation de sujets
                tokenized_texts.append(spacy_preprocessor.preprocess_text(text))
            
            # Sauvegarder dans le cache
            with open(cache_file, 'wb') as f:
                pickle.dump({
                    'texts': texts,
                    'tokenized_texts': tokenized_texts,
                    'cache_key_data': cache_key_data
                }, f)
            
            # Retourner un message de succès
            return dbc.Alert([
                html.P([
                    f"Prétraitement Spacy terminé avec succès pour {len(filtered)} articles. ",
                    "Résultats sauvegardés dans le cache : ",
                    html.Code(str(cache_file.name))
                ]),
                html.P([
                    "Pour utiliser ce cache avec la modélisation de sujets, exécutez : ",
                    html.Code(f"python src/scripts/run_topic_modeling.py --use-cache")
                ])
            ], color="success")
            
        except Exception as e:
            import traceback
            print(f"Erreur lors du prétraitement Spacy : {str(e)}")
            print(traceback.format_exc())
            return dbc.Alert([
                html.P(f"Erreur lors du prétraitement Spacy : {str(e)}"),
                html.Pre(traceback.format_exc())
            ], color="danger")
    
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
        State("topics-checklist", "value"),
        State("sample-size-input", "value"),
        State("sample-type", "value"),
        State("stratified-sampling", "value"),
        State("source-file-path", "value"),
        prevent_initial_call=True
    )
    def apply_filters(
        n_clicks, year_range, journals, keywords, keywords_mode, length_range, cluster_file, clusters,
        topics, sample_size, sample_type, stratified, source_path
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
            
            # Filtrage par clusters
            if cluster_file and clusters:
                print(f"Filtrage par clusters: {clusters} dans {cluster_file}")
                try:
                    # Charger le fichier de clusters
                    with open(cluster_file, 'r', encoding='utf-8') as f:
                        cluster_data = json.load(f)
                    
                    # Déterminer le format du fichier de clusters et extraire les articles correspondants
                    cluster_filtered = []
                    
                    # Format avec doc_ids, labels et n_clusters
                    if "doc_ids" in cluster_data and "labels" in cluster_data:
                        doc_ids = cluster_data.get("doc_ids", [])
                        labels = cluster_data.get("labels", [])
                        
                        # Créer un dictionnaire pour mapper les doc_ids aux labels
                        doc_id_to_label = {}
                        for i, doc_id in enumerate(doc_ids):
                            if i < len(labels):
                                doc_id_to_label[str(doc_id)] = labels[i]
                        
                        # Filtrer les articles par cluster
                        for article in filtered:
                            article_id = str(article.get('id', ''))
                            if article_id in doc_id_to_label and str(doc_id_to_label[article_id]) in clusters:
                                cluster_filtered.append(article)
                    
                    # Format avec une liste de clusters
                    elif "clusters" in cluster_data:
                        clusters_list = cluster_data.get("clusters", [])
                        
                        # Créer un ensemble d'IDs d'articles pour chaque cluster sélectionné
                        selected_article_ids = set()
                        for cluster_info in clusters_list:
                            cluster_id = str(cluster_info.get("id", ""))
                            if cluster_id in clusters:
                                articles_in_cluster = cluster_info.get("articles", [])
                                for article_id in articles_in_cluster:
                                    selected_article_ids.add(str(article_id))
                        
                        # Filtrer les articles par ID
                        for article in filtered:
                            article_id = str(article.get('id', ''))
                            if article_id in selected_article_ids:
                                cluster_filtered.append(article)
                    
                    # Format avec des clés numériques
                    else:
                        # Vérifier si le fichier contient des clés numériques
                        numeric_keys = [k for k in cluster_data.keys() if str(k).isdigit()]
                        if numeric_keys:
                            # Créer un ensemble d'IDs d'articles pour chaque cluster sélectionné
                            selected_article_ids = set()
                            for cluster_id in clusters:
                                if cluster_id in cluster_data:
                                    articles_in_cluster = cluster_data.get(cluster_id, [])
                                    for article_id in articles_in_cluster:
                                        selected_article_ids.add(str(article_id))
                            
                            # Filtrer les articles par ID
                            for article in filtered:
                                article_id = str(article.get('id', ''))
                                if article_id in selected_article_ids:
                                    cluster_filtered.append(article)
                    
                    print(f"Articles après filtrage par clusters: {len(cluster_filtered)}/{len(filtered)}")
                    filtered = cluster_filtered
                except Exception as e:
                    print(f"Erreur lors du filtrage par clusters: {e}")
            
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
    # Réinitialiser les variables globales pour forcer la détection des années et journaux
    global available_years, available_newspapers
    
    print("\n==== Début de get_source_manager_layout ====")
    
    # Obtenir le chemin actuel du fichier source depuis la configuration
    current_source_path = str(project_root / config['data']['processed_dir'] / "articles.json")
    print(f"Chemin du fichier source: {current_source_path}")
    
    # Détecter les années directement ici pour s'assurer qu'elles sont disponibles immédiatement
    available_years = detect_years_from_file(current_source_path)
    available_newspapers = []
    
    # Stocker les années min et max pour le slider
    min_year = min(available_years) if available_years else 2000
    max_year = max(available_years) if available_years else 2023
    
    print(f"Années détectées dans get_source_manager_layout: {available_years[:10] if len(available_years) > 10 else available_years}")
    
    # Calculer le nombre d'articles dans le fichier source
    try:
        with open(current_source_path, 'r', encoding='utf-8') as f:
            articles = json.load(f)
        article_count = len(articles)
        print(f"Nombre d'articles trouvés: {article_count}")
        
        # Afficher quelques exemples d'articles pour débogage
        if len(articles) > 0:
            print("Exemple d'article:")
            sample_article = articles[0]
            print(f"  - id: {sample_article.get('id', 'pas id')}")
            print(f"  - date: {sample_article.get('date', 'pas de date')}")
            print(f"  - newspaper: {sample_article.get('newspaper', 'pas de journal')}")
            print("==== Fin de get_source_manager_layout ====\n")
        
        # Afficher un résumé des années détectées
        if available_years:
            print(f"Années détectées: {available_years[:10] if len(available_years) > 10 else available_years}")
        else:
            print("Aucune année valide n'a été détectée dans le fichier articles.json")
        
        # Utiliser les années déjà détectées plus haut
        if available_years:
            print(f"Années détectées dans get_source_manager_layout: {min(available_years)}-{max(available_years)}")
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
            
            # Sélection du fichier source depuis la liste des fichiers JSON disponibles
            html.Div([
                html.H5("Sélectionner un fichier JSON", className="mb-3"),
                
                # Dropdown pour sélectionner un fichier JSON
                dbc.Row([
                    dbc.Col([
                        dcc.Dropdown(
                            id="json-files-dropdown",
                            options=get_processed_json_files(),
                            placeholder="Sélectionnez un fichier JSON",
                            clearable=True,
                            style={"width": "100%"}
                        )
                    ], width=12, className="mb-3")
                ]),
                
                # Boutons d'action
                dbc.Row([
                    dbc.Col([
                        dbc.Button(
                            "Rafraîchir la liste", 
                            id="refresh-json-files", 
                            color="secondary", 
                            className="w-100"
                        )
                    ], width=6),
                    dbc.Col([
                        dbc.Button(
                            "Appliquer", 
                            id="apply-dropdown-selection", 
                            color="primary", 
                            className="w-100"
                        )
                    ], width=6)
                ]),
                
                # Champ caché pour stocker le chemin sélectionné
                dbc.Input(id="source-file-path", type="hidden", value=current_source_path)
            ], className="mb-4"),
            
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
                            min=min_year,
                            max=max_year,
                            value=[min_year, max_year],
                            marks={i: str(i) for i in range(min_year, max_year + 1, max(1, (max_year - min_year) // 5))},
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
                        dbc.Button("Rafraîchir les clusters", id="refresh-clusters", color="secondary", size="sm", className="mb-2"),
                        dbc.Select(
                            id="cluster-file-dropdown",
                            options=[
                                {"label": "Aucun", "value": ""}
                            ],
                            value="",
                            className="mb-2"
                        ),
                        html.Div([
                            dbc.Button("Tout sélectionner", id="select-all-clusters", color="secondary", size="sm", className="me-2 mb-2"),
                            dbc.Button("Tout désélectionner", id="deselect-all-clusters", color="secondary", size="sm", className="mb-2")
                        ]),
                        dbc.Checklist(
                            id="cluster-checklist",
                            options=[],
                            value=[],
                            className="mb-4"
                        )
                    ], className="mb-4"),
                    
                    # Filtre par topics
                    html.Div([
                        html.H5("Topics", className="mb-2"),
                        html.Div([
                            dbc.Button("Détecter les topics", id="detect-topics", color="secondary", size="sm", className="me-2 mb-2"),
                            dbc.Button("Tout sélectionner", id="select-all-topics", color="secondary", size="sm", className="me-2 mb-2"),
                            dbc.Button("Tout désélectionner", id="deselect-all-topics", color="secondary", size="sm", className="mb-2")
                        ]),
                        dbc.Checklist(
                            id="topics-checklist",
                            options=[],
                            value=[],
                            className="mb-2"
                        ),
                        dbc.FormText("Sélectionnez les topics pour filtrer les articles", className="mb-4")
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
                    ),
                    dbc.Button(
                        "Prétraiter avec Spacy et mettre en cache",
                        id="preprocess-spacy",
                        color="primary",
                        className="ms-2"
                    )
                ])
            ], className="mb-3"),
            
            # Feedback pour le filtrage
            html.Div(id="filter-feedback"),
            
            # Section pour le prétraitement Spacy
            html.Div([
                html.H4("Prétraitement Spacy", className="mt-4 mb-3"),
                html.P("Prétraitez les articles avec Spacy et stockez les résultats dans le cache pour une utilisation ultérieure par la modélisation de sujets."),
                
                # Options de prétraitement Spacy
                dbc.Row([
                    dbc.Col([
                        html.H5("Options de prétraitement", className="mb-2"),
                        dbc.Label("Modèle Spacy:"),
                        dbc.Select(
                            id="spacy-model-select",
                            options=[
                                {"label": "fr_core_news_sm (petit, rapide)", "value": "fr_core_news_sm"},
                                {"label": "fr_core_news_md (moyen, recommandé)", "value": "fr_core_news_md"},
                                {"label": "fr_core_news_lg (grand, précis mais lent)", "value": "fr_core_news_lg"}
                            ],
                            value="fr_core_news_md",
                            className="mb-2"
                        ),
                        
                        dbc.Label("Catégories grammaticales à conserver:"),
                        dbc.Checklist(
                            id="pos-tags-checklist",
                            options=[
                                {"label": "Noms (NOUN)", "value": "NOUN"},
                                {"label": "Noms propres (PROPN)", "value": "PROPN"},
                                {"label": "Adjectifs (ADJ)", "value": "ADJ"},
                                {"label": "Verbes (VERB)", "value": "VERB"},
                                {"label": "Adverbes (ADV)", "value": "ADV"}
                            ],
                            value=["NOUN", "PROPN", "ADJ"],
                            className="mb-2"
                        ),
                        
                        dbc.Label("Longueur minimale des tokens:"),
                        dbc.Input(
                            id="min-token-length",
                            type="number",
                            min=1,
                            max=10,
                            value=3,
                            className="mb-3"
                        ),
                        
                        dbc.Checkbox(
                            id="use-multiprocessing",
                            label="Utiliser le multiprocessing (plus rapide)",
                            value=True,
                            className="mb-2"
                        ),
                        
                        dbc.Label("Nombre de processus:"),
                        dbc.Input(
                            id="num-processes",
                            type="number",
                            min=1,
                            max=32,
                            value=4,
                            className="mb-3"
                        )
                    ], width=6),
                    
                    dbc.Col([
                        html.H5("Informations sur le cache", className="mb-2"),
                        html.Div(id="cache-info", className="mb-3"),
                        dbc.Button(
                            "Rafraîchir les informations du cache",
                            id="refresh-cache-info",
                            color="secondary",
                            size="sm",
                            className="mb-3"
                        ),

                        html.Div(id="spacy-preprocessing-feedback")
                    ], width=6)
                ])
            ])
        ]),
        
        # Modal pour prévisualiser les articles
        create_articles_modal(id_prefix="source-preview")
    ])




