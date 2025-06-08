"""
Fonctions utilitaires pour le module de suivi de termes.
"""

import os
import re
import json
import yaml
import pathlib
from pathlib import Path
from typing import List, Dict, Any, Optional
import pandas as pd
import argparse

def get_term_tracking_args():
    """
    Récupère les arguments du script de suivi de termes.
    
    Returns:
        Liste des arguments et leurs descriptions
    """
    # Créer un parser similaire à celui du script run_term_tracking.py
    parser = argparse.ArgumentParser(description="Analyse de suivi des termes dans un corpus d'articles")
    
    parser.add_argument(
        "--term-file", 
        type=str, 
        help="Chemin vers le fichier JSON contenant les termes à rechercher"
    )
    
    parser.add_argument(
        "--output", 
        type=str, 
        default="term_tracking_results.csv",
        help="Nom du fichier de sortie CSV (par défaut: term_tracking_results.csv)"
    )
    
    parser.add_argument(
        "--by-year", 
        action="store_true",
        help="Agréger les résultats par année"
    )
    
    parser.add_argument(
        "--by-newspaper", 
        action="store_true",
        help="Agréger les résultats par journal"
    )
    
    parser.add_argument(
        "--limit", 
        type=int, 
        default=0,
        help="Limite le nombre d'articles à analyser (0 = pas de limite)"
    )
    
    # Ajouter les options pour l'analyse sémantique
    parser.add_argument(
        "--semantic-drift",
        action="store_true",
        help="Activer l'analyse de drift sémantique avec Word2Vec"
    )
    
    parser.add_argument(
        "--period-type",
        type=str,
        choices=["year", "decade", "custom"],
        default="decade",
        help="Type de période pour l'analyse sémantique (année, décennie ou personnalisé)"
    )
    
    parser.add_argument(
        "--custom-periods",
        type=str,
        help="Périodes personnalisées au format JSON: [[début1, fin1], [début2, fin2], ...]"
    )
    
    parser.add_argument(
        "--vector-size",
        type=int,
        default=100,
        help="Taille des vecteurs Word2Vec (par défaut: 100)"
    )
    
    parser.add_argument(
        "--window",
        type=int,
        default=5,
        help="Taille de la fenêtre contextuelle Word2Vec (par défaut: 5)"
    )
    
    parser.add_argument(
        "--min-count",
        type=int,
        default=5,
        help="Nombre minimum d'occurrences pour inclure un mot (par défaut: 5)"
    )
    
    parser.add_argument(
        "--reference-period",
        type=str,
        help="Période de référence pour l'alignement des modèles Word2Vec"
    )
    
    parser.add_argument(
        "--similar-terms",
        type=str,
        default='ordinateur,informatique',
        help='Liste de termes (séparés par des virgules) pour lesquels trouver les mots les plus similaires'
    )
    
    parser.add_argument(
        "--top-n",
        type=int,
        default=10,
        help='Nombre de mots similaires à retourner pour chaque terme'
    )
    
    # Extraire les arguments et leurs descriptions
    args = []
    for action in parser._actions:
        if action.dest != 'help':
            args.append({
                'name': action.dest,
                'help': action.help,
                'default': action.default if hasattr(action, 'default') else None,
                'type': action.type.__name__ if hasattr(action, 'type') and action.type else 'bool',
                'choices': action.choices if hasattr(action, 'choices') else None
            })
    
    return args

def load_config(config_path=None):
    """
    Charge la configuration depuis le fichier config.yaml.
    
    Args:
        config_path: Chemin vers le fichier de configuration (optionnel)
        
    Returns:
        Dictionnaire de configuration
    """
    if config_path is None:
        project_root = pathlib.Path(__file__).resolve().parents[3]
        config_path = project_root / 'config' / 'config.yaml'
    
    with open(config_path, encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    return config

def get_term_files():
    """
    Récupère les fichiers de termes disponibles dans le répertoire data/terms.
    
    Returns:
        Liste de dictionnaires avec label et value pour chaque fichier JSON
    """
    import os
    import json
    
    project_root = pathlib.Path(__file__).resolve().parents[3]
    
    # Chercher uniquement dans le répertoire data/terms
    terms_dir = project_root / 'data' / 'terms'
    term_files = []
    
    # Vérifier si le répertoire data/terms existe
    if terms_dir.exists():
        print(f"Recherche de fichiers JSON dans {terms_dir}")
        term_files = list(terms_dir.glob('*.json'))
        print(f"Trouvé {len(term_files)} fichiers JSON dans le répertoire terms: {[f.name for f in term_files]}")
    else:
        print(f"Répertoire terms non trouvé: {terms_dir}")
        # Créer le répertoire s'il n'existe pas
        terms_dir.mkdir(parents=True, exist_ok=True)
        print(f"Répertoire terms créé: {terms_dir}")
    
    # Si aucun fichier de termes n'est trouvé, créer un exemple par défaut
    if not term_files:
        print("Aucun fichier JSON trouvé, création d'un exemple par défaut")
        default_terms = {
            "exemple": ["terme1", "terme2", "terme3"]
        }
        default_path = terms_dir / "default_terms.json"
        with open(default_path, 'w', encoding='utf-8') as f:
            json.dump(default_terms, f, ensure_ascii=False, indent=2)
        term_files.append(default_path)
    
    # Sort by modification time (newest first)
    term_files.sort(key=lambda x: os.path.getmtime(x), reverse=True)
    
    # Format for dropdown
    options = [
        {'label': f"{f.stem} ({pd.to_datetime(os.path.getmtime(f), unit='s').strftime('%Y-%m-%d %H:%M')})", 
         'value': str(f)}
        for f in term_files
    ]
    
    print(f"Final term file options: {options}")
    return options

def clean_file_path(file_path):
    """
    Nettoie un chemin de fichier en supprimant les paramètres de cache-busting.
    
    Args:
        file_path: Chemin de fichier qui peut contenir un paramètre de cache-busting
        
    Returns:
        Chemin de fichier nettoyé
    """
    if isinstance(file_path, str) and '?' in file_path:
        return file_path.split('?')[0]
    return file_path


def get_term_tracking_results():
    """
    Récupère les fichiers de résultats de suivi de termes disponibles.
    
    Returns:
        Liste de dictionnaires avec label et value pour chaque fichier de résultats
    """
    project_root = pathlib.Path(__file__).resolve().parents[3]
    config = load_config()
    
    results_dir = project_root / config['data']['results_dir'] / 'term_tracking'
    
    if not results_dir.exists():
        return []
    
    # Get all term tracking result files
    result_files = list(results_dir.glob('*.csv'))
    
    # Sort by modification time (newest first)
    # Forcer la mise à jour des horodatages en utilisant os.stat au lieu de os.path.getmtime
    result_files.sort(key=lambda x: os.stat(x).st_mtime, reverse=True)
    
    # Format for dropdown with timestamp and cache-busting parameter
    import time
    current_time = int(time.time())  # Timestamp actuel pour éviter les problèmes de cache
    
    options = [
        {
            'label': f"{f.stem} ({pd.to_datetime(os.stat(f).st_mtime, unit='s').strftime('%Y-%m-%d %H:%M')})",
            'value': f"{str(f)}?t={current_time}"  # Ajouter un paramètre pour éviter le cache
        }
        for f in result_files
    ]
    
    return options

def get_semantic_drift_results():
    """
    Récupère les fichiers de résultats de dérive sémantique disponibles.
    
    Returns:
        Liste de dictionnaires avec label et value pour chaque fichier de résultats
    """
    project_root = pathlib.Path(__file__).resolve().parents[3]
    config = load_config()
    
    results_dir = project_root / config['data']['results_dir'] / 'term_tracking'
    
    if not results_dir.exists():
        return []
    
    # Get all semantic drift result files
    result_files = list(results_dir.glob('*semantic_drift*.csv'))
    
    # Sort by modification time (newest first)
    result_files.sort(key=lambda x: os.path.getmtime(x), reverse=True)
    
    # Format for dropdown
    options = [
        {'label': f"{f.stem} ({pd.to_datetime(os.path.getmtime(f), unit='s').strftime('%Y-%m-%d %H:%M')})", 
         'value': str(f)}
        for f in result_files
    ]
    
    return options

def get_similar_terms_results():
    """
    Récupère les fichiers de résultats de termes similaires disponibles.
    
    Returns:
        Liste de dictionnaires avec label et value pour chaque fichier de résultats
    """
    try:
        project_root = pathlib.Path(__file__).resolve().parents[3]
        config = load_config()
        
        results_dir = project_root / config['data']['results_dir'] / 'term_tracking'
        
        print(f"Recherche de fichiers de termes similaires dans: {results_dir}")
        
        if not results_dir.exists():
            print(f"Le répertoire {results_dir} n'existe pas")
            return []
        
        # Get all CSV files in the directory
        all_csv_files = list(results_dir.glob('*.csv'))
        print(f"Tous les fichiers CSV trouvés: {[f.name for f in all_csv_files]}")
        
        # Get all similar terms result files
        result_files = list(results_dir.glob('*similar_terms*.csv'))
        print(f"Fichiers de termes similaires trouvés: {[f.name for f in result_files]}")
        
        # Si aucun fichier n'est trouvé avec le motif, vérifier si le fichier spécifique existe
        if not result_files:
            specific_file = results_dir / 'similar_terms_term_tracking_results.csv'
            if specific_file.exists():
                print(f"Fichier spécifique trouvé: {specific_file.name}")
                result_files = [specific_file]
        
        # Sort by modification time (newest first)
        result_files.sort(key=lambda x: os.path.getmtime(x), reverse=True)
        
        # Format for dropdown
        options = [
            {'label': f"{f.stem} ({pd.to_datetime(os.path.getmtime(f), unit='s').strftime('%Y-%m-%d %H:%M')})", 
             'value': str(f)}
            for f in result_files
        ]
        
        print(f"Options de dropdown générées: {options}")
        
        return options
    
    except Exception as e:
        print(f"Error getting similar terms results: {e}")
        return []

def get_articles_by_filter(term=None, period=None, filter_type=None, filter_value=None, articles=None, config=None):
    """
    Récupère les articles correspondant à un filtre donné.
    
    Args:
        term: Terme spécifique à rechercher (optionnel)
        period: Période à filtrer (optionnel)
        filter_type: Type de filtre (année, journal, terme) (optionnel)
        filter_value: Valeur du filtre (optionnel)
        articles: Liste d'articles (optionnel, sera chargée depuis le fichier si non fournie)
        config: Configuration (optionnel, sera chargée si non fournie)
        
    Returns:
        Liste d'articles filtrés
    """
    import json
    import re
    
    # Charger la configuration si non fournie
    if config is None:
        config = load_config()
    
    # Charger les articles si non fournis
    if articles is None:
        project_root = pathlib.Path(__file__).resolve().parents[3]
        articles_path = project_root / config['data']['processed_dir'] / "articles.json"
        
        try:
            with open(articles_path, 'r', encoding='utf-8') as f:
                articles = json.load(f)
        except Exception as e:
            print(f"Erreur lors du chargement des articles: {e}")
            return []
    
    filtered_articles = []
    
    # Si period est fourni, le convertir en filter_type et filter_value
    if period and not filter_type:
        filter_type = 'année'
        # Si la période est au format "YYYY-YYYY", prendre la première année
        if '-' in period and not period.startswith('article_'):
            filter_value = period.split('-')[0]
        else:
            filter_value = period
    
    for article in articles:
        article_id = str(article.get('id', article.get('base_id', '')))
        if not article_id.startswith('article_'):
            continue
        
        # Extraire les informations de l'article
        try:
            parts = article_id.split('_')
            if len(parts) >= 3:
                date_part = parts[1]
                journal_part = parts[2]
                year_part = date_part.split('-')[0] if '-' in date_part else date_part
            else:
                continue
        except IndexError:
            continue
        
        # Appliquer les filtres
        match = True
        
        # Filtre par type
        if filter_type == 'année' and filter_value and str(filter_value) != year_part:
            match = False
        elif filter_type == 'journal' and filter_value and filter_value.lower() != journal_part.lower():
            match = False
        
        # Filtre par terme
        if term and match:
            text = article.get('text', article.get('content', ''))
            if not text or not re.search(r'\b' + re.escape(term) + r'\b', text, re.IGNORECASE):
                match = False
        
        # Si tous les filtres sont passés, ajouter l'article
        if match:
            filtered_articles.append(article)
    
    return filtered_articles

def highlight_term_in_text(text, term):
    """
    Met en évidence toutes les occurrences du terme (ou liste de termes) dans le texte avec une balise <mark>.
    - Recherche insensible à la casse.
    - Ne surligne que les mots entiers (\b).
    Args:
        text (str): Le texte dans lequel surligner le terme.
        term (str or list): Le terme ou la liste de termes à surligner.
    Returns:
        str: Le texte avec les termes surlignés par <mark>.
    """
    import re
    if not text or not term:
        return text
    terms = term if isinstance(term, list) else [term]
    # Trier les termes par longueur décroissante pour éviter les conflits de sous-chaînes
    terms = sorted(set(terms), key=len, reverse=True)
    # Construire le pattern regex pour tous les termes
    pattern = r"\b(" + "|".join(re.escape(t) for t in terms) + r")\b"
    def repl(match):
        return f"<mark>{match.group(0)}</mark>"
    return re.sub(pattern, repl, text, flags=re.IGNORECASE)

def extract_excerpt(text, term, context_size=100):
    """
    Extrait un extrait de texte contenant le terme recherché.
    
    Args:
        text: Le texte complet
        term: Le terme à rechercher
        context_size: Nombre de caractères à inclure avant et après le terme
        
    Returns:
        Extrait de texte avec le terme mis en évidence
    """
    import re
    
    if not text or not term:
        return ""
    
    # Créer un pattern insensible à la casse
    pattern = re.compile(re.escape(term), re.IGNORECASE)
    
    # Trouver la première occurrence du terme
    match = pattern.search(text)
    if not match:
        return text[:200] + "..."  # Retourner le début du texte si le terme n'est pas trouvé
    
    # Déterminer les indices de début et de fin de l'extrait
    start = max(0, match.start() - context_size)
    end = min(len(text), match.end() + context_size)
    
    # Extraire l'extrait
    excerpt = text[start:end]
    
    # Ajouter des ellipses si nécessaire
    if start > 0:
        excerpt = "..." + excerpt
    if end < len(text):
        excerpt = excerpt + "..."
    
    # Mettre en évidence le terme dans l'extrait
    highlighted_excerpt = pattern.sub(lambda m: f"**{m.group(0)}**", excerpt)
    
    return highlighted_excerpt

def get_cluster_files():
    """
    Récupère les fichiers de clusters disponibles.
    
    Returns:
        Liste de dictionnaires avec label et value pour chaque fichier
    """
    import os
    import json
    from pathlib import Path
    
    project_root = pathlib.Path(__file__).resolve().parents[3]
    config = load_config()
    
    # Chercher les fichiers de clusters dans le répertoire des résultats
    cluster_files = []
    
    # Vérifier le répertoire des résultats de clusters
    clusters_dir = project_root / config['data']['results_dir'] / 'clusters'
    if clusters_dir.exists():
        print(f"Recherche de fichiers de clusters dans {clusters_dir}")
        # Chercher les fichiers JSON qui contiennent "cluster" dans leur nom
        cluster_files_found = list(clusters_dir.glob('*cluster*.json'))
        print(f"Trouvé {len(cluster_files_found)} fichiers de clusters: {[f.name for f in cluster_files_found]}")
        cluster_files.extend(cluster_files_found)
    else:
        print(f"Répertoire de clusters non trouvé: {clusters_dir}")
    
    # Vérifier aussi le répertoire des résultats de topic modeling
    topic_dir = project_root / config['data']['results_dir'] / 'doc_topic_matrix'
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
    import json
    
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
