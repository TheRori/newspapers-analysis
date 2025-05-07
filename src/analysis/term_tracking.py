"""
term_tracking.py
Module dédié à la recherche et l'agrégation temporelle de termes prédéfinis dans les textes.

Fonctions :
- Comptage d'occurrences de termes dans les articles
- Agrégation des occurrences par année
- Export des fréquences de termes vers CSV
- (optionnel) Agrégation des occurrences par journal

Exemple d'utilisation :
    from analysis.term_tracking import count_term_occurrences, count_terms_by_year, export_term_frequencies_to_csv
"""

import re
import csv
import json
import logging
import pandas as pd
from collections import Counter, defaultdict
from typing import Dict, List, Any, Optional, Union, Tuple, Set
from pathlib import Path
import os

# Configuration du logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)


def count_term_occurrences(articles: List[Dict[str, Any]], term_list: List[str]) -> Dict[str, Dict[str, int]]:
    """
    Compte les occurrences de chaque terme de la liste dans chaque article.
    
    Args:
        articles: Liste de dictionnaires représentant les articles
        term_list: Liste de termes à rechercher
    
    Returns:
        Dictionnaire avec les IDs d'articles comme clés et un sous-dictionnaire 
        {terme: nombre d'occurrences} comme valeurs
    """
    logger.info(f"Comptage des occurrences pour {len(term_list)} termes dans {len(articles)} articles")
    
    # Compiler les expressions régulières pour une recherche plus rapide
    # Utiliser \b pour s'assurer que les termes sont des mots complets
    # S'assurer que seuls les termes de type string sont traités
    term_patterns = {}
    for term in term_list:
        if isinstance(term, str):
            term_patterns[term] = re.compile(r'\b' + re.escape(term) + r'\b', re.IGNORECASE)
        elif isinstance(term, dict) and 'term' in term:
            # Si le terme est un dictionnaire avec une clé 'term', utiliser cette valeur
            term_text = term['term']
            if isinstance(term_text, str):
                term_patterns[term_text] = re.compile(r'\b' + re.escape(term_text) + r'\b', re.IGNORECASE)
        else:
            logger.warning(f"Type de terme non pris en charge ignoré: {type(term)} - {term}")
    
    results = {}
    for article in articles:
        article_id = str(article.get('id', article.get('base_id', '')))
        text = article.get('text', article.get('content', ''))
        
        if not text or not article_id:
            continue
        
        # Compter les occurrences de chaque terme
        term_counts = {}
        for term, pattern in term_patterns.items():
            count = len(pattern.findall(text))
            if count > 0:
                term_counts[term] = count
        
        if term_counts:
            results[article_id] = term_counts
    
    logger.info(f"Terminé: {len(results)} articles contiennent au moins un des termes recherchés")
    return results


def count_terms_by_year(articles: List[Dict[str, Any]], term_dict: Dict[str, List[str]]) -> Dict[int, Dict[str, int]]:
    """
    Compte les occurrences de termes par année pour un ensemble de catégories de termes.
    
    Args:
        articles: Liste de dictionnaires représentant les articles
        term_dict: Dictionnaire avec des catégories comme clés et des listes de termes comme valeurs
    
    Returns:
        Dictionnaire avec les années comme clés et un sous-dictionnaire 
        {catégorie: nombre total d'occurrences} comme valeurs
    """
    logger.info(f"Agrégation temporelle pour {len(term_dict)} catégories de termes")
    
    # Créer un dictionnaire plat de tous les termes avec leur catégorie
    term_to_category = {}
    for category, terms in term_dict.items():
        for term in terms:
            term_to_category[term] = category
    
    # Obtenir tous les termes dans une liste plate
    all_terms = list(term_to_category.keys())
    
    # Compter les occurrences de tous les termes
    term_occurrences = count_term_occurrences(articles, all_terms)
    
    # Agréger par année
    results = defaultdict(lambda: defaultdict(int))
    
    for article in articles:
        article_id = str(article.get('id', article.get('base_id', '')))
        
        # Extraire l'année de la date de l'article
        date_str = article.get('date', '')
        if not date_str:
            continue
        
        # Essayer différents formats de date courants
        year = None
        try:
            # Format ISO: 2023-01-01
            if '-' in date_str:
                year = int(date_str.split('-')[0])
            # Format avec slash: 01/01/2023
            elif '/' in date_str:
                date_parts = date_str.split('/')
                if len(date_parts[-1]) == 4:  # Année en dernier (format DD/MM/YYYY)
                    year = int(date_parts[-1])
                else:  # Année en premier (format YYYY/MM/DD)
                    year = int(date_parts[0])
        except (ValueError, IndexError):
            logger.warning(f"Format de date non reconnu: {date_str} pour l'article {article_id}")
            continue
        
        if not year:
            continue
        
        # Si l'article contient des termes, les agréger par catégorie
        if article_id in term_occurrences:
            for term, count in term_occurrences[article_id].items():
                category = term_to_category[term]
                results[year][category] += count
    
    # Convertir defaultdict en dict normal pour le retour
    return {year: dict(categories) for year, categories in results.items()}


def export_term_frequencies_to_csv(term_counts: Dict[Any, Dict[str, int]], 
                                  output_file: str, 
                                  results_dir: Optional[str] = None,
                                  source_file: Optional[str] = None,
                                  metadata: Optional[Dict[str, Any]] = None) -> str:
    """
    Exporte les fréquences de termes vers un fichier CSV.
    
    Args:
        term_counts: Dictionnaire avec des clés (années, articles, etc.) et des sous-dictionnaires 
                    {terme/catégorie: nombre d'occurrences}
        output_file: Nom du fichier de sortie
        results_dir: Répertoire de sortie (optionnel)
    
    Returns:
        Chemin vers le fichier CSV créé
    """
    # Déterminer le chemin de sortie
    if results_dir:
        output_path = Path(results_dir) / output_file
    else:
        # Utiliser le répertoire de résultats par défaut
        from src.utils.config_loader import load_config
        config = load_config()
        results_dir = config['data']['results_dir']
        output_path = Path(results_dir) / "term_tracking" / output_file
    
    # Créer le répertoire s'il n'existe pas
    os.makedirs(output_path.parent, exist_ok=True)
    
    # Convertir en DataFrame pour faciliter l'export
    rows = []
    for key, term_dict in term_counts.items():
        row = {'key': key}
        row.update(term_dict)
        rows.append(row)
    
    df = pd.DataFrame(rows)
    
    # Exporter directement vers CSV sans commentaire pour éviter de perturber la lecture
    df.to_csv(output_path, index=False)
    
    logger.info(f"Données exportées vers {output_path}")
    
    # Sauvegarder les métadonnées dans un fichier séparé
    meta_data = metadata or {}
    if source_file:
        meta_data["source_file"] = source_file
    
    # Ajouter des statistiques globales aux métadonnées
    if not meta_data.get("statistics"):
        meta_data["statistics"] = {}
    
    # Calculer le nombre total d'occurrences
    total_occurrences = 0
    for _, term_dict in term_counts.items():
        total_occurrences += sum(term_dict.values())
    
    meta_data["statistics"]["total_occurrences"] = total_occurrences
    meta_data["statistics"]["articles_with_terms"] = len(term_counts)
    
    # Sauvegarder les métadonnées dans un fichier JSON
    meta_path = output_path.with_suffix('.meta.json')
    with open(meta_path, 'w', encoding='utf-8') as f:
        json.dump(meta_data, f, ensure_ascii=False, indent=2)
    
    logger.info(f"Métadonnées exportées vers {meta_path}")
    
    return str(output_path)


def count_terms_by_newspaper(articles: List[Dict[str, Any]], term_dict: Dict[str, List[str]]) -> Dict[str, Dict[str, int]]:
    """
    Compte les occurrences de termes par journal pour un ensemble de catégories de termes.
    
    Args:
        articles: Liste de dictionnaires représentant les articles
        term_dict: Dictionnaire avec des catégories comme clés et des listes de termes comme valeurs
    
    Returns:
        Dictionnaire avec les noms de journaux comme clés et un sous-dictionnaire 
        {catégorie: nombre total d'occurrences} comme valeurs
    """
    logger.info(f"Agrégation par journal pour {len(term_dict)} catégories de termes")
    
    # Créer un dictionnaire plat de tous les termes avec leur catégorie
    term_to_category = {}
    for category, terms in term_dict.items():
        for term in terms:
            term_to_category[term] = category
    
    # Obtenir tous les termes dans une liste plate
    all_terms = list(term_to_category.keys())
    
    # Compter les occurrences de tous les termes
    term_occurrences = count_term_occurrences(articles, all_terms)
    
    # Agréger par journal
    results = defaultdict(lambda: defaultdict(int))
    
    for article in articles:
        article_id = str(article.get('id', article.get('base_id', '')))
        
        # Extraire le nom du journal
        newspaper = article.get('newspaper', article.get('source', ''))
        if not newspaper:
            continue
        
        # Si l'article contient des termes, les agréger par catégorie
        if article_id in term_occurrences:
            for term, count in term_occurrences[article_id].items():
                category = term_to_category[term]
                results[newspaper][category] += count
    
    # Convertir defaultdict en dict normal pour le retour
    return {newspaper: dict(categories) for newspaper, categories in results.items()}
