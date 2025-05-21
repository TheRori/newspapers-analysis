#!/usr/bin/env python3
"""
Script pour filtrer les articles contenant des publicités ou du bruit.

Ce script identifie et isole deux types d'articles problématiques :
1. Articles avec des titres contenant "Advertisement", "Ad", "Ads", etc. (avec ou sans majuscules)
2. Articles avec beaucoup de bruit (caractères spéciaux consécutifs, retours à la ligne excessifs, etc.)

Les articles filtrés sont sauvegardés dans un fichier JSON séparé pour référence ultérieure.
"""

import os
import sys
import json
import argparse
import logging
import re
from pathlib import Path
from datetime import datetime
import multiprocessing
from functools import partial
import tqdm

# Configure the logging system
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Add the project root to the path to allow imports from other modules
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from src.utils.config_loader import load_config

def is_advertisement(article):
    """
    Vérifie si un article est une publicité en se basant sur son titre.
    
    Args:
        article: Dictionnaire contenant les données de l'article
        
    Returns:
        True si l'article est une publicité, False sinon
    """
    title = article.get('title', '').lower()
    
    # Liste de mots-clés indiquant une publicité
    ad_keywords = [
        'advertisement', 'advertisements', 
        'ad ', ' ad', '^ad$',
        'ads ', ' ads', '^ads$',
        'publicité', 'publicites', 
        'annonce', 'annonces'
    ]
    
    # Vérifier si le titre contient un des mots-clés
    for keyword in ad_keywords:
        if re.search(keyword, title, re.IGNORECASE):
            return True
    
    return False

def has_excessive_noise(article):
    """
    Vérifie si un article contient beaucoup de bruit.
    
    Args:
        article: Dictionnaire contenant les données de l'article
        
    Returns:
        True si l'article contient beaucoup de bruit, False sinon
    """
    # Récupérer le contenu de l'article (essayer différents champs)
    content = article.get('content', '') or article.get('text', '') or article.get('original_content', '')
    
    if not content:
        return False
    
    # Vérifier les caractères spéciaux consécutifs
    if re.search(r'[^\w\s.,;:!?()-]{5,}', content):
        return True
    
    # Vérifier les retours à la ligne excessifs
    if re.search(r'(\n\s*){5,}', content):
        return True
    
    # Vérifier les séquences de ponctuation anormales
    if re.search(r'[.,;:!?]{5,}', content):
        return True
    
    # Vérifier le ratio de caractères spéciaux
    total_chars = len(content)
    if total_chars > 0:
        special_chars = len(re.findall(r'[^\w\s.,;:!?()-]', content))
        special_ratio = special_chars / total_chars
        
        # Si plus de 15% du contenu est composé de caractères spéciaux
        if special_ratio > 0.15:
            return True
    
    return False

def is_tv_program(article, min_channel_matches=1, min_time_mentions=5):
    """
    Vérifie si un article est probablement un programme TV en exigeant
    au moins `min_channel_matches` chaînes TV présentes dans le titre ou le contenu
    ET au moins `min_time_mentions` mentions d'heures.
    
    Args:
        article: Dictionnaire contenant les données de l'article
        min_channel_matches: Nombre minimum de chaînes TV qui doivent être présentes
        min_time_mentions: Nombre minimum de mentions d'heures qui doivent être présentes
        
    Returns:
        True si l'article est probablement un programme TV, False sinon
    """
    title = article.get('title', '').lower()
    content = article.get('content', '').lower()
    combined_text = f"{title} {content}"
    
    # Liste des chaînes TV à détecter
    tv_channels = [
        'tf1', 'france 2', 'france 3', 'm6', 'canal+', 'c8',
        'tsr 1', 'tsr2', 'télévision suisse romande',
        'tv5', 'tv5monde', 'w9','nrj12', '6ter', 'rtl9',
        'france 5', 'rts un', 'rts deux', 'fr3','antenne 2'
    ]
    
    # Compter les chaînes TV présentes
    matched_channels = [ch for ch in tv_channels if ch in combined_text]
    channel_matches = len(matched_channels)
    
    # Définir différents formats d'heures à détecter
    time_patterns = [
        r'\d{1,2}[.:]\d{2}',          # Format 11.30 ou 11:30
        r'\d{1,2}h\d{0,2}',           # Format 11h30 ou 11h
        r'\d{1,2}\s*h\s*\d{0,2}',     # Format 11 h 30 ou 11 h
        r'\d{1,2}\s*heures?\s*\d{0,2}' # Format 11 heures 30 ou 11 heures
    ]
    
    # Récupérer toutes les mentions d'heures
    all_time_mentions = []
    for pattern in time_patterns:
        all_time_mentions.extend(re.findall(pattern, combined_text))
    
    # Déduplication des heures trouvées (pour éviter de compter plusieurs fois la même heure)
    unique_time_mentions = set(all_time_mentions)
    
    # Retourner True si au moins une chaîne TV ET suffisamment de mentions d'heures sont présentes
    return channel_matches >= min_channel_matches and len(unique_time_mentions) >= min_time_mentions

def is_job_offer(article, min_matches=3):
    """
    Vérifie si un article est probablement une offre d'emploi en exigeant
    au moins `min_matches` mots-clés présents dans le titre ou le contenu.
    
    Args:
        article: Dictionnaire contenant les données de l'article
        min_matches: Nombre minimum de mots-clés qui doivent être présents
        
    Returns:
        True si l'article est probablement une offre d'emploi, False sinon
    """
    title = article.get('title', '').lower()
    content = article.get('content', '').lower()
    
    job_keywords = [
        'offre d\'emploi', 'poste à pourvoir', 'nous recrutons', 
        'candidature', 'cv et lettre', 'recrutement',
        'recherche', 'embauche', 'postuler', 'candidat',
        'expérience requise', 'profil recherché', 'cdi', 'cdd',
        'temps plein', 'temps partiel'
    ]
    
    combined_text = f"{title} {content}"
    
    matches = sum(1 for kw in job_keywords if kw in combined_text)
    
    return matches >= min_matches

def process_article_batch(articles_batch, batch_idx=0, total_batches=1):
    """
    Traite un lot d'articles pour détecter les publicités, le bruit, les programmes TV et les offres d'emploi.
    
    Args:
        articles_batch: Liste d'articles à traiter
        batch_idx: Index du lot actuel (pour le logging)
        total_batches: Nombre total de lots (pour le logging)
        
    Returns:
        Tuple (articles normaux, articles publicitaires, articles bruités, articles TV, articles emploi)
    """
    normal_articles = []
    ad_articles = []
    noisy_articles = []
    tv_articles = []
    job_articles = []
    
    for i, article in enumerate(articles_batch):
        if i % 1000 == 0 and i > 0:
            logger.debug(f"Batch {batch_idx+1}/{total_batches}: Processed {i}/{len(articles_batch)} articles")
        
        # Créer une copie de l'article pour ne pas modifier l'original
        article_copy = article.copy()
        
        if is_advertisement(article):
            # Ajouter les mots-clés de publicité détectés
            title = article.get('title', '').lower()
            ad_keywords = [
                'advertisement', 'advertisements', 
                'ad ', ' ad', '^ad$',
                'ads ', ' ads', '^ads$',
                'publicité', 'publicites', 
                'annonce', 'annonces'
            ]
            matched_keywords = [kw for kw in ad_keywords if re.search(kw, title, re.IGNORECASE)]
            article_copy["filter_reason"] = {
                "type": "advertisement",
                "matched_keywords": matched_keywords
            }
            ad_articles.append(article_copy)
            
        elif is_tv_program(article):
            # Ajouter les chaînes TV et mentions d'heures détectées
            title = article.get('title', '').lower()
            content = article.get('content', '').lower()
            combined_text = f"{title} {content}"
            
            # Détecter les chaînes TV
            tv_channels = [
                'tf1', 'france 2', 'france 3', 'm6', 'canal+', 'c8',
                'tsr 1', 'tsr2', 'télévision suisse romande',
                'tv5', 'tv5monde', 'w9', 'tfx', 'nrj12', '6ter', 'rtl9',
                'france 5', 'fr3','antenne 2'
            ]
            matched_channels = [ch for ch in tv_channels if ch in combined_text]
            
            # Détecter les formats d'heures
            time_patterns = [
                r'\d{1,2}[.:]\d{2}',          # Format 11.30 ou 11:30
                r'\d{1,2}h\d{0,2}',           # Format 11h30 ou 11h
                r'\d{1,2}\s*h\s*\d{0,2}',     # Format 11 h 30 ou 11 h
                r'\d{1,2}\s*heures?\s*\d{0,2}' # Format 11 heures 30 ou 11 heures
            ]
            
            # Récupérer tous les formats d'heures trouvés
            time_matches = {}
            all_time_mentions = []
            
            for i, pattern in enumerate(time_patterns):
                found_times = re.findall(pattern, combined_text)
                all_time_mentions.extend(found_times)
                if found_times:
                    pattern_name = [
                        "format_point_deux_points",  # 11.30 ou 11:30
                        "format_h",                 # 11h30 ou 11h
                        "format_h_espace",          # 11 h 30 ou 11 h
                        "format_heures"             # 11 heures 30 ou 11 heures
                    ][i]
                    time_matches[pattern_name] = found_times[:10]  # Limiter à 10 exemples par format
            
            # Déduplication des heures trouvées
            unique_time_mentions = set(all_time_mentions)
            
            article_copy["filter_reason"] = {
                "type": "tv_program",
                "matched_channels": matched_channels,
                "time_formats": time_matches,
                "unique_time_mentions": len(unique_time_mentions),
                "total_time_mentions": len(all_time_mentions)
            }
            tv_articles.append(article_copy)
            
        elif is_job_offer(article):
            # Ajouter les mots-clés d'offre d'emploi détectés
            title = article.get('title', '').lower()
            content = article.get('content', '').lower()
            combined_text = f"{title} {content}"
            
            job_keywords = [
                'offre d\'emploi', 'poste à pourvoir', 'nous recrutons', 
                'candidature', 'cv et lettre', 'recrutement',
                'recherche', 'embauche', 'postuler', 'candidat',
                'expérience requise', 'profil recherché', 'cdi', 'cdd',
                'temps plein', 'temps partiel'
            ]
            
            matched_keywords = [kw for kw in job_keywords if kw in combined_text]
            
            article_copy["filter_reason"] = {
                "type": "job_offer",
                "matched_keywords": matched_keywords
            }
            job_articles.append(article_copy)
            
        elif has_excessive_noise(article):
            # Ajouter les raisons de bruit détectées
            content = article.get('content', '') or article.get('text', '') or article.get('original_content', '')
            noise_reasons = []
            
            if re.search(r'[^\w\s.,;:!?()-]{5,}', content):
                noise_reasons.append("caractères spéciaux consécutifs")
                
            if re.search(r'(\n\s*){5,}', content):
                noise_reasons.append("retours à la ligne excessifs")
                
            if re.search(r'[.,;:!?]{5,}', content):
                noise_reasons.append("séquences de ponctuation anormales")
            
            total_chars = len(content)
            if total_chars > 0:
                special_chars = len(re.findall(r'[^\w\s.,;:!?()-]', content))
                special_ratio = special_chars / total_chars
                if special_ratio > 0.15:
                    noise_reasons.append(f"ratio élevé de caractères spéciaux ({special_ratio:.2%})")
            
            article_copy["filter_reason"] = {
                "type": "excessive_noise",
                "noise_patterns": noise_reasons
            }
            noisy_articles.append(article_copy)
            
        else:
            normal_articles.append(article_copy)
    
    return normal_articles, ad_articles, noisy_articles, tv_articles, job_articles

def filter_articles(articles, batch_size=1000, num_processes=None):
    """
    Filtre les articles contenant des publicités, du bruit, des programmes TV ou des offres d'emploi.
    
    Args:
        articles: Liste des articles à analyser
        batch_size: Taille des lots d'articles à traiter en parallèle
        num_processes: Nombre de processus à utiliser (None = auto)
        
    Returns:
        Tuple (articles normaux, articles publicitaires, articles bruités, articles TV, articles emploi)
    """
    logger.info(f"Filtering {len(articles)} articles...")
    
    # Déterminer le nombre de processus à utiliser
    if num_processes is None:
        num_processes = max(1, multiprocessing.cpu_count() - 1)
    
    # Diviser les articles en lots
    batches = [articles[i:i+batch_size] for i in range(0, len(articles), batch_size)]
    total_batches = len(batches)
    
    logger.info(f"Processing {total_batches} batches with {num_processes} processes")
    
    # Traiter les lots en parallèle
    if num_processes > 1 and total_batches > 1:
        # Créer une fonction partielle avec les arguments fixes
        process_batch_with_idx = partial(
            process_article_batch,
            total_batches=total_batches
        )
        
        # Ajouter l'index du lot comme argument
        batch_args = [(batch, idx) for idx, batch in enumerate(batches)]
        
        # Traiter en parallèle
        with multiprocessing.Pool(processes=num_processes) as pool:
            results = list(tqdm.tqdm(
                pool.starmap(process_batch_with_idx, batch_args),
                total=total_batches,
                desc="Filtering articles"
            ))
    else:
        # Traitement séquentiel
        results = []
        for idx, batch in enumerate(tqdm.tqdm(batches, desc="Filtering articles")):
            results.append(process_article_batch(batch, idx, total_batches))
    
    # Combiner les résultats
    normal_articles = []
    ad_articles = []
    noisy_articles = []
    tv_articles = []
    job_articles = []
    
    for normal_batch, ad_batch, noisy_batch, tv_batch, job_batch in results:
        normal_articles.extend(normal_batch)
        ad_articles.extend(ad_batch)
        noisy_articles.extend(noisy_batch)
        tv_articles.extend(tv_batch)
        job_articles.extend(job_batch)
    
    logger.info(f"Filtering complete: {len(normal_articles)} normal articles, "
                f"{len(ad_articles)} advertisement articles, "
                f"{len(noisy_articles)} noisy articles, "
                f"{len(tv_articles)} TV program articles, "
                f"{len(job_articles)} job offer articles")
    
    return normal_articles, ad_articles, noisy_articles, tv_articles, job_articles

# Garder l'ancienne fonction pour la compatibilité avec le code existant
def filter_ads_and_noise(articles, batch_size=1000, num_processes=None):
    """
    Filtre les articles contenant des publicités ou du bruit (fonction maintenue pour compatibilité).
    
    Args:
        articles: Liste des articles à analyser
        batch_size: Taille des lots d'articles à traiter en parallèle
        num_processes: Nombre de processus à utiliser (None = auto)
        
    Returns:
        Tuple (articles normaux, articles publicitaires, articles bruités)
    """
    logger.warning("La fonction filter_ads_and_noise est dépréciée. Utilisez filter_articles à la place.")
    normal, ads, noise, _, _ = filter_articles(articles, batch_size, num_processes)
    return normal, ads, noise

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Filter advertisements and noisy articles")
    
    parser.add_argument("--config", type=str, default=None,
                        help="Path to the configuration file")
    parser.add_argument("--input", type=str, default=None,
                        help="Path to the input articles JSON file")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Directory to save the output files")
    parser.add_argument("--batch-size", type=int, default=1000,
                        help="Batch size for parallel processing")
    parser.add_argument("--processes", type=int, default=None,
                        help="Number of processes to use (default: auto)")
    
    return parser.parse_args()

def main():
    """Fonction principale pour filtrer les articles contenant des publicités, du bruit, des programmes TV ou des offres d'emploi."""
    args = parse_arguments()
    
    # Charger la configuration
    if args.config:
        config_path = args.config
    else:
        config_path = str(project_root / "config" / "config.yaml")
    
    config = load_config(config_path)
    
    # Déterminer le chemin du fichier d'articles
    if args.input:
        articles_path = args.input
    else:
        articles_path = str(project_root / config["data"]["processed_dir"] / "articles.json")
    
    # Déterminer le répertoire de sortie
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = Path(articles_path).parent
    
    # S'assurer que le répertoire de sortie existe
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Charger les articles
    logger.info(f"Loading articles from {articles_path}")
    try:
        with open(articles_path, 'r', encoding='utf-8') as f:
            articles = json.load(f)
        logger.info(f"Loaded {len(articles)} articles")
    except Exception as e:
        logger.error(f"Error loading articles: {e}")
        sys.exit(1)
    
    # Filtrer les articles
    normal_articles, ad_articles, noisy_articles, tv_articles, job_articles = filter_articles(
        articles,
        batch_size=args.batch_size,
        num_processes=args.processes
    )
    
    # Générer les noms de fichiers de sortie avec timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    normal_file = output_dir / f"filtered_articles_{timestamp}.json"
    ads_file = output_dir / f"advertisement_articles_{timestamp}.json"
    noisy_file = output_dir / f"noisy_articles_{timestamp}.json"
    tv_file = output_dir / f"tv_program_articles_{timestamp}.json"
    job_file = output_dir / f"job_offer_articles_{timestamp}.json"
    
    # Sauvegarder les articles normaux
    logger.info(f"Saving {len(normal_articles)} normal articles to {normal_file}")
    with open(normal_file, 'w', encoding='utf-8') as f:
        json.dump(normal_articles, f, ensure_ascii=False, indent=2)
    
    # Sauvegarder les articles publicitaires
    logger.info(f"Saving {len(ad_articles)} advertisement articles to {ads_file}")
    with open(ads_file, 'w', encoding='utf-8') as f:
        json.dump(ad_articles, f, ensure_ascii=False, indent=2)
    
    # Sauvegarder les articles bruités
    logger.info(f"Saving {len(noisy_articles)} noisy articles to {noisy_file}")
    with open(noisy_file, 'w', encoding='utf-8') as f:
        json.dump(noisy_articles, f, ensure_ascii=False, indent=2)
    
    # Sauvegarder les articles de programmes TV
    logger.info(f"Saving {len(tv_articles)} TV program articles to {tv_file}")
    with open(tv_file, 'w', encoding='utf-8') as f:
        json.dump(tv_articles, f, ensure_ascii=False, indent=2)
    
    # Sauvegarder les articles d'offres d'emploi
    logger.info(f"Saving {len(job_articles)} job offer articles to {job_file}")
    with open(job_file, 'w', encoding='utf-8') as f:
        json.dump(job_articles, f, ensure_ascii=False, indent=2)
    
    logger.info("Filtering complete!")
    logger.info(f"Normal articles: {len(normal_articles)} ({len(normal_articles)/len(articles):.2%})")
    logger.info(f"Advertisement articles: {len(ad_articles)} ({len(ad_articles)/len(articles):.2%})")
    logger.info(f"Noisy articles: {len(noisy_articles)} ({len(noisy_articles)/len(articles):.2%})")
    logger.info(f"TV program articles: {len(tv_articles)} ({len(tv_articles)/len(articles):.2%})")
    logger.info(f"Job offer articles: {len(job_articles)} ({len(job_articles)/len(articles):.2%})")

if __name__ == "__main__":
    main()
