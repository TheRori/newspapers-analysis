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

def process_article_batch(articles_batch, batch_idx=0, total_batches=1):
    """
    Traite un lot d'articles pour détecter les publicités et le bruit.
    
    Args:
        articles_batch: Liste d'articles à traiter
        batch_idx: Index du lot actuel (pour le logging)
        total_batches: Nombre total de lots (pour le logging)
        
    Returns:
        Tuple (articles normaux, articles publicitaires, articles bruités)
    """
    normal_articles = []
    ad_articles = []
    noisy_articles = []
    
    for i, article in enumerate(articles_batch):
        if i % 1000 == 0 and i > 0:
            logger.debug(f"Batch {batch_idx+1}/{total_batches}: Processed {i}/{len(articles_batch)} articles")
        
        if is_advertisement(article):
            ad_articles.append(article)
        elif has_excessive_noise(article):
            noisy_articles.append(article)
        else:
            normal_articles.append(article)
    
    return normal_articles, ad_articles, noisy_articles

def filter_ads_and_noise(articles, batch_size=1000, num_processes=None):
    """
    Filtre les articles contenant des publicités ou du bruit.
    
    Args:
        articles: Liste des articles à analyser
        batch_size: Taille des lots d'articles à traiter en parallèle
        num_processes: Nombre de processus à utiliser (None = auto)
        
    Returns:
        Tuple (articles normaux, articles publicitaires, articles bruités)
    """
    logger.info(f"Filtering {len(articles)} articles for ads and noise...")
    
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
    
    for normal_batch, ad_batch, noisy_batch in results:
        normal_articles.extend(normal_batch)
        ad_articles.extend(ad_batch)
        noisy_articles.extend(noisy_batch)
    
    logger.info(f"Filtering complete: {len(normal_articles)} normal articles, "
                f"{len(ad_articles)} advertisement articles, "
                f"{len(noisy_articles)} noisy articles")
    
    return normal_articles, ad_articles, noisy_articles

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
    """Fonction principale pour filtrer les articles contenant des publicités ou du bruit."""
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
    normal_articles, ad_articles, noisy_articles = filter_ads_and_noise(
        articles,
        batch_size=args.batch_size,
        num_processes=args.processes
    )
    
    # Générer les noms de fichiers de sortie avec timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    normal_file = output_dir / f"filtered_articles_{timestamp}.json"
    ads_file = output_dir / f"advertisement_articles_{timestamp}.json"
    noisy_file = output_dir / f"noisy_articles_{timestamp}.json"
    
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
    
    logger.info("Filtering complete!")
    logger.info(f"Normal articles: {len(normal_articles)} ({len(normal_articles)/len(articles):.2%})")
    logger.info(f"Advertisement articles: {len(ad_articles)} ({len(ad_articles)/len(articles):.2%})")
    logger.info(f"Noisy articles: {len(noisy_articles)} ({len(noisy_articles)/len(articles):.2%})")

if __name__ == "__main__":
    main()
