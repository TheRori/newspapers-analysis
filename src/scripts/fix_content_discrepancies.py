#!/usr/bin/env python3
"""
Script pour détecter et corriger les articles qui présentent trop de différences
entre les champs 'content' et 'original_content' dans le fichier articles.json.
Dans ces cas, 'content' est remplacé par 'original_content'.
"""

import os
import sys
import json
import argparse
import logging
from pathlib import Path
from datetime import datetime
import difflib
import re
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

def normalize_text(text):
    """
    Normalise un texte pour la comparaison.
    
    Args:
        text: Texte à normaliser
        
    Returns:
        Texte normalisé
    """
    if not text:
        return ""
    
    # Normaliser les textes pour la comparaison
    return re.sub(r'\s+', ' ', text.strip().lower())

def calculate_similarity(text1, text2):
    """
    Calcule la similarité entre deux textes en utilisant difflib.
    
    Args:
        text1: Premier texte
        text2: Deuxième texte
        
    Returns:
        Score de similarité entre 0 et 1
    """
    if not text1 or not text2:
        return 0.0
    
    # Normaliser les textes pour la comparaison
    text1 = normalize_text(text1)
    text2 = normalize_text(text2)
    
    # Utiliser SequenceMatcher pour calculer la similarité
    similarity = difflib.SequenceMatcher(None, text1, text2).ratio()
    return similarity

def process_article_batch(articles_batch, threshold=0.7, dry_run=False):
    """
    Traite un lot d'articles pour détecter et corriger les différences entre 'content' et 'original_content'.
    
    Args:
        articles_batch: Liste d'articles à traiter
        threshold: Seuil de similarité
        dry_run: Mode simulation sans modification
        
    Returns:
        Tuple (articles modifiés, nombre d'articles modifiés, nombre d'articles détectés)
    """
    modified_count = 0
    detected_count = 0
    results = []
    
    for article in articles_batch:
        # Vérifier si les deux champs existent
        if 'content' not in article or 'original_content' not in article:
            results.append(article)
            continue
            
        content = article.get('content', '')
        original_content = article.get('original_content', '')
        
        # Ignorer les cas où l'un des champs est vide
        if not content or not original_content:
            results.append(article)
            continue
            
        # Calculer la similarité
        similarity = calculate_similarity(content, original_content)
        
        # Si la similarité est inférieure au seuil, remplacer 'content' par 'original_content'
        if similarity < threshold:
            article_id = article.get('_id', article.get('id', 'unknown'))
            detected_count += 1
            
            if not dry_run:
                # Sauvegarder l'ancien contenu pour référence
                article['original_cleaned_content'] = article['content']
                
                # Remplacer 'content' par 'original_content'
                article['content'] = article['original_content']
                
                # Si 'cleaned_text' existe, le supprimer pour qu'il soit recalculé
                if 'cleaned_text' in article:
                    article.pop('cleaned_text')
                    
                modified_count += 1
        
        results.append(article)
    
    return results, modified_count, detected_count

def detect_and_fix_content_discrepancies(articles, threshold=0.7, dry_run=False, batch_size=1000, num_processes=None):
    """
    Détecte et corrige les articles qui présentent trop de différences entre 'content' et 'original_content'.
    Utilise une approche parallèle pour traiter efficacement un grand nombre d'articles.
    
    Args:
        articles: Liste des articles à analyser
        threshold: Seuil de similarité en dessous duquel 'content' est remplacé par 'original_content'
        dry_run: Si True, n'effectue pas les modifications mais affiche seulement les articles à modifier
        batch_size: Taille des lots d'articles à traiter en parallèle
        num_processes: Nombre de processus à utiliser (None = auto)
        
    Returns:
        Tuple (articles modifiés, nombre d'articles modifiés, nombre d'articles détectés)
    """
    # Déterminer le nombre de processus à utiliser
    if num_processes is None:
        num_processes = max(1, multiprocessing.cpu_count() - 1)
    
    total_articles = len(articles)
    logger.info(f"Analyse de {total_articles} articles avec {num_processes} processus...")
    
    # Diviser les articles en lots pour le traitement parallèle
    batches = [articles[i:i+batch_size] for i in range(0, len(articles), batch_size)]
    logger.info(f"Créé {len(batches)} lots d'articles pour traitement parallèle")
    
    # Préparer la fonction de traitement avec les paramètres
    process_func = partial(process_article_batch, threshold=threshold, dry_run=dry_run)
    
    # Traiter les lots en parallèle
    total_modified = 0
    total_detected = 0
    processed_articles = []
    
    with multiprocessing.Pool(processes=num_processes) as pool:
        # Utiliser tqdm pour afficher une barre de progression
        results = list(tqdm.tqdm(pool.imap(process_func, batches), total=len(batches), desc="Traitement des articles"))
        
        # Collecter les résultats
        for batch_articles, batch_modified, batch_detected in results:
            processed_articles.extend(batch_articles)
            total_modified += batch_modified
            total_detected += batch_detected
    
    # Journaliser les résultats détaillés si des articles ont été détectés
    if total_detected > 0:
        logger.info(f"Détecté {total_detected} articles avec des différences significatives")
        if not dry_run:
            logger.info(f"Modifié {total_modified} articles")
        else:
            logger.info(f"Mode dry-run: aucune modification n'a été effectuée")
    else:
        logger.info("Aucun article avec des différences significatives n'a été détecté")
    
    return processed_articles, total_modified, total_detected

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Détecter et corriger les différences entre content et original_content')
    
    parser.add_argument('--input', type=str, help='Chemin vers le fichier articles.json d\'entrée (par défaut: utilise le chemin dans config.yaml)')
    parser.add_argument('--output', type=str, help='Chemin vers le fichier de sortie (par défaut: remplace le fichier d\'entrée)')
    parser.add_argument('--threshold', type=float, default=0.7, help='Seuil de similarité en dessous duquel content est remplacé (défaut: 0.7)')
    parser.add_argument('--batch-size', type=int, default=1000, help='Taille des lots d\'articles à traiter en parallèle (défaut: 1000)')
    parser.add_argument('--processes', type=int, default=None, help='Nombre de processus à utiliser (défaut: nombre de CPU - 1)')
    parser.add_argument('--dry-run', action='store_true', help='Exécuter sans effectuer de modifications')
    parser.add_argument('--config', type=str, default='config/config.yaml', help='Chemin vers le fichier de configuration YAML')
    
    return parser.parse_args()

def main():
    """Fonction principale pour détecter et corriger les différences entre content et original_content."""
    # Charger la configuration
    config_path = os.path.join(project_root, "config", "config.yaml")
    config = load_config(config_path)
    
    # Analyser les arguments
    args = parse_arguments()
    
    # Déterminer le chemin d'entrée
    input_path = args.input
    if not input_path:
        processed_dir = config.get('data', {}).get('processed_dir', 'data/processed')
        input_path = os.path.join(project_root, processed_dir, "articles.json")
    
    # Déterminer le chemin de sortie
    output_path = args.output
    if not output_path:
        if args.dry_run:
            # En mode dry-run, créer un fichier temporaire pour éviter d'écraser le fichier d'origine
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = os.path.join(os.path.dirname(input_path), f"filtered_articles_{timestamp}.json")
        else:
            # Par défaut, remplacer le fichier d'origine
            output_path = input_path
    
    logger.info(f"Lecture des articles depuis {input_path}")
    
    try:
        # Charger les articles
        with open(input_path, 'r', encoding='utf-8') as f:
            articles = json.load(f)
        
        logger.info(f"Chargé {len(articles)} articles")
        
        # Détecter et corriger les différences
        start_time = datetime.now()
        logger.info(f"Début de l'analyse: {start_time.strftime('%H:%M:%S')}")
        
        modified_articles, modified_count, detected_count = detect_and_fix_content_discrepancies(
            articles, 
            threshold=args.threshold,
            dry_run=args.dry_run,
            batch_size=args.batch_size,
            num_processes=args.processes
        )
        
        end_time = datetime.now()
        duration = end_time - start_time
        logger.info(f"Fin de l'analyse: {end_time.strftime('%H:%M:%S')} (durée: {duration})")
        logger.info(f"Détecté {detected_count} articles avec des différences significatives")
        
        if not args.dry_run and modified_count > 0:
            # Sauvegarder les articles modifiés
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(modified_articles, f, ensure_ascii=False, indent=2)
            
            logger.info(f"Sauvegardé {len(modified_articles)} articles dans {output_path}")
            logger.info(f"Modifié {modified_count} articles")
        elif args.dry_run and modified_count > 0:
            logger.info(f"Mode dry-run: aucune modification n'a été effectuée")
            logger.info(f"{modified_count} articles seraient modifiés")
        else:
            logger.info("Aucun article n'a été modifié")
    
    except Exception as e:
        logger.error(f"Erreur lors du traitement des articles: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
