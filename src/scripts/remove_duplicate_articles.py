#!/usr/bin/env python3
"""
Script pour détecter et supprimer les articles dupliqués dans le fichier articles.json.
Les articles sont considérés comme des doublons s'ils ont un contenu original ("original_content")
et une date trop similaires.
"""

import os
import sys
import json
import argparse
import logging
from pathlib import Path
from datetime import datetime, timedelta
import difflib
import re
from collections import defaultdict
import hashlib
import multiprocessing
from functools import partial
import tqdm
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

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

def calculate_content_hash(text):
    """
    Calcule un hash du contenu normalisé pour une comparaison rapide.
    
    Args:
        text: Texte à hasher
        
    Returns:
        Hash du texte
    """
    text = normalize_text(text)
    if not text:
        return ""
    
    # Prendre les 1000 premiers caractères pour un hash rapide
    text = text[:1000]
    return hashlib.md5(text.encode('utf-8')).hexdigest()

def calculate_content_similarity(text1, text2):
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
    
    # Normaliser les textes
    text1 = normalize_text(text1)
    text2 = normalize_text(text2)
    
    # Utiliser SequenceMatcher pour calculer la similarité
    similarity = difflib.SequenceMatcher(None, text1, text2).ratio()
    return similarity

def calculate_similarity_batch(articles_batch, content_threshold=0.85, max_days_diff=3):
    """
    Calcule la similarité entre les articles d'un lot et retourne les paires similaires.
    
    Args:
        articles_batch: Liste de tuples (index, article, date, hash)
        content_threshold: Seuil de similarité
        max_days_diff: Différence maximale en jours
        
    Returns:
        Liste de tuples (i, j, similarité) pour les articles similaires
    """
    similar_pairs = []
    
    # Trier les articles par date pour optimiser la comparaison
    articles_batch.sort(key=lambda x: x[2] if x[2] is not None else datetime.min)
    
    for idx1, (i, article_i, date_i, hash_i) in enumerate(articles_batch):
        if date_i is None:
            continue
            
        # Comparer uniquement avec les articles suivants dans le lot
        for j, article_j, date_j, hash_j in articles_batch[idx1+1:]:
            # Ignorer les articles sans date
            if date_j is None:
                continue
                
            # Vérifier si les dates sont trop éloignées
            if abs((date_j - date_i).days) > max_days_diff:
                # Si les articles sont triés par date, on peut sauter les articles restants
                if date_j > date_i:
                    break
                continue
                
            # Vérifier d'abord si les hash sont identiques
            if hash_i == hash_j and hash_i != "":
                similar_pairs.append((i, j, 1.0))
                continue
                
            # Calculer la similarité seulement si les dates sont proches
            content_i = article_i.get('original_content', '')
            content_j = article_j.get('original_content', '')
            similarity = calculate_content_similarity(content_i, content_j)
            
            if similarity >= content_threshold:
                similar_pairs.append((i, j, similarity))
                
    return similar_pairs

def extract_date_from_article(article):
    """
    Extrait la date d'un article à partir de son ID ou de ses métadonnées.
    
    Args:
        article: Dictionnaire contenant les données de l'article
        
    Returns:
        Objet datetime ou None si la date ne peut pas être extraite
    """
    # Essayer d'extraire la date à partir de l'ID de l'article
    article_id = str(article.get('_id', article.get('id', '')))
    
    # Format attendu: article_YYYY-MM-DD_journal_XXXX_source
    date_match = re.search(r'article_(\d{4}-\d{2}-\d{2})_', article_id)
    if date_match:
        try:
            return datetime.strptime(date_match.group(1), '%Y-%m-%d')
        except ValueError:
            pass
    
    # Essayer d'extraire la date à partir des métadonnées
    if 'date' in article:
        try:
            # Essayer différents formats de date
            date_formats = ['%Y-%m-%d', '%d/%m/%Y', '%Y/%m/%d', '%d-%m-%Y']
            for date_format in date_formats:
                try:
                    return datetime.strptime(article['date'], date_format)
                except ValueError:
                    continue
        except Exception:
            pass
    
    # Si aucune date n'est trouvée, retourner None
    return None

def are_dates_close(date1, date2, max_days_diff=3):
    """
    Vérifie si deux dates sont proches l'une de l'autre.
    
    Args:
        date1: Première date (datetime)
        date2: Deuxième date (datetime)
        max_days_diff: Différence maximale en jours
        
    Returns:
        True si les dates sont proches, False sinon
    """
    if date1 is None or date2 is None:
        return False
    
    return abs((date1 - date2).days) <= max_days_diff

def detect_duplicate_articles(articles, content_threshold=0.85, max_days_diff=3, dry_run=False, batch_size=1000, num_processes=None):
    """
    Détecte les articles dupliqués en fonction de la similarité du contenu et de la proximité des dates.
    Utilise une approche optimisée pour traiter un grand nombre d'articles.
    
    Args:
        articles: Liste des articles à analyser
        content_threshold: Seuil de similarité au-dessus duquel les articles sont considérés comme similaires
        max_days_diff: Différence maximale en jours entre les dates des articles
        dry_run: Si True, n'effectue pas les modifications mais affiche seulement les articles à supprimer
        batch_size: Taille des lots d'articles à traiter en parallèle
        num_processes: Nombre de processus à utiliser (None = auto)
        
    Returns:
        Tuple (articles filtrés, nombre d'articles supprimés, groupes de doublons)
    """
    # Déterminer le nombre de processus à utiliser
    if num_processes is None:
        num_processes = max(1, multiprocessing.cpu_count() - 1)
    
    # Dictionnaire pour stocker les groupes d'articles similaires
    duplicate_groups = defaultdict(list)
    
    # Identifier les articles qui ont un contenu original
    logger.info("Filtrage des articles avec contenu original...")
    articles_with_original_content = []
    for i, a in enumerate(articles):
        if 'original_content' in a and a['original_content']:
            articles_with_original_content.append((i, a))
    
    num_articles = len(articles_with_original_content)
    logger.info(f"Trouvé {num_articles} articles avec contenu original")
    
    if num_articles == 0:
        return articles, 0, {}
    
    # Extraire les dates et calculer les hash des articles
    logger.info("Extraction des dates et calcul des hash...")
    processed_articles = []
    for i, article in tqdm.tqdm(articles_with_original_content, desc="Préparation des articles"):
        date = extract_date_from_article(article)
        content_hash = calculate_content_hash(article.get('original_content', ''))
        processed_articles.append((i, article, date, content_hash))
    
    # Regrouper les articles par hash pour détecter rapidement les doublons exacts
    hash_groups = defaultdict(list)
    for i, article, date, content_hash in processed_articles:
        if content_hash:  # Ignorer les hash vides
            hash_groups[content_hash].append((i, article, date))
    
    # Traiter les groupes de hash identiques
    logger.info("Traitement des articles avec hash identiques...")
    exact_duplicate_count = 0
    for content_hash, group in hash_groups.items():
        if len(group) > 1:
            # Créer un groupe de doublons
            group_id = group[0][0]  # Utiliser l'index du premier article comme ID de groupe
            duplicate_groups[group_id] = [item[0] for item in group]  # Ajouter tous les indices
            exact_duplicate_count += len(group) - 1
            
            # Journaliser les doublons exacts trouvés
            article_ids = [item[1].get('_id', item[1].get('id', i)) for i, item in enumerate(group)]
            logger.info(f"Doublons exacts trouvés: {article_ids} (hash: {content_hash[:8]}...)")
    
    logger.info(f"Trouvé {exact_duplicate_count} doublons exacts par hash")
    
    # Pour les articles restants, utiliser une approche par lots avec multiprocessing
    remaining_articles = []
    for i, article, date, content_hash in processed_articles:
        # Vérifier si l'article est déjà dans un groupe de doublons exacts
        in_exact_group = False
        for group in duplicate_groups.values():
            if i in group and len(group) > 1:
                in_exact_group = True
                break
        
        if not in_exact_group and date is not None:
            remaining_articles.append((i, article, date, content_hash))
    
    # Trier les articles par date pour optimiser les comparaisons
    remaining_articles.sort(key=lambda x: x[2])
    
    # Diviser les articles en lots pour le traitement parallèle
    num_remaining = len(remaining_articles)
    logger.info(f"Analyse de {num_remaining} articles restants pour similarité de contenu...")
    
    if num_remaining > 0:
        # Créer des lots d'articles proches en date
        batches = []
        current_batch = []
        current_date = remaining_articles[0][2]
        
        for article_data in remaining_articles:
            article_date = article_data[2]
            
            # Si la date est trop éloignée ou le lot est plein, créer un nouveau lot
            if len(current_batch) >= batch_size or (article_date - current_date).days > max_days_diff * 2:
                if current_batch:
                    batches.append(current_batch)
                current_batch = [article_data]
                current_date = article_date
            else:
                current_batch.append(article_data)
        
        # Ajouter le dernier lot
        if current_batch:
            batches.append(current_batch)
        
        logger.info(f"Créé {len(batches)} lots d'articles pour traitement parallèle")
        
        # Traiter les lots en parallèle
        similar_pairs = []
        process_func = partial(calculate_similarity_batch, content_threshold=content_threshold, max_days_diff=max_days_diff)
        
        with multiprocessing.Pool(processes=num_processes) as pool:
            for batch_results in tqdm.tqdm(pool.imap(process_func, batches), total=len(batches), desc="Traitement des lots"):
                similar_pairs.extend(batch_results)
        
        # Ajouter les paires similaires aux groupes de doublons
        for i, j, similarity in similar_pairs:
            # Vérifier si i ou j est déjà dans un groupe
            group_i = None
            group_j = None
            
            for group_id, group in duplicate_groups.items():
                if i in group:
                    group_i = group_id
                if j in group:
                    group_j = group_id
                if group_i is not None and group_j is not None:
                    break
            
            # Si les deux articles sont déjà dans des groupes différents, fusionner les groupes
            if group_i is not None and group_j is not None and group_i != group_j:
                duplicate_groups[group_i].extend([idx for idx in duplicate_groups[group_j] if idx not in duplicate_groups[group_i]])
                del duplicate_groups[group_j]
            # Si seul i est dans un groupe, ajouter j à ce groupe
            elif group_i is not None:
                if j not in duplicate_groups[group_i]:
                    duplicate_groups[group_i].append(j)
            # Si seul j est dans un groupe, ajouter i à ce groupe
            elif group_j is not None:
                if i not in duplicate_groups[group_j]:
                    duplicate_groups[group_j].append(i)
            # Si aucun des deux n'est dans un groupe, créer un nouveau groupe
            else:
                duplicate_groups[i].append(i)
                duplicate_groups[i].append(j)
            
            # Journaliser la paire similaire
            article_i = articles[i]
            article_j = articles[j]
            article_i_id = article_i.get('_id', article_i.get('id', i))
            article_j_id = article_j.get('_id', article_j.get('id', j))
            logger.info(f"Articles {article_i_id} et {article_j_id} sont similaires (similarité: {similarity:.4f})")
    
    # Filtrer les groupes pour ne garder que ceux avec plus d'un article
    duplicate_groups = {k: v for k, v in duplicate_groups.items() if len(v) > 1}
    
    # Créer une liste des indices d'articles à supprimer
    articles_to_remove = set()
    for group in duplicate_groups.values():
        # Garder le premier article du groupe et supprimer les autres
        articles_to_keep = group[0]
        for article_idx in group[1:]:
            articles_to_remove.add(article_idx)
    
    # Créer une nouvelle liste d'articles sans les doublons
    if not dry_run:
        filtered_articles = []
        for i, article in enumerate(articles):
            # Vérifier si l'article est dans la liste des articles à supprimer
            is_duplicate = False
            for group in duplicate_groups.values():
                if i in group[1:]:  # Ignorer le premier article de chaque groupe
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                filtered_articles.append(article)
    else:
        filtered_articles = articles
    
    # Nombre d'articles supprimés
    removed_count = len(articles_to_remove)
    
    return filtered_articles, removed_count, duplicate_groups

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Détecter et supprimer les articles dupliqués')
    
    parser.add_argument('--input', type=str, help='Chemin vers le fichier articles.json d\'entrée (par défaut: utilise le chemin dans config.yaml)')
    parser.add_argument('--output', type=str, help='Chemin vers le fichier de sortie (par défaut: remplace le fichier d\'entrée)')
    parser.add_argument('--content-threshold', type=float, default=0.85, help='Seuil de similarité du contenu (défaut: 0.85)')
    parser.add_argument('--max-days-diff', type=int, default=3, help='Différence maximale en jours entre les dates (défaut: 3)')
    parser.add_argument('--batch-size', type=int, default=1000, help='Taille des lots d\'articles à traiter en parallèle (défaut: 1000)')
    parser.add_argument('--processes', type=int, default=None, help='Nombre de processus à utiliser (défaut: nombre de CPU - 1)')
    parser.add_argument('--dry-run', action='store_true', help='Exécuter sans effectuer de modifications')
    parser.add_argument('--config', type=str, default='config/config.yaml', help='Chemin vers le fichier de configuration YAML')
    
    return parser.parse_args()

def main():
    """Fonction principale pour détecter et supprimer les articles dupliqués."""
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
            output_path = os.path.join(os.path.dirname(input_path), f"deduplicated_articles_{timestamp}.json")
        else:
            # Par défaut, remplacer le fichier d'origine
            output_path = input_path
    
    logger.info(f"Lecture des articles depuis {input_path}")
    
    try:
        # Charger les articles
        with open(input_path, 'r', encoding='utf-8') as f:
            articles = json.load(f)
        
        logger.info(f"Chargé {len(articles)} articles")
        
        # Détecter et supprimer les doublons
        start_time = datetime.now()
        logger.info(f"Début de l'analyse: {start_time.strftime('%H:%M:%S')}")
        
        filtered_articles, removed_count, duplicate_groups = detect_duplicate_articles(
            articles, 
            content_threshold=args.content_threshold,
            max_days_diff=args.max_days_diff,
            dry_run=args.dry_run,
            batch_size=args.batch_size,
            num_processes=args.processes
        )
        
        end_time = datetime.now()
        duration = end_time - start_time
        logger.info(f"Fin de l'analyse: {end_time.strftime('%H:%M:%S')} (durée: {duration})")
        
        # Afficher les statistiques
        logger.info(f"Détecté {len(duplicate_groups)} groupes d'articles dupliqués")
        logger.info(f"Nombre total d'articles dupliqués à supprimer: {removed_count}")
        
        if not args.dry_run and removed_count > 0:
            # Sauvegarder les articles filtrés
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(filtered_articles, f, ensure_ascii=False, indent=2)
            
            logger.info(f"Sauvegardé {len(filtered_articles)} articles dans {output_path}")
            logger.info(f"Supprimé {removed_count} articles dupliqués")
        elif args.dry_run and removed_count > 0:
            logger.info(f"Mode dry-run: aucune modification n'a été effectuée")
            logger.info(f"{removed_count} articles seraient supprimés")
            
            # Afficher des détails sur les groupes de doublons
            for group_id, group in duplicate_groups.items():
                article_ids = [articles[idx].get('_id', articles[idx].get('id', idx)) for idx in group]
                logger.info(f"Groupe {group_id}: {article_ids}")
        else:
            logger.info("Aucun article dupliqué n'a été détecté")
    
    except Exception as e:
        logger.error(f"Erreur lors du traitement des articles: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
