#!/usr/bin/env python3
"""
Script pour exécuter l'analyse de suivi des termes sur un corpus d'articles.
Permet de rechercher des termes prédéfinis et d'analyser leur distribution temporelle.

Exemple d'utilisation:
    python run_term_tracking.py --term-file terms.json --output term_results.csv
"""

import os
import sys
import json
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Any

# Ajouter le répertoire parent au chemin pour permettre les imports relatifs
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from src.utils.config_loader import load_config
from src.preprocessing.data_loader import DataLoader
from src.analysis.term_tracking import (
    count_term_occurrences, 
    count_terms_by_year, 
    export_term_frequencies_to_csv,
    count_terms_by_newspaper
)

# Configuration du logger
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_terms(term_file: str) -> Dict[str, List[str]]:
    """
    Charge les termes à partir d'un fichier JSON.
    
    Args:
        term_file: Chemin vers le fichier JSON contenant les termes
        
    Returns:
        Dictionnaire avec des catégories comme clés et des listes de termes comme valeurs
    """
    try:
        with open(term_file, 'r', encoding='utf-8') as f:
            terms = json.load(f)
        
        # Vérifier la structure du fichier
        if isinstance(terms, list):
            # Si c'est une simple liste, créer une catégorie unique
            return {"default": terms}
        elif isinstance(terms, dict):
            # Si c'est déjà un dictionnaire de catégories, le retourner tel quel
            return terms
        else:
            raise ValueError("Format de fichier de termes non reconnu")
    
    except (json.JSONDecodeError, FileNotFoundError) as e:
        logger.error(f"Erreur lors du chargement du fichier de termes: {e}")
        sys.exit(1)


def main():
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
    
    args = parser.parse_args()
    
    # Vérifier si un fichier de termes a été spécifié
    if not args.term_file:
        logger.error("Vous devez spécifier un fichier de termes avec --term-file")
        parser.print_help()
        sys.exit(1)
    
    # Charger la configuration
    config = load_config()
    
    # Créer le répertoire de résultats s'il n'existe pas
    results_dir = Path(config['data']['results_dir']) / "term_tracking"
    results_dir.mkdir(exist_ok=True, parents=True)
    
    # Charger les articles
    logger.info("Chargement des articles...")
    data_loader = DataLoader(config['data'])
    articles_path = Path(config['data']['processed_dir']) / "articles.json"
    
    try:
        with open(articles_path, 'r', encoding='utf-8') as f:
            articles = json.load(f)
        logger.info(f"Chargé {len(articles)} articles depuis {articles_path}")
    except FileNotFoundError:
        logger.error(f"Fichier d'articles non trouvé: {articles_path}")
        sys.exit(1)
    
    # Limiter le nombre d'articles si demandé
    if args.limit > 0:
        articles = articles[:args.limit]
        logger.info(f"Limité à {len(articles)} articles")
    
    # Charger les termes
    logger.info(f"Chargement des termes depuis {args.term_file}...")
    terms = load_terms(args.term_file)
    logger.info(f"Chargé {sum(len(terms_list) for terms_list in terms.values())} termes dans {len(terms)} catégories")
    
    # Exécuter l'analyse appropriée
    if args.by_year:
        logger.info("Analyse des termes par année...")
        results = count_terms_by_year(articles, terms)
        logger.info(f"Analyse terminée pour {len(results)} années")
    elif args.by_newspaper:
        logger.info("Analyse des termes par journal...")
        results = count_terms_by_newspaper(articles, terms)
        logger.info(f"Analyse terminée pour {len(results)} journaux")
    else:
        # Par défaut, compter les occurrences par article
        logger.info("Analyse des termes par article...")
        # Aplatir le dictionnaire de termes en une seule liste
        flat_terms = [term for terms_list in terms.values() for term in terms_list]
        results = count_term_occurrences(articles, flat_terms)
        logger.info(f"Analyse terminée pour {len(results)} articles contenant les termes recherchés")
    
    # Exporter les résultats
    output_path = export_term_frequencies_to_csv(results, args.output, str(results_dir))
    logger.info(f"Résultats exportés vers {output_path}")


if __name__ == "__main__":
    main()
