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
sys.path.insert(0, str(project_root))

# Import using relative imports
try:
    from src.utils.config_loader import load_config
    from src.preprocessing.data_loader import DataLoader
    from src.analysis.term_tracking import (
        count_term_occurrences, 
        count_terms_by_year, 
        export_term_frequencies_to_csv,
        count_terms_by_newspaper
    )
except ModuleNotFoundError:
    # Alternative import path if the above fails
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from utils.config_loader import load_config
    from preprocessing.data_loader import DataLoader
    from analysis.term_tracking import (
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


def normalize_article_id(article_id):
    """
    Normalise un ID d'article pour s'assurer que les comparaisons fonctionnent correctement.
    
    Args:
        article_id: ID d'article à normaliser
        
    Returns:
        ID d'article normalisé
    """
    # Si l'ID est None ou vide, retourner une chaîne vide
    if not article_id:
        return ""
    
    # Convertir en chaîne de caractères
    article_id = str(article_id)
    
    # Supprimer les préfixes courants comme "article_" s'ils sont présents
    if article_id.startswith("article_"):
        article_id = article_id[8:]
    
    # Supprimer les suffixes comme "_mistral" s'ils sont présents
    if "_mistral" in article_id:
        article_id = article_id.split("_mistral")[0]
    
    return article_id


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
    
    # Ajouter les options pour l'analyse sémantique
    parser.add_argument(
        "--semantic-drift",
        action="store_true",
        help="Activer l'analyse de drift sémantique avec Word2Vec"
    )
    
    parser.add_argument(
        "--filter-redundant",
        action="store_true",
        default=True,
        help="Filtrer les termes redondants dans les résultats de similarité (par défaut: activé)"
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
    
    parser.add_argument(
        "--article-list",
        type=str,
        help="Chemin vers un fichier contenant une liste d'IDs d'articles à analyser (un ID par ligne)"
    )
    
    parser.add_argument(
        "--source-file",
        type=str,
        help="Chemin vers un fichier JSON d'articles alternatif (remplace celui de la config)"
    )
    
    args = parser.parse_args()
    
    # Vérifier si un fichier de termes a été spécifié
    if not args.term_file:
        logger.error("Vous devez spécifier un fichier de termes avec --term-file")
        parser.print_help()
        sys.exit(1)
    
    # Utiliser des chemins absolus pour éviter les problèmes lorsque le script est appelé depuis différents répertoires
    script_dir = Path(__file__).parent.absolute()
    project_root = script_dir.parent.parent
    config_path = project_root / "config" / "config.yaml"
    
    # Log des chemins pour le débogage
    logger.info(f"Répertoire du script: {script_dir}")
    logger.info(f"Répertoire racine du projet: {project_root}")
    logger.info(f"Chemin du fichier de configuration: {config_path}")
    
    # Charger la configuration
    config = load_config(str(config_path))
    
    # Créer le répertoire de résultats s'il n'existe pas
    results_dir = project_root / "data" / "results" / "term_tracking"
    results_dir.mkdir(exist_ok=True, parents=True)
    logger.info(f"Répertoire de résultats: {results_dir}")
    
    # Charger les articles
    logger.info("Chargement des articles...")
    data_loader = DataLoader(config['data'])
    
    # Déterminer le chemin du fichier d'articles (par défaut ou personnalisé)
    if args.source_file:
        articles_path = Path(args.source_file)
        logger.info(f"Utilisation d'un fichier d'articles personnalisé: {articles_path}")
    else:
        articles_path = project_root / "data" / "processed" / "articles.json"
        logger.info(f"Utilisation du fichier d'articles par défaut: {articles_path}")
    
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
    
    # Filtrer les articles si une liste est fournie
    if args.article_list:
        try:
            with open(args.article_list, 'r', encoding='utf-8') as f:
                article_ids = [line.strip() for line in f if line.strip()]
            
            logger.info(f"Fichier de liste d'articles trouvé: {args.article_list} avec {len(article_ids)} articles")
            
            # Afficher les premiers IDs pour débogage
            if article_ids:
                logger.info(f"Exemples d'IDs d'articles dans la liste: {article_ids[:5]}")
            
            # Filtrer les articles pour ne garder que ceux dans la liste
            filtered_articles = []
            for article in articles:
                # Construire l'ID complet de l'article pour la comparaison
                article_id = article.get('id', '')
                if not article_id:
                    # Si l'ID n'est pas présent, essayer de le construire à partir des autres champs
                    date = article.get('date', '')
                    journal = article.get('journal', '')
                    base_id = article.get('base_id', '')
                    if date and journal and base_id:
                        article_id = f"article_{date}_{journal}_{base_id}_mistral"
                
                if article_id in article_ids:
                    filtered_articles.append(article)
                    logger.debug(f"Article trouvé: {article_id}")
            
            # Afficher des informations de débogage
            logger.info(f"Nombre total d'articles avant filtrage: {len(articles)}")
            logger.info(f"Nombre d'articles après filtrage: {len(filtered_articles)}")
            
            if len(filtered_articles) == 0:
                # Si aucun article n'a été trouvé, afficher des informations supplémentaires pour le débogage
                logger.warning("Aucun article ne correspond aux IDs fournis dans la liste.")
                
                # Vérifier si les IDs sont au bon format
                if articles and article_ids:
                    # Construire l'ID complet du premier article pour la comparaison
                    sample_article = articles[0]
                    sample_id = sample_article.get('id', '')
                    if not sample_id:
                        date = sample_article.get('date', '')
                        journal = sample_article.get('journal', '')
                        base_id = sample_article.get('base_id', '')
                        if date and journal and base_id:
                            sample_id = f"article_{date}_{journal}_{base_id}_mistral"
                    
                    logger.info(f"Format d'ID d'article dans les données: {sample_id}")
                    logger.info(f"Format d'ID d'article dans la liste: {article_ids[0]}")
            
            articles = filtered_articles
            logger.info(f"Filtré à {len(articles)} articles à partir de la liste dans {args.article_list}")
            
            if not articles:
                logger.warning("Aucun article ne correspond aux IDs fournis dans la liste.")
        except Exception as e:
            logger.error(f"Erreur lors du filtrage des articles: {e}")
    
    # Charger les termes
    logger.info(f"Chargement des termes depuis {args.term_file}...")
    terms = load_terms(args.term_file)
    logger.info(f"Chargé {sum(len(terms_list) for terms_list in terms.values())} termes dans {len(terms)} catégories")
    
    # Aplatir le dictionnaire de termes en une seule liste pour certaines analyses
    flat_terms = [term for terms_list in terms.values() for term in terms_list]
    
    # Exécuter l'analyse de drift sémantique si demandée
    if args.semantic_drift:
        from src.analysis.semantic_drift import (
            create_temporal_word2vec_models,
            calculate_semantic_drift,
            export_semantic_drift_to_csv,
            find_most_similar_terms,
            export_similar_terms_to_csv
        )
        
        logger.info("Analyse de drift sémantique avec Word2Vec...")
        
        # Traiter les périodes personnalisées si spécifiées
        custom_periods = None
        if args.custom_periods:
            try:
                custom_periods = json.loads(args.custom_periods)
                logger.info(f"Utilisation de périodes personnalisées: {custom_periods}")
            except json.JSONDecodeError:
                logger.error(f"Format de périodes personnalisées invalide: {args.custom_periods}")
                sys.exit(1)
        
        # Créer les modèles Word2Vec par période
        models = create_temporal_word2vec_models(
            articles=articles,
            period_type=args.period_type,
            custom_periods=custom_periods,
            vector_size=args.vector_size,
            window=args.window,
            min_count=args.min_count,
            workers=4,
            sg=1  # Skip-gram
        )
        
        if not models:
            logger.error("Aucun modèle Word2Vec n'a pu être créé")
            sys.exit(1)
        
        # Calculer le drift sémantique
        drift_results = calculate_semantic_drift(
            models=models,
            terms=flat_terms,
            reference_period=args.reference_period
        )
        
        # Exporter les résultats
        semantic_output = f"semantic_drift_{os.path.splitext(args.output)[0]}.csv"
        output_path = export_semantic_drift_to_csv(
            drift_results=drift_results,
            output_file=semantic_output,
            results_dir=str(results_dir)
        )
        logger.info(f"Résultats de drift sémantique exportés vers {output_path}")
        
        # Trouver les termes similaires si demandé
        if args.similar_terms:
            similar_terms_list = [term.strip() for term in args.similar_terms.split(',')]
            logger.info(f"Recherche des {args.top_n} mots les plus similaires pour: {similar_terms_list}")
            
            similarity_results = find_most_similar_terms(
                models=models,
                terms=similar_terms_list,
                top_n=args.top_n,
                filter_redundant=args.filter_redundant
            )
            
            if similarity_results:
                similar_output = f"similar_terms_{os.path.splitext(args.output)[0]}.csv"
                # Préparer les métadonnées pour l'exportation des termes similaires
                similar_metadata = {
                    "source_file": str(articles_path),
                    "term_file": args.term_file,
                    "analysis_type": "similar_terms",
                    "model_parameters": {
                        "vector_size": args.vector_size,
                        "window": args.window,
                        "min_count": args.min_count,
                        "period_type": args.period_type,
                        "filter_redundant": args.filter_redundant
                    }
                }
                
                similar_path = export_similar_terms_to_csv(
                    similarity_results=similarity_results,
                    output_file=similar_output,
                    results_dir=str(results_dir),
                    source_file=str(articles_path),
                    metadata=similar_metadata
                )
                logger.info(f"Résultats des termes similaires exportés vers: {similar_path}")
            else:
                logger.warning("Aucun résultat de termes similaires n'a été généré")
    
    # Exécuter l'analyse appropriée pour le term tracking classique
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
        results = count_term_occurrences(articles, flat_terms)
        logger.info(f"Analyse terminée pour {len(results)} articles contenant les termes recherchés")
    
    # Préparer les métadonnées pour l'exportation
    metadata = {
        "source_file": str(articles_path),
        "term_file": args.term_file,
        "analysis_type": "by_year" if args.by_year else "by_newspaper" if args.by_newspaper else "by_article",
        "statistics": {
            # Ces statistiques seront complétées par la fonction d'exportation
        }
    }
    
    # Ajouter des statistiques sur les articles
    if len(articles) > 0:
        # Extraire les dates pour déterminer la période couverte
        dates = []
        journals = set()
        for article in articles:
            if 'date' in article and article['date']:
                dates.append(article['date'])
            if 'journal' in article and article['journal']:
                journals.add(article['journal'])
        
        if dates:
            metadata["statistics"]["period_start"] = min(dates)
            metadata["statistics"]["period_end"] = max(dates)
        
        metadata["statistics"]["total_articles"] = len(articles)
        metadata["statistics"]["total_journals"] = len(journals)
        metadata["statistics"]["journals"] = list(journals)
    
    # Exporter les résultats avec les métadonnées et le fichier source
    output_path = export_term_frequencies_to_csv(
        term_counts=results, 
        output_file=args.output, 
        results_dir=str(results_dir),
        source_file=str(articles_path),
        metadata=metadata
    )
    logger.info(f"Résultats exportés vers {output_path}")


if __name__ == "__main__":
    main()
