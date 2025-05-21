#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script pour détecter et filtrer les publicités, offres d'emploi/formation et programmes TV dans un topic spécifique.
"""

import os
import sys
import json
import yaml
import logging
import argparse
import pathlib
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple
from tqdm import tqdm

# Ajouter le répertoire parent au path pour pouvoir importer les modules
script_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.dirname(os.path.dirname(script_dir))
sys.path.insert(0, project_dir)

from src.analysis.llm_utils import LLMClient

# Configurer le logger
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Chemin vers le fichier de configuration
config_path = os.path.join(project_dir, "config", "config.yaml")

# Fonction pour charger la configuration
def load_config(config_path):
    """Charge la configuration depuis un fichier YAML."""
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        return config
    except Exception as e:
        logger.error(f"Erreur lors du chargement de la configuration: {str(e)}")
        return {"llm": {"provider": "mistral", "model": "mistral-small"}}


def process_batch_for_categories(batch: List[Dict[str, Any]], llm_client: LLMClient, batch_size: int) -> List[Tuple[str, str]]:
    """
    Traite un lot d'articles pour déterminer leur catégorie.
    
    Args:
        batch: Liste des articles à analyser
        llm_client: Client LLM pour l'analyse
        batch_size: Taille du lot
        
    Returns:
        Liste de tuples (catégorie, explication) où catégorie est l'une des valeurs suivantes:
        - "article" (article légitime)
        - "ads" (publicité commerciale)
        - "offreemploi" (offre d'emploi ou formation)
        - "programmetv" (programme TV ou agenda)
    """
    # Construire le prompt pour le lot
    prompt = """Tu es chargé de classifier des textes en français en différentes catégories.

Pour chaque texte, tu dois déterminer s'il s'agit:
1. D'un **article journalistique** (catégorie "ARTICLE")
2. D'une **publicité commerciale** (catégorie "ADS")
3. D'une **offre d'emploi ou formation** (catégorie "OFFREEMPLOI")
4. D'un **programme TV ou agenda culturel** (catégorie "PROGRAMMETV")

Critères de classification:
- "ARTICLE": Texte journalistique, éditorial, reportage, analyse, opinion, interview, etc.
- "ADS": Publicité pour un produit, service, magasin, marque, promotion, soldes, etc.
- "OFFREEMPLOI": Offre d'emploi, recrutement, formation professionnelle, stage, etc.
- "PROGRAMMETV": Programme TV, cinéma, théâtre, agenda culturel, horaires d'événements, etc.

### Format de réponse:
Pour chaque texte, réponds UNIQUEMENT avec l'une des 4 catégories suivantes:

Texte 1: ARTICLE
Texte 2: ADS
Texte 3: OFFREEMPLOI
Texte 4: PROGRAMMETV
...

### Exemples:
Texte: "Le Conseil fédéral a annoncé hier de nouvelles mesures économiques." → ARTICLE
Texte: "Ordinateur portable HP à 599 CHF chez MediaMarkt. Offre valable jusqu'à dimanche." → ADS
Texte: "Recherche comptable à 80%, expérience 3 ans minimum, CV à envoyer à rh@entreprise.ch" → OFFREEMPLOI
Texte: "Ce soir sur TF1: 20h15 Film 'Avatar', 22h30 Journal, 23h00 Série 'Dr House'" → PROGRAMMETV
"""
    
    # Ajouter chaque article au prompt
    for i, article in enumerate(batch):
        # Obtenir le contenu de l'article
        content = None
        for key in ['content', 'text', 'original_content']:
            if key in article and article[key]:
                content = article[key]
                break
        
        if not content:
            continue
            
        # Limiter la taille du contenu pour éviter de dépasser les limites du LLM
        content = content[:1000]  # Limiter à 1000 caractères
        
        prompt += f"\nTexte {i+1}: {content}\n"
    
    # Envoyer la requête au LLM
    response = llm_client.ask(prompt, max_tokens=1000)
    
    # Analyser la réponse
    results = []
    lines = response.strip().split('\n')
    
    # Extraire les réponses (ARTICLE, ADS, OFFREEMPLOI, PROGRAMMETV)
    current_text_num = 1
    for line in lines:
        line = line.strip()
        if not line:
            continue
            
        # Essayer de trouver le format "Texte X: CATÉGORIE"
        if line.lower().startswith(f"texte {current_text_num}") or line.lower().startswith(f"texte{current_text_num}"):
            # Extraire la réponse et l'explication
            parts = line.split(':', 1)
            if len(parts) > 1:
                response_part = parts[1].strip().upper()
                
                # Déterminer la catégorie
                if "ARTICLE" in response_part:
                    category = "article"
                    explanation = "Détecté comme article journalistique"
                elif "OFFREEMPLOI" in response_part:
                    category = "offreemploi"
                    explanation = "Détecté comme offre d'emploi ou formation"
                elif "PROGRAMMETV" in response_part:
                    category = "programmetv"
                    explanation = "Détecté comme programme TV ou agenda"
                elif "ADS" in response_part or "PUBLICITÉ" in response_part or "PUBLICITE" in response_part:
                    category = "ads"
                    explanation = "Détecté comme publicité commerciale"
                else:
                    # Par défaut, si la catégorie n'est pas reconnue
                    category = "article"
                    explanation = "Catégorie non reconnue, considéré comme article par défaut"
                
                results.append((category, explanation))
                current_text_num += 1
    
    # S'assurer que nous avons une réponse pour chaque article
    while len(results) < len(batch):
        # Par défaut, considérer comme article si pas de réponse
        results.append(("article", "Pas de réponse du LLM, considéré comme article par défaut"))
    
    # Limiter les résultats à la taille du lot
    results = results[:len(batch)]
    
    # Journaliser les résultats pour débogage
    for i, (article, (category, explanation)) in enumerate(zip(batch, results)):
        article_id = article.get('doc_id', article.get('id', f'article_{i}'))
        logger.info(f"Classification pour document {article_id}: {category.upper()} ({explanation})")
    
    return results


def filter_ads_from_topic_categories(
    articles_path: str,
    doc_topic_matrix_path: str,
    topic_id: int,
    min_topic_value: float = 0.5,
    output_dir: Optional[str] = None,
    dry_run: bool = False,
    batch_size: int = 10,
    max_articles: Optional[int] = None
) -> Dict[str, Any]:
    """
    Filtre et catégorise les articles d'un topic spécifique.
    
    Args:
        articles_path: Chemin vers le fichier JSON contenant les articles
        doc_topic_matrix_path: Chemin vers le fichier JSON contenant la matrice document-topic
        topic_id: ID du topic à analyser
        min_topic_value: Valeur minimale du topic pour considérer un article (0.0-1.0)
        output_dir: Répertoire où sauvegarder les fichiers JSON filtrés (si None, utilise le répertoire d'entrée)
        dry_run: Si True, n'écrit pas les fichiers de sortie
        batch_size: Nombre d'articles à traiter par lot pour les requêtes LLM
        max_articles: Nombre maximum d'articles à analyser (pour les tests)
        
    Returns:
        Dict contenant les statistiques du filtrage
    """
    # Charger la configuration avec le chemin explicite
    config = load_config(config_path)
    
    # Initialiser le client LLM
    if 'llm' not in config:
        raise ValueError("Configuration LLM manquante dans le fichier config.yaml")
    
    llm_client = LLMClient(config['llm'])
    logger.info(f"Client LLM initialisé: {config['llm'].get('provider', 'mistral')}/{config['llm'].get('model', 'mistral-small')}")
    
    # Charger les articles
    logger.info(f"Chargement des articles depuis {articles_path}")
    with open(articles_path, 'r', encoding='utf-8') as f:
        articles = json.load(f)
    
    # Charger la matrice document-topic
    logger.info(f"Chargement de la matrice document-topic depuis {doc_topic_matrix_path}")
    with open(doc_topic_matrix_path, 'r', encoding='utf-8') as f:
        doc_topic_data = json.load(f)
    
    # Vérifier la structure du fichier
    if not isinstance(doc_topic_data, list) and 'doc_topic_matrix' in doc_topic_data:
        doc_topic_matrix = doc_topic_data['doc_topic_matrix']
    else:
        doc_topic_matrix = doc_topic_data
    
    # Initialiser les compteurs et les résultats
    article_count = 0       # Articles légitimes
    ads_count = 0           # Publicités commerciales
    offreemploi_count = 0   # Offres d'emploi ou formations
    programmetv_count = 0   # Programmes TV ou agendas
    short_ads_count = 0     # Articles courts considérés comme publicités

    # Initialiser les listes d'articles par catégorie
    article_articles = []   # Articles légitimes à conserver
    ads_articles = []       # Publicités commerciales
    offreemploi_articles = [] # Offres d'emploi ou formations
    programmetv_articles = [] # Programmes TV ou agendas
    short_ads_articles = [] # Articles courts considérés comme publicités
    
    # Créer un dictionnaire pour accéder rapidement aux valeurs de topic par doc_id
    doc_topic_dict = {}
    for item in doc_topic_matrix:
        doc_id = item.get('doc_id', '')
        topic_distribution = item.get('topic_distribution', [])
        if doc_id and len(topic_distribution) > topic_id:
            doc_topic_dict[str(doc_id)] = topic_distribution[topic_id]
    
    # Filtrer les articles appartenant au topic spécifié avec une valeur minimale
    topic_articles = []
    for article in articles:
        article_id = article.get('doc_id', article.get('id', ''))
        article_id_str = str(article_id)
        
        if article_id_str in doc_topic_dict and doc_topic_dict[article_id_str] >= min_topic_value:
            topic_articles.append(article)
    
    logger.info(f"Nombre d'articles dans le topic {topic_id} (valeur >= {min_topic_value}): {len(topic_articles)}")
    
    # Séparer les articles courts (considérés automatiquement comme publicités)
    # des articles à analyser par le LLM
    short_articles = []
    articles_to_analyze = []
    
    for article in topic_articles:
        # Vérifier si le document a suffisamment de texte
        text_for_check = None
        for key in ['cleaned_text', 'content', 'text', 'original_content']:
            if key in article and article[key]:
                text_for_check = article[key]
                break
        
        if not text_for_check or len(text_for_check.split()) < 80:
            # Document trop court, considéré comme une publicité
            article['category'] = "short_ads"
            article['category_explanation'] = "Article trop court (moins de 80 mots)"
            short_articles.append(article)
            short_ads_count += 1
            # Ajouter aux articles courts
            short_ads_articles.append(article)
        else:
            # Document à analyser par le LLM
            articles_to_analyze.append(article)
    
    logger.info(f"Articles trop courts (filtrés automatiquement comme publicités): {short_ads_count}")
    
    # Conserver tous les articles courts pour le traitement
    all_short_articles = short_articles.copy()
    
    # Limiter le nombre d'articles si max_articles est spécifié
    total_articles_available = len(articles_to_analyze)
    
    if max_articles is not None and max_articles > 0:
        # Calculer combien d'articles nous pouvons analyser
        articles_to_analyze_limit = min(max_articles, total_articles_available)
        logger.info(f"Limitation à {articles_to_analyze_limit} articles sur {total_articles_available} disponibles")
        articles_to_analyze = articles_to_analyze[:articles_to_analyze_limit]
    
    logger.info(f"Articles à analyser: {len(articles_to_analyze)}")
    
    # Si max_articles est utilisé, limiter le nombre d'articles courts
    if max_articles is not None and max_articles > 0:
        # Calculer combien d'articles courts nous pouvons inclure
        remaining_slots = max(0, max_articles - len(articles_to_analyze))
        # Limiter le nombre d'articles courts pour les statistiques
        short_ads_count = min(short_ads_count, remaining_slots)
        # Limiter le nombre d'articles courts dans la liste
        short_ads_articles = short_ads_articles[:remaining_slots]
        logger.info(f"Articles courts inclus dans la limite max_articles: {short_ads_count} sur {len(all_short_articles)} disponibles")
    
    # Traiter les articles par lots
    if articles_to_analyze:
        logger.info(f"Analyse des articles par lots de {batch_size}...")
        
        # Diviser les articles en lots
        batches = [articles_to_analyze[i:i+batch_size] for i in range(0, len(articles_to_analyze), batch_size)]
        
        with tqdm(total=len(articles_to_analyze), desc=f"Analyse du topic {topic_id}") as pbar:
            for batch in batches:
                try:
                    # Traiter le lot
                    results = process_batch_for_categories(batch, llm_client, batch_size)
                    
                    # Traiter les résultats
                    for article, (category, explanation) in zip(batch, results):
                        # Ajouter la catégorie et l'explication dans l'article
                        article['category'] = category
                        article['category_explanation'] = explanation
                        
                        # Ajouter l'article à la liste correspondante
                        if category == "article":
                            article_count += 1
                            article_articles.append(article)
                        elif category == "ads":
                            ads_count += 1
                            ads_articles.append(article)
                        elif category == "offreemploi":
                            offreemploi_count += 1
                            offreemploi_articles.append(article)
                        elif category == "programmetv":
                            programmetv_count += 1
                            programmetv_articles.append(article)
                    
                    # Mettre à jour la barre de progression
                    pbar.update(len(batch))
                    
                except Exception as e:
                    logger.error(f"Erreur lors du traitement d'un lot: {str(e)}")
                    # En cas d'erreur, on garde tous les articles du lot comme articles légitimes par précaution
                    for article in batch:
                        logger.warning(f"Erreur de traitement, article conservé comme légitime par précaution: {article.get('doc_id', article.get('id', 'unknown'))}")
                        article_count += 1
                        article['category'] = "article"
                        article['category_explanation'] = "Erreur de traitement, conservé comme article par précaution"
                        article_articles.append(article)
                    pbar.update(len(batch))
    
    # Préparer les statistiques
    total_analyzed = len(articles_to_analyze) + len(short_ads_articles)
    total_non_articles = ads_count + offreemploi_count + programmetv_count + short_ads_count
    
    # Préparer les chemins de sortie
    if not output_dir:
        output_dir = os.path.dirname(articles_path)
    
    # Créer le répertoire de sortie s'il n'existe pas
    if not os.path.exists(output_dir) and not dry_run:
        os.makedirs(output_dir)
    
    # Construire les noms de fichiers
    base_name = os.path.basename(articles_path).split('.')[0]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    article_output_path = os.path.join(output_dir, f"{base_name}_articles_topic{topic_id}.json")
    ads_output_path = os.path.join(output_dir, f"{base_name}_ads_topic{topic_id}.json")
    offreemploi_output_path = os.path.join(output_dir, f"{base_name}_offreemploi_topic{topic_id}.json")
    programmetv_output_path = os.path.join(output_dir, f"{base_name}_programmetv_topic{topic_id}.json")
    short_ads_output_path = os.path.join(output_dir, f"{base_name}_short_ads_topic{topic_id}.json")
    
    # Préparer les statistiques
    stats = {
        "topic_id": topic_id,
        "min_topic_value": min_topic_value,
        "total_articles": len(topic_articles),
        "analyzed_articles": total_analyzed,
        "article_count": article_count,
        "ads_count": ads_count,
        "offreemploi_count": offreemploi_count,
        "programmetv_count": programmetv_count,
        "short_ads_count": short_ads_count,
        "non_article_percentage": round(total_non_articles / total_analyzed * 100, 2) if total_analyzed > 0 else 0,
        "article_output_path": article_output_path,
        "ads_output_path": ads_output_path,
        "offreemploi_output_path": offreemploi_output_path,
        "programmetv_output_path": programmetv_output_path,
        "short_ads_output_path": short_ads_output_path
    }
    
    # Sauvegarder les résultats si ce n'est pas un dry run
    if not dry_run:
        logger.info(f"=== Résumé du filtrage (topic {topic_id}) ===")
        logger.info(f"Articles analysés: {stats['analyzed_articles']}")
        logger.info(f"Articles légitimes: {stats['article_count']}")
        logger.info(f"Publicités commerciales: {stats['ads_count']}")
        logger.info(f"Offres d'emploi/formation: {stats['offreemploi_count']}")
        logger.info(f"Programmes TV/agenda: {stats['programmetv_count']}")
        logger.info(f"Articles courts (< 80 mots): {stats['short_ads_count']}")
        logger.info(f"Pourcentage de non-articles: {stats['non_article_percentage']}%")
        
        # Sauvegarder les articles par catégorie
        logger.info(f"Sauvegarde des articles légitimes dans {article_output_path}")
        with open(article_output_path, 'w', encoding='utf-8') as f:
            json.dump(article_articles, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Sauvegarde des publicités commerciales dans {ads_output_path}")
        with open(ads_output_path, 'w', encoding='utf-8') as f:
            json.dump(ads_articles, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Sauvegarde des offres d'emploi/formation dans {offreemploi_output_path}")
        with open(offreemploi_output_path, 'w', encoding='utf-8') as f:
            json.dump(offreemploi_articles, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Sauvegarde des programmes TV/agenda dans {programmetv_output_path}")
        with open(programmetv_output_path, 'w', encoding='utf-8') as f:
            json.dump(programmetv_articles, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Sauvegarde des articles courts dans {short_ads_output_path}")
        with open(short_ads_output_path, 'w', encoding='utf-8') as f:
            json.dump(short_ads_articles, f, ensure_ascii=False, indent=2)
        
        # Sauvegarder aussi les statistiques
        stats_path = os.path.join(output_dir, f"filter_stats_topic{topic_id}.json")
        with open(stats_path, 'w', encoding='utf-8') as f:
            json.dump(stats, f, ensure_ascii=False, indent=2)
    
    return stats


def get_parser():
    """
    Crée le parser d'arguments pour le script.
    """
    parser = argparse.ArgumentParser(description='Filtre et catégorise les articles d\'un topic spécifique.')
    
    parser.add_argument('--articles', '-a', required=True, help='Chemin vers le fichier JSON contenant les articles')
    parser.add_argument('--doc-topic-matrix', '-m', required=True, help='Chemin vers le fichier JSON contenant la matrice document-topic')
    parser.add_argument('--topic-id', '-t', type=int, required=True, help='ID du topic à analyser')
    parser.add_argument('--min-topic-value', '-v', type=float, default=0.5, help='Valeur minimale du topic pour considérer un article (0.0-1.0)')
    parser.add_argument('--output-dir', '-o', help='Répertoire où sauvegarder les fichiers JSON filtrés')
    parser.add_argument('--dry-run', '-d', action='store_true', help='Ne pas écrire les fichiers de sortie')
    parser.add_argument('--batch-size', '-b', type=int, default=10, help='Nombre d\'articles à traiter par lot pour les requêtes LLM')
    parser.add_argument('--max-articles', '-n', type=int, help='Nombre maximum d\'articles à analyser (pour les tests)')
    
    return parser


def main():
    """
    Fonction principale du script.
    """
    parser = get_parser()
    args = parser.parse_args()
    
    try:
        # Vérifier que les fichiers d'entrée existent
        if not os.path.exists(args.articles):
            logger.error(f"Le fichier d'articles n'existe pas: {args.articles}")
            return 1
        
        if not os.path.exists(args.doc_topic_matrix):
            logger.error(f"Le fichier de matrice document-topic n'existe pas: {args.doc_topic_matrix}")
            return 1
        
        # Vérifier que le topic ID est valide
        if args.topic_id < 0:
            logger.error(f"L'ID du topic doit être positif: {args.topic_id}")
            return 1
        
        # Vérifier que la valeur minimale du topic est valide
        if args.min_topic_value < 0.0 or args.min_topic_value > 1.0:
            logger.error(f"La valeur minimale du topic doit être entre 0.0 et 1.0: {args.min_topic_value}")
            return 1
        
        # Exécuter le filtrage
        stats = filter_ads_from_topic_categories(
            articles_path=args.articles,
            doc_topic_matrix_path=args.doc_topic_matrix,
            topic_id=args.topic_id,
            min_topic_value=args.min_topic_value,
            output_dir=args.output_dir,
            dry_run=args.dry_run,
            batch_size=args.batch_size,
            max_articles=args.max_articles
        )
        
        # Afficher un résumé des résultats
        logger.info("Filtrage terminé avec succès.")
        
        return 0
        
    except Exception as e:
        logger.error(f"Erreur lors de l'exécution du script: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return 1


if __name__ == "__main__":
    sys.exit(main())
