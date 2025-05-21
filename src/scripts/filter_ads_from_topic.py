#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script pour détecter et filtrer les publicités dans un topic spécifique.
"""

import os
import sys
import json
import yaml
import logging
import argparse
import pathlib
from datetime import datetime
from typing import List, Dict, Any, Optional
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


def process_batch_for_ads(batch: List[Dict[str, Any]], llm_client: LLMClient, batch_size: int) -> List[tuple]:
    """
    Traite un lot d'articles pour déterminer s'ils sont des publicités.
    
    Args:
        batch: Liste des articles à analyser
        llm_client: Client LLM pour l'analyse
        batch_size: Taille du lot
        
    Returns:
        Liste de tuples (booléen, explication) indiquant si chaque article est une publicité (True) ou non (False)
        et l'explication fournie par le LLM
    """
    # Construire le prompt pour le lot
    prompt = """You are tasked with determining whether a given text is a **non-editorial announcement** (such as a commercial advertisement or a procedural notice), or an **editorial/article-type content** relevant for topic modeling and media analysis.

Classify as **“OUI”** only if the text is one of the following:
- A **commercial advertisement** (product, service, shop, brand),
- A **job offer** or **real estate listing**,
- A **program schedule** (TV, cinema, radio, administrative notices),
- A **very short announcement** clearly lacking editorial content.

⚠️ Do NOT classify as "OUI" (do NOT treat as ad) if:
- The text is a **journalistic article**, an **editorial**, a **reflection**, or a **news report**,
- It is an **announcement of a conference, colloquium, or scientific/cultural event** with meaningful context or analysis,
- It is a **short article or news brief** summarizing a public event or statement,
- It contains **opinion**, **contextual interpretation**, or **descriptive analysis**.

### Format of response:
For each text, respond **only** with:

Texte 1 : NON  
Texte 2 : OUI  
Texte 3 : NON  
...

### Examples:
Texte : "Ce soir, conférence sur l’impact de l’IA sur la médecine. Entrée libre." → NON  
Texte : "PC portable Lenovo, 799 CHF chez Infomarkt. Stock limité." → OUI  
Texte : "La presse face au numérique : débat au Club 44 avec Jean-Claude Nicole." → NON  
Texte : "Recherche comptable bilingue 80 %, CV à envoyer." → OUI  
Texte : "Colloque international : L'informatique et l’aménagement du territoire." → NON

"""
    
    # Ajouter chaque article au prompt
    for i, article in enumerate(batch):
        # Obtenir le contenu de l'article
        content = None
        for key in ['cleaned_text', 'content', 'text', 'original_content']:
            if key in article and article[key]:
                content = article[key]
                break
        
        if not content:
            # Si aucun contenu n'est trouvé, considérer comme une publicité par défaut
            continue
            
        # Limiter la taille du contenu pour éviter de dépasser les limites du LLM
        content = content[:1000]  # Limiter à 1000 caractères
        
        prompt += f"\nTexte {i+1} : {content}\n"
    
    # Les instructions finales sont déjà incluses dans le prompt principal
    
    # Envoyer la requête au LLM
    response = llm_client.ask(prompt, max_tokens=1000)
    
    # Analyser la réponse
    results = []
    lines = response.strip().split('\n')
    
    # Extraire les réponses OUI/NON et les explications
    current_text_num = 1
    for line in lines:
        line = line.strip()
        if not line:
            continue
            
        # Essayer de trouver le format "Texte X: OUI/NON - [Explication]"
        if line.lower().startswith(f"texte {current_text_num}") or line.lower().startswith(f"texte{current_text_num}"):
            # Extraire la réponse et l'explication
            parts = line.split(':', 1)
            if len(parts) > 1:
                response_part = parts[1].strip()
                
                # Séparer la réponse OUI/NON de l'explication
                if '-' in response_part:
                    decision, explanation = response_part.split('-', 1)
                    decision = decision.strip().upper()
                    explanation = explanation.strip()
                else:
                    # Si pas de tiret, essayer de trouver OUI ou NON au début
                    decision = 'OUI' if 'OUI' in response_part.upper() else 'NON' if 'NON' in response_part.upper() else ''
                    explanation = response_part
                
                is_ad = 'OUI' in decision
                results.append((is_ad, explanation))
                current_text_num += 1
        elif 'OUI' in line.upper() or 'NON' in line.upper():
            # Format alternatif, juste OUI ou NON
            is_ad = 'OUI' in line.upper()
            explanation = line.replace('OUI', '').replace('NON', '').replace('oui', '').replace('non', '').strip()
            if not explanation:
                explanation = "Détecté comme publicité" if is_ad else "Détecté comme article légitime"
            results.append((is_ad, explanation))
            current_text_num += 1
    
    # S'assurer que nous avons une réponse pour chaque article
    while len(results) < len(batch):
        # Par défaut, considérer comme non-publicité si pas de réponse
        results.append((False, "Pas de réponse du LLM, considéré comme article légitime par défaut"))
    
    # Limiter les résultats à la taille du lot
    results = results[:len(batch)]
    
    # Journaliser les résultats pour débogage
    for i, (article, (is_ad, explanation)) in enumerate(zip(batch, results)):
        article_id = article.get('doc_id', article.get('id', f'article_{i}'))
        logger.info(f"Analyse publicité pour document {article_id}: {'PUBLICITÉ' if is_ad else 'NON-PUBLICITÉ'} (réponse: {'oui' if is_ad else 'non'})")
    
    return results


def is_probable_publicite_llm(doc: Dict[str, Any], llm_client: LLMClient) -> bool:
    """
    Détermine si un document est probablement une publicité en utilisant un LLM.
    
    Args:
        doc: Le document à analyser (dictionnaire contenant une clé 'cleaned_text' ou 'content')
        llm_client: Instance de LLMClient pour interagir avec le modèle de langage
        
    Returns:
        bool: True si le document est probablement une publicité, False sinon
    """
    # Obtenir le contenu du document
    content = None
    for key in ['cleaned_text', 'content', 'text', 'original_content']:
        if key in doc and doc[key]:
            content = doc[key]
            break
    
    if not content:
        logger.warning(f"Document sans contenu pour l'analyse de publicité: {doc.get('_id', doc.get('id', 'unknown'))}")
        return False
    
    # Limiter la taille du contenu pour éviter de dépasser les limites du LLM
    content = content[:1000]  # Limiter à 1000 caractères
    
    # Construire le prompt
    prompt = """Tu es chargé de détecter si un texte est une publicité ou une annonce commerciale non rédactionnelle.

Cela inclut :
- publicités commerciales (produits, services, offres),
- offres d'emploi,
- annonces immobilières,
- annonces administratives ou programmes automatiques (horaires, programmes TV, etc.).

Ne classe PAS comme publicité :
- les annonces de conférences, colloques ou débats publics,
- les articles courts à propos d'événements culturels, scientifiques ou politiques,
- les articles d'opinion ou les comptes rendus même brefs à propos d'un événement.

Pour le texte suivant, réponds uniquement par OUI si c'est une publicité ou une annonce non éditoriale, ou par NON si c'est un article ou une annonce informative pertinente pour l'analyse de contenu.

Texte : """ + content
    
    # Envoyer la requête au LLM
    response = llm_client.ask(prompt, max_tokens=50)
    
    # Analyser la réponse
    is_ad = 'OUI' in response.upper() or 'YES' in response.upper()
    
    return is_ad


def filter_ads_from_topic(
    articles_path: str,
    doc_topic_matrix_path: str,
    topic_id: int,
    min_topic_value: float = 0.5,
    output_path: Optional[str] = None,
    dry_run: bool = False,
    batch_size: int = 10,
    max_articles: Optional[int] = None
) -> Dict[str, Any]:
    """
    Filtre les publicités d'un topic spécifique.
    
    Args:
        articles_path: Chemin vers le fichier JSON contenant les articles
        doc_topic_matrix_path: Chemin vers le fichier JSON contenant la matrice document-topic
        topic_id: ID du topic à analyser
        min_topic_value: Valeur minimale du topic pour considérer un article (0.0-1.0)
        output_path: Chemin où sauvegarder le fichier JSON filtré (si None, utilise le chemin d'entrée)
        dry_run: Si True, n'écrit pas le fichier de sortie
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
    ads_count = 0          # Publicités détectées par le LLM
    non_ads_count = 0      # Articles légitimes détectés par le LLM
    short_ads_count = 0    # Articles courts considérés comme publicités
    filtered_articles = [] # Articles non-publicités à conserver
    ads_articles = []      # Articles publicités détectées par le LLM
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
            article['ad_criteria'] = "Article trop court (moins de 80 mots)"
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
                    results = process_batch_for_ads(batch, llm_client, batch_size)
                    
                    # Traiter les résultats
                    for article, (is_ad, explanation) in zip(batch, results):
                        if is_ad:
                            ads_count += 1
                            # Ajouter l'explication dans l'article
                            article['ad_criteria'] = explanation
                            # Ajouter les publicités au tableau des publicités
                            ads_articles.append(article)
                        else:
                            non_ads_count += 1
                            # Ajouter uniquement les articles non-publicités au tableau filtré
                            filtered_articles.append(article)
                    
                    # Mettre à jour la barre de progression
                    pbar.update(len(batch))
                    
                except Exception as e:
                    logger.error(f"Erreur lors du traitement d'un lot: {str(e)}")
                    # En cas d'erreur, on garde tous les articles du lot par précaution
                    for article in batch:
                        logger.warning(f"Erreur de traitement, article conservé par précaution: {article.get('doc_id', article.get('id', 'unknown'))}")
                        non_ads_count += 1
                        # Ajouter uniquement les articles non-publicités au tableau filtré
                        filtered_articles.append(article)
                    pbar.update(len(batch))
    
    # Préparer les statistiques
    total_analyzed = len(articles_to_analyze) + len(short_ads_articles)
    total_ads = ads_count + len(short_ads_articles)  # Total des publicités (détectées par LLM + articles courts)
    
    if max_articles is not None and max_articles > 0:
        # Limiter le nombre total d'articles analysés à max_articles
        total_analyzed = min(total_analyzed, max_articles)
        
        # S'assurer que le nombre de publicités et d'articles non-publicités ne dépasse pas le nombre total d'articles analysés
        if total_ads + non_ads_count > total_analyzed:
            logger.warning(f"Incohérence dans les statistiques: {total_ads + non_ads_count} articles classés pour {total_analyzed} analysés")
            
            # Ajuster les statistiques pour éviter les pourcentages supérieurs à 100%
            if total_analyzed > 0:
                # Calculer la proportion de chaque type
                total_classified = total_ads + non_ads_count
                ads_ratio = total_ads / total_classified
                non_ads_ratio = non_ads_count / total_classified
                
                # Ajuster les compteurs pour respecter la limite
                adjusted_ads = int(total_analyzed * ads_ratio)
                adjusted_non_ads = total_analyzed - adjusted_ads
                
                # Utiliser les valeurs ajustées pour les statistiques
                stats = {
                    "topic_id": topic_id,
                    "min_topic_value": min_topic_value,
                    "total_articles": len(topic_articles),
                    "analyzed_articles": total_analyzed,
                    "ads_detected": adjusted_ads,
                    "non_ads": adjusted_non_ads,
                    "ads_percentage": round(adjusted_ads / total_analyzed * 100, 2) if total_analyzed > 0 else 0,
                    "max_articles_limit": max_articles
                }
            else:
                stats = {
                    "topic_id": topic_id,
                    "min_topic_value": min_topic_value,
                    "total_articles": len(topic_articles),
                    "analyzed_articles": 0,
                    "ads_detected": 0,
                    "llm_ads_detected": 0,
                    "short_ads_detected": 0,
                    "non_ads": 0,
                    "ads_percentage": 0,
                    "max_articles_limit": max_articles
                }
        else:
            stats = {
                "topic_id": topic_id,
                "min_topic_value": min_topic_value,
                "total_articles": len(topic_articles),
                "analyzed_articles": total_analyzed,
                "ads_detected": total_ads,
                "llm_ads_detected": ads_count,
                "short_ads_detected": len(short_ads_articles),
                "non_ads": non_ads_count,
                "ads_percentage": round(total_ads / total_analyzed * 100, 2) if total_analyzed > 0 else 0,
                "max_articles_limit": max_articles
            }
    else:
        stats = {
            "topic_id": topic_id,
            "min_topic_value": min_topic_value,
            "total_articles": len(topic_articles),
            "ads_detected": total_ads,
            "llm_ads_detected": ads_count,
            "short_ads_detected": len(short_ads_articles),
            "non_ads": non_ads_count,
            "ads_percentage": round(total_ads / len(topic_articles) * 100, 2) if len(topic_articles) > 0 else 0
        }

    logger.info(f"=== Résumé du filtrage (limité à {max_articles} articles) ===")

    # Sauvegarder les résultats si ce n'est pas un dry run
    if not dry_run:
        # Déterminer le chemin de sortie si non spécifié
        if not output_path:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            base_name = os.path.basename(articles_path).split('.')[0]
            output_path = os.path.join(os.path.dirname(articles_path), f"{base_name}_filtered_topic{topic_id}.json")
            ads_output_path = os.path.join(os.path.dirname(articles_path), f"{base_name}_ads_topic{topic_id}.json")
            short_ads_output_path = os.path.join(os.path.dirname(articles_path), f"{base_name}_short_ads_topic{topic_id}.json")
        else:
            # Construire le chemin pour les publicités basé sur le chemin de sortie
            base_path = os.path.splitext(output_path)[0]
            ads_output_path = f"{base_path}_ads.json"
            short_ads_output_path = f"{base_path}_short_ads.json"

        if max_articles is not None and max_articles > 0:
            logger.info(f"Articles analysés: {stats['analyzed_articles']}")
        logger.info(f"Publicités détectées (total): {stats['ads_detected']} ({stats['ads_percentage']}%)")
        logger.info(f"  - Publicités détectées par LLM: {len(ads_articles)}")
        logger.info(f"  - Articles courts (< 80 mots): {len(short_ads_articles)}")
        logger.info(f"Articles non-publicités: {stats['non_ads']}")
        logger.info(f"Taille de lot utilisée: {batch_size} articles par requête LLM")

        # Ajouter les chemins de sortie aux statistiques
        stats["output_path"] = output_path
        stats["ads_output_path"] = ads_output_path
        stats["short_ads_output_path"] = short_ads_output_path

        # Sauvegarder les articles filtrés (non-publicités)
        logger.info(f"Sauvegarde des articles filtrés (non-publicités) dans {output_path}")
        logger.info(f"Nombre d'articles filtrés (non-publicités) à sauvegarder: {len(filtered_articles)}")
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(filtered_articles, f, ensure_ascii=False, indent=2)

        # Sauvegarder les publicités détectées par le LLM
        logger.info(f"Sauvegarde des publicités détectées par le LLM dans {ads_output_path}")
        logger.info(f"Nombre de publicités détectées par le LLM: {len(ads_articles)}")
        with open(ads_output_path, 'w', encoding='utf-8') as f:
            json.dump(ads_articles, f, ensure_ascii=False, indent=2)
            
        # Sauvegarder les articles courts (considérés comme publicités)
        logger.info(f"Sauvegarde des articles courts dans {short_ads_output_path}")
        logger.info(f"Nombre d'articles courts: {len(short_ads_articles)}")
        with open(short_ads_output_path, 'w', encoding='utf-8') as f:
            json.dump(short_ads_articles, f, ensure_ascii=False, indent=2)

        # Sauvegarder aussi les statistiques
        stats_path = os.path.join(os.path.dirname(output_path), f"filter_stats_topic{topic_id}.json")
        with open(stats_path, 'w', encoding='utf-8') as f:
            json.dump(stats, f, ensure_ascii=False, indent=2)

    return stats

def get_parser():
    """
    Crée le parser d'arguments pour le script.
    """
    parser = argparse.ArgumentParser(description="Filtre les publicités d'un topic spécifique")
    
    parser.add_argument("--articles", type=str, required=True,
                        help="Chemin vers le fichier JSON contenant les articles")
    
    parser.add_argument("--doc-topic-matrix", type=str, required=True,
                        help="Chemin vers le fichier JSON contenant la matrice document-topic")
    
    parser.add_argument("--topic-id", type=int, required=True,
                        help="ID du topic à analyser")
    
    parser.add_argument("--min-topic-value", type=float, default=0.5,
                        help="Valeur minimale du topic pour considérer un article (0.0-1.0)")
    
    parser.add_argument("--output", type=str, default=None,
                        help="Chemin où sauvegarder le fichier JSON filtré")
    
    parser.add_argument("--dry-run", action="store_true",
                        help="Si spécifié, n'écrit pas le fichier de sortie")
    
    parser.add_argument("--batch-size", type=int, default=10,
                        help="Nombre d'articles à traiter par lot pour les requêtes LLM (défaut: 10)")
    
    parser.add_argument("--max-articles", type=int, default=None,
                        help="Nombre maximum d'articles à analyser (pour les tests)")
    
    return parser


def main():
    """
    Fonction principale du script.
    """
    parser = get_parser()
    args = parser.parse_args()
    
    try:
        stats = filter_ads_from_topic(
            articles_path=args.articles,
            doc_topic_matrix_path=args.doc_topic_matrix,
            topic_id=args.topic_id,
            min_topic_value=args.min_topic_value,
            output_path=args.output,
            dry_run=args.dry_run,
            batch_size=args.batch_size,
            max_articles=args.max_articles
        )
        
        logger.info("=== Résumé du filtrage ===")
        logger.info(f"Topic: {stats['topic_id']}")
        logger.info(f"Articles dans le topic: {stats['total_articles']}")
        logger.info(f"Publicités détectées (total): {stats['ads_detected']} ({stats['ads_percentage']}%)")
        logger.info(f"  - Publicités détectées par LLM: {stats['llm_ads_detected']}")
        logger.info(f"  - Articles courts (< 80 mots): {stats['short_ads_detected']}")
        logger.info(f"Articles non-publicités: {stats['non_ads']}")
        logger.info(f"Taille de lot utilisée: {args.batch_size} articles par requête LLM")
        
        if not args.dry_run:
            print(f"Fichier filtré sauvegardé: {stats['output_path']}")
        
        return 0
        
    except Exception as e:
        logger.error(f"Erreur lors de l'exécution du script: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
