#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Script pour générer des noms de topics en utilisant un LLM.
Ce script peut être exécuté indépendamment du processus de topic modeling.
Il utilise les résultats existants (doc_topic_matrix.json, etc.) pour générer les noms de topics.
"""

import os
import sys
import json
import yaml
import logging
import argparse
from pathlib import Path
import time
from typing import Dict, List, Tuple, Any

# Ajouter le répertoire parent au path pour importer les modules du projet
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from src.analysis.llm_utils import LLMClient

# Configuration du logger
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Génération de noms de topics avec LLM")
    
    parser.add_argument("--source-file", type=str, required=True,
                        help="Fichier source contenant les articles (JSON)")
    
    parser.add_argument("--doc-topic-matrix", type=str, default="data/results/doc_topic_matrix.json",
                        help="Fichier contenant la matrice document-topic (JSON). Par défaut: data/results/doc_topic_matrix.json")
    
    parser.add_argument("--method", type=str, choices=["articles", "keywords"], default="articles",
                        help="Méthode de génération des noms: 'articles' (utilise les articles représentatifs) ou 'keywords' (utilise les mots-clés)")
    
    parser.add_argument("--output-file", type=str, default="data/results/topic_names_llm.json",
                        help="Fichier de sortie pour les noms de topics générés. Par défaut: data/results/topic_names_llm.json")
    
    parser.add_argument("--top-words-file", type=str, default="data/results/advanced_topic/advanced_topic_analysis.json",
                        help="Fichier contenant les mots-clés des topics. Requis si method=keywords. Par défaut: data/results/advanced_topic/advanced_topic_analysis.json")
    
    parser.add_argument("--config", type=str, default="config/config.yaml",
                        help="Fichier de configuration (YAML)")
    
    parser.add_argument("--num-articles", type=int, default=10,
                        help="Nombre d'articles représentatifs à utiliser par topic")
    
    parser.add_argument("--threshold", type=float, default=0.5,
                        help="Seuil de probabilité pour considérer un article comme représentatif d'un topic")
    
    return parser.parse_args()

def load_file(file_path: str) -> Dict:
    """Charger un fichier JSON ou YAML."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            if file_path.endswith('.json'):
                return json.load(f)
            elif file_path.endswith('.yaml') or file_path.endswith('.yml'):
                return yaml.safe_load(f)
            else:
                # Essayer de déterminer le format en fonction du contenu
                content = f.read()
                try:
                    return json.loads(content)
                except json.JSONDecodeError:
                    try:
                        return yaml.safe_load(content)
                    except yaml.YAMLError:
                        raise ValueError(f"Format de fichier non reconnu: {file_path}")
    except Exception as e:
        logger.error(f"Erreur lors du chargement du fichier {file_path}: {e}")
        sys.exit(1)

def get_topic_names_from_representative_docs(
    representative_docs: Dict[str, List],
    llm_client: LLMClient,
    num_articles: int = 10
) -> Dict[str, Tuple[str, str]]:
    """
    Générer des noms de topics en utilisant un LLM à partir des articles représentatifs.
    
    Args:
        representative_docs: Dictionnaire avec les IDs de topics comme clés et des listes d'articles comme valeurs
        llm_client: Client LLM initialisé
        num_articles: Nombre d'articles représentatifs à utiliser par topic
        
    Returns:
        Dictionnaire avec les IDs de topics comme clés et des tuples (titre, résumé) comme valeurs
    """
    # Dictionnaire pour stocker les noms et résumés générés
    topic_names_and_summaries = {}
    
    # Pour chaque topic, générer un nom
    for topic_id, articles_data in representative_docs.items():
        try:
            # Extraire les articles en fonction de la structure des données
            articles = []
            
            # Log de la structure des données pour débogage
            logger.info(f"Topic #{topic_id} - Structure des données: {type(articles_data)}, longueur: {len(articles_data) if isinstance(articles_data, (list, dict)) else 'N/A'}")
            
            # Vérifier si articles_data est une liste
            if isinstance(articles_data, list):
                # Chercher spécifiquement les articles correspondant au topic_id
                # D'après l'exemple, les articles sont stockés dans une liste où chaque élément est un dictionnaire
                # avec une clé correspondant au topic_id
                for i, item in enumerate(articles_data):
                    # Log du type de chaque élément
                    logger.info(f"Topic #{topic_id} - Élément {i}: {type(item)}")
                    
                    # Si l'élément est un dictionnaire (structure observée dans les données)
                    if isinstance(item, dict):
                        # Log des clés du dictionnaire
                        logger.info(f"Topic #{topic_id} - Élément {i} (dict) - Clés: {list(item.keys())}")
                        
                        # Chercher les articles correspondant au topic_id
                        # Essayer d'abord avec le topic_id exact
                        if topic_id in item:
                            texts = item[topic_id]
                            if isinstance(texts, list):
                                logger.info(f"Topic #{topic_id} - Trouvé {len(texts)} articles pour la clé exacte '{topic_id}'")
                                articles.extend([text for text in texts if isinstance(text, str)])
                        # Sinon, parcourir toutes les clés du dictionnaire
                        else:
                            for key, texts in item.items():
                                if isinstance(texts, list):
                                    logger.info(f"Topic #{topic_id} - Élément {i} - Clé {key}: {len(texts)} articles")
                                    articles.extend([text for text in texts if isinstance(text, str)])
                    # Si l'élément est directement une liste d'articles
                    elif isinstance(item, list):
                        # Log de la longueur de la liste
                        logger.info(f"Topic #{topic_id} - Élément {i} (list): {len(item)} éléments")
                        articles.extend([text for text in item if isinstance(text, str)])
                    # Si l'élément est directement un texte d'article
                    elif isinstance(item, str):
                        # Log des premiers caractères du texte
                        preview = item[:50] + "..." if len(item) > 50 else item
                        logger.info(f"Topic #{topic_id} - Élément {i} (str): {preview}")
                        articles.append(item)
            
            # Limiter le nombre d'articles si nécessaire
            best_articles = articles[:num_articles]
            
            # Si nous avons des articles pour ce topic, générer un nom
            if best_articles:
                logger.info(f"Topic #{topic_id}: {len(best_articles)} articles représentatifs trouvés")
                
                # Log des débuts d'articles pour vérifier s'ils sont similaires entre topics
                for i, article in enumerate(best_articles):
                    preview = article[:100].replace('\n', ' ') + "..." if len(article) > 100 else article.replace('\n', ' ')
                    logger.info(f"Topic #{topic_id} - Article {i}: {preview}")
                
                # Générer un nom et un résumé pour ce topic
                title, summary = llm_client.get_topic_name_from_articles(
                    best_articles, 
                    max_tokens=100,
                    temperature=0.3
                )
                
                topic_names_and_summaries[str(topic_id)] = (title, summary)
                logger.info(f"Topic #{topic_id} - Titre généré: {title}")
                logger.info(f"Topic #{topic_id} - Résumé: {summary}")
            else:
                logger.warning(f"Aucun article représentatif trouvé pour le topic {topic_id}")
                topic_names_and_summaries[str(topic_id)] = (f"Topic {topic_id}", "Pas de description disponible")
        
        except Exception as e:
            logger.error(f"Erreur lors de la génération du nom pour le topic {topic_id}: {e}")
            topic_names_and_summaries[str(topic_id)] = (f"Topic {topic_id}", f"Erreur: {str(e)}")
    
    return topic_names_and_summaries

def get_topic_names_from_articles(
    articles: List[Dict],
    doc_topic_matrix: List[Dict],
    llm_client: LLMClient,
    num_articles: int = 10,
    threshold: float = 0.5
) -> Dict[str, Tuple[str, str]]:
    """
    Générer des noms de topics en utilisant un LLM à partir des articles représentatifs.
    
    Args:
        articles: Liste des articles (dictionnaires avec clés 'text', 'content', etc.)
        doc_topic_matrix: Matrice document-topic (liste de dictionnaires avec 'doc_id' et 'topic_distribution')
        llm_client: Client LLM initialisé
        num_articles: Nombre d'articles représentatifs à utiliser par topic
        threshold: Seuil de probabilité pour considérer un article comme représentatif
        
    Returns:
        Dictionnaire avec les IDs de topics comme clés et des tuples (titre, résumé) comme valeurs
    """
    # Créer un mapping des ID d'articles aux articles complets
    article_id_map = {}
    for article in articles:
        # Vérifier si l'article est un dictionnaire ou une chaîne de caractères
        if isinstance(article, dict):
            doc_id = article.get('doc_id', article.get('id', None))
            if doc_id:
                article_id_map[doc_id] = article
        else:
            logger.warning(f"Format d'article inattendu: {type(article)}")
    
    # Créer un dictionnaire pour stocker les distributions de topics par document
    doc_topics = {}
    for doc in doc_topic_matrix:
        # Vérifier si le document est un dictionnaire
        if isinstance(doc, dict):
            doc_id = doc.get('doc_id')
            topic_distribution = doc.get('topic_distribution')
            if doc_id and topic_distribution:
                doc_topics[doc_id] = {
                    'topic_distribution': topic_distribution,
                    'dominant_topic': int(max(range(len(topic_distribution)), key=lambda i: topic_distribution[i]))
                }
        else:
            logger.warning(f"Format de document inattendu dans doc_topic_matrix: {type(doc)}")
    
    # Déterminer le nombre de topics
    num_topics = 0
    
    # Vérifier si doc_topic_matrix est une liste ou un dictionnaire
    if isinstance(doc_topic_matrix, list) and len(doc_topic_matrix) > 0:
        first_doc = doc_topic_matrix[0]
        if isinstance(first_doc, dict) and 'topic_distribution' in first_doc:
            num_topics = len(first_doc['topic_distribution'])
        else:
            logger.warning("Format inattendu pour le premier document dans doc_topic_matrix")
    elif isinstance(doc_topic_matrix, dict):
        # Si c'est un dictionnaire, essayer de trouver les informations sur les topics
        if 'topics' in doc_topic_matrix:
            num_topics = len(doc_topic_matrix['topics'])
            logger.info(f"Trouvé {num_topics} topics dans le dictionnaire doc_topic_matrix")
    else:
        logger.error(f"Format inattendu pour doc_topic_matrix: {type(doc_topic_matrix)}")
    
    # Dictionnaire pour stocker les noms et résumés générés
    topic_names_and_summaries = {}
    
    # Si aucun topic n'a été trouvé, utiliser une approche alternative
    if num_topics == 0:
        logger.warning("Aucun topic trouvé dans doc_topic_matrix, utilisation d'une approche alternative")
        # Générer un nom générique pour un seul topic
        topic_names_and_summaries["0"] = ("Articles de presse", "Collection d'articles de presse divers")
        return topic_names_and_summaries
    
    # Pour chaque topic, générer un nom
    for topic_id in range(num_topics):
        try:
            # Trouver les articles les plus représentatifs pour ce topic
            topic_articles = []
            topic_articles_scores = []
            
            # Pour chaque document, vérifier sa probabilité d'appartenance au topic
            for doc_id, doc_info in doc_topics.items():
                # Vérifier si le document a une distribution de topics
                if 'topic_distribution' in doc_info:
                    # Vérifier si le topic_id est dans la plage valide
                    if 0 <= topic_id < len(doc_info['topic_distribution']):
                        # Récupérer la probabilité d'appartenance au topic
                        topic_prob = doc_info['topic_distribution'][topic_id]
                        
                        # Si la probabilité est significative, ajouter à la liste
                        if topic_prob > threshold:
                            # Vérifier si l'article existe dans notre mapping
                            if doc_id in article_id_map:
                                # Récupérer le texte de l'article
                                article = article_id_map[doc_id]
                                article_text = None
                                
                                # Vérifier le format de l'article et extraire le texte
                                if isinstance(article, dict):
                                    # Extraire le texte de l'article selon les clés disponibles
                                    if 'text' in article:
                                        article_text = article['text']
                                    elif 'content' in article:
                                        article_text = article['content']
                                    elif 'body' in article:
                                        article_text = article['body']
                                    elif 'full_text' in article:
                                        article_text = article['full_text']
                                elif isinstance(article, str):
                                    # Si l'article est déjà une chaîne de caractères, l'utiliser directement
                                    article_text = article
                                if article_text:
                                    topic_articles_scores.append((article_text, topic_prob))
            
            # Trier les articles par score et prendre les N meilleurs
            topic_articles_scores.sort(key=lambda x: x[1], reverse=True)
            best_articles = [article for article, _ in topic_articles_scores[:num_articles]]
            
            # Si nous avons des articles pour ce topic, générer un nom
            if best_articles:
                logger.info(f"Topic #{topic_id}: {len(best_articles)} articles représentatifs trouvés")
                
                # Générer un nom et un résumé pour ce topic
                title, summary = llm_client.get_topic_name_from_articles(
                    best_articles, 
                    max_tokens=100,
                    temperature=0.3
                )
                
                topic_names_and_summaries[str(topic_id)] = (title, summary)
                logger.info(f"Topic #{topic_id} - Titre généré: {title}")
            else:
                logger.warning(f"Aucun article représentatif trouvé pour le topic {topic_id}")
                topic_names_and_summaries[str(topic_id)] = (f"Topic {topic_id}", "Pas de description disponible")
        
        except Exception as e:
            logger.error(f"Erreur lors de la génération du nom pour le topic {topic_id}: {e}")
            topic_names_and_summaries[str(topic_id)] = (f"Topic {topic_id}", f"Erreur: {str(e)}")
    
    return topic_names_and_summaries

def get_topic_names_from_keywords(
    advanced_analysis: Dict,
    llm_client: LLMClient
) -> Dict[str, str]:
    """
    Générer des noms de topics en utilisant un LLM à partir des mots-clés.
    
    Args:
        advanced_analysis: Dictionnaire contenant les résultats d'analyse avancée, avec une clé 'weighted_words'
        llm_client: Client LLM initialisé
        
    Returns:
        Dictionnaire avec les IDs de topics comme clés et des noms de topics comme valeurs
    """
    topic_names = {}
    
    # Extraire les mots-clés pondérés des topics
    weighted_words = advanced_analysis.get('weighted_words', {})
    
    if not weighted_words:
        logger.error("Aucun mot-clé pondéré trouvé dans le fichier d'analyse avancée")
        return topic_names
    
    logger.info(f"Trouvé {len(weighted_words)} topics avec des mots-clés")
    
    for topic_id, word_weights in weighted_words.items():
        try:
            # Extraire les mots sans les poids
            if isinstance(word_weights, list):
                # Si c'est une liste de tuples (mot, poids)
                if all(isinstance(item, (list, tuple)) and len(item) >= 1 for item in word_weights):
                    words = [item[0] for item in word_weights]
                else:
                    # Si c'est une liste de mots
                    words = word_weights
            else:
                logger.warning(f"Format inattendu pour les mots-clés du topic {topic_id}: {type(word_weights)}")
                continue
            
            # Générer un nom pour ce topic
            title = llm_client.get_topic_name(
                words, 
                max_tokens=20,
                temperature=0.3
            )
            
            topic_names[topic_id] = title
            logger.info(f"Topic #{topic_id} - Titre généré: {title}")
        
        except Exception as e:
            logger.error(f"Erreur lors de la génération du nom pour le topic {topic_id}: {e}")
            topic_names[topic_id] = f"Topic {topic_id}"
    
    return topic_names

def main():
    """Point d'entrée principal du script."""
    start_time = time.time()
    
    # Analyser les arguments de la ligne de commande
    args = parse_args()
    
    # Déterminer les chemins des fichiers
    project_root = Path(__file__).resolve().parent.parent.parent
    source_file = Path(args.source_file)
    
    if not args.doc_topic_matrix:
        doc_topic_matrix_file = project_root / "data" / "results" / "doc_topic_matrix.json"
    else:
        doc_topic_matrix_file = Path(args.doc_topic_matrix)
    
    if not args.output_file:
        output_file = project_root / "data" / "results" / "topic_names_llm.json"
    else:
        output_file = Path(args.output_file)
    
    if args.method == "keywords" and not args.top_words_file:
        top_words_file = project_root / "data" / "results" / "topic_words.json"
    else:
        top_words_file = Path(args.top_words_file) if args.top_words_file else None
    
    config_file = project_root / args.config
    
    # Vérifier que les fichiers existent
    if not source_file.exists():
        logger.error(f"Le fichier source {source_file} n'existe pas")
        sys.exit(1)
    
    if not doc_topic_matrix_file.exists() and args.method == "articles":
        logger.error(f"Le fichier de matrice document-topic {doc_topic_matrix_file} n'existe pas")
        sys.exit(1)
    
    if args.method == "keywords" and (not top_words_file or not top_words_file.exists()):
        logger.error(f"Le fichier de mots-clés {top_words_file} n'existe pas")
        sys.exit(1)
    
    if not config_file.exists():
        logger.error(f"Le fichier de configuration du LLM {config_file} n'existe pas")
        sys.exit(1)
    
    # Charger les données
    logger.info(f"Chargement des articles depuis {source_file}")
    articles = load_file(str(source_file))
    
    logger.info(f"Chargement de la configuration depuis {config_file}")
    config = load_file(str(config_file))
    
    # Extraire la configuration du LLM
    llm_config = config.get('llm', {})
    
    # Initialiser le client LLM
    llm_client = LLMClient(llm_config)
    logger.info(f"Client LLM initialisé: {llm_client.provider}/{llm_client.model}")
    
    # Générer les noms de topics
    if args.method == "articles":
        # Vérifier d'abord si le fichier d'analyse avancée contient des articles représentatifs
        logger.info(f"Vérification du fichier d'analyse avancée pour des articles représentatifs")
        advanced_analysis = load_file(str(top_words_file))
        
        if isinstance(advanced_analysis, dict) and 'representative_docs' in advanced_analysis:
            logger.info("Articles représentatifs trouvés dans le fichier d'analyse avancée")
            representative_docs = advanced_analysis['representative_docs']
            
            logger.info(f"Génération des noms de topics à partir des articles représentatifs pré-calculés")
            topic_names_and_summaries = get_topic_names_from_representative_docs(
                representative_docs,
                llm_client,
                num_articles=args.num_articles
            )
        else:
            # Utiliser la méthode traditionnelle avec la matrice document-topic
            logger.info("Aucun article représentatif trouvé dans le fichier d'analyse avancée, utilisation de la matrice document-topic")
            logger.info(f"Chargement de la matrice document-topic depuis {doc_topic_matrix_file}")
            doc_topic_data = load_file(str(doc_topic_matrix_file))
            
            # Extraire la liste doc_topic_matrix du dictionnaire si nécessaire
            if isinstance(doc_topic_data, dict) and 'doc_topic_matrix' in doc_topic_data:
                logger.info("Extraction de la matrice document-topic du dictionnaire")
                doc_topic_matrix = doc_topic_data['doc_topic_matrix']
            else:
                doc_topic_matrix = doc_topic_data
            
            logger.info(f"Génération des noms de topics à partir des articles représentatifs")
            topic_names_and_summaries = get_topic_names_from_articles(
                articles,
                doc_topic_matrix,
                llm_client,
                num_articles=args.num_articles,
                threshold=args.threshold
            )
        
        # Préparer les résultats
        results = {
            "method": "articles",
            "llm": f"{llm_client.provider}/{llm_client.model}",
            "topic_names": {topic_id: title for topic_id, (title, _) in topic_names_and_summaries.items()},
            "topic_summaries": {topic_id: summary for topic_id, (_, summary) in topic_names_and_summaries.items()},
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "runtime_seconds": time.time() - start_time
        }
    
    else:  # method == "keywords"
        logger.info(f"Chargement des mots-clés depuis {top_words_file}")
        advanced_analysis = load_file(str(top_words_file))
        
        # Vérifier si nous avons besoin de charger la matrice document-topic pour obtenir les IDs de topics
        if 'weighted_words' not in advanced_analysis:
            logger.warning("La clé 'weighted_words' n'a pas été trouvée dans le fichier d'analyse avancée")
            logger.info(f"Chargement de la matrice document-topic depuis {doc_topic_matrix_file} pour obtenir les IDs de topics")
            doc_topic_data = load_file(str(doc_topic_matrix_file))
            
            # Extraire la liste doc_topic_matrix du dictionnaire si nécessaire
            if isinstance(doc_topic_data, dict) and 'doc_topic_matrix' in doc_topic_data:
                logger.info("Extraction de la matrice document-topic du dictionnaire")
                doc_topic_matrix = doc_topic_data['doc_topic_matrix']
            else:
                doc_topic_matrix = doc_topic_data
            
            # Extraire les IDs de topics à partir de la matrice document-topic
            if doc_topic_matrix and isinstance(doc_topic_matrix, list) and len(doc_topic_matrix) > 0:
                first_doc = doc_topic_matrix[0]
                if 'topic_distribution' in first_doc:
                    num_topics = len(first_doc['topic_distribution'])
                    logger.info(f"Trouvé {num_topics} topics dans la matrice document-topic")
                    
                    # Créer un dictionnaire de mots-clés vide pour chaque topic
                    advanced_analysis['weighted_words'] = {}
                    for i in range(num_topics):
                        advanced_analysis['weighted_words'][str(i)] = [f"topic_{i}", f"mot_clé_{i}", f"terme_{i}", f"concept_{i}", f"sujet_{i}"]
        
        logger.info(f"Génération des noms de topics à partir des mots-clés")
        topic_names = get_topic_names_from_keywords(advanced_analysis, llm_client)
        
        # Préparer les résultats
        results = {
            "method": "keywords",
            "llm": f"{llm_client.provider}/{llm_client.model}",
            "topic_names": topic_names,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "runtime_seconds": time.time() - start_time
        }
    
    # Enregistrer les résultats
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    logger.info(f"Noms de topics générés et enregistrés dans {output_file}")
    logger.info(f"Temps d'exécution total: {time.time() - start_time:.2f} secondes")

if __name__ == "__main__":
    main()
