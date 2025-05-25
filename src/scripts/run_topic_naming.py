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
    
    parser.add_argument("--doc-topic-dir", type=str, default="data/results/doc_topic_matrix",
                        help="Répertoire contenant les fichiers de matrice document-topic. Par défaut: data/results/doc_topic_matrix")
    
    parser.add_argument("--advanced-analysis-dir", type=str, default="data/results/advanced_analysis",
                        help="Répertoire contenant les fichiers d'analyse avancée. Par défaut: data/results/advanced_analysis")
    
    parser.add_argument("--method", type=str, choices=["articles", "keywords"], default="articles",
                        help="Méthode de génération des noms: 'articles' (utilise les articles représentatifs) ou 'keywords' (utilise les mots-clés)")
    
    parser.add_argument("--output-file", type=str, default="data/results/topic_names_llm.json",
                        help="Fichier de sortie pour les noms de topics générés. Par défaut: data/results/topic_names_llm.json")
    
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

def find_latest_file(directory, pattern):
    """Find the latest file in a directory matching a pattern."""
    directory = Path(directory)
    if not directory.exists() or not directory.is_dir():
        return None
    
    matching_files = list(directory.glob(pattern))
    if not matching_files:
        return None
    
    # Sort by modification time, newest first
    matching_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
    return matching_files[0]

def is_gensim_file(file_path):
    """Check if a file is related to a Gensim model."""
    return "gensim" in file_path.name.lower()

def extract_representative_docs_from_matrix(doc_topic_matrix, articles, num_per_topic=3):
    """Extract representative documents for each topic from a doc-topic matrix."""
    # Create a mapping from article IDs to articles with multiple ID formats
    article_id_map = {}
    for article in articles:
        if isinstance(article, dict):
            # Try different ID fields that might be present
            article_id = article.get('id', None)
            base_id = article.get('base_id', None)
            doc_id = article.get('doc_id', None)
            
            # Store the article under all its possible IDs
            for id_val in [article_id, base_id, doc_id]:
                if id_val is not None:
                    # Convert ID to string to ensure consistent comparison
                    article_id_map[str(id_val)] = article
    
    logger.info(f"Mapped {len(article_id_map)} unique article IDs from {len(articles)} articles")
    
    # Extract doc_topics from the matrix
    doc_topics = {}
    if isinstance(doc_topic_matrix, dict) and 'doc_topics' in doc_topic_matrix:
        doc_topics = doc_topic_matrix['doc_topics']
    
    logger.info(f"Found {len(doc_topics)} documents in the topic matrix")
    
    # Organize articles by topic
    topics_to_articles = {}
    matched_articles = 0
    
    for doc_id, doc_info in doc_topics.items():
        if isinstance(doc_info, dict) and 'topic_distribution' in doc_info and 'dominant_topic' in doc_info:
            topic_id = str(doc_info['dominant_topic'])
            topic_score = max(doc_info['topic_distribution'])
            
            if topic_id not in topics_to_articles:
                topics_to_articles[topic_id] = []
            
            # Try to find the article in our mapping
            article = None
            article_id_str = str(doc_id)
            
            # Direct match
            if article_id_str in article_id_map:
                article = article_id_map[article_id_str]
            else:
                # Try matching by substring (some IDs might be partial)
                for mapped_id, mapped_article in article_id_map.items():
                    if article_id_str in mapped_id or mapped_id in article_id_str:
                        article = mapped_article
                        break
            
            if article:
                matched_articles += 1
                # Get the article content, preferring 'content' field
                article_text = None
                if 'content' in article:
                    article_text = article['content']
                elif 'text' in article:
                    article_text = article['text']
                elif 'cleaned_text' in article:
                    article_text = article['cleaned_text']
                elif 'original_content' in article:
                    article_text = article['original_content']
                
                if article_text:
                    # Truncate very long articles for logging purposes
                    display_text = article_text
                    if len(display_text) > 1000:
                        display_text = display_text[:1000] + "..."
                    
                    topics_to_articles[topic_id].append((article_text, topic_score, doc_id, display_text))
    
    logger.info(f"Successfully matched {matched_articles} articles from the topic matrix to the source file")
    
    # Sort articles by topic score and take the top N for each topic
    representative_docs = {}
    for topic_id, articles_with_scores in topics_to_articles.items():
        articles_with_scores.sort(key=lambda x: x[1], reverse=True)
        # Only take the full article text for the representative docs
        representative_docs[topic_id] = [article for article, _, _, _ in articles_with_scores[:num_per_topic]]
        
        if representative_docs[topic_id]:
            logger.info(f"Topic #{topic_id} - Trouvé {len(representative_docs[topic_id])} articles pour la clé exacte '{topic_id}'")
            logger.info(f"Topic #{topic_id} - {len(representative_docs[topic_id])} articles représentatifs trouvés")
            
            # Log sample articles (using the truncated display text)
            for i, (_, _, _, display_text) in enumerate(articles_with_scores[:num_per_topic]):
                logger.info(f"Topic #{topic_id} - Article {i}: {display_text[:100]}...")
        else:
            logger.warning(f"Topic #{topic_id} - Aucun article représentatif trouvé")
    
    return representative_docs

def main():
    """Point d'entrée principal du script."""
    start_time = time.time()
    
    # Analyser les arguments de la ligne de commande
    args = parse_args()
    
    # Déterminer les chemins des fichiers
    project_root = Path(__file__).resolve().parent.parent.parent
    source_file = Path(args.source_file)
    
    # Directories for results
    advanced_analysis_dir = Path(args.advanced_analysis_dir)
    doc_topic_matrix_dir = Path(args.doc_topic_dir)
    
    # Make sure the directories exist
    if not advanced_analysis_dir.exists() or not advanced_analysis_dir.is_dir():
        logger.warning(f"Le répertoire d'analyse avancée {advanced_analysis_dir} n'existe pas ou n'est pas un répertoire")
        advanced_analysis_dir = project_root / "data" / "results" / "advanced_analysis"
        logger.info(f"Utilisation du répertoire par défaut: {advanced_analysis_dir}")
    
    if not doc_topic_matrix_dir.exists() or not doc_topic_matrix_dir.is_dir():
        logger.warning(f"Le répertoire de matrices document-topic {doc_topic_matrix_dir} n'existe pas ou n'est pas un répertoire")
        doc_topic_matrix_dir = project_root / "data" / "results" / "doc_topic_matrix"
        logger.info(f"Utilisation du répertoire par défaut: {doc_topic_matrix_dir}")
    
    # Find the latest files in each directory
    latest_advanced_file = find_latest_file(advanced_analysis_dir, "*.json")
    latest_matrix_file = find_latest_file(doc_topic_matrix_dir, "*.json")
    
    logger.info(f"Fichier d'analyse avancée le plus récent: {latest_advanced_file}")
    logger.info(f"Fichier de matrice document-topic le plus récent: {latest_matrix_file}")
    
    if not args.output_file:
        output_file = project_root / "data" / "results" / "topic_names_llm.json"
    else:
        output_file = Path(args.output_file)
    
    config_file = project_root / args.config
    
    # Vérifier que les fichiers existent
    if not source_file.exists():
        logger.error(f"Le fichier source {source_file} n'existe pas")
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
        logger.info("Méthode de génération des noms: articles représentatifs")
        
        # Vérifier si c'est un modèle Gensim en regardant le nom du fichier d'analyse avancée
        is_gensim_model = latest_advanced_file and is_gensim_file(latest_advanced_file)
        logger.info(f"Modèle Gensim détecté: {is_gensim_model}")
        
        representative_docs = None
        
        # Si nous avons un fichier d'analyse avancée, essayons de l'utiliser
        if latest_advanced_file:
            logger.info(f"Analyse du fichier d'analyse avancée: {latest_advanced_file}")
            advanced_analysis = load_file(str(latest_advanced_file))
            
            # Vérifier si le fichier contient des articles représentatifs (pour BERTopic)
            if isinstance(advanced_analysis, dict) and 'representative_docs' in advanced_analysis and not is_gensim_model:
                logger.info("Articles représentatifs trouvés dans le fichier d'analyse avancée (BERTopic)")
                representative_docs = advanced_analysis['representative_docs']
        
        # Pour les modèles Gensim, nous devons extraire les articles représentatifs à partir de la matrice document-topic
        if is_gensim_model and latest_matrix_file:
            logger.info(f"Modèle Gensim: utilisation de la matrice document-topic pour trouver les articles représentatifs")
            logger.info(f"Chargement de la matrice document-topic depuis {latest_matrix_file}")
            
            doc_topic_data = load_file(str(latest_matrix_file))
            
            # Extraire les articles représentatifs à partir de la matrice document-topic
            logger.info("Extraction des articles représentatifs à partir de la matrice document-topic")
            representative_docs = extract_representative_docs_from_matrix(doc_topic_data, articles, num_per_topic=3)
        
        # Si nous n'avons toujours pas d'articles représentatifs, utiliser la méthode traditionnelle
        if representative_docs is None:
            # Utiliser la méthode traditionnelle avec la matrice document-topic
            logger.info("Aucun article représentatif trouvé, utilisation de la méthode traditionnelle")
            
            if not latest_matrix_file:
                logger.error("Aucun fichier de matrice document-topic trouvé")
                sys.exit(1)
            
            logger.info(f"Chargement de la matrice document-topic depuis {latest_matrix_file}")
            doc_topic_data = load_file(str(latest_matrix_file))
            
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
        else:
            # Utiliser les articles représentatifs que nous avons trouvés
            logger.info(f"Génération des noms de topics à partir des articles représentatifs")
            topic_names_and_summaries = get_topic_names_from_representative_docs(
                representative_docs,
                llm_client,
                num_articles=args.num_articles
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
        logger.info("Méthode de génération des noms: mots-clés")
        
        # Vérifier si nous avons un fichier d'analyse avancée
        if not latest_advanced_file:
            logger.error("Aucun fichier d'analyse avancée trouvé")
            sys.exit(1)
            
        logger.info(f"Chargement des mots-clés depuis {latest_advanced_file}")
        advanced_analysis = load_file(str(latest_advanced_file))
        
        # Vérifier si nous avons besoin de charger la matrice document-topic pour obtenir les IDs de topics
        if 'weighted_words' not in advanced_analysis:
            logger.warning("La clé 'weighted_words' n'a pas été trouvée dans le fichier d'analyse avancée")
            
            # Vérifier si nous avons un fichier de matrice document-topic
            if not latest_matrix_file:
                logger.error("Aucun fichier de matrice document-topic trouvé")
                sys.exit(1)
                
            logger.info(f"Chargement de la matrice document-topic depuis {latest_matrix_file} pour obtenir les IDs de topics")
            doc_topic_data = load_file(str(latest_matrix_file))
            
            # Extraire les informations de topics
            if isinstance(doc_topic_data, dict):
                if 'doc_topics' in doc_topic_data:
                    # Format Gensim
                    logger.info("Format de matrice document-topic Gensim détecté")
                    doc_topics = doc_topic_data['doc_topics']
                    if doc_topics and len(doc_topics) > 0:
                        # Prendre le premier document pour déterminer le nombre de topics
                        first_doc_id = list(doc_topics.keys())[0]
                        first_doc = doc_topics[first_doc_id]
                        if 'topic_distribution' in first_doc:
                            num_topics = len(first_doc['topic_distribution'])
                            logger.info(f"Trouvé {num_topics} topics dans la matrice document-topic Gensim")
                            
                            # Créer un dictionnaire de mots-clés vide pour chaque topic
                            advanced_analysis['weighted_words'] = {}
                            for i in range(num_topics):
                                advanced_analysis['weighted_words'][str(i)] = [f"topic_{i}", f"mot_clé_{i}", f"terme_{i}", f"concept_{i}", f"sujet_{i}"]
                elif 'doc_topic_matrix' in doc_topic_data:
                    # Format traditionnel
                    logger.info("Format de matrice document-topic traditionnel détecté")
                    doc_topic_matrix = doc_topic_data['doc_topic_matrix']
                    if isinstance(doc_topic_matrix, list) and len(doc_topic_matrix) > 0:
                        first_doc = doc_topic_matrix[0]
                        if isinstance(first_doc, dict) and 'topic_distribution' in first_doc:
                            num_topics = len(first_doc['topic_distribution'])
                            logger.info(f"Trouvé {num_topics} topics dans la matrice document-topic")
                            
                            # Créer un dictionnaire de mots-clés vide pour chaque topic
                            advanced_analysis['weighted_words'] = {}
                            for i in range(num_topics):
                                advanced_analysis['weighted_words'][str(i)] = [f"topic_{i}", f"mot_clé_{i}", f"terme_{i}", f"concept_{i}", f"sujet_{i}"]
            elif isinstance(doc_topic_data, list) and len(doc_topic_data) > 0:
                # Format liste simple
                first_doc = doc_topic_data[0]
                if isinstance(first_doc, dict) and 'topic_distribution' in first_doc:
                    num_topics = len(first_doc['topic_distribution'])
                    logger.info(f"Trouvé {num_topics} topics dans la matrice document-topic (format liste)")
                    
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
