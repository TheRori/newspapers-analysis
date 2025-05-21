#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script pour classifier et filtrer différents types de contenu (publicités, offres d'emploi, programmes TV, etc.)
à l'aide de modèles ML entraînés.
"""

import os
import sys
import json
import yaml
import logging
import argparse
import pathlib
import pickle
import numpy as np
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple
from tqdm import tqdm
from sklearn.base import BaseEstimator, TransformerMixin


# Ajouter le répertoire parent au path pour pouvoir importer les modules
script_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.dirname(os.path.dirname(script_dir))
sys.path.insert(0, project_dir)

# Configurer le logger
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Chemin vers le fichier de configuration
config_path = os.path.join(project_dir, "config", "config.yaml")


# Dictionnaires de mots-clés pour différents types de contenu
CONTENT_KEYWORDS = {
    'ads': [
        'fr.', 'tél.', 'chf', 'prix', 'offre', 'vente', 'achat', 'promotion',
        'soldes', 'catalogue', 'magasin', 'boutique', 'commande', 'livraison',
        'gratuit', 'remise', 'réduction', 'stock', 'disponible', 'nouveau',
        'contact', 'adresse', 'horaire', 'ouvert', 'fermé', 'service', 'client',
        'garantie', 'qualité', 'produit', 'modèle', 'marque', 'référence'
    ],
    'job_offers': [
        'offre d\'emploi', 'poste', 'candidature', 'cv', 'curriculum vitae',
        'salaire', 'rémunération', 'expérience', 'formation', 'diplôme',
        'compétence', 'qualification', 'recrutement', 'embauche', 'cherche',
        'recherche', 'engageons', 'engager', 'postuler', 'candidat', 'emploi',
        'travail', 'temps plein', 'temps partiel', 'CDI', 'CDD', 'stage',
        'apprentissage', 'alternance', 'contrat', 'entreprise', 'société',
        'département', 'service', 'équipe', 'collaborateur', 'responsable'
    ],
    'tv_programs': [
        'programme', 'tv', 'télévision', 'chaîne', 'émission', 'film', 'série',
        'documentaire', 'journal', 'météo', 'sport', 'divertissement', 'jeu',
        'téléfilm', 'reportage', 'débat', 'interview', 'direct', 'rediffusion',
        'horaire', 'soirée', 'après-midi', 'matinée', 'nuit', 'h', 'heures',
        'minutes', 'TF1', 'France 2', 'France 3', 'Canal+', 'Arte', 'M6',
        'lundi', 'mardi', 'mercredi', 'jeudi', 'vendredi', 'samedi', 'dimanche'
    ]
}


class TextFeatureExtractor(BaseEstimator, TransformerMixin):
    """Extracteur de caractéristiques textuelles supplémentaires."""
    
    def __init__(self, content_type='ads'):
        self.content_type = content_type
        self.keywords = CONTENT_KEYWORDS.get(content_type, CONTENT_KEYWORDS['ads'])
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        """Extrait des caractéristiques textuelles supplémentaires."""
        features = np.zeros((len(X), 5))
        
        for i, text in enumerate(X):
            # Longueur du texte (nombre de caractères)
            features[i, 0] = len(text)
            
            # Nombre de mots
            features[i, 1] = len(text.split())
            
            # Proportion de chiffres
            features[i, 2] = sum(c.isdigit() for c in text) / max(len(text), 1)
            
            # Proportion de majuscules
            features[i, 3] = sum(c.isupper() for c in text) / max(len(text), 1)
            
            # Présence de mots-clés spécifiques au type de contenu
            features[i, 4] = sum(1 for keyword in self.keywords if keyword.lower() in text.lower()) / len(self.keywords)
        
        return features


# Fonction pour charger la configuration
def load_config(config_path):
    """Charge la configuration depuis un fichier YAML."""
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        return config
    except Exception as e:
        logger.error(f"Erreur lors du chargement de la configuration: {str(e)}")
        return {}


def load_model(model_path: str):
    """
    Charge le modèle de classification.
    
    Args:
        model_path: Chemin vers le fichier du modèle
        
    Returns:
        Modèle chargé
    """
    try:
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        logger.info(f"Modèle chargé avec succès: {model_path}")
        return model
    except Exception as e:
        logger.error(f"Erreur lors du chargement du modèle: {str(e)}")
        return None


def classify_articles_file(
    articles_path: str,
    model_path: str,
    content_type: str = 'ads',
    content_key: str = 'content',
    min_word_count: int = 80,
    output_path: Optional[str] = None,
    dry_run: bool = False,
    max_articles: Optional[int] = None
) -> Dict[str, Any]:
    """
    Classifie tous les articles d'un fichier JSON en utilisant le modèle ML.
    
    Args:
        articles_path: Chemin vers le fichier JSON contenant les articles
        model_path: Chemin vers le fichier du modèle de classification
        content_type: Type de contenu à classifier (ads, job_offers, tv_programs)
        content_key: Clé à utiliser en priorité pour le contenu des articles
        min_word_count: Nombre minimum de mots pour ne pas être automatiquement classé comme court
        output_path: Chemin où sauvegarder le fichier JSON filtré (si None, utilise le chemin d'entrée)
        dry_run: Si True, n'écrit pas le fichier de sortie
        max_articles: Nombre maximum d'articles à analyser (pour les tests)
        
    Returns:
        Dict contenant les statistiques du filtrage
    """
    # Charger le modèle
    model = load_model(model_path)
    if model is None:
        logger.error("Impossible de continuer sans modèle")
        return {"error": "Modèle non chargé"}
    
    # Charger les articles
    try:
        with open(articles_path, 'r', encoding='utf-8') as f:
            articles = json.load(f)
        logger.info(f"Articles chargés: {len(articles)}")
    except Exception as e:
        logger.error(f"Erreur lors du chargement des articles: {str(e)}")
        return {"error": f"Erreur lors du chargement des articles: {str(e)}"}
    
    # Limiter le nombre d'articles si nécessaire
    if max_articles is not None and max_articles > 0:
        articles = articles[:max_articles]
        logger.info(f"Limitation à {max_articles} articles pour les tests")
    
    # Définir l'ordre de priorité des clés de contenu
    content_keys = [content_key]
    for key in ['cleaned_text', 'original_cleaned_content', 'text', 'original_content', 'content']:
        if key != content_key and key not in content_keys:
            content_keys.append(key)
    
    # Filtrer les articles
    filtered_articles = []  # Articles qui ne correspondent pas au type de contenu recherché
    matched_articles = []   # Articles qui correspondent au type de contenu recherché
    short_articles = []     # Articles trop courts pour être analysés
    
    # Statistiques
    stats = {
        "total_articles": len(articles),
        "filtered_articles": 0,
        "matched_articles": 0,
        "short_articles": 0
    }
    
    # Traiter les articles
    for article in tqdm(articles, desc=f"Classification des articles ({content_type})"):
        # Obtenir le contenu de l'article selon l'ordre de priorité des clés
        content = None
        for key in content_keys:
            if key in article and article[key]:
                content = article[key]
                break
        
        if not content:
            logger.warning(f"Article sans contenu: {article.get('id', article.get('_id', 'unknown'))}")
            continue
        
        # Calculer le nombre de mots
        word_count = article.get('word_count', len(content.split()))
        
        # Traiter automatiquement les articles courts
        if word_count < min_word_count:
            article['classification_criteria'] = "COURT"
            short_articles.append(article)
            stats["short_articles"] += 1
            continue
        
        # Prédire si l'article correspond au type de contenu recherché
        is_match = model.predict([content])[0] == 1
        
        # Stocker le résultat
        if is_match:
            article['classification_criteria'] = f"{content_type.upper()}_ML"
            matched_articles.append(article)
            stats["matched_articles"] += 1
        else:
            filtered_articles.append(article)
            stats["filtered_articles"] += 1
    
    logger.info(f"Articles ne correspondant pas au type {content_type}: {len(filtered_articles)}")
    logger.info(f"Articles correspondant au type {content_type}: {len(matched_articles)}")
    logger.info(f"Articles courts: {len(short_articles)}")
    
    # Sauvegarder les résultats
    if not dry_run:
        # Générer les noms de fichiers de sortie
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        if output_path:
            base_output_path = output_path
        else:
            output_dir = os.path.dirname(articles_path)
            base_output_path = os.path.join(output_dir, f"classified_{content_type}_{timestamp}")
        
        # Sauvegarder les articles filtrés
        filtered_output_path = f"{base_output_path}_filtered.json"
        with open(filtered_output_path, 'w', encoding='utf-8') as f:
            json.dump(filtered_articles, f, indent=2, ensure_ascii=False)
        logger.info(f"Articles filtrés sauvegardés: {filtered_output_path}")
        
        # Sauvegarder les articles correspondant au type recherché
        matched_output_path = f"{base_output_path}_matched.json"
        with open(matched_output_path, 'w', encoding='utf-8') as f:
            json.dump(matched_articles, f, indent=2, ensure_ascii=False)
        logger.info(f"Articles correspondant au type {content_type} sauvegardés: {matched_output_path}")
        
        # Sauvegarder les articles courts
        short_output_path = f"{base_output_path}_short.json"
        with open(short_output_path, 'w', encoding='utf-8') as f:
            json.dump(short_articles, f, indent=2, ensure_ascii=False)
        logger.info(f"Articles courts sauvegardés: {short_output_path}")
    
    return stats


def filter_content_by_topic(
    articles_path: str,
    doc_topic_matrix_path: str,
    topic_id: int,
    model_path: str,
    content_type: str = 'ads',
    content_key: str = 'content',
    min_topic_value: float = 0.5,
    min_word_count: int = 80,
    output_path: Optional[str] = None,
    dry_run: bool = False,
    max_articles: Optional[int] = None
) -> Dict[str, Any]:
    """
    Filtre les articles d'un topic spécifique en utilisant le modèle ML.
    
    Args:
        articles_path: Chemin vers le fichier JSON contenant les articles
        doc_topic_matrix_path: Chemin vers le fichier JSON contenant la matrice document-topic
        topic_id: ID du topic à analyser
        model_path: Chemin vers le fichier du modèle de classification
        content_type: Type de contenu à classifier (ads, job_offers, tv_programs)
        content_key: Clé à utiliser en priorité pour le contenu des articles
        min_topic_value: Valeur minimale du topic pour considérer un article (0.0-1.0)
        min_word_count: Nombre minimum de mots pour ne pas être automatiquement classé comme court
        output_path: Chemin où sauvegarder le fichier JSON filtré (si None, utilise le chemin d'entrée)
        dry_run: Si True, n'écrit pas le fichier de sortie
        max_articles: Nombre maximum d'articles à analyser (pour les tests)
        
    Returns:
        Dict contenant les statistiques du filtrage
    """
    # Charger le modèle
    model = load_model(model_path)
    if model is None:
        logger.error("Impossible de continuer sans modèle")
        return {"error": "Modèle non chargé"}
    
    # Charger les articles
    try:
        with open(articles_path, 'r', encoding='utf-8') as f:
            articles = json.load(f)
        logger.info(f"Articles chargés: {len(articles)}")
    except Exception as e:
        logger.error(f"Erreur lors du chargement des articles: {str(e)}")
        return {"error": f"Erreur lors du chargement des articles: {str(e)}"}
    
    # Charger la matrice document-topic
    try:
        with open(doc_topic_matrix_path, 'r', encoding='utf-8') as f:
            doc_topic_matrix = json.load(f)
        logger.info(f"Matrice document-topic chargée: {len(doc_topic_matrix)} documents")
    except Exception as e:
        logger.error(f"Erreur lors du chargement de la matrice document-topic: {str(e)}")
        return {"error": f"Erreur lors du chargement de la matrice document-topic: {str(e)}"}
    
    # Filtrer les articles par topic
    topic_articles = []
    for article in articles:
        article_id = article.get('id', article.get('_id', ''))
        
        # Vérifier si l'article est dans la matrice document-topic
        if article_id in doc_topic_matrix:
            # Vérifier si le topic est suffisamment présent dans l'article
            topic_value = doc_topic_matrix[article_id][topic_id]
            if topic_value >= min_topic_value:
                topic_articles.append(article)
    
    logger.info(f"Articles du topic {topic_id} (valeur >= {min_topic_value}): {len(topic_articles)}")
    
    # Limiter le nombre d'articles si nécessaire
    if max_articles is not None and max_articles > 0:
        topic_articles = topic_articles[:max_articles]
        logger.info(f"Limitation à {max_articles} articles pour les tests")
    
    # Définir l'ordre de priorité des clés de contenu
    content_keys = [content_key]
    for key in ['cleaned_text', 'original_cleaned_content', 'text', 'original_content', 'content']:
        if key != content_key and key not in content_keys:
            content_keys.append(key)
    
    # Filtrer les articles
    filtered_articles = []  # Articles qui ne correspondent pas au type de contenu recherché
    matched_articles = []   # Articles qui correspondent au type de contenu recherché
    short_articles = []     # Articles trop courts pour être analysés
    
    # Statistiques
    stats = {
        "total_topic_articles": len(topic_articles),
        "filtered_articles": 0,
        "matched_articles": 0,
        "short_articles": 0
    }
    
    # Traiter les articles
    for article in tqdm(topic_articles, desc=f"Classification des articles du topic {topic_id} ({content_type})"):
        # Obtenir le contenu de l'article selon l'ordre de priorité des clés
        content = None
        for key in content_keys:
            if key in article and article[key]:
                content = article[key]
                break
        
        if not content:
            logger.warning(f"Article sans contenu: {article.get('id', article.get('_id', 'unknown'))}")
            continue
        
        # Calculer le nombre de mots
        word_count = article.get('word_count', len(content.split()))
        
        # Traiter automatiquement les articles courts
        if word_count < min_word_count:
            article['classification_criteria'] = "COURT"
            short_articles.append(article)
            stats["short_articles"] += 1
            continue
        
        # Prédire si l'article correspond au type de contenu recherché
        is_match = model.predict([content])[0] == 1
        
        # Stocker le résultat
        if is_match:
            article['classification_criteria'] = f"{content_type.upper()}_ML"
            matched_articles.append(article)
            stats["matched_articles"] += 1
        else:
            filtered_articles.append(article)
            stats["filtered_articles"] += 1
    
    logger.info(f"Articles ne correspondant pas au type {content_type}: {len(filtered_articles)}")
    logger.info(f"Articles correspondant au type {content_type}: {len(matched_articles)}")
    logger.info(f"Articles courts: {len(short_articles)}")
    
    # Sauvegarder les résultats
    if not dry_run:
        # Générer les noms de fichiers de sortie
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        if output_path:
            base_output_path = output_path
        else:
            output_dir = os.path.dirname(articles_path)
            base_output_path = os.path.join(output_dir, f"classified_{content_type}_topic{topic_id}_{timestamp}")
        
        # Sauvegarder les articles filtrés
        filtered_output_path = f"{base_output_path}_filtered.json"
        with open(filtered_output_path, 'w', encoding='utf-8') as f:
            json.dump(filtered_articles, f, indent=2, ensure_ascii=False)
        logger.info(f"Articles filtrés sauvegardés: {filtered_output_path}")
        
        # Sauvegarder les articles correspondant au type recherché
        matched_output_path = f"{base_output_path}_matched.json"
        with open(matched_output_path, 'w', encoding='utf-8') as f:
            json.dump(matched_articles, f, indent=2, ensure_ascii=False)
        logger.info(f"Articles correspondant au type {content_type} sauvegardés: {matched_output_path}")
        
        # Sauvegarder les articles courts
        short_output_path = f"{base_output_path}_short.json"
        with open(short_output_path, 'w', encoding='utf-8') as f:
            json.dump(short_articles, f, indent=2, ensure_ascii=False)
        logger.info(f"Articles courts sauvegardés: {short_output_path}")
    
    return stats


def get_parser():
    """
    Crée le parser d'arguments pour le script.
    """
    parser = argparse.ArgumentParser(description='Classifie et filtre différents types de contenu à l\'aide de modèles ML.')
    
    parser.add_argument('--articles', type=str, required=True,
                        help='Chemin vers le fichier JSON contenant les articles')
    
    parser.add_argument('--doc-topic-matrix', type=str, default=None,
                        help='Chemin vers le fichier JSON contenant la matrice document-topic')
    
    parser.add_argument('--topic-id', type=int, default=None,
                        help='ID du topic à analyser')
    
    parser.add_argument('--model', type=str, required=True,
                        help='Chemin vers le fichier du modèle de classification')
    
    parser.add_argument('--content-type', type=str, default='ads', choices=['ads', 'job_offers', 'tv_programs'],
                        help='Type de contenu à classifier (défaut: ads)')
    
    parser.add_argument('--content-key', type=str, default='content',
                        help='Clé à utiliser en priorité pour le contenu des articles (défaut: content)')
    
    parser.add_argument('--min-topic-value', type=float, default=0.5,
                        help='Valeur minimale du topic pour considérer un article (0.0-1.0)')
    
    parser.add_argument('--min-word-count', type=int, default=80,
                        help='Nombre minimum de mots pour ne pas être automatiquement classé comme court')
    
    parser.add_argument('--output', type=str, default=None,
                        help='Chemin où sauvegarder les fichiers JSON filtrés')
    
    parser.add_argument('--dry-run', action='store_true',
                        help='Ne pas écrire les fichiers de sortie')
    
    parser.add_argument('--max-articles', type=int, default=None,
                        help='Nombre maximum d\'articles à analyser (pour les tests)')
    
    return parser


def main():
    """
    Fonction principale du script.
    """
    # Charger la configuration
    config = load_config(config_path)
    
    # Parser les arguments
    parser = get_parser()
    args = parser.parse_args()
    
    # Déterminer le mode de fonctionnement
    if args.doc_topic_matrix is None or args.topic_id is None:
        # Mode classification de tous les articles
        logger.info(f"Mode classification de tous les articles (type: {args.content_type})")
        stats = classify_articles_file(
            articles_path=args.articles,
            model_path=args.model,
            content_type=args.content_type,
            content_key=args.content_key,
            min_word_count=args.min_word_count,
            output_path=args.output,
            dry_run=args.dry_run,
            max_articles=args.max_articles
        )
    else:
        # Mode filtrage par topic
        logger.info(f"Mode filtrage par topic (topic: {args.topic_id}, type: {args.content_type})")
        stats = filter_content_by_topic(
            articles_path=args.articles,
            doc_topic_matrix_path=args.doc_topic_matrix,
            topic_id=args.topic_id,
            model_path=args.model,
            content_type=args.content_type,
            content_key=args.content_key,
            min_topic_value=args.min_topic_value,
            min_word_count=args.min_word_count,
            output_path=args.output,
            dry_run=args.dry_run,
            max_articles=args.max_articles
        )
    
    # Afficher les statistiques
    logger.info("Statistiques de la classification:")
    for key, value in stats.items():
        logger.info(f"  {key}: {value}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
