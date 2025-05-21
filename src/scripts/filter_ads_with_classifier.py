#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script pour filtrer les publicités des articles à l'aide du classifieur ML entraîné.
Alternative au script filter_ads_from_topic.py qui utilise un LLM.
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


class TextFeatureExtractor(BaseEstimator, TransformerMixin):
    """Extracteur de caractéristiques textuelles supplémentaires."""
    
    def __init__(self):
        self.ad_keywords = [
            'fr.', 'tél.', 'chf', 'prix', 'offre', 'vente', 'achat', 'promotion',
            'soldes', 'catalogue', 'magasin', 'boutique', 'commande', 'livraison',
            'gratuit', 'remise', 'réduction', 'stock', 'disponible', 'nouveau',
            'contact', 'adresse', 'horaire', 'ouvert', 'fermé', 'service', 'client',
            'garantie', 'qualité', 'produit', 'modèle', 'marque', 'référence',
            'offre d\'emploi', 'poste', 'candidature', 'cv', 'curriculum vitae',
            'salaire', 'rémunération', 'expérience', 'formation', 'diplôme',
            'compétence', 'qualification', 'recrutement', 'embauche'
        ]
    
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
            
            # Présence de mots-clés publicitaires
            features[i, 4] = sum(1 for keyword in self.ad_keywords if keyword.lower() in text.lower()) / len(self.ad_keywords)
        
        return features

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
    Charge le modèle de classification des publicités.
    
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
    output_path: Optional[str] = None,
    dry_run: bool = False,
    max_articles: Optional[int] = None
) -> Dict[str, Any]:
    """
    Classifie tous les articles d'un fichier JSON en utilisant le modèle ML.
    
    Args:
        articles_path: Chemin vers le fichier JSON contenant les articles
        model_path: Chemin vers le fichier du modèle de classification
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
    
    # Filtrer les publicités
    filtered_articles = []
    ads_articles = []
    short_ads_articles = []
    
    # Statistiques
    stats = {
        "total_articles": len(articles),
        "filtered_articles": 0,
        "ads_articles": 0,
        "short_ads_articles": 0
    }
    
    # Traiter les articles
    for article in tqdm(articles, desc="Classification des articles"):
        # Obtenir le contenu de l'article (content en priorité)
        content = None
        for key in ['content', 'cleaned_text', 'original_cleaned_content', 'text', 'original_content']:
            if key in article and article[key]:
                content = article[key]
                break
        
        if not content:
            logger.warning(f"Article sans contenu: {article.get('id', article.get('_id', 'unknown'))}")
            continue
        
        # Calculer le nombre de mots
        word_count = article.get('word_count', len(content.split()))
        
        # Traiter automatiquement les articles courts comme des publicités
        if word_count < 80:
            article['ad_criteria'] = "COURT"
            short_ads_articles.append(article)
            stats["short_ads_articles"] += 1
            continue
        
        # Prédire si l'article est une publicité
        is_ad = model.predict([content])[0] == 1
        
        # Stocker le résultat
        if is_ad:
            article['ad_criteria'] = "ML"
            ads_articles.append(article)
            stats["ads_articles"] += 1
        else:
            filtered_articles.append(article)
            stats["filtered_articles"] += 1
    
    logger.info(f"Articles filtrés (non-publicités): {len(filtered_articles)}")
    logger.info(f"Articles identifiés comme publicités: {len(ads_articles)}")
    logger.info(f"Articles courts traités comme publicités: {len(short_ads_articles)}")
    
    # Sauvegarder les résultats
    if not dry_run:
        # Générer les noms de fichiers de sortie
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        if output_path:
            base_output_path = output_path
        else:
            output_dir = os.path.dirname(articles_path)
            base_output_path = os.path.join(output_dir, f"classified_articles_{timestamp}")
        
        # Sauvegarder les articles filtrés
        filtered_output_path = f"{base_output_path}_filtered.json"
        with open(filtered_output_path, 'w', encoding='utf-8') as f:
            json.dump(filtered_articles, f, indent=2, ensure_ascii=False)
        logger.info(f"Articles filtrés sauvegardés: {filtered_output_path}")
        
        # Sauvegarder les publicités
        ads_output_path = f"{base_output_path}_ads.json"
        with open(ads_output_path, 'w', encoding='utf-8') as f:
            json.dump(ads_articles, f, indent=2, ensure_ascii=False)
        logger.info(f"Publicités sauvegardées: {ads_output_path}")
        
        # Sauvegarder les publicités courtes
        short_ads_output_path = f"{base_output_path}_short_ads.json"
        with open(short_ads_output_path, 'w', encoding='utf-8') as f:
            json.dump(short_ads_articles, f, indent=2, ensure_ascii=False)
        logger.info(f"Publicités courtes sauvegardées: {short_ads_output_path}")
    
    return stats


def filter_ads_with_classifier(
    articles_path: str,
    doc_topic_matrix_path: str,
    topic_id: int,
    model_path: str,
    min_topic_value: float = 0.5,
    output_path: Optional[str] = None,
    dry_run: bool = False,
    max_articles: Optional[int] = None
) -> Dict[str, Any]:
    """
    Filtre les publicités d'un topic spécifique en utilisant le classifieur ML.
    
    Args:
        articles_path: Chemin vers le fichier JSON contenant les articles
        doc_topic_matrix_path: Chemin vers le fichier JSON contenant la matrice document-topic
        topic_id: ID du topic à analyser
        model_path: Chemin vers le fichier du modèle de classification
        min_topic_value: Valeur minimale du topic pour considérer un article (0.0-1.0)
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
    
    # Filtrer les publicités
    filtered_articles = []
    ads_articles = []
    short_ads_articles = []
    
    # Statistiques
    stats = {
        "total_articles": len(topic_articles),
        "filtered_articles": 0,
        "ads_articles": 0,
        "short_ads_articles": 0
    }
    
    # Traiter les articles
    for article in tqdm(topic_articles, desc="Filtrage des publicités"):
        # Obtenir le contenu de l'article (content en priorité)
        content = None
        for key in ['content', 'cleaned_text', 'original_cleaned_content', 'text', 'original_content']:
            if key in article and article[key]:
                content = article[key]
                break
        
        if not content:
            logger.warning(f"Article sans contenu: {article.get('id', article.get('_id', 'unknown'))}")
            continue
        
        # Calculer le nombre de mots
        word_count = article.get('word_count', len(content.split()))
        
        # Traiter automatiquement les articles courts comme des publicités
        if word_count < 80:
            article['ad_criteria'] = "COURT"
            short_ads_articles.append(article)
            stats["short_ads_articles"] += 1
            continue
        
        # Prédire si l'article est une publicité
        is_ad = model.predict([content])[0] == 1
        
        # Stocker le résultat
        if is_ad:
            article['ad_criteria'] = "ML"
            ads_articles.append(article)
            stats["ads_articles"] += 1
        else:
            filtered_articles.append(article)
            stats["filtered_articles"] += 1
    
    logger.info(f"Articles filtrés (non-publicités): {len(filtered_articles)}")
    logger.info(f"Articles identifiés comme publicités: {len(ads_articles)}")
    logger.info(f"Articles courts traités comme publicités: {len(short_ads_articles)}")
    
    # Sauvegarder les résultats
    if not dry_run:
        # Générer les noms de fichiers de sortie
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        if output_path:
            base_output_path = output_path
        else:
            output_dir = os.path.dirname(articles_path)
            base_output_path = os.path.join(output_dir, f"filtered_articles_{timestamp}")
        
        # Sauvegarder les articles filtrés
        filtered_output_path = f"{base_output_path}_filtered_topic{topic_id}.json"
        with open(filtered_output_path, 'w', encoding='utf-8') as f:
            json.dump(filtered_articles, f, indent=2, ensure_ascii=False)
        logger.info(f"Articles filtrés sauvegardés: {filtered_output_path}")
        
        # Sauvegarder les publicités
        ads_output_path = f"{base_output_path}_ads_topic{topic_id}.json"
        with open(ads_output_path, 'w', encoding='utf-8') as f:
            json.dump(ads_articles, f, indent=2, ensure_ascii=False)
        logger.info(f"Publicités sauvegardées: {ads_output_path}")
        
        # Sauvegarder les publicités courtes
        short_ads_output_path = f"{base_output_path}_short_ads_topic{topic_id}.json"
        with open(short_ads_output_path, 'w', encoding='utf-8') as f:
            json.dump(short_ads_articles, f, indent=2, ensure_ascii=False)
        logger.info(f"Publicités courtes sauvegardées: {short_ads_output_path}")
    
    return stats


def get_parser():
    """
    Crée le parser d'arguments pour le script.
    """
    parser = argparse.ArgumentParser(description='Filtre les publicités des articles à l\'aide du classifieur ML.')
    
    parser.add_argument('--articles', type=str, required=True,
                        help='Chemin vers le fichier JSON contenant les articles')
    
    parser.add_argument('--doc-topic-matrix', type=str, default=None,
                        help='Chemin vers le fichier JSON contenant la matrice document-topic')
    
    parser.add_argument('--topic-id', type=int, default=None,
                        help='ID du topic à analyser')
    
    parser.add_argument('--model', type=str, required=True,
                        help='Chemin vers le fichier du modèle de classification')
    
    parser.add_argument('--min-topic-value', type=float, default=0.5,
                        help='Valeur minimale du topic pour considérer un article (0.0-1.0)')
    
    parser.add_argument('--output', type=str, default=None,
                        help='Chemin où sauvegarder le fichier JSON filtré')
    
    parser.add_argument('--dry-run', action='store_true',
                        help='Ne pas écrire le fichier de sortie')
    
    parser.add_argument('--max-articles', type=int, default=None,
                        help='Nombre maximum d\'articles à analyser (pour les tests)')
    
    parser.add_argument('--classify-all', action='store_true',
                        help='Classifier tous les articles sans filtrer par topic')
    
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
    if args.classify_all or (args.doc_topic_matrix is None or args.topic_id is None):
        # Mode classification de tous les articles
        logger.info("Mode classification de tous les articles")
        stats = classify_articles_file(
            articles_path=args.articles,
            model_path=args.model,
            output_path=args.output,
            dry_run=args.dry_run,
            max_articles=args.max_articles
        )
    else:
        # Mode filtrage par topic
        logger.info("Mode filtrage par topic")
        stats = filter_ads_with_classifier(
            articles_path=args.articles,
            doc_topic_matrix_path=args.doc_topic_matrix,
            topic_id=args.topic_id,
            model_path=args.model,
            min_topic_value=args.min_topic_value,
            output_path=args.output,
            dry_run=args.dry_run,
            max_articles=args.max_articles
        )
    
    # Afficher les statistiques
    logger.info("Statistiques du filtrage:")
    for key, value in stats.items():
        logger.info(f"  {key}: {value}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
