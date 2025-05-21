#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script pour entraîner un classifieur automatique de publicités basé sur les données étiquetées.
Utilise une régression logistique avec des features TF-IDF et d'autres caractéristiques textuelles.
"""

import os
import sys
import json
import yaml
import logging
import argparse
import pathlib
import numpy as np
import pandas as pd
import pickle
import re
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple
from tqdm import tqdm
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

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


def load_articles(file_path: str) -> List[Dict[str, Any]]:
    """
    Charge les articles depuis un fichier JSON.
    
    Args:
        file_path: Chemin vers le fichier JSON
        
    Returns:
        Liste des articles
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            articles = json.load(f)
        return articles
    except Exception as e:
        logger.error(f"Erreur lors du chargement des articles: {str(e)}")
        return []


def prepare_dataset(ads_file: str, non_ads_file: str) -> Tuple[pd.DataFrame, np.ndarray]:
    """
    Prépare le dataset à partir des fichiers d'articles étiquetés.
    
    Args:
        ads_file: Chemin vers le fichier JSON contenant les publicités
        non_ads_file: Chemin vers le fichier JSON contenant les non-publicités
        
    Returns:
        Tuple (DataFrame, labels) contenant les textes et les étiquettes (1 pour publicité, 0 pour non-publicité)
    """
    # Charger les articles
    ads_articles = load_articles(ads_file)
    non_ads_articles = load_articles(non_ads_file)
    
    logger.info(f"Nombre d'articles publicités: {len(ads_articles)}")
    logger.info(f"Nombre d'articles non-publicités: {len(non_ads_articles)}")
    
    # Créer le DataFrame
    data = []
    
    # Ajouter les publicités
    for article in ads_articles:
        # Utiliser content en priorité, puis les autres contenus disponibles
        content = None
        for key in ['content', 'cleaned_text', 'original_cleaned_content', 'text', 'original_content']:
            if key in article and article[key]:
                content = article[key]
                break
        
        if content:
            data.append({
                'id': article.get('id', article.get('_id', '')),
                'text': content,
                'word_count': article.get('word_count', len(content.split())),
                'label': 1  # 1 pour publicité
            })
    
    # Ajouter les non-publicités
    for article in non_ads_articles:
        # Utiliser content en priorité, puis les autres contenus disponibles
        content = None
        for key in ['content', 'cleaned_text', 'original_cleaned_content', 'text', 'original_content']:
            if key in article and article[key]:
                content = article[key]
                break
        
        if content:
            data.append({
                'id': article.get('id', article.get('_id', '')),
                'text': content,
                'word_count': article.get('word_count', len(content.split())),
                'label': 0  # 0 pour non-publicité
            })
    
    # Créer le DataFrame
    df = pd.DataFrame(data)
    
    # Extraire les étiquettes
    labels = df['label'].values
    
    logger.info(f"Dataset préparé avec {len(df)} articles au total")
    logger.info(f"Distribution des étiquettes: {np.bincount(labels)}")
    
    return df, labels


def train_model(df: pd.DataFrame, labels: np.ndarray, test_size: float = 0.2, random_state: int = 42) -> Tuple[Pipeline, Dict[str, float]]:
    """
    Entraîne un modèle de classification des publicités.
    
    Args:
        df: DataFrame contenant les textes
        labels: Étiquettes (1 pour publicité, 0 pour non-publicité)
        test_size: Proportion du dataset à utiliser pour le test
        random_state: Graine aléatoire pour la reproductibilité
        
    Returns:
        Tuple (modèle entraîné, métriques d'évaluation)
    """
    # Diviser le dataset en ensembles d'entraînement et de test
    X_train, X_test, y_train, y_test = train_test_split(
        df['text'].values, labels, test_size=test_size, random_state=random_state, stratify=labels
    )
    
    logger.info(f"Ensemble d'entraînement: {len(X_train)} articles")
    logger.info(f"Ensemble de test: {len(X_test)} articles")
    
    # Créer le pipeline avec TF-IDF et caractéristiques textuelles
    pipeline = Pipeline([
        ('features', FeatureUnion([
            ('tfidf', Pipeline([
                ('vect', TfidfVectorizer(
                    ngram_range=(1, 3),
                    max_features=5000,
                    min_df=2,
                    max_df=0.9,
                    sublinear_tf=True
                ))
            ])),
            ('text_features', TextFeatureExtractor())
        ])),
        ('classifier', LogisticRegression(
            C=1.0,
            class_weight='balanced',
            max_iter=1000,
            random_state=random_state
        ))
    ])
    
    # Entraîner le modèle
    logger.info("Entraînement du modèle...")
    pipeline.fit(X_train, y_train)
    
    # Évaluer le modèle
    logger.info("Évaluation du modèle...")
    y_pred = pipeline.predict(X_test)
    
    # Calculer les métriques
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    logger.info(f"Accuracy: {accuracy:.4f}")
    logger.info(f"Precision: {precision:.4f}")
    logger.info(f"Recall: {recall:.4f}")
    logger.info(f"F1-score: {f1:.4f}")
    
    logger.info("\nRapport de classification:")
    logger.info(classification_report(y_test, y_pred))
    
    logger.info("\nMatrice de confusion:")
    logger.info(confusion_matrix(y_test, y_pred))
    
    # Validation croisée
    logger.info("\nValidation croisée (5-fold):")
    cv_scores = cross_val_score(pipeline, df['text'].values, labels, cv=5, scoring='f1')
    logger.info(f"Scores F1 par fold: {cv_scores}")
    logger.info(f"Score F1 moyen: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
    
    # Retourner le modèle et les métriques
    metrics = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'cv_f1_mean': cv_scores.mean(),
        'cv_f1_std': cv_scores.std()
    }
    
    return pipeline, metrics


def save_model(model: Pipeline, output_path: str):
    """
    Sauvegarde le modèle entraîné.
    
    Args:
        model: Modèle à sauvegarder
        output_path: Chemin où sauvegarder le modèle
    """
    try:
        with open(output_path, 'wb') as f:
            pickle.dump(model, f)
        logger.info(f"Modèle sauvegardé avec succès: {output_path}")
    except Exception as e:
        logger.error(f"Erreur lors de la sauvegarde du modèle: {str(e)}")


def get_parser():
    """
    Crée le parser d'arguments pour le script.
    """
    parser = argparse.ArgumentParser(description='Entraîne un classifieur automatique de publicités.')
    
    parser.add_argument('--ads-file', type=str, required=True,
                        help='Chemin vers le fichier JSON contenant les publicités')
    
    parser.add_argument('--non-ads-file', type=str, required=True,
                        help='Chemin vers le fichier JSON contenant les non-publicités')
    
    parser.add_argument('--output-model', type=str, default=None,
                        help='Chemin où sauvegarder le modèle entraîné')
    
    parser.add_argument('--test-size', type=float, default=0.2,
                        help='Proportion du dataset à utiliser pour le test (défaut: 0.2)')
    
    parser.add_argument('--random-state', type=int, default=42,
                        help='Graine aléatoire pour la reproductibilité (défaut: 42)')
    
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
    
    # Préparer le dataset
    df, labels = prepare_dataset(args.ads_file, args.non_ads_file)
    
    # Entraîner le modèle
    model, metrics = train_model(df, labels, test_size=args.test_size, random_state=args.random_state)
    
    # Sauvegarder le modèle
    if args.output_model:
        output_model_path = args.output_model
    else:
        # Créer un nom de fichier basé sur la date et l'heure
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = os.path.join(project_dir, "models")
        os.makedirs(output_dir, exist_ok=True)
        output_model_path = os.path.join(output_dir, f"ads_classifier_{timestamp}.pkl")
    
    save_model(model, output_model_path)
    
    # Sauvegarder les métriques
    metrics_path = output_model_path.replace('.pkl', '_metrics.json')
    try:
        with open(metrics_path, 'w', encoding='utf-8') as f:
            json.dump(metrics, f, indent=2)
        logger.info(f"Métriques sauvegardées avec succès: {metrics_path}")
    except Exception as e:
        logger.error(f"Erreur lors de la sauvegarde des métriques: {str(e)}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
