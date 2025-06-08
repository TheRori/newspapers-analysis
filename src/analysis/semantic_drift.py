"""
semantic_drift.py
Module dédié à l'analyse de l'évolution sémantique des termes dans le temps (drift sémantique).

Fonctions :
- Création de modèles Word2Vec par période temporelle
- Alignement des espaces vectoriels entre périodes
- Calcul de la distance sémantique entre représentations d'un terme
- Visualisation de l'évolution sémantique

Exemple d'utilisation :
    from analysis.semantic_drift import create_temporal_word2vec_models, align_embeddings, calculate_semantic_drift
"""

import os
import re
import json
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional, Union
from collections import defaultdict
import matplotlib.pyplot as plt
from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence
from scipy.spatial.distance import cosine
from scipy.linalg import orthogonal_procrustes

# Configuration du logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)


def preprocess_text_for_word2vec(text: str) -> List[str]:
    """
    Prétraite le texte pour l'entraînement Word2Vec.
    
    Args:
        text: Texte brut à prétraiter
    
    Returns:
        Liste de tokens (mots)
    """
    # Convertir en minuscules
    text = text.lower()
    
    # Remplacer les caractères spéciaux par des espaces
    text = re.sub(r'[^\w\s]', ' ', text)
    
    # Diviser en tokens et supprimer les espaces supplémentaires
    tokens = [token.strip() for token in text.split() if token.strip()]
    
    return tokens


def extract_year_from_article(article: Dict[str, Any]) -> Optional[int]:
    """
    Extrait l'année d'un article, soit de la date, soit de l'ID.
    
    Args:
        article: Dictionnaire représentant un article
    
    Returns:
        Année sous forme d'entier ou None si non trouvée
    """
    # Essayer d'extraire de la date
    date_str = article.get('date', '')
    if date_str:
        try:
            # Format ISO: 2023-01-01
            if '-' in date_str:
                return int(date_str.split('-')[0])
            # Format avec slash: 01/01/2023
            elif '/' in date_str:
                date_parts = date_str.split('/')
                if len(date_parts[-1]) == 4:  # Année en dernier (format DD/MM/YYYY)
                    return int(date_parts[-1])
                else:  # Année en premier (format YYYY/MM/DD)
                    return int(date_parts[0])
        except (ValueError, IndexError):
            pass
    
    # Essayer d'extraire de l'ID (format: article_YYYY-MM-DD_journal_XXXX_source)
    article_id = str(article.get('id', article.get('base_id', '')))
    if article_id:
        match = re.search(r'article_(\d{4})-\d{2}-\d{2}', article_id)
        if match:
            return int(match.group(1))
    
    return None


def group_articles_by_period(articles: List[Dict[str, Any]], 
                            period_type: str = 'year', 
                            custom_periods: Optional[List[Tuple[int, int]]] = None) -> Dict[str, List[Dict[str, Any]]]:
    """
    Groupe les articles par période temporelle.
    
    Args:
        articles: Liste de dictionnaires représentant les articles
        period_type: Type de période ('year', 'decade', ou 'custom')
        custom_periods: Liste de tuples (début, fin) pour les périodes personnalisées
    
    Returns:
        Dictionnaire avec les périodes comme clés et les listes d'articles comme valeurs
    """
    periods = defaultdict(list)
    
    for article in articles:
        year = extract_year_from_article(article)
        if year is None:
            continue
        
        if period_type == 'year':
            period_key = str(year)
        elif period_type == 'decade':
            decade = (year // 10) * 10
            period_key = f"{decade}s"
        elif period_type == 'custom' and custom_periods:
            period_key = None
            for start, end in custom_periods:
                if start <= year <= end:
                    period_key = f"{start}-{end}"
                    break
            if period_key is None:
                continue
        else:
            # Par défaut, utiliser l'année
            period_key = str(year)
        
        periods[period_key].append(article)
    
    # Filtrer les périodes avec trop peu d'articles
    return {k: v for k, v in periods.items() if len(v) >= 50}


def create_temporal_word2vec_models(articles: List[Dict[str, Any]], 
                                   period_type: str = 'year', 
                                   custom_periods: Optional[List[Tuple[int, int]]] = None,
                                   vector_size: int = 100,
                                   window: int = 5,
                                   min_count: int = 5,
                                   workers: int = 4,
                                   sg: int = 1) -> Dict[str, Word2Vec]:
    """
    Crée des modèles Word2Vec distincts pour chaque période temporelle.
    
    Args:
        articles: Liste de dictionnaires représentant les articles
        period_type: Type de période ('year', 'decade', ou 'custom')
        custom_periods: Liste de tuples (début, fin) pour les périodes personnalisées
        vector_size: Taille des vecteurs Word2Vec
        window: Taille de la fenêtre contextuelle
        min_count: Nombre minimum d'occurrences pour inclure un mot
        workers: Nombre de threads pour l'entraînement
        sg: 1 pour Skip-gram, 0 pour CBOW
    
    Returns:
        Dictionnaire avec les périodes comme clés et les modèles Word2Vec comme valeurs
    """
    logger.info(f"Création de modèles Word2Vec par période ({period_type})")
    
    # Grouper les articles par période
    periods = group_articles_by_period(articles, period_type, custom_periods)
    
    # Créer un modèle Word2Vec pour chaque période
    models = {}
    for period, period_articles in periods.items():
        logger.info(f"Entraînement du modèle pour la période {period} ({len(period_articles)} articles)")
        
        # Préparer les phrases pour l'entraînement
        sentences = []
        for article in period_articles:
            text = article.get('text', article.get('content', ''))
            if text:
                tokens = preprocess_text_for_word2vec(text)
                if tokens:
                    sentences.append(tokens)
        
        if not sentences:
            logger.warning(f"Aucune phrase trouvée pour la période {period}")
            continue
        
        # Entraîner le modèle
        model = Word2Vec(
            sentences=sentences,
            vector_size=vector_size,
            window=window,
            min_count=min_count,
            workers=workers,
            sg=sg
        )
        
        models[period] = model
        logger.info(f"Modèle pour {period} entraîné avec {len(model.wv.key_to_index)} mots")
    
    return models


def align_embeddings(source_model: Word2Vec, target_model: Word2Vec, shared_vocab: Optional[List[str]] = None) -> Tuple[np.ndarray, float]:
    """
    Aligne les espaces vectoriels de deux modèles Word2Vec en utilisant Orthogonal Procrustes.
    
    Args:
        source_model: Modèle Word2Vec source
        target_model: Modèle Word2Vec cible
        shared_vocab: Liste de mots communs à utiliser pour l'alignement (si None, utilise l'intersection des vocabulaires)
    
    Returns:
        Tuple (matrice de transformation, erreur d'alignement)
    """
    # Trouver les mots communs aux deux modèles
    if shared_vocab is None:
        source_vocab = set(source_model.wv.key_to_index.keys())
        target_vocab = set(target_model.wv.key_to_index.keys())
        shared_vocab = list(source_vocab.intersection(target_vocab))
    
    if len(shared_vocab) < 10:
        logger.warning(f"Vocabulaire commun trop petit pour un alignement fiable: {len(shared_vocab)} mots")
        return np.eye(source_model.wv.vector_size), float('inf')
    
    # Extraire les vecteurs pour les mots communs
    source_vecs = np.vstack([source_model.wv[word] for word in shared_vocab])
    target_vecs = np.vstack([target_model.wv[word] for word in shared_vocab])
    
    # Calculer la transformation orthogonale
    R, s = orthogonal_procrustes(source_vecs, target_vecs)
    
    # Calculer l'erreur d'alignement
    error = np.sum((source_vecs @ R - target_vecs) ** 2)
    
    return R, error


def is_redundant(term: str, candidate: str, threshold: float = 0.7) -> bool:
    """
    Détermine si un terme candidat est une redondance lexicale du terme d'origine.
    
    Args:
        term: Le terme d'origine
        candidate: Le terme candidat à vérifier
        threshold: Seuil de similarité (entre 0 et 1) au-delà duquel un terme est considéré redondant
        
    Returns:
        True si le terme est redondant, False sinon
    """
    # Convertir en minuscules pour la comparaison
    term = term.lower()
    candidate = candidate.lower()
    
    # Vérifier si l'un est contenu dans l'autre
    if term in candidate or candidate in term:
        return True
    
    # Vérifier les variantes avec/sans espaces ou tirets
    term_normalized = term.replace(' ', '').replace('-', '')
    candidate_normalized = candidate.replace(' ', '').replace('-', '')
    
    if term_normalized in candidate_normalized or candidate_normalized in term_normalized:
        return True
    
    # Calculer la distance de Levenshtein normalisée
    max_len = max(len(term), len(candidate))
    if max_len == 0:
        return False
    
    # Calculer la distance d'édition
    distance = 0
    for i in range(min(len(term), len(candidate))):
        if term[i] != candidate[i]:
            distance += 1
    
    # Ajouter la différence de longueur
    distance += abs(len(term) - len(candidate))
    
    # Normaliser
    similarity = 1 - (distance / max_len)
    
    return similarity > threshold

def find_most_similar_terms(models: Dict[str, Word2Vec], 
                           terms: List[str], 
                           top_n: int = 10,
                           filter_redundant: bool = True,
                           extra_terms: int = 5) -> Dict[str, Dict[str, List[Tuple[str, float]]]]:
    """
    Trouve les N mots les plus proches vectoriellement pour chaque terme dans chaque période.
    
    Args:
        models: Dictionnaire avec les périodes comme clés et les modèles Word2Vec comme valeurs
        terms: Liste de termes à analyser
        top_n: Nombre de mots similaires à retourner pour chaque terme
    
    Returns:
        Dictionnaire avec les termes comme clés, puis les périodes comme sous-clés, 
        et les listes de tuples (mot, similarité) comme valeurs
    """
    if not models:
        logger.warning("Aucun modèle fourni pour la recherche de termes similaires")
        return {}
    
    # Trier les périodes chronologiquement
    periods = sorted(models.keys())
    
    # Filtrer les termes présents dans au moins un modèle
    valid_terms = []
    for term in terms:
        if any(term in model.wv for model in models.values()):
            valid_terms.append(term)
    
    if not valid_terms:
        logger.warning(f"Aucun des termes fournis n'est présent dans les modèles")
        return {}
    
    logger.info(f"Recherche des {top_n} mots les plus similaires pour {len(valid_terms)} termes")
    
    # Trouver les mots similaires pour chaque terme dans chaque période
    similarity_results = {}
    
    for term in valid_terms:
        term_similarities = {}
        
        for period in periods:
            model = models[period]
            
            if term not in model.wv:
                logger.warning(f"Terme '{term}' non trouvé dans le modèle pour la période {period}")
                continue
            
            # Trouver les mots les plus similaires
            try:
                # Demander plus de mots si le filtrage est activé pour compenser ceux qui seront filtrés
                search_top_n = top_n + extra_terms if filter_redundant else top_n
                all_similar_words = model.wv.most_similar(term, topn=search_top_n)
                
                # Filtrer les redondances si demandé
                if filter_redundant:
                    filtered_similar_words = []
                    for word, similarity in all_similar_words:
                        if not is_redundant(term, word):
                            filtered_similar_words.append((word, similarity))
                            if len(filtered_similar_words) >= top_n:
                                break
                    
                    # Si on n'a pas assez de mots après filtrage, compléter avec les mots originaux
                    if len(filtered_similar_words) < top_n:
                        for word, similarity in all_similar_words:
                            if (word, similarity) not in filtered_similar_words:
                                filtered_similar_words.append((word, similarity))
                                if len(filtered_similar_words) >= top_n:
                                    break
                    
                    similar_words = filtered_similar_words[:top_n]
                    logger.info(f"Période {period}, terme '{term}': {len(all_similar_words) - len(similar_words)} mots redondants filtrés")
                else:
                    similar_words = all_similar_words[:top_n]
                
                term_similarities[period] = similar_words
            except KeyError:
                logger.warning(f"Erreur lors de la recherche de mots similaires pour '{term}' dans la période {period}")
                continue
        
        if term_similarities:
            similarity_results[term] = term_similarities
    
    return similarity_results


def export_similar_terms_to_csv(similarity_results: Dict[str, Dict[str, List[Tuple[str, float]]]], 
                              output_file: str, 
                              results_dir: Optional[str] = None,
                              source_file: Optional[str] = None,
                              metadata: Optional[Dict[str, Any]] = None) -> str:
    """
    Exporte les résultats des termes similaires vers un fichier CSV.
    
    Args:
        similarity_results: Dictionnaire avec les termes comme clés, les périodes comme sous-clés,
                           et les listes de tuples (mot, similarité) comme valeurs
        output_file: Nom du fichier de sortie
        results_dir: Répertoire de sortie (optionnel)
        source_file: Chemin vers le fichier source des articles (optionnel)
        metadata: Dictionnaire de métadonnées supplémentaires à inclure (optionnel)
    
    Returns:
        Chemin vers le fichier CSV créé
    """
    # Déterminer le chemin de sortie
    if results_dir:
        output_path = Path(results_dir) / output_file
    else:
        # Utiliser le répertoire de résultats par défaut
        from src.utils.config_loader import load_config
        config = load_config()
        results_dir = config['data']['results_dir']
        output_path = Path(results_dir) / "term_tracking" / output_file
    
    # Créer le répertoire s'il n'existe pas
    os.makedirs(output_path.parent, exist_ok=True)
    
    # Convertir en DataFrame pour faciliter l'export
    rows = []
    for term, period_dict in similarity_results.items():
        for period, similar_words in period_dict.items():
            for rank, (similar_word, similarity) in enumerate(similar_words, 1):
                rows.append({
                    'term': term,
                    'period': period,
                    'rank': rank,
                    'similar_word': similar_word,
                    'similarity': similarity
                })
    
    df = pd.DataFrame(rows)
    
    # Sauvegarder les métadonnées supplémentaires si fournies
    if metadata is None:
        metadata = {}
    
    # Ajouter le fichier source aux métadonnées s'il est fourni
    if source_file:
        metadata['source_file'] = source_file
    
    # Sauvegarder les métadonnées dans un fichier JSON séparé
    if metadata:
        metadata_path = output_path.with_suffix('.meta.json')
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)
        logger.info(f"Métadonnées exportées vers {metadata_path}")
    
    # Exporter vers CSV
    df.to_csv(output_path, index=False)
    logger.info(f"Données de termes similaires exportées vers {output_path}")
    
    return str(output_path)


def calculate_semantic_drift(models: Dict[str, Word2Vec], 
                           terms: List[str], 
                           reference_period: Optional[str] = None) -> Dict[str, Dict[str, float]]:
    """
    Calcule la distance sémantique (drift) pour chaque terme entre périodes.
    
    Args:
        models: Dictionnaire avec les périodes comme clés et les modèles Word2Vec comme valeurs
        terms: Liste de termes à analyser
        reference_period: Période de référence pour l'alignement (si None, utilise la première période)
    
    Returns:
        Dictionnaire avec les termes comme clés et un sous-dictionnaire {période: distance} comme valeurs
    """
    if not models:
        logger.warning("Aucun modèle fourni pour le calcul du drift sémantique")
        return {}
    
    # Trier les périodes chronologiquement
    periods = sorted(models.keys())
    
    if not reference_period:
        reference_period = periods[0]
    
    if reference_period not in models:
        logger.error(f"Période de référence {reference_period} non trouvée dans les modèles")
        return {}
    
    reference_model = models[reference_period]
    
    # Filtrer les termes présents dans le modèle de référence
    valid_terms = [term for term in terms if term in reference_model.wv]
    if not valid_terms:
        logger.warning(f"Aucun des termes fournis n'est présent dans le modèle de référence")
        return {}
    
    logger.info(f"Calcul du drift sémantique pour {len(valid_terms)} termes")
    
    # Calculer les distances pour chaque terme
    drift_results = {}
    
    for term in valid_terms:
        term_drift = {reference_period: 0.0}  # Distance à soi-même = 0
        
        for period in periods:
            if period == reference_period:
                continue
            
            if term not in models[period].wv:
                logger.warning(f"Terme '{term}' non trouvé dans le modèle pour la période {period}")
                continue
            
            # Aligner le modèle de la période avec le modèle de référence
            R, error = align_embeddings(models[period], reference_model)
            
            # Appliquer la transformation et calculer la distance cosinus
            aligned_vector = models[period].wv[term] @ R
            reference_vector = reference_model.wv[term]
            
            distance = cosine(aligned_vector, reference_vector)
            term_drift[period] = distance
        
        drift_results[term] = term_drift
    
    return drift_results


def export_semantic_drift_to_csv(drift_results: Dict[str, Dict[str, float]], 
                               output_file: str, 
                               results_dir: Optional[str] = None) -> str:
    """
    Exporte les résultats de drift sémantique vers un fichier CSV.
    
    Args:
        drift_results: Dictionnaire avec les termes comme clés et un sous-dictionnaire {période: distance} comme valeurs
        output_file: Nom du fichier de sortie
        results_dir: Répertoire de sortie (optionnel)
    
    Returns:
        Chemin vers le fichier CSV créé
    """
    # Déterminer le chemin de sortie
    if results_dir:
        output_path = Path(results_dir) / output_file
    else:
        # Utiliser le répertoire de résultats par défaut
        from src.utils.config_loader import load_config
        config = load_config()
        results_dir = config['data']['results_dir']
        output_path = Path(results_dir) / "term_tracking" / output_file
    
    # Créer le répertoire s'il n'existe pas
    os.makedirs(output_path.parent, exist_ok=True)
    
    # Convertir en DataFrame pour faciliter l'export
    rows = []
    for term, period_dict in drift_results.items():
        for period, distance in period_dict.items():
            rows.append({
                'term': term,
                'period': period,
                'semantic_distance': distance
            })
    
    df = pd.DataFrame(rows)
    
    # Exporter vers CSV
    df.to_csv(output_path, index=False)
    logger.info(f"Données de drift sémantique exportées vers {output_path}")
    
    return str(output_path)


def combine_frequency_and_drift(frequency_results: pd.DataFrame, 
                              drift_results: Dict[str, Dict[str, float]]) -> pd.DataFrame:
    """
    Combine les résultats de fréquence et de drift sémantique.
    
    Args:
        frequency_results: DataFrame avec les résultats de fréquence
        drift_results: Dictionnaire avec les résultats de drift sémantique
    
    Returns:
        DataFrame combiné
    """
    # Convertir les résultats de drift en DataFrame
    drift_rows = []
    for term, period_dict in drift_results.items():
        for period, distance in period_dict.items():
            drift_rows.append({
                'term': term,
                'period': period,
                'semantic_distance': distance
            })
    
    drift_df = pd.DataFrame(drift_rows)
    
    # Déterminer le type de résultats de fréquence (par année, par journal, etc.)
    if 'key' in frequency_results.columns:
        # Renommer la colonne 'key' en fonction du type de données
        try:
            # Essayer de convertir en entier pour voir si c'est une année
            frequency_results['key'].iloc[0] = int(frequency_results['key'].iloc[0])
            frequency_results = frequency_results.rename(columns={'key': 'period'})
        except (ValueError, TypeError):
            # Si ce n'est pas une année, c'est probablement un journal ou un article
            pass
    
    # Fusionner les DataFrames si possible
    if 'period' in frequency_results.columns:
        # Convertir period en string pour la fusion
        frequency_results['period'] = frequency_results['period'].astype(str)
        drift_df['period'] = drift_df['period'].astype(str)
        
        # Fusionner sur period et term
        result = pd.merge(
            frequency_results, 
            drift_df, 
            on=['period', 'term'] if 'term' in frequency_results.columns else 'period',
            how='outer'
        )
    else:
        # Si la fusion n'est pas possible, retourner les deux DataFrames côte à côte
        result = pd.concat([frequency_results, drift_df], axis=1)
    
    return result
