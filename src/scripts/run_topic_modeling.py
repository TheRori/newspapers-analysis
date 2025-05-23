import json
import os
import argparse
import sys
import uuid
import time
import pickle
import hashlib
from pathlib import Path
import logging

# Add project root to path
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(project_root))

from src.analysis.topic_modeling import TopicModeler
from src.utils.config_loader import load_config
# Importer la classe SpacyPreprocessor directement
from src.preprocessing.spacy_preprocessor import SpacyPreprocessor
from bertopic import BERTopic

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s", stream=sys.stdout)
logger = logging.getLogger()


def get_parser():
    """Argument parser for the script."""
    parser = argparse.ArgumentParser(description="Run topic modeling on articles.")
    parser.add_argument('--source-file', type=str, help='Path to the source JSON file of articles.')
    parser.add_argument('--cache-file', type=str, help='Path to a specific cache file to use or create.')
    parser.add_argument('--engine', choices=['gensim', 'bertopic'], default='bertopic', help='Topic modeling engine.')
    parser.add_argument('--algorithm', type=str, default=None, help='Algorithm (e.g., gensim_lda, bertopic). Overrides engine default.')
    parser.add_argument('--num-topics', type=str, default='auto', help='Number of topics. Use "auto" for BERTopic.')
    parser.add_argument('--workers', type=int, default=-1, help='Number of CPU workers for modeling.')
    return parser

# --- LOGIQUE DE CACHE CORRIGÉE ---
def load_or_preprocess_data(config: dict, articles: list, cache_file: Path) -> dict:
    """
    Loads preprocessed (tokenized) data from cache if it exists.
    Otherwise, runs SpaCy preprocessing using the robust 'process_documents' method
    and saves the result to the cache.
    """
    if cache_file.exists():
        logger.info(f"Loading preprocessed data from cache: {cache_file}")
        with open(cache_file, 'rb') as f:
            return pickle.load(f)

    # --- CORRECTION MAJEURE : Utilisation de la méthode de batch 'process_documents' ---
    logger.info(f"Cache not found. Preprocessing {len(articles)} documents with SpaCy...")
    preproc_config = config.get('analysis', {}).get('topic_modeling', {}).get('preprocessing', {})
    spacy_preprocessor = SpacyPreprocessor(preproc_config)
    
    # On utilise la méthode de traitement par lots, qui est plus riche et efficace
    # Elle prend la liste de dictionnaires et ajoute une clé 'tokens'
    processed_articles = spacy_preprocessor.process_documents(
        articles,
        text_key='content', # Clé primaire à utiliser
        output_key='tokens', # Nom de la clé pour les résultats
        alternative_keys=['text', 'cleaned_text'] # Clés de secours
    )
    
    # On extrait les textes et les tokens des documents traités pour la mise en cache
    texts = [doc.get('content', doc.get('text', '')) for doc in processed_articles]
    tokenized_texts = [doc.get('tokens', []) for doc in processed_articles]
    
    preprocessed_data = {'texts': texts, 'tokenized_texts': tokenized_texts}
    
    logger.info(f"Saving preprocessed data to cache: {cache_file}")
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_file, 'wb') as f:
        pickle.dump(preprocessed_data, f)
        
    return preprocessed_data

# (Le reste du fichier est identique à la version précédente)
# ... (Fonction perform_and_save_advanced_analysis) ...
# Dans run_topic_modeling.py

def perform_and_save_advanced_analysis(modeler: TopicModeler, run_id: str, config: dict, results_dir: Path, tokenized_texts: list = None, articles: list = None):
    """
    Performs advanced analysis (coherence, distribution, etc.) and saves the results.
    """
    logger.info("Performing advanced analysis...")
    advanced_results = {"run_id": run_id, "config": config}
    
    engine = config.get('algorithm')

    if engine in ['gensim_lda', 'hdp']:
        # ... (la logique pour Gensim reste inchangée)
        if not hasattr(modeler, 'gensim_dictionary'):
            logger.warning("Gensim model not fitted. Skipping advanced analysis.")
            return
            
        coherence = modeler.get_topic_coherence(texts=tokenized_texts)
        topic_dist = modeler.get_topic_distribution()
        weighted_words = modeler.get_topic_word_weights()

        advanced_results.update({
            "coherence_score": coherence,
            "topic_distribution": topic_dist,
            "weighted_words": {str(k): v for k, v in weighted_words.items()},
        })

    elif engine == 'bertopic':
        if not isinstance(modeler.model, BERTopic):
            logger.warning("BERTopic model not fitted. Skipping advanced analysis.")
            return
        
        # --- PARTIE CORRIGÉE ---
        
        # Obtenir la cohérence si possible
        try:
            raw_texts = [doc.get('content', doc.get('text', '')) for doc in articles]
            tokenized_texts_for_coherence = [text.split() for text in raw_texts]
            coherence = modeler.get_bertopic_coherence(texts=tokenized_texts_for_coherence)
            advanced_results["coherence_score"] = coherence
        except Exception as e:
            logger.warning(f"Could not calculate BERTopic coherence: {e}")

        # Obtenir les autres métriques
        topic_dist = modeler.get_bertopic_topic_distribution()
        weighted_words = modeler.get_bertopic_word_weights()
        
        # CORRECTION: BERTopic renvoie déjà le contenu des documents, pas les indices.
        # On récupère donc directement le dictionnaire de {topic_id: [liste de textes]}.
        # Le nom de la variable est changé pour être plus clair.
        rep_docs_content = modeler.get_bertopic_representative_docs()

        # Il faut juste s'assurer que les clés (topic_id) sont des chaînes de caractères pour le JSON
        rep_docs_content_str_keys = {str(k): v for k, v in rep_docs_content.items()}

        advanced_results.update({
            "topic_distribution": topic_dist,
            "weighted_words": weighted_words,
            "representative_docs": rep_docs_content_str_keys, # Utiliser le dictionnaire corrigé
        })
        
    if len(advanced_results) > 2:
        advanced_dir = results_dir / "advanced_analysis"
        advanced_dir.mkdir(parents=True, exist_ok=True)
        file_path = advanced_dir / f'advanced_analysis_{run_id}.json'
        
        # Utiliser un `default=str` pour gérer les types non sérialisables comme les np.float32
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(advanced_results, f, ensure_ascii=False, indent=2, default=str)
        logger.info(f"Saved advanced analysis to {file_path}")

def main():
    """Main execution function."""
    parser = get_parser()
    args = parser.parse_args()
    
    logger.info(f"Executing command: python {' '.join(sys.argv)}")

    # --- 1. Configuration and Paths ---
    config = load_config(str(project_root / 'config' / 'config.yaml'))
    topic_config = config.get('analysis', {}).get('topic_modeling', {})

    topic_config['algorithm'] = args.algorithm if args.algorithm else args.engine
    if topic_config['algorithm'] == 'gensim':
        topic_config['algorithm'] = 'gensim_lda'
    if args.num_topics.lower() == 'auto':
        topic_config['num_topics'] = 'auto'
    else:
        topic_config['num_topics'] = int(args.num_topics)
    topic_config['workers'] = args.workers

    articles_path = Path(args.source_file or project_root / 'data' / 'processed' / 'articles.json')
    results_dir = project_root / 'data' / 'results'
    cache_dir = project_root / 'data' / 'cache'
    
    logger.info(f"Using articles from: {articles_path}")
    logger.info(f"Topic modeling configuration: {topic_config}")

    # --- 2. Load Data ---
    with open(articles_path, 'r', encoding='utf-8') as f:
        articles = json.load(f)
    
    if not articles:
        logger.error("No articles found in source file. Exiting.")
        sys.exit(1)

    # --- 3. Modeling Workflow ---
    modeler = TopicModeler(topic_config)
    preprocessed_data = None
    
    # --- CORRECTION : La logique de cache est maintenant en dehors du 'if/else' de l'engine ---
    is_cache_used = False
    if args.cache_file and Path(args.cache_file).exists():
        logger.info(f"Cache file provided: {args.cache_file}. Loading data from cache.")
        preprocessed_data = load_or_preprocess_data(config, articles, Path(args.cache_file))
        is_cache_used = True
        # Ajout d'un flag pour informer le modeler
        modeler.using_cache = True

    # Si le moteur est Gensim et qu'aucun cache n'a été fourni, on lance le prétraitement
    elif topic_config['algorithm'] in ['gensim_lda', 'hdp'] and not is_cache_used:
        source_hash = hashlib.md5(str(articles_path).encode()).hexdigest()[:8]
        cache_file = cache_dir / f"preprocessed_gensim_{source_hash}.pkl"
        preprocessed_data = load_or_preprocess_data(config, articles, cache_file)

    # Pour BERTopic, si aucun cache n'est utilisé, on ne fait rien ici. 
    # Le prétraitement se fera en interne.
    
    # Lancer l'entraînement
    results = modeler.fit_transform(articles, preprocessed_data=preprocessed_data)

    # ... (le reste de la fonction main est identique)
    # --- 4. Save Core Results ---
    run_id = str(uuid.uuid4())[:8]
    
    top_words_path = results_dir / 'top_words'
    top_words_path.mkdir(parents=True, exist_ok=True)
    top_words_file = top_words_path / f'top_words_{run_id}.json'
    with open(top_words_file, 'w', encoding='utf-8') as f:
        json.dump({"run_id": run_id, "config": topic_config, "top_words_per_topic": results['top_terms']}, f, ensure_ascii=False, indent=2)
    logger.info(f"Top words saved to {top_words_file}")

    doc_matrix_path = results_dir / 'doc_topic_matrix'
    doc_matrix_path.mkdir(parents=True, exist_ok=True)
    doc_matrix_file = doc_matrix_path / f'doc_topic_matrix_{run_id}.json'
    with open(doc_matrix_file, 'w', encoding='utf-8') as f:
        json.dump({"run_id": run_id, "config": topic_config, "doc_topics": results['doc_topics']}, f, ensure_ascii=False, indent=2)
    logger.info(f"Document-topic matrix saved to {doc_matrix_file}")

    models_dir = project_root / 'data' / 'models'
    modeler.save_model(str(models_dir), prefix=f"model_{run_id}")

    # --- 5. Perform and Save Advanced Analysis ---
    perform_and_save_advanced_analysis(
        modeler=modeler,
        run_id=run_id,
        config=topic_config,
        results_dir=results_dir,
        tokenized_texts=preprocessed_data['tokenized_texts'] if preprocessed_data else None,
        articles=articles
    )

    logger.info(f"Topic modeling run {run_id} completed successfully.")
if __name__ == "__main__":
    main()