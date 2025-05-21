import json
import os
import argparse
from pathlib import Path
import numpy as np
import logging
import sys
import uuid
from gensim.models import LdaMulticore, CoherenceModel
import time
try:
    import psutil
except ImportError:
    psutil = None

# Add the project root to the path to allow imports from src
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from src.analysis.topic_modeling import TopicModeler
from src.utils.config_loader import load_config
from src.analysis.utils import get_french_stopwords, get_stopwords
from src.preprocessing import SpacyPreprocessor, preprocess_with_spacy
from src.utils.filter_utils import apply_all_filters, get_filter_summary

def find_best_num_topics_bisect(corpus, id2word, texts, k_min=5, k_max=20, tol=1, logger=None):
    results = {}
    # Évaluer les extrêmes
    for k in [k_min, k_max]:
        model_k = LdaMulticore(corpus=corpus, id2word=id2word, num_topics=k, workers=2, random_state=42)
        cm = CoherenceModel(model=model_k, texts=texts, dictionary=id2word, coherence='c_v')
        results[k] = cm.get_coherence()
        if logger:
            logger.info(f"num_topics={k}, coherence={results[k]:.4f}")
        else:
            print(f"num_topics={k}, coherence={results[k]:.4f}")
    # Recherche dichotomique
    while k_max - k_min > tol:
        k_mid = (k_min + k_max) // 2
        if k_mid in results:
            break
        model_k = LdaMulticore(corpus=corpus, id2word=id2word, num_topics=k_mid, workers=2, random_state=42)
        cm = CoherenceModel(model=model_k, texts=texts, dictionary=id2word, coherence='c_v')
        results[k_mid] = cm.get_coherence()
        if logger:
            logger.info(f"num_topics={k_mid}, coherence={results[k_mid]:.4f}")
        else:
            print(f"num_topics={k_mid}, coherence={results[k_mid]:.4f}")
        # Choisir la moitié la plus prometteuse
        vals = [(kk, results[kk]) for kk in sorted(results.keys())]
        best_k, best_score = max(vals, key=lambda x: x[1])
        if k_mid == best_k:
            # Explorer autour du maximum local
            if abs(k_mid - k_min) > abs(k_max - k_mid):
                k_max = k_mid
            else:
                k_min = k_mid
        elif k_mid < best_k:
            k_min = k_mid
        else:
            k_max = k_mid
    best_k = max(results.items(), key=lambda x: x[1])
    if logger:
        logger.info(f"Best num_topics: {best_k[0]} with coherence: {best_k[1]:.4f}")
    else:
        print(f"Best num_topics: {best_k[0]} with coherence: {best_k[1]:.4f}")
    return best_k[0], results

def save_coherence_stats(run_id, coherences, coherence_dir, start_time, end_time, cpu_times):
    stats = {
        "run_id": run_id,
        "coherence_trials": [
            {"num_topics": int(k), "coherence": float(score)} for k, score in sorted(coherences.items())
        ],
        "start_time": start_time,
        "end_time": end_time,
        "duration_sec": end_time - start_time,
        "cpu_times": cpu_times
    }
    path = coherence_dir / f"coherence_trials_{run_id}.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    return path

def get_parser():
    parser = argparse.ArgumentParser(description="Run topic modeling on articles.")
    parser.add_argument('--versioned', action='store_true', help='Save results with unique versioned filenames (default: True)')
    parser.add_argument('--no-versioned', dest='versioned', action='store_false', help='Save results only with generic names (overwrites previous)')
    parser.set_defaults(versioned=True)
    parser.add_argument('--engine', choices=['sklearn', 'gensim', 'bertopic'], default='bertopic', help='Choose topic modeling engine: sklearn, gensim, or bertopic (default: gensim)')
    parser.add_argument('--algorithm', type=str, default=None, help='Topic modeling algorithm (e.g., lda, hdp, nmf). Overrides config file.')
    parser.add_argument('--auto-num-topics', action='store_true', help='Automatically find the best num_topics (LDA only, Gensim)')
    parser.add_argument('--num-topics', 
        type=str,
        default=None, 
        help='Nombre de topics à utiliser. Utiliser "auto" pour BERTopic automatique.')
    parser.add_argument('--k-min', type=int, default=5, help='Min num_topics for search (default: 5)')
    parser.add_argument('--k-max', type=int, default=20, help='Max num_topics for search (default: 20)')
    parser.add_argument('--search-mode', choices=['linear', 'bisect'], default='linear', help='Mode de recherche du meilleur num_topics (linear ou bisect, défaut: linear)')
    parser.add_argument('--bisect-tol', type=int, default=1, help='Tolérance (écart min) pour arrêt dichotomique (défaut: 1)')
    parser.add_argument('--llm-topic-names', action='store_true', help='Générer automatiquement les noms de topics via LLM (cf. config llm)')
    parser.add_argument('--workers', type=int, default=2, help='Nombre de workers CPU à utiliser pour la modélisation (défaut: 2)')
    
    # Les options de filtrage et l'option input-file ont été supprimées
    
    return parser

def main():
    # Parse arguments for versioning
    parser = get_parser()
    args = parser.parse_args()
    # Set up logging to always print to stdout for Dash
    import sys
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    handler.setFormatter(formatter)
    logger = logging.getLogger()
    logger.handlers = [handler]
    logger.setLevel(logging.INFO)

    # Paths
    config_path = project_root / 'config' / 'config.yaml'
    articles_path = project_root / 'data' / 'processed' / 'articles.json'
    results_dir = project_root / 'data' / 'results'
    os.makedirs(results_dir, exist_ok=True)
    
    # Vérifier si une variable d'environnement spécifie un fichier source personnalisé
    custom_source = os.environ.get('TOPIC_MODELING_SOURCE_FILE')
    if custom_source and os.path.exists(custom_source):
        articles_path = Path(custom_source)
        logger.info(f"Utilisation du fichier d'articles personnalisé (via variable d'environnement): {articles_path}")
    else:
        logger.info(f"Utilisation du fichier d'articles par défaut: {articles_path}")
    
    # Dossier pour sauvegarder les modèles et le cache
    models_dir = project_root / 'data' / 'models'
    cache_dir = project_root / 'data' / 'cache'
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(cache_dir, exist_ok=True)

    # Ensure result subdirectories exist
    advanced_dir = results_dir / "advanced_topic"
    topwords_dir = results_dir / "top_words"
    coherence_dir = results_dir / "coherence_trials"
    for d in [advanced_dir, topwords_dir, coherence_dir]:
        d.mkdir(parents=True, exist_ok=True)

    # Load config
    config = load_config(str(config_path))
    topic_config = config.get('analysis', {}).get('topic_modeling', {})
    
    # Override config with command-line arguments if provided
    if args.algorithm:
        topic_config['algorithm'] = args.algorithm
    elif args.engine == 'bertopic':
        topic_config['algorithm'] = 'bertopic'
    
    # Explicitly override num_topics if provided via command line
    if args.num_topics is not None and not args.auto_num_topics:
        if args.engine == "bertopic" and str(args.num_topics).lower() == "auto":
            topic_config['num_topics'] = "auto"
            logger.info("Nombre de topics en mode automatique (BERTopic)")
        else:
            try:
                topic_config['num_topics'] = int(args.num_topics)
                logger.info(f"Setting num_topics to {args.num_topics} (from command line)")
            except ValueError:
                logger.warning(f"Invalid num_topics value: {args.num_topics}. Using config value: {topic_config.get('num_topics', 'default')}")
    
    logger.info(f"Loaded topic modeling config: {topic_config}")

    # Load articles
    with open(articles_path, 'r', encoding='utf-8') as f:
        articles = json.load(f)
    logger.info(f"Loaded {len(articles)} articles from {articles_path}")
    
    # Les filtres ont été supprimés
    filtered_articles = articles
    logger.info(f"Utilisation de tous les articles ({len(articles)})")

    # Check if we have enough articles after filtering
    if len(filtered_articles) < 10:
        logger.warning(f"Only {len(filtered_articles)} articles remain after filtering. This may be too few for meaningful topic modeling.")
        if len(filtered_articles) == 0:
            logger.error("No articles remain after filtering. Exiting.")
            sys.exit(1)
    
    # Update articles to use filtered set
    articles = filtered_articles
    logger.info(f"Using {len(articles)} articles after applying filters")

    # Ajouter le paramètre workers à la configuration
    topic_config['workers'] = args.workers
    logger.info(f"Utilisation de {args.workers} workers CPU pour la modélisation")
    
    # Initialize topic modeler
    modeler = TopicModeler(topic_config)

    # Preprocess articles for topic modeling
    doc_ids = [doc.get('id', f"doc_{i}") for i, doc in enumerate(articles)]
    
    # Add preprocessing configuration to topic_config if not present
    if 'preprocessing' not in topic_config:
        topic_config['preprocessing'] = {
            'spacy_model': 'fr_core_news_md',
            'allowed_pos': ["NOUN", "PROPN", "ADJ"],
            'min_token_length': 3
        }
    
    # Initialize SpaCy preprocessor for tokenization
    spacy_preprocessor = SpacyPreprocessor(topic_config.get('preprocessing', {}))
    
    # Check for cache and use it if available
    import pickle
    import hashlib
    
    # Create a cache key based on preprocessing parameters and input file
    preproc_config = topic_config.get('preprocessing', {})
    cache_key_data = {
        'articles_path': str(articles_path),
        'spacy_model': preproc_config.get('spacy_model', 'fr_core_news_md'),
        'allowed_pos': preproc_config.get('allowed_pos', ["NOUN", "PROPN", "ADJ"]),
        'min_token_length': preproc_config.get('min_token_length', 3),
        'articles_count': len(articles),
        'articles_last_modified': os.path.getmtime(articles_path)
    }
    
    # Ajouter des paramètres optionnels s'ils existent
    for param in ['min_doc_length', 'min_word_length', 'max_word_length']:
        if hasattr(args, param):
            cache_key_data[param] = getattr(args, param)
    
    # Create a hash of the cache key
    cache_key = hashlib.md5(json.dumps(cache_key_data, sort_keys=True).encode()).hexdigest()
    cache_file = cache_dir / f"preprocessed_docs_{cache_key}.pkl"
    
    # Extract text from articles and preprocess
    texts = []
    tokenized_texts = []
    
    # Vérifier si un fichier de cache spécifique a été sélectionné via l'interface
    cache_config_path = project_root / "config" / "cache_config.json"
    selected_cache_file = None
    
    if cache_config_path.exists():
        try:
            with open(cache_config_path, 'r', encoding='utf-8') as f:
                cache_config = json.load(f)
                selected_cache = cache_config.get("selected_cache")
                if selected_cache:
                    selected_cache_file = cache_dir / selected_cache
                    logger.info(f"Fichier de cache spécifique sélectionné via l'interface: {selected_cache_file}")
        except Exception as e:
            logger.warning(f"Erreur lors de la lecture du fichier de configuration du cache: {e}")
    
    # Essayer de charger depuis le cache (toujours tenter d'utiliser le cache s'il existe)
    cache_loaded = False
    cache_to_use = selected_cache_file if selected_cache_file and selected_cache_file.exists() else cache_file
    
    if cache_to_use.exists():
        try:
            logger.info(f"Chargement des documents prétraités depuis le cache: {cache_to_use}")
            with open(cache_to_use, 'rb') as f:
                cache_data = pickle.load(f)
                texts = cache_data['texts']
                tokenized_texts = cache_data['tokenized_texts']
                logger.info(f"{len(texts)} documents chargés avec succès depuis le cache")
                cache_loaded = True
        except Exception as e:
            logger.warning(f"Impossible de charger le cache: {e}")
            cache_loaded = False
    
    # Preprocess if cache not loaded
    if not cache_loaded:
        logger.info("Preprocessing documents with spaCy (this may take a while)...")
        for doc in articles:
            if 'cleaned_text' in doc:
                text = doc['cleaned_text']
            elif 'content' in doc:
                text = doc['content']
            elif 'text' in doc:
                text = doc['text']
            else:
                text = ""
                logger.warning(f"No text content found for document {doc.get('id', 'unknown')}")
            
            texts.append(text)
            # Tokenize with SpaCy for better topic modeling
            tokenized_texts.append(spacy_preprocessor.preprocess_text(text))
        
        logger.info(f"Preprocessed {len(texts)} articles with SpaCy")
        
        # Save to cache
        try:
            logger.info(f"Saving preprocessed documents to cache: {cache_file}")
            with open(cache_file, 'wb') as f:
                pickle.dump({
                    'texts': texts,
                    'tokenized_texts': tokenized_texts,
                    'cache_key_data': cache_key_data
                }, f)
            logger.info("Cache saved successfully")
        except Exception as e:
            logger.warning(f"Failed to save cache: {e}")
    
    logger.info(f"Sample tokens from first document: {tokenized_texts[0][:10] if tokenized_texts else []}")

    # If requested, automatically find the best num_topics using coherence (LDA + Gensim engine only)
    if args.auto_num_topics and args.engine == 'gensim' and topic_config.get('algorithm', 'lda') == 'lda':
        logger.info(f"Searching best num_topics (coherence) in [{args.k_min}, {args.k_max}] with mode {args.search_mode}...")
        # Use the SpaCy tokenized texts directly for Gensim
        from gensim.corpora import Dictionary
        id2word = Dictionary(tokenized_texts)
        corpus = [id2word.doc2bow(text) for text in tokenized_texts]
        run_id = str(uuid.uuid4())
        start_time = time.time()
        cpu_times_before = psutil.cpu_times_percent() if psutil else None
        if args.search_mode == 'bisect':
            best_k, coherence_dict = find_best_num_topics_bisect(corpus, id2word, tokenized_texts, k_min=args.k_min, k_max=args.k_max, tol=args.bisect_tol, logger=logger)
        else:
            coherence_dict = {}
            for k in range(args.k_min, args.k_max + 1):
                model_k = LdaMulticore(corpus=corpus, id2word=id2word, num_topics=k, workers=args.workers, random_state=42)
                cm = CoherenceModel(model=model_k, texts=tokenized_texts, dictionary=id2word, coherence='c_v')
                score = cm.get_coherence()
                coherence_dict[k] = score
                logger.info(f"num_topics={k}, coherence={score:.4f}")
            best_k = max(coherence_dict.items(), key=lambda x: x[1])[0]
            logger.info(f"Best num_topics: {best_k} with coherence: {coherence_dict[best_k]:.4f}")
        end_time = time.time()
        cpu_times_after = psutil.cpu_times_percent() if psutil else None
        cpu_times = {
            "before": cpu_times_before._asdict() if psutil and cpu_times_before else None,
            "after": cpu_times_after._asdict() if psutil and cpu_times_after else None
        }
        # Set the best number of topics in the config AND override any user-provided num_topics
        topic_config['num_topics'] = best_k
        # If num_topics was explicitly provided, log that we're overriding it
        if args.num_topics is not None:
            logger.info(f"Auto-search found best num_topics={best_k}. Overriding user-provided num_topics={args.num_topics}")
        else:
            logger.info(f"Auto-search found best num_topics={best_k}")
        
        # Save all stats for this run
        stats_path = save_coherence_stats(run_id, coherence_dict, coherence_dir, start_time, end_time, cpu_times)
        logger.info(f"Saved coherence trials to {stats_path}")
    elif args.num_topics is not None:
        if args.engine == "bertopic" and str(args.num_topics).lower() == "auto":
            topic_config['num_topics'] = "auto"
            logger.info("Nombre de topics en mode automatique (BERTopic)")
        else:
            topic_config['num_topics'] = int(args.num_topics)
            logger.info(f"Nombre de topics fixé à {args.num_topics} (pas de recherche du meilleur k)")

    # Fit and transform using selected engine
    if args.engine == 'sklearn':
        doc_topic_matrix = modeler.fit_transform_sklearn(texts)
        feature_names = modeler.feature_names
        model_components = modeler.model.components_
    elif args.engine == 'bertopic':
        # Chemin pour sauvegarder/charger le modèle
        model_dir = models_dir / 'bertopic'
        os.makedirs(model_dir, exist_ok=True)
        model_path = model_dir / f"topic_model_bertopic.pkl"
        
        # Vérifier si un modèle existe déjà
        if os.path.exists(model_path):
            try:
                logger.info(f"[BERTopic] Tentative de chargement du modèle depuis {model_path}")
                loaded_modeler = TopicModeler.load_model(str(model_path), algorithm='bertopic')
                logger.info(f"[BERTopic] Modèle chargé avec succès depuis {model_path}")
                # Nouvelle logique : lire la config d'entraînement du modèle
                trained_num_topics = None
                if hasattr(loaded_modeler, 'training_config') and loaded_modeler.training_config:
                    trained_num_topics = str(loaded_modeler.training_config.get('num_topics', 'auto'))
                current_num_topics = str(topic_config['num_topics'])
                # Si les deux sont 'auto', pas de réentraînement
                if trained_num_topics == "auto" and current_num_topics == "auto":
                    logger.info("[BERTopic] Modèle existant et config tous deux en 'auto' (vérifié via config d'entraînement) : pas de réentraînement.")
                    modeler = loaded_modeler
                    results = modeler.transform_with_bertopic(texts)
                elif trained_num_topics == current_num_topics:
                    modeler = loaded_modeler
                    results = modeler.transform_with_bertopic(texts)
                else:
                    logger.info(f"[BERTopic] Le modèle existant a été entraîné avec num_topics={trained_num_topics}, mais {current_num_topics} sont demandés. Réentraînement...")
                    results = modeler.fit_transform(articles)
                    logger.info(f"[BERTopic] Sauvegarde immédiate du modèle dans {model_path}")
                    modeler.save_model(str(model_dir), prefix='topic_model')
            except Exception as e:
                logger.warning(f"[BERTopic] Erreur lors du chargement du modèle : {e}")
                results = modeler.fit_transform(articles)
                logger.info(f"[BERTopic] Sauvegarde immédiate du modèle dans {model_path}")
                modeler.save_model(str(model_dir), prefix='topic_model')
        else:
            logger.info("[BERTopic] Aucun modèle existant, entraînement d'un nouveau modèle")
            results = modeler.fit_transform(articles)
            logger.info(f"[BERTopic] Sauvegarde immédiate du modèle dans {model_path}")
            modeler.save_model(str(model_dir), prefix='topic_model')
        
        # Extraire les résultats
        # Vérifier la structure des résultats pour éviter KeyError
        logger.info(f"DEBUG: Structure des résultats: {list(results.keys())}")
        
        if 'doc_topics' in results and results['doc_topics']:
            # Vérifier la structure du premier document pour déterminer la clé correcte
            first_doc = next(iter(results['doc_topics'].values()))
            logger.info(f"DEBUG: Structure du premier document: {list(first_doc.keys())}")
            logger.info(f"DEBUG: Premier document ID: {next(iter(results['doc_topics'].keys()))}")
            
            # Log the topic assignments from BERTopic directly
            if hasattr(modeler, 'bertopic_topics') and modeler.bertopic_topics is not None:
                # Count topic assignments
                topic_counts = {}
                for topic in modeler.bertopic_topics:
                    topic_counts[topic] = topic_counts.get(topic, 0) + 1
                logger.info(f"DEBUG: Topic assignments from BERTopic.bertopic_topics: {topic_counts}")
                
                # Check for outliers
                outlier_count = topic_counts.get(-1, 0)
                logger.info(f"DEBUG: Outlier count from BERTopic: {outlier_count} ({outlier_count/len(modeler.bertopic_topics)*100:.2f}%)")
            
            if 'topic_distribution' in first_doc:
                logger.info("DEBUG: Using 'topic_distribution' key from results")
                doc_topic_matrix = [doc['topic_distribution'] for doc in results['doc_topics'].values()]
                
                # Log information about the document-topic matrix
                if doc_topic_matrix:
                    logger.info(f"DEBUG: doc_topic_matrix length: {len(doc_topic_matrix)}")
                    logger.info(f"DEBUG: First document topic distribution length: {len(doc_topic_matrix[0])}")
                    logger.info(f"DEBUG: First document topic distribution: {doc_topic_matrix[0]}")
                    
                    # Count documents by dominant topic
                    dominant_topics = {}
                    for dist in doc_topic_matrix:
                        # Convert to numpy array if it's not already
                        dist_array = np.array(dist) if not isinstance(dist, np.ndarray) else dist
                        dominant_topic = int(np.argmax(dist_array))
                        dominant_topics[dominant_topic] = dominant_topics.get(dominant_topic, 0) + 1
                    logger.info(f"DEBUG: Dominant topics in doc_topic_matrix: {dominant_topics}")
            elif 'topic_probs' in first_doc:  # Format utilisé par transform_with_bertopic
                logger.info("Utilisation du format transform_with_bertopic (topic_probs)")
                doc_topic_matrix = [doc['topic_probs'] for doc in results['doc_topics'].values()]
                
                # Log information about the document-topic matrix
                if doc_topic_matrix:
                    logger.info(f"DEBUG: doc_topic_matrix length: {len(doc_topic_matrix)}")
                    logger.info(f"DEBUG: First document topic distribution length: {len(doc_topic_matrix[0])}")
                    logger.info(f"DEBUG: First document topic distribution: {doc_topic_matrix[0]}")
            else:
                # Utiliser le contenu directement si aucune clé attendue n'existe
                logger.warning("Aucune clé de distribution de topics trouvée, utilisation des valeurs directement")
                doc_topic_matrix = list(results['doc_topics'].values())
        else:
            logger.error("La clé 'doc_topics' n'existe pas dans les résultats ou est vide")
            logger.info(f"DEBUG: Available keys in results: {list(results.keys())}")
            doc_topic_matrix = []
        
        feature_names = None  # Not used for BERTopic
        
        # Gérer les différentes structures de résultats entre fit_transform et transform_with_bertopic
        if 'top_terms' in results:
            model_components = results['top_terms']
        elif 'topic_words' in results:  # Format utilisé par transform_with_bertopic
            logger.info("Utilisation du format transform_with_bertopic (topic_words)")
            # Convertir topic_words au format attendu par le reste du code
            model_components = {}
            for topic_id, topic_data in results['topic_words'].items():
                try:
                    if isinstance(topic_id, int):
                        model_components[f'topic_{topic_id}'] = [(word, weight) for word, weight in zip(topic_data['words'], topic_data['weights'])]
                    else:
                        model_components[topic_id] = [(word, weight) for word, weight in zip(topic_data['words'], topic_data['weights'])]
                except Exception as e:
                    logger.warning(f"Erreur lors de la récupération des mots pour le topic {topic_id}: {e}")
                    # Provide a fallback with empty lists if there's an error
                    if isinstance(topic_id, int):
                        model_components[f'topic_{topic_id}'] = []
                    else:
                        model_components[topic_id] = []
        else:
            logger.error("Ni 'top_terms' ni 'topic_words' n'existent dans les résultats")
            model_components = {}
    else:  # gensim
        # Pass the tokenized texts directly to the Gensim engine
        doc_topic_matrix = modeler.fit_transform_gensim(tokenized_texts)
        feature_names = modeler.feature_names
        model_components = modeler.model.get_topics()

    # Generate a unique run ID (without date/time, just a UUID)
    run_id = str(uuid.uuid4())
    logger.info(f"Run ID for this result set: {run_id}")

    # Print and save top words per topic
    num_top_words = 10
    top_words_per_topic = {}
    if args.engine == 'sklearn':
        for topic_idx, topic in enumerate(model_components):
            top_features = [feature_names[i] for i in topic.argsort()[:-num_top_words - 1:-1]]
            logger.info(f"Topic #{topic_idx}: {' '.join(top_features)}")
            top_words_per_topic[f"topic_{topic_idx}"] = top_features
    elif args.engine == 'bertopic':
        for topic_idx, top_features in model_components.items():
            # Vérifier si top_features contient des tuples (word, weight) ou juste des mots
            if top_features and isinstance(top_features[0], tuple):
                # Extraire seulement les mots des tuples (word, weight)
                words_only = [word for word, weight in top_features]
                logger.info(f"Topic #{topic_idx}: {' '.join(words_only)}")
                top_words_per_topic[topic_idx] = words_only
            else:
                # Utiliser directement la liste de mots
                logger.info(f"Topic #{topic_idx}: {' '.join(top_features)}")
                top_words_per_topic[topic_idx] = top_features
    else:  # gensim
        for topic_idx, topic in enumerate(model_components):
            top_features = [word for word, _ in modeler.model.show_topic(topic_idx, num_top_words)]
            logger.info(f"Topic #{topic_idx}: {' '.join(top_features)}")
            top_words_per_topic[f"topic_{topic_idx}"] = top_features

    # Include run_id in the results
    results_summary = {
        "run_id": run_id,
        "top_words_per_topic": top_words_per_topic
    }

    # Génération automatique des noms de topics via LLM si demandé (appel direct, sans llm_utils)
    if args.llm_topic_names:
        logger.info("Génération des noms de topics via LLM (appel direct)...")
        llm_config = config.get('llm', {})
        top_words_lists = list(top_words_per_topic.values())
        topic_names = modeler.get_topic_names_llm_direct(top_words_lists, llm_config=llm_config)
        results_summary['topic_names_llm'] = {f"topic_{i}": name for i, name in enumerate(topic_names)}
        # Ajoute le nom du modèle LLM utilisé
        llm_name = llm_config.get('model', None)
        results_summary['llm_name'] = llm_name
        # Propagation explicite pour analyses avancées
        modeler.llm_name_used = llm_name
        modeler.topic_names_llm = {f"topic_{i}": name for i, name in enumerate(topic_names)}
        logger.info(f"Noms de topics générés par LLM : {results_summary['topic_names_llm']}")
        # Sauvegarde avec les noms de topics
        top_words_path = topwords_dir / f'top_words_per_topic_{run_id}.json' if args.versioned else topwords_dir / 'top_words_per_topic.json'
        with open(top_words_path, 'w', encoding='utf-8') as f:
            json.dump(results_summary, f, ensure_ascii=False, indent=2)
        logger.info(f"Top words + noms LLM sauvegardés dans {top_words_path}")
    else:
        top_words_path = topwords_dir / f'top_words_per_topic_{run_id}.json' if args.versioned else topwords_dir / 'top_words_per_topic.json'
        with open(top_words_path, 'w', encoding='utf-8') as f:
            json.dump(results_summary, f, ensure_ascii=False, indent=2)
        logger.info(f"Saved top words per topic to {top_words_path}")

    # Also save/update latest version for convenience
    latest_top_words_path = topwords_dir / 'top_words_per_topic.json'
    with open(latest_top_words_path, 'w', encoding='utf-8') as f:
        json.dump(results_summary, f, ensure_ascii=False, indent=2)
    logger.info(f"Updated latest top words per topic at {latest_top_words_path}")

    # Save document-topic matrix with doc ids and run_id
    # Add debug logging to understand the structure of the doc_topic_matrix
    logger.info(f"DEBUG: doc_topic_matrix type: {type(doc_topic_matrix)}")
    logger.info(f"DEBUG: doc_topic_matrix length: {len(doc_topic_matrix)}")
    if doc_topic_matrix and len(doc_topic_matrix) > 0:
        logger.info(f"DEBUG: First document topic distribution type: {type(doc_topic_matrix[0])}")
        logger.info(f"DEBUG: First document topic distribution length: {len(doc_topic_matrix[0])}")
        logger.info(f"DEBUG: First document topic distribution values: {doc_topic_matrix[0]}")
    
    # For BERTopic, check if we need to add an outlier topic column
    if args.engine == 'bertopic':
        # Get the actual number of topics from the model
        if hasattr(modeler, 'model') and modeler.model is not None:
            topic_info = modeler.model.get_topic_info()
            logger.info(f"DEBUG: BERTopic topic_info: {topic_info.head()}")
            actual_num_topics = len(topic_info[topic_info['Topic'] != -1])
            logger.info(f"DEBUG: BERTopic actual number of topics (excluding outliers): {actual_num_topics}")
            
            # Check if outlier topic exists
            has_outlier_topic = -1 in topic_info['Topic'].values
            logger.info(f"DEBUG: BERTopic has outlier topic (-1): {has_outlier_topic}")
            
            # Check if we need to add an outlier topic column
            if has_outlier_topic and doc_topic_matrix and len(doc_topic_matrix) > 0:
                expected_columns = actual_num_topics + 1  # +1 for outlier topic
                actual_columns = len(doc_topic_matrix[0])
                logger.info(f"DEBUG: Expected columns (including outlier): {expected_columns}, Actual columns: {actual_columns}")
                
                # If the outlier topic is missing, we should add it
                if actual_columns < expected_columns:
                    logger.warning(f"DEBUG: Outlier topic is missing from doc_topic_matrix. Adding zero column for outlier topic.")
                    # Add the outlier topic column (with zeros) to each document's topic distribution
                    doc_topic_matrix = [list(dist) + [0.0] for dist in doc_topic_matrix]
                    logger.info(f"DEBUG: Added outlier topic column. New distribution length: {len(doc_topic_matrix[0])}")
                    logger.info(f"DEBUG: Updated first document topic distribution: {doc_topic_matrix[0]}")
                    
                    # Update the number of topics in the model configuration
                    if isinstance(topic_config.get('num_topics'), int):
                        topic_config['num_topics'] += 1
                        logger.info(f"DEBUG: Updated num_topics in config to {topic_config['num_topics']}")
                    
                    # Log the updated topic count
                    logger.info(f"DEBUG: Total topics after adding outlier: {len(doc_topic_matrix[0])}")
                    
                    # Update the topic info to include the outlier topic
                    topic_counts = modeler.get_bertopic_article_counts()
                    for topic_id, count in topic_counts.items():
                        logger.info(f"Topic #{topic_id}: {count} articles ({(count/len(doc_ids))*100:.2f}%)")
                    
                    # Make sure the outlier topic is included in the top words
                    if 'topic_5' not in top_words_per_topic and 5 not in top_words_per_topic:
                        # Add a placeholder for the outlier topic
                        top_words_per_topic['topic_5'] = ['outlier', 'divers', 'inclassable']
                        logger.info(f"Topic #topic_5: outlier divers inclassable")
    
    # Check if there's a mismatch between the topic distribution in the logs and the doc_topic_matrix
    # This is where we need to fix the issue with outlier documents being assigned to topic 0
    if args.engine == 'bertopic' and hasattr(modeler, 'bertopic_topics'):
        # Count the number of documents assigned to each topic by BERTopic
        bertopic_counts = {}
        for topic in modeler.bertopic_topics:
            bertopic_counts[topic] = bertopic_counts.get(topic, 0) + 1
        
        # Count the number of documents assigned to each topic in the doc_topic_matrix
        matrix_counts = {}
        for dist in doc_topic_matrix:
            # Convert to numpy array if it's not already
            dist_array = np.array(dist) if not isinstance(dist, np.ndarray) else dist
            dominant_topic = int(np.argmax(dist_array))
            matrix_counts[dominant_topic] = matrix_counts.get(dominant_topic, 0) + 1
        
        # Log the comparison
        logger.info(f"DEBUG: Topic counts from BERTopic.bertopic_topics: {bertopic_counts}")
        logger.info(f"DEBUG: Topic counts from doc_topic_matrix: {matrix_counts}")
        
        # Check if there's a major discrepancy
        if -1 in bertopic_counts and bertopic_counts[-1] > 0:
            outlier_count = bertopic_counts[-1]
            logger.info(f"DEBUG: Found {outlier_count} outlier documents in BERTopic.bertopic_topics")
            
            # Check if these outliers are being incorrectly assigned to topic 0 in the matrix
            if 0 in matrix_counts and matrix_counts[0] > (bertopic_counts.get(0, 0) + outlier_count/2):
                logger.warning(f"DEBUG: Detected that outlier documents are being incorrectly assigned to topic 0 in the doc_topic_matrix")
                
                # Create a new doc_topic_matrix with correct outlier assignment
                logger.info(f"DEBUG: Creating a new doc_topic_matrix with correct outlier assignment")
                new_doc_topic_matrix = []
                
                for i, (topic_idx, dist) in enumerate(zip(modeler.bertopic_topics, doc_topic_matrix)):
                    if topic_idx == -1:  # This is an outlier document
                        # Create a distribution with high probability for the outlier topic
                        new_dist = [0.0] * len(dist)
                        if len(new_dist) > actual_num_topics:  # If we have an outlier column
                            new_dist[actual_num_topics] = 1.0  # Set high probability for outlier topic
                        new_doc_topic_matrix.append(new_dist)
                    else:
                        # For non-outlier documents, ensure the dominant topic matches the BERTopic assignment
                        # First, create a copy of the original distribution
                        new_dist = list(dist)
                        
                        # Find the maximum probability in the current distribution
                        max_prob = max(new_dist)
                        
                        # Set all probabilities to a small value
                        new_dist = [0.001] * len(new_dist)
                        
                        # Set a high probability for the assigned topic
                        new_dist[topic_idx] = max(max_prob, 0.9)  # Use at least 0.9 to ensure it's dominant
                        
                        new_doc_topic_matrix.append(new_dist)
                
                # Replace the doc_topic_matrix with the corrected one
                doc_topic_matrix = new_doc_topic_matrix
                logger.info(f"DEBUG: Corrected doc_topic_matrix. First document: {doc_topic_matrix[0]}")
                
                # Count the number of documents assigned to each topic in the corrected matrix
                corrected_counts = {}
                for dist in doc_topic_matrix:
                    # Convert to numpy array if it's not already
                    dist_array = np.array(dist) if not isinstance(dist, np.ndarray) else dist
                    dominant_topic = int(np.argmax(dist_array))
                    corrected_counts[dominant_topic] = corrected_counts.get(dominant_topic, 0) + 1
                logger.info(f"DEBUG: Topic counts after correction: {corrected_counts}")
    
    # Create the output structure
    doc_topic_output = [
        {"doc_id": doc_id, "topic_distribution": [float(x) for x in topic_dist]}
        for doc_id, topic_dist in zip(doc_ids, doc_topic_matrix)
    ]
    
    # Log the first few documents for debugging
    logger.info(f"DEBUG: First document in doc_topic_output: {doc_topic_output[0]}")
    
    # Create the final summary
    doc_topic_matrix_summary = {
        "run_id": run_id,
        "doc_topic_matrix": doc_topic_output
    }
    doc_topic_matrix_path = results_dir / f'doc_topic_matrix_{run_id}.json' if args.versioned else results_dir / 'doc_topic_matrix.json'
    with open(doc_topic_matrix_path, 'w', encoding='utf-8') as f:
        json.dump(doc_topic_matrix_summary, f, ensure_ascii=False, indent=2)
    logger.info(f"Saved document-topic matrix to {doc_topic_matrix_path}")

    # Also save/update latest version for convenience
    latest_doc_topic_path = results_dir / 'doc_topic_matrix.json'
    with open(latest_doc_topic_path, 'w', encoding='utf-8') as f:
        json.dump(doc_topic_matrix_summary, f, ensure_ascii=False, indent=2)
    logger.info(f"Updated latest document-topic matrix at {latest_doc_topic_path}")

    # --- Analyse avancée Gensim : pondération, cohérence, distribution, docs représentatifs ---
    if args.engine == 'gensim':
        # 1. Mots-clés pondérés par sujet
        weighted_words = modeler.get_topic_word_weights(n_terms=num_top_words)
        logger.info("\n--- Mots-clés pondérés par sujet ---")
        for topic_idx, words in weighted_words.items():
            logger.info(f"Topic #{topic_idx}: {[(w, f'{wgt:.3f}') for w, wgt in words]}")

        # 2. Score de cohérence
        if hasattr(modeler, 'tokenized_texts'):
            coherence = modeler.get_topic_coherence(texts=modeler.tokenized_texts)
        else:
            coherence = modeler.get_topic_coherence(texts=[t.split() for t in texts])
        logger.info(f"Coherence score (c_v): {coherence:.3f}")

        # 3. Distribution globale des sujets
        topic_dist = modeler.get_topic_distribution()
        logger.info("\n--- Distribution globale des sujets (importance relative) ---")
        for topic_idx, frac in enumerate(topic_dist):
            logger.info(f"Topic #{topic_idx}: {frac:.2%}")

        # 4. Documents les plus représentatifs par sujet
        try:
            # Vérifier quelle méthode est disponible selon l'algorithme utilisé
            if hasattr(modeler, 'get_bertopic_representative_docs'):
                rep_docs = modeler.get_bertopic_representative_docs(n_docs=3)
            elif hasattr(modeler, 'get_representative_documents'):
                rep_docs = modeler.get_representative_documents(n_docs=3)
            else:
                # Utiliser une approche générique si aucune méthode spécifique n'est disponible
                logger.warning("Méthode pour obtenir les documents représentatifs non disponible pour cet algorithme")
                rep_docs = {}
                
            if rep_docs:
                logger.info("\n--- Documents représentatifs par sujet (indices) ---")
                for topic_idx, doc_indices in rep_docs.items():
                    logger.info(f"Topic #{topic_idx}: {doc_indices}")
        except Exception as e:
            logger.warning(f"Erreur lors de la récupération des documents représentatifs: {e}")
            rep_docs = {}

        # 5. Sauvegarde des scores et analyses avancées dans results
        # Utilise le nom du LLM effectivement utilisé pour nommer les topics
        llm_name = getattr(modeler, 'llm_name_used', None)
        topic_names_llm = getattr(modeler, 'topic_names_llm', None)
        advanced_results = {
            "run_id": run_id,
            "coherence_score": coherence,
            "topic_distribution": topic_dist,
            "weighted_words": {str(k): [(w, float(f"{wgt:.5f}")) for w, wgt in v] for k, v in weighted_words.items()},
            "representative_docs": {str(k): v for k, v in rep_docs.items()},
            "llm_name": llm_name,
            "topic_names_llm": topic_names_llm,
            "topic_article_counts": modeler.get_topic_article_counts()
        }
        
        advanced_path = advanced_dir / f'advanced_topic_analysis_{run_id}.json' if args.versioned else advanced_dir / 'advanced_topic_analysis.json'
        with open(advanced_path, 'w', encoding='utf-8') as f:
            json.dump(advanced_results, f, ensure_ascii=False, indent=2)
        logger.info(f"Saved advanced topic analysis to {advanced_path}")
        
        # Also save/update latest version for convenience
        latest_advanced_path = advanced_dir / 'advanced_topic_analysis.json'
        with open(latest_advanced_path, 'w', encoding='utf-8') as f:
            json.dump(advanced_results, f, ensure_ascii=False, indent=2)
        logger.info(f"Updated latest advanced topic analysis at {latest_advanced_path}")

    # --- Analyse avancée BERTopic : pondération, cohérence, distribution, docs représentatifs ---
    elif args.engine == 'bertopic':
        # 1. Mots-clés pondérés par sujet
        weighted_words = modeler.get_bertopic_word_weights(n_terms=num_top_words)
        logger.info("\n--- Mots-clés pondérés par sujet (BERTopic) ---")
        for topic_idx, words in weighted_words.items():
            logger.info(f"Topic #{topic_idx}: {[(w, f'{wgt:.3f}') for w, wgt in words]}")

        # 2. Score de cohérence
        # Tokenize texts for coherence calculation
        tokenized_texts = [text.split() for text in texts]
        try:
            coherence = modeler.get_bertopic_coherence(texts=tokenized_texts)
            logger.info(f"Coherence score (c_v): {coherence:.3f}")
        except Exception as e:
            logger.warning(f"Could not calculate coherence score: {e}")
            coherence = None

        # 3. Distribution globale des sujets
        topic_dist = modeler.get_bertopic_topic_distribution()
        logger.info("\n--- Distribution globale des sujets (importance relative) ---")
        for topic_idx, frac in enumerate(topic_dist):
            logger.info(f"Topic #{topic_idx}: {frac:.2%}")

        # 4. Documents les plus représentatifs par sujet
        rep_docs = modeler.get_bertopic_representative_docs(n_docs=3)
        logger.info("\n--- Documents représentatifs par sujet (indices) ---")
        for topic_idx, doc_indices in rep_docs.items():
            logger.info(f"Topic #{topic_idx}: {doc_indices}")

        # 5. Nombre d'articles par sujet
        topic_article_counts = modeler.get_bertopic_article_counts()
        logger.info("\n--- Nombre d'articles par sujet ---")
        for topic_idx, count in topic_article_counts.items():
            logger.info(f"Topic #{topic_idx}: {count} articles")

        # 6. Sauvegarde des scores et analyses avancées dans results
        # Utilise le nom du LLM effectivement utilisé pour nommer les topics
        llm_name = getattr(modeler, 'llm_name_used', None)
        topic_names_llm = getattr(modeler, 'topic_names_llm', None)
        advanced_results = {
            "run_id": run_id,
            "coherence_score": coherence,
            "topic_distribution": topic_dist,
            "weighted_words": weighted_words,
            "representative_docs": rep_docs,
            "llm_name": llm_name,
            "topic_names_llm": topic_names_llm,
            "topic_article_counts": topic_article_counts
        }
        
        advanced_path = advanced_dir / f'advanced_topic_analysis_{run_id}.json' if args.versioned else advanced_dir / 'advanced_topic_analysis.json'
        with open(advanced_path, 'w', encoding='utf-8') as f:
            json.dump(advanced_results, f, ensure_ascii=False, indent=2)
        logger.info(f"Saved advanced topic analysis to {advanced_path}")
        
        # Also save/update latest version for convenience
        latest_advanced_path = advanced_dir / 'advanced_topic_analysis.json'
        with open(latest_advanced_path, 'w', encoding='utf-8') as f:
            json.dump(advanced_results, f, ensure_ascii=False, indent=2)
        logger.info(f"Updated latest advanced topic analysis at {latest_advanced_path}")

if __name__ == "__main__":
    main()
