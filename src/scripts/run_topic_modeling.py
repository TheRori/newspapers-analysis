import json
import os
import argparse
from pathlib import Path
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

def main():
    # Parse arguments for versioning
    parser = argparse.ArgumentParser(description="Run topic modeling on articles.")
    parser.add_argument('--versioned', action='store_true', help='Save results with unique versioned filenames (default: True)')
    parser.add_argument('--no-versioned', dest='versioned', action='store_false', help='Save results only with generic names (overwrites previous)')
    parser.set_defaults(versioned=True)
    parser.add_argument('--engine', choices=['sklearn', 'gensim', 'bertopic'], default='gensim', help='Choose topic modeling engine: sklearn, gensim, or bertopic (default: gensim)')
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
    args = parser.parse_args()

    # Configure logging
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    logger = logging.getLogger(__name__)

    # Paths
    config_path = project_root / 'config' / 'config.yaml'
    articles_path = project_root / 'data' / 'processed' / 'articles.json'
    results_dir = project_root / 'data' / 'results'
    os.makedirs(results_dir, exist_ok=True)
    
    # Dossier pour sauvegarder les modèles
    models_dir = project_root / 'data' / 'models'
    os.makedirs(models_dir, exist_ok=True)

    # Ensure result subdirectories exist
    advanced_dir = results_dir / "advanced_topic"
    topwords_dir = results_dir / "top_words"
    coherence_dir = results_dir / "coherence_trials"
    for d in [advanced_dir, topwords_dir, coherence_dir]:
        d.mkdir(parents=True, exist_ok=True)

    # Load config
    config = load_config(str(config_path))
    topic_config = config.get('analysis', {}).get('topic_modeling', {})
    # Override algorithm if provided via CLI
    if args.algorithm:
        topic_config['algorithm'] = args.algorithm
    elif args.engine == 'bertopic':
        topic_config['algorithm'] = 'bertopic'
    logger.info(f"Loaded topic modeling config: {topic_config}")

    # Load articles
    with open(articles_path, 'r', encoding='utf-8') as f:
        articles = json.load(f)
    logger.info(f"Loaded {len(articles)} articles from {articles_path}")

    # Initialize topic modeler
    modeler = TopicModeler(topic_config)

    # Preprocess articles for sklearn (using 'content' as text field)
    doc_ids = [doc.get('id', f"doc_{i}") for i, doc in enumerate(articles)]
    texts = [doc['content'] for doc in articles]

    # If requested, automatically find the best num_topics using coherence (LDA + Gensim engine only)
    if args.auto_num_topics and args.engine == 'gensim' and topic_config.get('algorithm', 'lda') == 'lda':
        logger.info(f"Searching best num_topics (coherence) in [{args.k_min}, {args.k_max}] with mode {args.search_mode}...")
        # Prepare texts and corpus for Gensim
        lang = "fr"  # default language
        if lang == "fr":
            stopwords = get_french_stopwords()
        else:
            stopwords = get_stopwords(lang)
        import re
        tokenized_texts = [
            [
                word for word in re.findall(r"\b\w{2,}\b", text.lower())
                if word not in stopwords and not word.isdigit()
            ]
            for text in texts
        ]
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
                model_k = LdaMulticore(corpus=corpus, id2word=id2word, num_topics=k, workers=2, random_state=42)
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
        topic_config['num_topics'] = best_k
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
        doc_topic_matrix = [doc['topic_distribution'] for doc in results['doc_topics'].values()]
        feature_names = None  # Not used for BERTopic
        model_components = results['top_terms']
    else:  # gensim
        doc_topic_matrix = modeler.fit_transform_gensim(texts)
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
            logger.info(f"Topic #{topic_idx}: {' '.join(top_features)}")
            top_words_per_topic[f"topic_{topic_idx}"] = top_features
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
    doc_topic_output = [
        {"doc_id": doc_id, "topic_distribution": [float(x) for x in topic_dist]}
        for doc_id, topic_dist in zip(doc_ids, doc_topic_matrix)
    ]
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
        rep_docs = modeler.get_representative_docs(n_docs=3)
        logger.info("\n--- Documents représentatifs par sujet (indices) ---")
        for topic_idx, doc_indices in rep_docs.items():
            logger.info(f"Topic #{topic_idx}: {doc_indices}")

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
