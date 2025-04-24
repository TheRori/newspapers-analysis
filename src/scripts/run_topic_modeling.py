import json
import os
import argparse
from pathlib import Path
import logging
import sys
import uuid

# Add the project root to the path to allow imports from src
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from src.analysis.topic_modeling import TopicModeler
from src.utils.config_loader import load_config

def main():
    # Parse arguments for versioning
    parser = argparse.ArgumentParser(description="Run topic modeling on articles.")
    parser.add_argument('--versioned', action='store_true', help='Save results with unique versioned filenames (default: True)')
    parser.add_argument('--no-versioned', dest='versioned', action='store_false', help='Save results only with generic names (overwrites previous)')
    parser.set_defaults(versioned=True)
    parser.add_argument('--engine', choices=['sklearn', 'gensim'], default='sklearn', help='Choose topic modeling engine: sklearn or gensim (default: sklearn)')
    args = parser.parse_args()

    # Configure logging
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    logger = logging.getLogger(__name__)

    # Paths
    config_path = project_root / 'config' / 'config.yaml'
    articles_path = project_root / 'data' / 'processed' / 'articles.json'
    results_dir = project_root / 'data' / 'results'
    os.makedirs(results_dir, exist_ok=True)

    # Load config
    config = load_config(str(config_path))
    topic_config = config.get('analysis', {}).get('topic_modeling', {})
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

    # Fit and transform using selected engine
    if args.engine == 'sklearn':
        doc_topic_matrix = modeler.fit_transform_sklearn(texts)
        feature_names = modeler.feature_names
        model_components = modeler.model.components_
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
    top_words_path = results_dir / f'top_words_per_topic_{run_id}.json' if args.versioned else results_dir / 'top_words_per_topic.json'
    with open(top_words_path, 'w', encoding='utf-8') as f:
        json.dump(results_summary, f, ensure_ascii=False, indent=2)
    logger.info(f"Saved top words per topic to {top_words_path}")

    # Also save/update latest version for convenience
    latest_top_words_path = results_dir / 'top_words_per_topic.json'
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
        advanced_results = {
            "run_id": run_id,
            "coherence_score": coherence,
            "topic_distribution": topic_dist,
            "weighted_words": {str(k): [(w, float(f"{wgt:.5f}")) for w, wgt in v] for k, v in weighted_words.items()},
            "representative_docs": {str(k): v for k, v in rep_docs.items()}
        }
        adv_path = results_dir / f'advanced_topic_analysis_{run_id}.json' if args.versioned else results_dir / 'advanced_topic_analysis.json'
        with open(adv_path, 'w', encoding='utf-8') as f:
            json.dump(advanced_results, f, ensure_ascii=False, indent=2)
        logger.info(f"Saved advanced topic analysis to {adv_path}")
        # Always update latest
        latest_adv_path = results_dir / 'advanced_topic_analysis.json'
        with open(latest_adv_path, 'w', encoding='utf-8') as f:
            json.dump(advanced_results, f, ensure_ascii=False, indent=2)
        logger.info(f"Updated latest advanced topic analysis at {latest_adv_path}")

        # 6. Comptage des articles par topic (avec seuil)
        threshold = 0.2  # seuil par défaut, modifiable
        topic_article_counts = modeler.get_topic_article_counts(threshold=threshold)
        logger.info(f"\n--- Nombre d'articles par topic (seuil={threshold}) ---")
        for topic_idx, count in topic_article_counts.items():
            logger.info(f"Topic #{topic_idx}: {count} articles")
        # Ajoute au JSON
        advanced_results["topic_article_counts"] = {str(k): v for k, v in topic_article_counts.items()}
        # Réécrit les fichiers
        with open(adv_path, 'w', encoding='utf-8') as f:
            json.dump(advanced_results, f, ensure_ascii=False, indent=2)
        with open(latest_adv_path, 'w', encoding='utf-8') as f:
            json.dump(advanced_results, f, ensure_ascii=False, indent=2)
        logger.info(f"Updated advanced topic analysis with article counts (threshold={threshold})")

if __name__ == "__main__":
    main()
