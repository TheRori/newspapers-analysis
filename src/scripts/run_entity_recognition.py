import json
import os
import argparse
from pathlib import Path
import logging
import sys
import uuid
import time
try:
    import psutil
except ImportError:
    psutil = None

# Add the project root to the path to allow imports from src
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from src.analysis.entity_recognition import EntityRecognizer
from src.utils.config_loader import load_config
from src.utils.filter_utils import apply_all_filters, get_filter_summary

def get_parser():
    parser = argparse.ArgumentParser(description="Run entity recognition on articles.")
    parser.add_argument('--versioned', action='store_true', help='Save results with unique versioned filenames (default: True)')
    parser.add_argument('--no-versioned', dest='versioned', action='store_false', help='Save results only with generic names (overwrites previous)')
    parser.set_defaults(versioned=True)
    parser.add_argument('--model', type=str, help='SpaCy model to use (overrides config)')
    parser.add_argument('--entities', type=str, help='Comma-separated list of entity types to extract (e.g., PERSON,ORG,GPE)')
    parser.add_argument('--use-cache', action='store_true', help='Use cached preprocessed documents if available')
    
    # Add filtering options
    parser.add_argument('--start-date', type=str, help='Filter articles starting from this date (format: YYYY-MM-DD)')
    parser.add_argument('--end-date', type=str, help='Filter articles until this date (format: YYYY-MM-DD)')
    parser.add_argument('--newspaper', type=str, help='Filter articles by newspaper name')
    parser.add_argument('--canton', type=str, help='Filter articles by canton (e.g., FR, VD)')
    parser.add_argument('--topic', type=str, help='Filter articles by existing topic tag')
    parser.add_argument('--min-words', type=int, help='Filter articles with at least this many words')
    parser.add_argument('--max-words', type=int, help='Filter articles with at most this many words')
    
    return parser

def main():
    # Parse arguments
    parser = get_parser()
    args = parser.parse_args()
    
    # Set up logging to always print to stdout for Dash
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    handler.setFormatter(formatter)
    logger = logging.getLogger()
    logger.handlers = [handler]
    logger.setLevel(logging.INFO)

    # Generate run ID for versioning
    run_id = str(uuid.uuid4())[:8]
    
    # Paths
    config_path = project_root / 'config' / 'config.yaml'
    articles_path = project_root / 'data' / 'processed' / 'articles.json'
    results_dir = project_root / 'data' / 'results'
    os.makedirs(results_dir, exist_ok=True)
    
    # Ensure result subdirectories exist
    ner_dir = results_dir / "entity_recognition"
    ner_dir.mkdir(parents=True, exist_ok=True)

    # Load config
    config = load_config(str(config_path))
    ner_config = config.get('analysis', {}).get('ner', {})
    
    # Override config with command-line arguments if provided
    if args.model:
        ner_config['model'] = args.model
        logger.info(f"Using model from command line: {args.model}")
    
    if args.entities:
        entity_types = [e.strip() for e in args.entities.split(',')]
        ner_config['entities'] = entity_types
        logger.info(f"Using entity types from command line: {entity_types}")
    
    logger.info(f"Loaded entity recognition config: {ner_config}")

    # Load articles
    with open(articles_path, 'r', encoding='utf-8') as f:
        articles = json.load(f)
    logger.info(f"Loaded {len(articles)} articles from {articles_path}")
    
    # Apply filters if specified
    original_count = len(articles)
    filtered_articles = apply_all_filters(
        articles,
        start_date=args.start_date,
        end_date=args.end_date,
        newspaper=args.newspaper,
        canton=args.canton,
        topic=args.topic,
        min_words=args.min_words,
        max_words=args.max_words
    )
    
    # Log filter results
    if original_count != len(filtered_articles):
        filter_summary = get_filter_summary(
            original_count,
            len(filtered_articles),
            start_date=args.start_date,
            end_date=args.end_date,
            newspaper=args.newspaper,
            canton=args.canton,
            topic=args.topic,
            min_words=args.min_words,
            max_words=args.max_words
        )
        logger.info(f"Applied filters: {filter_summary}")
    
    # Initialize entity recognizer
    start_time = time.time()
    logger.info(f"Initializing entity recognizer with model: {ner_config.get('model', 'en_core_web_lg')}")
    entity_recognizer = EntityRecognizer(ner_config)
    
    # Process articles
    logger.info(f"Running entity recognition on {len(filtered_articles)} articles...")
    articles_with_entities = entity_recognizer.process_documents(filtered_articles)
    
    # Get summary statistics
    entity_summary = entity_recognizer.get_entity_summary(articles_with_entities)
    logger.info(f"Entity recognition complete. Found {sum(entity_summary['total_by_type'].values())} entities.")
    
    # Get entity frequencies for top entities
    entity_frequencies = {}
    for entity_type in entity_summary['total_by_type']:
        if entity_summary['total_by_type'][entity_type] > 0:
            freq = entity_recognizer.get_entity_frequency(articles_with_entities, entity_type)
            # Get top 50 entities by frequency
            top_entities = sorted(freq.items(), key=lambda x: x[1], reverse=True)[:50]
            entity_frequencies[entity_type] = dict(top_entities)
    
    # Save results
    end_time = time.time()
    duration = end_time - start_time
    
    # CPU usage stats if available
    cpu_times = None
    if psutil:
        try:
            cpu_times = {
                "user": psutil.Process().cpu_times().user,
                "system": psutil.Process().cpu_times().system,
                "total": psutil.Process().cpu_times().user + psutil.Process().cpu_times().system
            }
        except Exception as e:
            logger.warning(f"Could not get CPU times: {e}")
    
    # Create results object
    results = {
        "run_id": run_id,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "duration_seconds": duration,
        "cpu_times": cpu_times,
        "model": ner_config.get('model', 'en_core_web_lg'),
        "target_entities": list(ner_config.get('entities', ['PERSON', 'ORG', 'GPE', 'LOC', 'DATE'])),
        "num_articles": len(articles_with_entities),
        "summary": entity_summary,
        "entity_frequencies": entity_frequencies,
        "filter_settings": {
            "start_date": args.start_date,
            "end_date": args.end_date,
            "newspaper": args.newspaper,
            "canton": args.canton,
            "topic": args.topic,
            "min_words": args.min_words,
            "max_words": args.max_words
        }
    }
    
    # Save summary results
    if args.versioned:
        summary_file = ner_dir / f"entity_summary_{run_id}.json"
        articles_file = ner_dir / f"articles_with_entities_{run_id}.json"
    else:
        summary_file = ner_dir / "entity_summary.json"
        articles_file = ner_dir / "articles_with_entities.json"
    
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    logger.info(f"Saved entity summary to {summary_file}")
    
    with open(articles_file, 'w', encoding='utf-8') as f:
        json.dump(articles_with_entities, f, ensure_ascii=False, indent=2)
    logger.info(f"Saved articles with entities to {articles_file}")
    
    # Print final summary
    logger.info(f"Entity Recognition Results:")
    logger.info(f"  Model: {ner_config.get('model', 'en_core_web_lg')}")
    logger.info(f"  Target entities: {', '.join(ner_config.get('entities', ['PERSON', 'ORG', 'GPE', 'LOC', 'DATE']))}")
    logger.info(f"  Articles analyzed: {len(articles_with_entities)}")
    logger.info(f"  Total entities found: {sum(entity_summary['total_by_type'].values())}")
    
    # Print entity counts by type
    logger.info(f"  Entity counts by type:")
    for entity_type, count in sorted(entity_summary['total_by_type'].items(), key=lambda x: x[1], reverse=True):
        logger.info(f"    {entity_type}: {count}")
    
    logger.info(f"  Processing time: {duration:.2f} seconds")
    
    return results

if __name__ == "__main__":
    main()
