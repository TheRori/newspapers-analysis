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
sys.path.insert(0, str(project_root))

# Import using relative imports
try:
    from src.analysis.sentiment_analysis import SentimentAnalyzer
    from src.utils.config_loader import load_config
    from src.utils.filter_utils import apply_all_filters, get_filter_summary
except ModuleNotFoundError:
    # Alternative import path if the above fails
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from analysis.sentiment_analysis import SentimentAnalyzer
    from utils.config_loader import load_config
    from utils.filter_utils import apply_all_filters, get_filter_summary

def get_parser():
    parser = argparse.ArgumentParser(description="Run sentiment analysis on articles.")
    parser.add_argument('--versioned', action='store_true', help='Save results with unique versioned filenames (default: True)')
    parser.add_argument('--no-versioned', dest='versioned', action='store_false', help='Save results only with generic names (overwrites previous)')
    parser.set_defaults(versioned=True)
    parser.add_argument('--model', choices=['vader', 'transformers'], help='Sentiment analysis model to use (overrides config)')
    parser.add_argument('--transformer-model', type=str, help='Transformer model to use if model=transformers (overrides config)')
    parser.add_argument('--use-cache', action='store_true', help='Use cached preprocessed documents if available')
    parser.add_argument('--source-file', type=str, help='Chemin vers un fichier JSON d\'articles alternatif (remplace celui de la config)')
    
    # Add filtering options
    parser.add_argument('--start-date', type=str, help='Filter articles starting from this date (format: YYYY-MM-DD)')
    parser.add_argument('--end-date', type=str, help='Filter articles until this date (format: YYYY-MM-DD)')
    parser.add_argument('--newspaper', type=str, help='Filter articles by newspaper name')
    parser.add_argument('--canton', type=str, help='Filter articles by canton (e.g., FR, VD)')
    parser.add_argument('--topic', type=str, help='Filter articles by existing topic tag')
    parser.add_argument('--min-words', type=int, help='Filter articles with at least this many words')
    parser.add_argument('--max-words', type=int, help='Filter articles with at most this many words')
    
    # Add cluster filtering options
    parser.add_argument('--cluster-file', type=str, help='Path to the cluster file for filtering articles by cluster')
    parser.add_argument('--cluster-id', type=str, help='Cluster ID to filter articles by')
    
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
    
    # Paths - use absolute paths to avoid issues when called from different directories
    script_dir = Path(__file__).parent.absolute()
    project_root = script_dir.parent.parent
    config_path = project_root / 'config' / 'config.yaml'
    
    # Déterminer le chemin du fichier d'articles (par défaut ou personnalisé)
    if args.source_file:
        articles_path = Path(args.source_file)
        logger.info(f"Using custom articles file: {articles_path}")
    else:
        articles_path = project_root / 'data' / 'processed' / 'articles.json'
        logger.info(f"Using default articles file from config")
    
    results_dir = project_root / 'data' / 'results'
    os.makedirs(results_dir, exist_ok=True)
    
    # Log paths for debugging
    logger.info(f"Script directory: {script_dir}")
    logger.info(f"Project root: {project_root}")
    logger.info(f"Config path: {config_path}")
    logger.info(f"Articles path: {articles_path}")
    logger.info(f"Results directory: {results_dir}")
    
    # Ensure result subdirectories exist
    sentiment_dir = results_dir / "sentiment_analysis"
    sentiment_dir.mkdir(parents=True, exist_ok=True)

    # Load config
    config = load_config(str(config_path))
    sentiment_config = config.get('analysis', {}).get('sentiment', {})
    
    # Override config with command-line arguments if provided
    if args.model:
        sentiment_config['model'] = args.model
        logger.info(f"Using model from command line: {args.model}")
    
    if args.transformer_model and sentiment_config.get('model') == 'transformers':
        sentiment_config['transformer_model'] = args.transformer_model
        logger.info(f"Using transformer model from command line: {args.transformer_model}")
    
    logger.info(f"Loaded sentiment analysis config: {sentiment_config}")

    # Load articles
    with open(articles_path, 'r', encoding='utf-8') as f:
        articles = json.load(f)
    logger.info(f"Loaded {len(articles)} articles from {articles_path}")
    
    # Load cluster data if specified
    cluster_data = None
    if args.cluster_file and args.cluster_id:
        try:
            logger.info(f"Loading cluster data from {args.cluster_file}")
            with open(args.cluster_file, 'r', encoding='utf-8') as f:
                cluster_data = json.load(f)
            logger.info(f"Loaded cluster data with keys: {list(cluster_data.keys())}")
        except Exception as e:
            logger.error(f"Error loading cluster file: {e}")
            cluster_data = None
    
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
        max_words=args.max_words,
        cluster=args.cluster_id,
        cluster_data=cluster_data
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
            max_words=args.max_words,
            cluster=args.cluster_id
        )
        logger.info(f"Applied filters: {filter_summary}")
    
    # Initialize sentiment analyzer
    start_time = time.time()
    logger.info(f"Initializing sentiment analyzer with model: {sentiment_config.get('model', 'vader')}")
    sentiment_analyzer = SentimentAnalyzer(sentiment_config)
    
    # Process articles
    logger.info(f"Running sentiment analysis on {len(filtered_articles)} articles...")
    articles_with_sentiment = sentiment_analyzer.analyze_documents(filtered_articles)
    
    # Get summary statistics
    sentiment_summary = sentiment_analyzer.get_sentiment_summary(articles_with_sentiment)
    logger.info(f"Sentiment analysis complete. Summary: {sentiment_summary}")
    
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
        "model": sentiment_config.get('model', 'vader'),
        "transformer_model": sentiment_config.get('transformer_model') if sentiment_config.get('model') == 'transformers' else None,
        "num_articles": len(articles_with_sentiment),
        "summary": sentiment_summary,
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
        summary_file = sentiment_dir / f"sentiment_summary_{run_id}.json"
        articles_file = sentiment_dir / f"articles_with_sentiment_{run_id}.json"
    else:
        summary_file = sentiment_dir / "sentiment_summary.json"
        articles_file = sentiment_dir / "articles_with_sentiment.json"
    
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    logger.info(f"Saved sentiment summary to {summary_file}")
    
    with open(articles_file, 'w', encoding='utf-8') as f:
        json.dump(articles_with_sentiment, f, ensure_ascii=False, indent=2)
    logger.info(f"Saved articles with sentiment to {articles_file}")
    
    # Print final summary
    logger.info(f"Sentiment Analysis Results:")
    logger.info(f"  Model: {sentiment_config.get('model', 'vader')}")
    if sentiment_config.get('model') == 'transformers':
        logger.info(f"  Transformer model: {sentiment_config.get('transformer_model')}")
    logger.info(f"  Articles analyzed: {len(articles_with_sentiment)}")
    logger.info(f"  Average compound sentiment: {sentiment_summary['mean_compound']:.4f}")
    logger.info(f"  Positive articles: {sentiment_summary['positive_count']} ({sentiment_summary['positive_percentage']:.1f}%)")
    logger.info(f"  Neutral articles: {sentiment_summary['neutral_count']} ({sentiment_summary['neutral_percentage']:.1f}%)")
    logger.info(f"  Negative articles: {sentiment_summary['negative_count']} ({sentiment_summary['negative_percentage']:.1f}%)")
    logger.info(f"  Processing time: {duration:.2f} seconds")
    
    return results

if __name__ == "__main__":
    main()
