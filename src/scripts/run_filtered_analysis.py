#!/usr/bin/env python3
"""
Script pour exécuter des analyses filtrées par topic ou cluster.
"""

import os
import sys
import json
import argparse
import logging
import uuid
from pathlib import Path
from datetime import datetime

# Add the project root to the path to allow imports from src
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from src.analysis.topic_filter import TopicFilter, run_filtered_analysis
from src.utils.config_loader import load_config

def get_parser():
    parser = argparse.ArgumentParser(description="Run analysis filtered by topic or cluster.")
    
    # Type d'analyse
    parser.add_argument('--analysis-type', 
                        choices=['sentiment', 'entity', 'lexical', 'term_tracking'],
                        required=True,
                        help='Type of analysis to run')
    
    # Filtres de topic/cluster
    parser.add_argument('--topic-id', type=int, help='ID of topic to include')
    parser.add_argument('--cluster-id', type=int, help='ID of cluster to include')
    parser.add_argument('--exclude-topic-id', type=int, help='ID of topic to exclude')
    parser.add_argument('--exclude-cluster-id', type=int, help='ID of cluster to exclude')
    
    # Chemins de fichiers
    parser.add_argument('--topic-results', type=str, help='Path to topic modeling results')
    parser.add_argument('--output-dir', type=str, help='Directory to save results')
    parser.add_argument('--versioned', action='store_true', default=True, 
                        help='Save results with unique versioned filenames (default: True)')
    parser.add_argument('--no-versioned', dest='versioned', action='store_false', 
                        help='Save results only with generic names (overwrites previous)')
    
    # Options spécifiques à l'analyse de sentiment
    parser.add_argument('--sentiment-model', choices=['vader', 'transformers'], 
                        help='Sentiment analysis model to use (overrides config)')
    
    # Options spécifiques à l'analyse d'entités
    parser.add_argument('--entity-types', type=str, nargs='+',
                        help='Entity types to include (e.g., PERSON, ORG, LOC)')
    
    # Options spécifiques au suivi de termes
    parser.add_argument('--terms', type=str, nargs='+',
                        help='Terms to track (for term_tracking analysis)')
    parser.add_argument('--case-sensitive', action='store_true',
                        help='Case sensitive term matching (for term_tracking)')
    
    return parser

def main():
    # Parse arguments
    parser = get_parser()
    args = parser.parse_args()
    
    # Set up logging
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    handler.setFormatter(formatter)
    logger = logging.getLogger()
    logger.handlers = [handler]
    logger.setLevel(logging.INFO)
    
    # Generate run ID for versioning
    run_id = str(uuid.uuid4())[:8]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Paths
    config_path = project_root / 'config' / 'config.yaml'
    articles_path = project_root / 'data' / 'processed' / 'articles.json'
    results_dir = project_root / 'data' / 'results'
    
    # Determine topic results path if not provided
    if not args.topic_results:
        topic_results_dir = results_dir / "advanced_topic"
        # Try to find the most recent topic results file
        topic_files = list(topic_results_dir.glob("advanced_topic_analysis*.json"))
        if topic_files:
            # Sort by modification time (most recent first)
            topic_files.sort(key=lambda x: os.path.getmtime(x), reverse=True)
            args.topic_results = str(topic_files[0])
            logger.info(f"Using most recent topic results file: {args.topic_results}")
        else:
            logger.error("No topic results file found. Please specify with --topic-results")
            sys.exit(1)
    
    # Determine output directory if not provided
    if not args.output_dir:
        args.output_dir = str(results_dir / "filtered_analysis")
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Determine output filename
    filter_suffix = ""
    if args.topic_id is not None:
        filter_suffix += f"_topic{args.topic_id}"
    if args.cluster_id is not None:
        filter_suffix += f"_cluster{args.cluster_id}"
    if args.exclude_topic_id is not None:
        filter_suffix += f"_exclTopic{args.exclude_topic_id}"
    if args.exclude_cluster_id is not None:
        filter_suffix += f"_exclCluster{args.exclude_cluster_id}"
    
    if args.versioned:
        output_filename = f"{args.analysis_type}{filter_suffix}_{timestamp}_{run_id}.json"
    else:
        output_filename = f"{args.analysis_type}{filter_suffix}.json"
    
    output_path = os.path.join(args.output_dir, output_filename)
    
    # Load config
    config = load_config(str(config_path))
    
    # Prepare additional arguments based on analysis type
    additional_args = {}
    
    if args.analysis_type == 'sentiment' and args.sentiment_model:
        additional_args['model'] = args.sentiment_model
    
    elif args.analysis_type == 'entity' and args.entity_types:
        additional_args['entity_types'] = args.entity_types
    
    elif args.analysis_type == 'term_tracking' and args.terms:
        additional_args['terms'] = args.terms
        additional_args['case_sensitive'] = args.case_sensitive
    
    # Log analysis parameters
    logger.info(f"Running {args.analysis_type} analysis with the following filters:")
    if args.topic_id is not None:
        logger.info(f"  Including topic ID: {args.topic_id}")
    if args.cluster_id is not None:
        logger.info(f"  Including cluster ID: {args.cluster_id}")
    if args.exclude_topic_id is not None:
        logger.info(f"  Excluding topic ID: {args.exclude_topic_id}")
    if args.exclude_cluster_id is not None:
        logger.info(f"  Excluding cluster ID: {args.exclude_cluster_id}")
    
    # Run the filtered analysis
    try:
        results = run_filtered_analysis(
            analysis_type=args.analysis_type,
            topic_results_path=args.topic_results,
            output_path=output_path,
            topic_id=args.topic_id,
            cluster_id=args.cluster_id,
            exclude_topic_id=args.exclude_topic_id,
            exclude_cluster_id=args.exclude_cluster_id,
            config_path=str(config_path),
            articles_path=str(articles_path),
            **additional_args
        )
        
        # Print summary of results
        if args.analysis_type == 'sentiment' and 'summary' in results:
            summary = results['summary']
            logger.info(f"Sentiment Analysis Results (Filtered):")
            logger.info(f"  Articles analyzed: {len(results['articles'])}")
            logger.info(f"  Average compound sentiment: {summary['mean_compound']:.4f}")
            logger.info(f"  Positive articles: {summary['positive_count']} ({summary['positive_percentage']:.1f}%)")
            logger.info(f"  Neutral articles: {summary['neutral_count']} ({summary['neutral_percentage']:.1f}%)")
            logger.info(f"  Negative articles: {summary['negative_count']} ({summary['negative_percentage']:.1f}%)")
        
        elif args.analysis_type == 'entity' and 'summary' in results:
            summary = results['summary']
            logger.info(f"Entity Recognition Results (Filtered):")
            logger.info(f"  Articles analyzed: {len(results['articles'])}")
            logger.info(f"  Total entities found: {summary['total_entities']}")
            logger.info(f"  Entity types: {', '.join(summary['total_by_type'].keys())}")
            for entity_type, count in summary['total_by_type'].items():
                logger.info(f"    {entity_type}: {count}")
        
        elif args.analysis_type == 'lexical':
            logger.info(f"Lexical Analysis Results (Filtered):")
            logger.info(f"  Articles analyzed: {results['filter_info']['filtered_articles']}")
            if 'word_counts' in results:
                logger.info(f"  Total words analyzed: {sum(results['word_counts'].values())}")
                logger.info(f"  Unique words: {len(results['word_counts'])}")
        
        elif args.analysis_type == 'term_tracking':
            logger.info(f"Term Tracking Results (Filtered):")
            logger.info(f"  Articles analyzed: {results['filter_info']['filtered_articles']}")
            if 'term_counts' in results:
                for term, count in results['term_counts'].items():
                    logger.info(f"    '{term}': {count} occurrences")
        
        logger.info(f"Results saved to: {output_path}")
        
    except Exception as e:
        logger.error(f"Error running filtered analysis: {e}")
        import traceback
        logger.error(traceback.format_exc())
        sys.exit(1)

if __name__ == "__main__":
    main()
