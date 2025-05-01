"""
Integrated analysis pipeline that combines topic clustering with sentiment analysis and entity recognition.

This script takes a cluster file as input, then runs sentiment analysis and entity recognition
on each cluster, providing comprehensive statistics and insights for each topic cluster.

Usage:
    python -m src.scripts.run_integrated_analysis --cluster-file path/to/clusters.json --output path/to/output.json
"""

import json
import os
import argparse
from pathlib import Path
import logging
import sys
import uuid
import time
import numpy as np
import pandas as pd
from collections import Counter, defaultdict
try:
    import psutil
except ImportError:
    psutil = None

# Add the project root to the path to allow imports from src
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from src.analysis.sentiment_analysis import SentimentAnalyzer
from src.analysis.entity_recognition import EntityRecognizer
from src.utils.config_loader import load_config

def get_parser():
    parser = argparse.ArgumentParser(description="Run integrated analysis (sentiment + entity recognition) on topic clusters.")
    parser.add_argument('--cluster-file', type=str, required=True, help='Path to the cluster file (JSON)')
    parser.add_argument('--output', type=str, help='Output file path (JSON). Default is integrated_analysis_{timestamp}.json')
    parser.add_argument('--articles-file', type=str, help='Path to the articles file (JSON). Default is from config')
    parser.add_argument('--sentiment-model', choices=['vader', 'transformers'], default=None, 
                      help='Sentiment analysis model to use (overrides config)')
    parser.add_argument('--ner-model', type=str, default=None, 
                      help='Named Entity Recognition model to use (overrides config)')
    parser.add_argument('--entity-types', type=str, default=None, 
                      help='Comma-separated list of entity types to extract (e.g., PERSON,ORG,GPE)')
    
    return parser

def load_articles(file_path):
    """Load articles from JSON file."""
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def load_clusters(file_path):
    """Load cluster data from JSON file."""
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def group_articles_by_cluster(articles, cluster_data):
    """Group articles by their cluster assignment."""
    # Create a mapping from article ID to cluster
    article_to_cluster = {}
    
    # Handle different cluster file formats
    if 'document_clusters' in cluster_data:
        # New format with document_clusters list
        for doc in cluster_data['document_clusters']:
            article_id = str(doc.get('id', doc.get('base_id', '')))
            if article_id:
                article_to_cluster[article_id] = doc['cluster']
    elif 'clusters' in cluster_data:
        # Old format with separate clusters and document_ids lists
        for i, doc_id in enumerate(cluster_data.get('document_ids', [])):
            article_id = str(doc_id)
            if i < len(cluster_data['clusters']):
                article_to_cluster[article_id] = cluster_data['clusters'][i]
    
    # Group articles by cluster
    clusters = defaultdict(list)
    for article in articles:
        article_id = str(article.get('id', article.get('base_id', '')))
        if article_id in article_to_cluster:
            cluster_id = article_to_cluster[article_id]
            clusters[cluster_id].append(article)
    
    return clusters

def analyze_clusters(clusters, sentiment_analyzer, entity_recognizer):
    """Run sentiment analysis and entity recognition on each cluster."""
    cluster_analysis = {}
    
    for cluster_id, articles in clusters.items():
        # Skip empty clusters
        if not articles:
            continue
        
        # Run sentiment analysis
        articles_with_sentiment = sentiment_analyzer.analyze_documents(articles)
        sentiment_summary = sentiment_analyzer.get_sentiment_summary(articles_with_sentiment)
        
        # Run entity recognition
        articles_with_entities = entity_recognizer.process_documents(articles_with_sentiment)
        entity_summary = entity_recognizer.get_entity_summary(articles_with_entities)
        
        # Get entity frequencies for top entities
        entity_frequencies = {}
        for entity_type in entity_summary['total_by_type']:
            if entity_summary['total_by_type'][entity_type] > 0:
                freq = entity_recognizer.get_entity_frequency(articles_with_entities, entity_type)
                # Get top 20 entities by frequency
                top_entities = sorted(freq.items(), key=lambda x: x[1], reverse=True)[:20]
                entity_frequencies[entity_type] = dict(top_entities)
        
        # Calculate additional cluster statistics
        avg_length = sum(len(article.get('text', '').split()) for article in articles) / len(articles)
        newspapers = Counter(article.get('newspaper', 'Unknown') for article in articles)
        dates = sorted([article.get('date', '') for article in articles if article.get('date')])
        date_range = (dates[0], dates[-1]) if dates else ('Unknown', 'Unknown')
        
        # Store cluster analysis
        cluster_analysis[cluster_id] = {
            'num_articles': len(articles),
            'avg_article_length': avg_length,
            'newspaper_distribution': dict(newspapers),
            'date_range': date_range,
            'sentiment': sentiment_summary,
            'entities': entity_summary,
            'top_entities': entity_frequencies,
            'articles': [
                {
                    'id': article.get('id', article.get('base_id', '')),
                    'title': article.get('title', 'No title'),
                    'date': article.get('date', ''),
                    'newspaper': article.get('newspaper', 'Unknown'),
                    'sentiment': article.get('sentiment', {}),
                    'entity_counts': article.get('entity_counts', {})
                }
                for article in articles_with_entities
            ]
        }
    
    return cluster_analysis

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
    
    # Load config
    config_path = project_root / 'config' / 'config.yaml'
    config = load_config(str(config_path))
    
    # Determine paths
    cluster_file = args.cluster_file
    
    if args.articles_file:
        articles_file = args.articles_file
    else:
        articles_file = project_root / config['data']['processed_dir'] / 'articles.json'
    
    if args.output:
        output_file = args.output
    else:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        output_dir = project_root / config['data']['results_dir'] / 'integrated_analysis'
        os.makedirs(output_dir, exist_ok=True)
        output_file = output_dir / f"integrated_analysis_{timestamp}.json"
    
    # Load data
    logger.info(f"Loading cluster data from {cluster_file}")
    cluster_data = load_clusters(cluster_file)
    
    logger.info(f"Loading articles from {articles_file}")
    articles = load_articles(articles_file)
    
    # Group articles by cluster
    logger.info("Grouping articles by cluster")
    clusters = group_articles_by_cluster(articles, cluster_data)
    logger.info(f"Found {len(clusters)} clusters")
    
    # Configure analyzers
    sentiment_config = config.get('analysis', {}).get('sentiment', {})
    if args.sentiment_model:
        sentiment_config['model'] = args.sentiment_model
    
    ner_config = config.get('analysis', {}).get('ner', {})
    if args.ner_model:
        ner_config['model'] = args.ner_model
    
    if args.entity_types:
        entity_types = [e.strip() for e in args.entity_types.split(',')]
        ner_config['entities'] = entity_types
    
    # Initialize analyzers
    logger.info(f"Initializing sentiment analyzer with model: {sentiment_config.get('model', 'vader')}")
    sentiment_analyzer = SentimentAnalyzer(sentiment_config)
    
    logger.info(f"Initializing entity recognizer with model: {ner_config.get('model', 'en_core_web_lg')}")
    entity_recognizer = EntityRecognizer(ner_config)
    
    # Analyze clusters
    start_time = time.time()
    logger.info("Running integrated analysis on clusters")
    cluster_analysis = analyze_clusters(clusters, sentiment_analyzer, entity_recognizer)
    
    # Calculate overall statistics
    total_articles = sum(data['num_articles'] for data in cluster_analysis.values())
    
    # Prepare cluster summaries for visualization
    cluster_summaries = []
    for cluster_id, data in cluster_analysis.items():
        # Get top entities across all types
        all_entities = []
        for entity_type, entities in data['top_entities'].items():
            for entity, count in entities.items():
                all_entities.append((entity, entity_type, count))
        
        top_entities = sorted(all_entities, key=lambda x: x[2], reverse=True)[:10]
        
        cluster_summaries.append({
            'cluster_id': cluster_id,
            'num_articles': data['num_articles'],
            'percentage': data['num_articles'] / total_articles * 100 if total_articles > 0 else 0,
            'avg_sentiment': data['sentiment']['mean_compound'],
            'positive_percentage': data['sentiment']['positive_percentage'],
            'negative_percentage': data['sentiment']['negative_percentage'],
            'neutral_percentage': data['sentiment']['neutral_percentage'],
            'top_entities': [{'text': e[0], 'type': e[1], 'count': e[2]} for e in top_entities],
            'top_newspapers': dict(Counter(data['newspaper_distribution']).most_common(3)),
            'date_range': data['date_range']
        })
    
    # Create results object
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
    
    results = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "duration_seconds": duration,
        "cpu_times": cpu_times,
        "cluster_file": cluster_file,
        "articles_file": str(articles_file),
        "sentiment_model": sentiment_config.get('model', 'vader'),
        "ner_model": ner_config.get('model', 'en_core_web_lg'),
        "entity_types": list(ner_config.get('entities', ['PERSON', 'ORG', 'GPE', 'LOC', 'DATE'])),
        "num_clusters": len(cluster_analysis),
        "total_articles": total_articles,
        "cluster_summaries": cluster_summaries,
        "cluster_analysis": cluster_analysis
    }
    
    # Save results
    logger.info(f"Saving integrated analysis results to {output_file}")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    # Print summary
    logger.info(f"Integrated Analysis Results:")
    logger.info(f"  Clusters analyzed: {len(cluster_analysis)}")
    logger.info(f"  Total articles: {total_articles}")
    logger.info(f"  Sentiment model: {sentiment_config.get('model', 'vader')}")
    logger.info(f"  NER model: {ner_config.get('model', 'en_core_web_lg')}")
    logger.info(f"  Processing time: {duration:.2f} seconds")
    
    # Print cluster summaries
    logger.info("\nCluster Summaries:")
    for summary in sorted(cluster_summaries, key=lambda x: x['cluster_id']):
        logger.info(f"  Cluster {summary['cluster_id']}:")
        logger.info(f"    Articles: {summary['num_articles']} ({summary['percentage']:.1f}%)")
        logger.info(f"    Avg. Sentiment: {summary['avg_sentiment']:.4f}")
        logger.info(f"    Sentiment Distribution: +{summary['positive_percentage']:.1f}% / 0{summary['neutral_percentage']:.1f}% / -{summary['negative_percentage']:.1f}%")
        if summary['top_entities']:
            top_entity = summary['top_entities'][0]
            logger.info(f"    Top Entity: {top_entity['text']} ({top_entity['type']}, {top_entity['count']} occurrences)")
    
    return results

if __name__ == "__main__":
    main()
