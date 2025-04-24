"""
Script to run BERTopic analysis on newspaper articles.

This script demonstrates how to use the BERTopic algorithm
within the existing TopicModeler framework.
"""
import os
import sys
import json
import logging
from pathlib import Path

# Add the parent directory to sys.path to import modules
sys.path.append(str(Path(__file__).parent.parent))

from analysis.topic_modeling import TopicModeler

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Path to the articles data
DATA_PATH = os.path.join(os.path.dirname(__file__), '../../data/processed/articles.json')

def load_articles(path, max_articles=None):
    """Load articles from JSON file."""
    logger.info(f"Loading articles from {path}")
    with open(path, encoding='utf-8') as f:
        data = json.load(f)
    
    # Convert to list of dictionaries with doc_id and text
    articles = []
    for i, article in enumerate(data):
        if "content" in article:
            articles.append({
                "doc_id": article.get("id", f"doc_{i}"),
                "text": article["content"],
                "date": article.get("date", ""),
                "newspaper": article.get("newspaper", ""),
                "title": article.get("title", "")
            })
    
    if max_articles:
        articles = articles[:max_articles]
    
    logger.info(f"Loaded {len(articles)} articles")
    return articles

def run_bertopic_analysis(articles, num_topics=10, save_model=True):
    """Run BERTopic analysis on the articles."""
    # Configure the topic modeler
    config = {
        "algorithm": "bertopic",
        "num_topics": num_topics,
        "max_df": 0.7,
        "min_df": 2
    }
    
    # Create and fit the model
    topic_modeler = TopicModeler(config)
    results = topic_modeler.fit_transform(articles)
    
    # Display results
    logger.info(f"Found {results['num_topics']} topics")
    
    # Print top terms for each topic
    print("\nTop terms for each topic:")
    for topic_id, terms in results['top_terms'].items():
        print(f"Topic {topic_id}: {', '.join(terms)}")
    
    # Print a few document-topic assignments
    print("\nSample document-topic assignments:")
    for i, (doc_id, doc_info) in enumerate(list(results['doc_topics'].items())[:5]):
        print(f"Document: {doc_id}")
        print(f"  Title: {articles[i].get('title', '')[:50]}")
        print(f"  Dominant topic: {doc_info['dominant_topic']}")
        print(f"  Top terms: {results['top_terms'].get(doc_info['dominant_topic'], ['N/A'])[:5]}")
        print()
    
    # Save the model if requested
    if save_model:
        output_dir = os.path.join(os.path.dirname(__file__), '../../models')
        os.makedirs(output_dir, exist_ok=True)
        model_path = os.path.join(output_dir, 'bertopic_model')
        topic_modeler.save_model(model_path)
        logger.info(f"Model saved to {model_path}")
    
    return results, topic_modeler

def main():
    """Main function to run the analysis."""
    # Load articles
    articles = load_articles(DATA_PATH, max_articles=100)  # Limit to 100 articles for faster processing
    
    # Run BERTopic analysis
    results, model = run_bertopic_analysis(articles, num_topics="auto")
    
    # You can also visualize topics if you're running this in a notebook environment
    # Uncomment these lines if you want to save visualizations
    # if hasattr(model.model, 'visualize_topics'):
    #     fig = model.model.visualize_topics()
    #     fig.write_html("topic_visualization.html")
    
    print("\nAnalysis complete!")

if __name__ == "__main__":
    main()
