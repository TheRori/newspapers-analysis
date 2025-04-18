#!/usr/bin/env python3
"""
Script for loading articles from MongoDB with various filtering options.
This script demonstrates how to use the MongoDBClient to retrieve and filter articles.
"""

import os
import sys
import json
import argparse
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime

# Add the project root to the path to allow imports from other modules
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from src.utils.config_loader import load_config
from src.utils.mongodb_client import MongoDBClient

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Load articles from MongoDB with filtering options')
    
    # Basic filtering options
    parser.add_argument('--limit', type=int, default=10, help='Maximum number of articles to retrieve')
    parser.add_argument('--output', type=str, default='articles.json', help='Output file path')
    
    # Date filtering
    parser.add_argument('--start-date', type=str, help='Start date in YYYY-MM-DD format')
    parser.add_argument('--end-date', type=str, help='End date in YYYY-MM-DD format')
    
    # Content filtering
    parser.add_argument('--topics', type=str, nargs='+', help='Filter by topics')
    parser.add_argument('--match-all-topics', action='store_true', help='Match all topics (AND) instead of any (OR)')
    parser.add_argument('--search-text', type=str, help='Search for text in article content')
    
    # Metadata filtering
    parser.add_argument('--newspaper', type=str, help='Filter by newspaper name')
    parser.add_argument('--canton', type=str, help='Filter by canton code (e.g., NE, VD, GE)')
    
    # List options
    parser.add_argument('--list-newspapers', action='store_true', help='List all unique newspaper names')
    parser.add_argument('--list-cantons', action='store_true', help='List all unique canton codes')
    parser.add_argument('--list-topics', action='store_true', help='List all unique topics')
    
    # Article by ID
    parser.add_argument('--article-id', type=str, help='Retrieve a specific article by ID')
    
    return parser.parse_args()

def format_article_for_display(article: Dict[str, Any]) -> Dict[str, Any]:
    """
    Format an article for display, handling MongoDB-specific fields.
    
    Args:
        article: The article document from MongoDB
        
    Returns:
        Formatted article dictionary
    """
    # Create a copy to avoid modifying the original
    formatted = article.copy()
    
    # Handle ObjectId
    if '_id' in formatted and not isinstance(formatted['_id'], str):
        formatted['_id'] = str(formatted['_id'])
    
    # Handle other MongoDB-specific types if needed
    
    return formatted

def save_articles_to_file(articles: List[Dict[str, Any]], output_path: str):
    """
    Save articles to a JSON file.
    
    Args:
        articles: List of article documents
        output_path: Path to save the output file
    """
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    
    # Format articles for display
    formatted_articles = [format_article_for_display(article) for article in articles]
    
    # Save to file
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(formatted_articles, f, ensure_ascii=False, indent=2)
    
    print(f"Saved {len(formatted_articles)} articles to {output_path}")

def main():
    """Main function to load and filter articles from MongoDB."""
    args = parse_arguments()
    
    # Load configuration
    config = load_config()
    
    # Initialize MongoDB client
    mongo_client = MongoDBClient(config)
    
    try:
        # Handle list options
        if args.list_newspapers:
            newspapers = mongo_client.get_unique_values('newspaper')
            print("Available newspapers:")
            for newspaper in sorted(newspapers):
                print(f"  - {newspaper}")
            return
        
        if args.list_cantons:
            cantons = mongo_client.get_unique_values('canton')
            print("Available cantons:")
            for canton in sorted(cantons):
                print(f"  - {canton}")
            return
        
        if args.list_topics:
            topics = mongo_client.get_unique_values('topics')
            print("Available topics:")
            for topic in sorted(topics):
                print(f"  - {topic}")
            return
        
        # Handle article by ID
        if args.article_id:
            article = mongo_client.get_article_by_id(args.article_id)
            if article:
                formatted_article = format_article_for_display(article)
                print(json.dumps(formatted_article, ensure_ascii=False, indent=2))
                
                # Save to file if output is specified
                if args.output:
                    save_articles_to_file([article], args.output)
            else:
                print(f"No article found with ID: {args.article_id}")
            return
        
        # Handle combined filters
        date_range = None
        if args.start_date and args.end_date:
            date_range = (args.start_date, args.end_date)
        
        # Get articles with combined filters
        articles = mongo_client.get_articles_by_combined_filters(
            date_range=date_range,
            topics=args.topics,
            newspaper=args.newspaper,
            canton=args.canton,
            search_text=args.search_text,
            match_all_topics=args.match_all_topics,
            limit=args.limit
        )
        
        # Print summary
        print(f"Retrieved {len(articles)} articles")
        
        # Save to file if output is specified
        if args.output:
            save_articles_to_file(articles, args.output)
        
    finally:
        # Close MongoDB connection
        mongo_client.close()

if __name__ == "__main__":
    main()
