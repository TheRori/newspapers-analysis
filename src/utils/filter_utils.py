"""
Utility functions for filtering articles based on various criteria.
This module provides functions to filter articles based on criteria
such as date range, newspaper, canton, topic tags, and word count.
"""

import os
import json
from typing import List, Dict, Any, Optional, Union
from datetime import datetime
from pathlib import Path

def filter_articles_by_date_range(
    articles: List[Dict[str, Any]], 
    start_date: Optional[str] = None, 
    end_date: Optional[str] = None
) -> List[Dict[str, Any]]:
    """
    Filter articles by date range.
    
    Args:
        articles: List of article dictionaries
        start_date: Start date in YYYY-MM-DD format (inclusive)
        end_date: End date in YYYY-MM-DD format (inclusive)
        
    Returns:
        Filtered list of articles
    """
    filtered_articles = articles
    
    if start_date:
        start_date_obj = datetime.strptime(start_date, "%Y-%m-%d").date()
        filtered_articles = [
            article for article in filtered_articles 
            if article.get('date') and datetime.strptime(article['date'], "%Y-%m-%d").date() >= start_date_obj
        ]
    
    if end_date:
        end_date_obj = datetime.strptime(end_date, "%Y-%m-%d").date()
        filtered_articles = [
            article for article in filtered_articles 
            if article.get('date') and datetime.strptime(article['date'], "%Y-%m-%d").date() <= end_date_obj
        ]
    
    return filtered_articles

def filter_articles_by_newspaper(
    articles: List[Dict[str, Any]], 
    newspaper: Optional[str] = None
) -> List[Dict[str, Any]]:
    """
    Filter articles by newspaper name (partial match).
    
    Args:
        articles: List of article dictionaries
        newspaper: Newspaper name to filter by (partial match)
        
    Returns:
        Filtered list of articles
    """
    if not newspaper:
        return articles
    
    return [
        article for article in articles 
        if article.get('newspaper') and newspaper.lower() in article['newspaper'].lower()
    ]

def filter_articles_by_canton(
    articles: List[Dict[str, Any]], 
    canton: Optional[str] = None
) -> List[Dict[str, Any]]:
    """
    Filter articles by canton.
    
    Args:
        articles: List of article dictionaries
        canton: Canton code (e.g., FR, VD)
        
    Returns:
        Filtered list of articles
    """
    if not canton:
        return articles
    
    return [
        article for article in articles 
        if article.get('canton') and article['canton'] == canton
    ]

def filter_articles_by_topic(
    articles: List[Dict[str, Any]], 
    topic: Optional[str] = None
) -> List[Dict[str, Any]]:
    """
    Filter articles by topic tag.
    
    Args:
        articles: List of article dictionaries
        topic: Topic tag to filter by
        
    Returns:
        Filtered list of articles
    """
    if not topic:
        return articles
    
    return [
        article for article in articles 
        if article.get('topics') and topic.lower() in [t.lower() for t in article['topics']]
    ]

def filter_articles_by_word_count(
    articles: List[Dict[str, Any]], 
    min_words: Optional[int] = None, 
    max_words: Optional[int] = None
) -> List[Dict[str, Any]]:
    """
    Filter articles by word count range.
    
    Args:
        articles: List of article dictionaries
        min_words: Minimum word count (inclusive)
        max_words: Maximum word count (inclusive)
        
    Returns:
        Filtered list of articles
    """
    filtered_articles = articles
    
    if min_words is not None:
        filtered_articles = [
            article for article in filtered_articles 
            if article.get('word_count', 0) >= min_words
        ]
    
    if max_words is not None:
        filtered_articles = [
            article for article in filtered_articles 
            if article.get('word_count', 0) <= max_words
        ]
    
    return filtered_articles

def apply_all_filters(
    articles: List[Dict[str, Any]],
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    newspaper: Optional[str] = None,
    canton: Optional[str] = None,
    topic: Optional[str] = None,
    min_words: Optional[int] = None,
    max_words: Optional[int] = None
) -> List[Dict[str, Any]]:
    """
    Apply all filters to a list of articles.
    
    Args:
        articles: List of article dictionaries
        start_date: Start date in YYYY-MM-DD format (inclusive)
        end_date: End date in YYYY-MM-DD format (inclusive)
        newspaper: Newspaper name to filter by
        canton: Canton code (e.g., FR, VD)
        topic: Topic tag to filter by
        min_words: Minimum word count (inclusive)
        max_words: Maximum word count (inclusive)
        
    Returns:
        Filtered list of articles
    """
    filtered = filter_articles_by_date_range(articles, start_date, end_date)
    filtered = filter_articles_by_newspaper(filtered, newspaper)
    filtered = filter_articles_by_canton(filtered, canton)
    filtered = filter_articles_by_topic(filtered, topic)
    filtered = filter_articles_by_word_count(filtered, min_words, max_words)
    
    return filtered

def get_filter_summary(
    original_count: int,
    filtered_count: int,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    newspaper: Optional[str] = None,
    canton: Optional[str] = None,
    topic: Optional[str] = None,
    min_words: Optional[int] = None,
    max_words: Optional[int] = None
) -> Dict[str, Any]:
    """
    Generate a summary of applied filters and their effect.
    
    Args:
        original_count: Original number of articles
        filtered_count: Number of articles after filtering
        start_date: Start date filter
        end_date: End date filter
        newspaper: Newspaper filter
        canton: Canton filter
        topic: Topic filter
        min_words: Minimum word count filter
        max_words: Maximum word count filter
        
    Returns:
        Dictionary with filter summary information
    """
    filters_applied = {}
    if start_date:
        filters_applied['start_date'] = start_date
    if end_date:
        filters_applied['end_date'] = end_date
    if newspaper:
        filters_applied['newspaper'] = newspaper
    if canton:
        filters_applied['canton'] = canton
    if topic:
        filters_applied['topic'] = topic
    if min_words is not None:
        filters_applied['min_words'] = min_words
    if max_words is not None:
        filters_applied['max_words'] = max_words
    
    return {
        'original_count': original_count,
        'filtered_count': filtered_count,
        'reduction_percentage': round((1 - filtered_count / original_count) * 100, 2) if original_count > 0 else 0,
        'filters_applied': filters_applied
    }

def load_articles(config_or_path: Union[Dict[str, Any], str, Path]) -> List[Dict[str, Any]]:
    """
    Load articles from a JSON file.
    
    Args:
        config_or_path: Either a config dictionary with data.processed_dir or a direct path
        
    Returns:
        List of article dictionaries
    """
    if isinstance(config_or_path, dict):
        processed_dir = config_or_path.get('data', {}).get('processed_dir', 'data/processed')
        articles_path = os.path.join(processed_dir, 'articles.json')
    else:
        articles_path = config_or_path
    
    if not os.path.exists(articles_path):
        raise FileNotFoundError(f"Articles file not found at {articles_path}")
    
    with open(articles_path, 'r', encoding='utf-8') as f:
        articles = json.load(f)
    
    return articles
