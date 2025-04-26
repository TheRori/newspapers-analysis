"""
Utility functions for grouping articles by different criteria.
This module provides functions to filter and group articles based on criteria
such as year, newspaper, or other metadata for further analysis like topic modeling.
"""

import os
import json
import pandas as pd
from typing import List, Dict, Any, Optional, Union, Tuple
from pathlib import Path
import re
from datetime import datetime

def load_articles(config: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Load articles from the configured JSON file.
    
    Args:
        config: Application configuration dictionary
        
    Returns:
        List of article dictionaries
    """
    processed_dir = config.get('data', {}).get('processed_dir', 'data/processed')
    articles_path = os.path.join(processed_dir, 'articles.json')
    
    if not os.path.exists(articles_path):
        raise FileNotFoundError(f"Articles file not found at {articles_path}")
    
    with open(articles_path, 'r', encoding='utf-8') as f:
        articles = json.load(f)
    
    return articles

def extract_year_from_id(article_id: Union[str, int]) -> Optional[int]:
    """
    Extract year from article ID if it contains a date pattern.
    
    Args:
        article_id: Article ID which may contain a date pattern
        
    Returns:
        Extracted year as integer or None if not found
    """
    # Convert to string if it's not already
    article_id = str(article_id)
    
    # Try to extract YYYY-MM-DD pattern
    date_pattern = r'(\d{4})-\d{2}-\d{2}'
    match = re.search(date_pattern, article_id)
    if match:
        return int(match.group(1))
    
    # Try to extract YYYY pattern
    year_pattern = r'_(\d{4})_'
    match = re.search(year_pattern, article_id)
    if match:
        return int(match.group(1))
    
    # Try to extract from 'date' field if it exists in the article
    return None

def extract_newspaper_from_id(article_id: Union[str, int]) -> Optional[str]:
    """
    Extract newspaper name from article ID.
    
    Args:
        article_id: Article ID which may contain newspaper information
        
    Returns:
        Extracted newspaper name or None if not found
    """
    # Convert to string if it's not already
    article_id = str(article_id)
    
    # Format expected: article_YYYY-MM-DD_newspaper_id
    parts = article_id.split('_')
    if len(parts) >= 3:
        return parts[2]
    
    return None

def group_articles_by_year(articles: List[Dict[str, Any]]) -> Dict[int, List[Dict[str, Any]]]:
    """
    Group articles by publication year.
    
    Args:
        articles: List of article dictionaries
        
    Returns:
        Dictionary with years as keys and lists of articles as values
    """
    grouped = {}
    
    for article in articles:
        # Try to get year from article metadata
        year = None
        
        # First try to get from date field if it exists
        if 'date' in article:
            try:
                date_str = article['date']
                if isinstance(date_str, str):
                    # Try different date formats
                    for fmt in ['%Y-%m-%d', '%Y/%m/%d', '%d/%m/%Y', '%Y']:
                        try:
                            date_obj = datetime.strptime(date_str, fmt)
                            year = date_obj.year
                            break
                        except ValueError:
                            continue
            except (ValueError, TypeError):
                pass
        
        # If year not found in date field, try to extract from ID
        if year is None:
            article_id = article.get('doc_id', article.get('id', article.get('_id', '')))
            year = extract_year_from_id(article_id)
        
        # If still no year, skip this article
        if year is None:
            continue
        
        # Add article to the appropriate year group
        if year not in grouped:
            grouped[year] = []
        grouped[year].append(article)
    
    return grouped

def group_articles_by_newspaper(articles: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    """
    Group articles by newspaper source.
    
    Args:
        articles: List of article dictionaries
        
    Returns:
        Dictionary with newspaper names as keys and lists of articles as values
    """
    grouped = {}
    
    for article in articles:
        # Try to get newspaper from article metadata
        newspaper = None
        
        # First check if there's a source or newspaper field
        if 'source' in article:
            newspaper = article['source']
        elif 'newspaper' in article:
            newspaper = article['newspaper']
        
        # If newspaper not found in metadata, try to extract from ID
        if newspaper is None or newspaper == '':
            article_id = article.get('doc_id', article.get('id', article.get('_id', '')))
            newspaper = extract_newspaper_from_id(article_id)
        
        # If still no newspaper, use 'unknown'
        if newspaper is None or newspaper == '':
            newspaper = 'unknown'
        
        # Add article to the appropriate newspaper group
        if newspaper not in grouped:
            grouped[newspaper] = []
        grouped[newspaper].append(article)
    
    return grouped

def filter_articles_by_years(articles: List[Dict[str, Any]], 
                            start_year: int, 
                            end_year: Optional[int] = None) -> List[Dict[str, Any]]:
    """
    Filter articles by a range of years.
    
    Args:
        articles: List of article dictionaries
        start_year: Start year (inclusive)
        end_year: End year (inclusive), if None, only articles from start_year are returned
        
    Returns:
        Filtered list of articles
    """
    if end_year is None:
        end_year = start_year
    
    filtered = []
    grouped_by_year = group_articles_by_year(articles)
    
    for year in range(start_year, end_year + 1):
        if year in grouped_by_year:
            filtered.extend(grouped_by_year[year])
    
    return filtered

def filter_articles_by_newspapers(articles: List[Dict[str, Any]], 
                                newspapers: List[str]) -> List[Dict[str, Any]]:
    """
    Filter articles by a list of newspaper names.
    
    Args:
        articles: List of article dictionaries
        newspapers: List of newspaper names to include
        
    Returns:
        Filtered list of articles
    """
    filtered = []
    grouped_by_newspaper = group_articles_by_newspaper(articles)
    
    for newspaper in newspapers:
        if newspaper in grouped_by_newspaper:
            filtered.extend(grouped_by_newspaper[newspaper])
    
    return filtered

def get_year_distribution(articles: List[Dict[str, Any]]) -> Dict[int, int]:
    """
    Get the distribution of articles by year.
    
    Args:
        articles: List of article dictionaries
        
    Returns:
        Dictionary with years as keys and article counts as values
    """
    grouped = group_articles_by_year(articles)
    return {year: len(articles_list) for year, articles_list in grouped.items()}

def get_newspaper_distribution(articles: List[Dict[str, Any]]) -> Dict[str, int]:
    """
    Get the distribution of articles by newspaper.
    
    Args:
        articles: List of article dictionaries
        
    Returns:
        Dictionary with newspaper names as keys and article counts as values
    """
    grouped = group_articles_by_newspaper(articles)
    return {newspaper: len(articles_list) for newspaper, articles_list in grouped.items()}

def save_filtered_articles(articles: List[Dict[str, Any]], 
                          output_path: str) -> str:
    """
    Save filtered articles to a JSON file.
    
    Args:
        articles: List of article dictionaries to save
        output_path: Path where to save the filtered articles
        
    Returns:
        Path to the saved file
    """
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(articles, f, ensure_ascii=False, indent=2)
    
    return output_path
