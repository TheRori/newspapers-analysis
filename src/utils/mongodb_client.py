"""
MongoDB client module for newspaper articles analysis.
Provides functionality to connect to MongoDB and retrieve articles with filtering capabilities.
"""

import os
import json
from typing import Dict, List, Any, Optional, Union, Tuple
from datetime import datetime
import logging
from pymongo import MongoClient
from pymongo.errors import ConnectionFailure, OperationFailure

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class MongoDBClient:
    """
    Client for interacting with MongoDB to retrieve newspaper articles.
    Provides methods for filtering and retrieving articles with various criteria.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the MongoDB client with configuration settings.
        
        Args:
            config: Dictionary containing MongoDB configuration
        """
        self.config = config.get('mongodb', {})
        self.client = None
        self.db = None
        self.collection = None
        self._connect()
    
    def _connect(self):
        """
        Establish connection to MongoDB using configuration settings.
        """
        try:
            # Get connection parameters from config
            uri = self.config.get('uri')
            db_name = self.config.get('database')
            collection_name = self.config.get('collection')
            
            if not uri or not db_name or not collection_name:
                logger.error("Missing MongoDB configuration parameters")
                raise ValueError("Missing MongoDB configuration parameters")
            
            # Connect to MongoDB
            logger.info(f"Attempting to connect to MongoDB with URI: {uri[:20]}...{uri[-20:]} (middle part hidden)")
            self.client = MongoClient(uri, serverSelectionTimeoutMS=5000)
            # Test connection
            self.client.admin.command('ping')
            logger.info("Successfully connected to MongoDB")
            
            # Get database and collection
            self.db = self.client[db_name]
            self.collection = self.db[collection_name]
            
        except ConnectionFailure as e:
            logger.error(f"Failed to connect to MongoDB: {e}")
            raise
        except OperationFailure as e:
            logger.error(f"Authentication failed: {e}")
            raise
    
    def get_articles(self, 
                    filters: Optional[Dict[str, Any]] = None, 
                    projection: Optional[Dict[str, Any]] = None,
                    sort: Optional[List[Tuple[str, int]]] = None,
                    limit: Optional[int] = None,
                    skip: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Retrieve articles from MongoDB with optional filtering, projection, sorting, and pagination.
        
        Args:
            filters: MongoDB query dictionary for filtering documents
            projection: Fields to include or exclude in the results
            sort: List of (field, direction) tuples for sorting
            limit: Maximum number of documents to return
            skip: Number of documents to skip
            
        Returns:
            List of article documents
        """
        try:
            # Default empty filter if none provided
            query = filters or {}
            
            # Create cursor with query
            cursor = self.collection.find(query, projection)
            
            # Apply sorting if specified
            if sort:
                cursor = cursor.sort(sort)
            
            # Apply pagination if specified
            if skip:
                cursor = cursor.skip(skip)
            if limit:
                cursor = cursor.limit(limit)
            
            # Convert cursor to list
            articles = list(cursor)
            logger.info(f"Retrieved {len(articles)} articles matching the query")
            
            return articles
            
        except Exception as e:
            logger.error(f"Error retrieving articles: {e}")
            raise
    
    def get_article_by_id(self, article_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve a single article by its ID.
        
        Args:
            article_id: The ID of the article to retrieve
            
        Returns:
            Article document or None if not found
        """
        try:
            # Check if article_id is in ObjectId format
            if article_id.startswith("article_"):
                # Using the base_id or id field
                article = self.collection.find_one({"$or": [{"id": article_id}, {"base_id": article_id}]})
            else:
                # Using the MongoDB _id field
                from bson.objectid import ObjectId
                try:
                    article = self.collection.find_one({"_id": ObjectId(article_id)})
                except:
                    # If conversion fails, try as a string
                    article = self.collection.find_one({"_id": article_id})
            
            return article
            
        except Exception as e:
            logger.error(f"Error retrieving article by ID: {e}")
            return None
    
    def get_articles_by_date_range(self, 
                                  start_date: str, 
                                  end_date: str,
                                  limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Retrieve articles within a specific date range.
        
        Args:
            start_date: Start date in format YYYY-MM-DD
            end_date: End date in format YYYY-MM-DD
            limit: Maximum number of articles to return
            
        Returns:
            List of article documents
        """
        try:
            # Create date range filter
            date_filter = {"date": {"$gte": start_date, "$lte": end_date}}
            
            # Get articles with date filter
            return self.get_articles(filters=date_filter, limit=limit)
            
        except Exception as e:
            logger.error(f"Error retrieving articles by date range: {e}")
            return []
    
    def get_articles_by_topics(self, 
                              topics: List[str], 
                              match_all: bool = False,
                              limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Retrieve articles containing specific topics.
        
        Args:
            topics: List of topics to filter by
            match_all: If True, articles must contain all topics; if False, any topic
            limit: Maximum number of articles to return
            
        Returns:
            List of article documents
        """
        try:
            # Create topic filter based on match_all parameter
            if match_all:
                # Articles must contain all specified topics
                topic_filter = {"topics": {"$all": topics}}
            else:
                # Articles must contain any of the specified topics
                topic_filter = {"topics": {"$in": topics}}
            
            # Get articles with topic filter
            return self.get_articles(filters=topic_filter, limit=limit)
            
        except Exception as e:
            logger.error(f"Error retrieving articles by topics: {e}")
            return []
    
    def get_articles_by_newspaper(self, 
                                 newspaper: str,
                                 limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Retrieve articles from a specific newspaper.
        
        Args:
            newspaper: Name of the newspaper
            limit: Maximum number of articles to return
            
        Returns:
            List of article documents
        """
        try:
            # Create newspaper filter
            newspaper_filter = {"newspaper": newspaper}
            
            # Get articles with newspaper filter
            return self.get_articles(filters=newspaper_filter, limit=limit)
            
        except Exception as e:
            logger.error(f"Error retrieving articles by newspaper: {e}")
            return []
    
    def get_articles_by_canton(self, 
                              canton: str,
                              limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Retrieve articles from a specific canton.
        
        Args:
            canton: Canton code (e.g., "NE", "VD", "GE")
            limit: Maximum number of articles to return
            
        Returns:
            List of article documents
        """
        try:
            # Create canton filter
            canton_filter = {"canton": canton}
            
            # Get articles with canton filter
            return self.get_articles(filters=canton_filter, limit=limit)
            
        except Exception as e:
            logger.error(f"Error retrieving articles by canton: {e}")
            return []
    
    def get_articles_by_text_search(self, 
                                   search_text: str,
                                   limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Retrieve articles containing specific text in content.
        
        Args:
            search_text: Text to search for in article content
            limit: Maximum number of articles to return
            
        Returns:
            List of article documents
        """
        try:
            # Create text search filter
            text_filter = {"content": {"$regex": search_text, "$options": "i"}}
            
            # Get articles with text filter
            return self.get_articles(filters=text_filter, limit=limit)
            
        except Exception as e:
            logger.error(f"Error retrieving articles by text search: {e}")
            return []
    
    def get_articles_by_combined_filters(self,
                                        date_range: Optional[Tuple[str, str]] = None,
                                        topics: Optional[List[str]] = None,
                                        newspaper: Optional[str] = None,
                                        canton: Optional[str] = None,
                                        search_text: Optional[str] = None,
                                        match_all_topics: bool = False,
                                        limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Retrieve articles using a combination of filters.
        
        Args:
            date_range: Tuple of (start_date, end_date) in format (YYYY-MM-DD, YYYY-MM-DD)
            topics: List of topics to filter by
            newspaper: Name of the newspaper
            canton: Canton code
            search_text: Text to search for in article content
            match_all_topics: If True, articles must contain all topics; if False, any topic
            limit: Maximum number of articles to return
            
        Returns:
            List of article documents
        """
        try:
            # Build combined filter
            combined_filter = {}
            
            # Add date range filter if provided
            if date_range:
                start_date, end_date = date_range
                combined_filter["date"] = {"$gte": start_date, "$lte": end_date}
            
            # Add topics filter if provided
            if topics:
                if match_all_topics:
                    combined_filter["topics"] = {"$all": topics}
                else:
                    combined_filter["topics"] = {"$in": topics}
            
            # Add newspaper filter if provided
            if newspaper:
                combined_filter["newspaper"] = newspaper
            
            # Add canton filter if provided
            if canton:
                combined_filter["canton"] = canton
            
            # Add text search filter if provided
            if search_text:
                combined_filter["content"] = {"$regex": search_text, "$options": "i"}
            
            # Get articles with combined filter
            return self.get_articles(filters=combined_filter, limit=limit)
            
        except Exception as e:
            logger.error(f"Error retrieving articles by combined filters: {e}")
            return []
    
    def get_unique_values(self, field: str) -> List[str]:
        """
        Get unique values for a specific field across all articles.
        
        Args:
            field: Field name to get unique values for
            
        Returns:
            List of unique values
        """
        try:
            # Use distinct to get unique values
            unique_values = self.collection.distinct(field)
            return unique_values
            
        except Exception as e:
            logger.error(f"Error retrieving unique values for field {field}: {e}")
            return []
    
    def get_article_count(self, filters: Optional[Dict[str, Any]] = None) -> int:
        """
        Get the count of articles matching the specified filters.
        
        Args:
            filters: MongoDB query dictionary for filtering documents
            
        Returns:
            Count of matching articles
        """
        try:
            # Default empty filter if none provided
            query = filters or {}
            
            # Get count of matching documents
            count = self.collection.count_documents(query)
            return count
            
        except Exception as e:
            logger.error(f"Error getting article count: {e}")
            return 0
    
    def close(self):
        """
        Close the MongoDB connection.
        """
        if self.client:
            self.client.close()
            logger.info("MongoDB connection closed")
