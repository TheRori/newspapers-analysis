"""
Data loading utilities for OCR-processed newspaper articles.
"""

import os
import json
import csv
import pandas as pd
from typing import List, Dict, Any, Optional, Union
import pymongo

class DataLoader:
    """Class for loading and handling OCR-processed newspaper data."""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the DataLoader with configuration settings.
        
        Args:
            config: Dictionary containing data configuration
        """
        self.config = config
        self.raw_dir = config.get('raw_dir', '../data/raw')
        self.processed_dir = config.get('processed_dir', '../data/processed')
        self.mongodb_config = config.get('mongodb', {})
        self.mongo_client = None
        self.mongo_db = None
        self.mongo_collection = None
        if self.mongodb_config:
            self._init_mongo()

    def _init_mongo(self):
        """
        Initialize MongoDB client and collection.
        """
        uri = self.mongodb_config.get('uri', 'mongodb://localhost:27017')
        username = self.mongodb_config.get('username')
        password = self.mongodb_config.get('password')
        if username and password:
            uri = uri.replace('mongodb://', f'mongodb://{username}:{password}@')
        self.mongo_client = pymongo.MongoClient(uri)
        db_name = self.mongodb_config.get('database', 'newspaper_db')
        collection_name = self.mongodb_config.get('collection', 'articles')
        self.mongo_db = self.mongo_client[db_name]
        self.mongo_collection = self.mongo_db[collection_name]

    
    def load_from_mongodb(self, query: Optional[Dict[str, Any]] = None, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Load documents from MongoDB.
        Args:
            query: MongoDB query dict
            limit: Maximum number of documents to load; if None or 0, loads all
        Returns:
            List of documents
        """
        if not self.mongo_collection:
            raise RuntimeError("MongoDB collection is not initialized.")
        cursor = self.mongo_collection.find(query or {})
        if limit is not None and limit > 0:
            cursor = cursor.limit(limit)
        documents = list(cursor)
        for doc in documents:
            doc['doc_id'] = str(doc.get('_id', ''))
            if 'text' not in doc and 'content' in doc:
                doc['text'] = doc['content']
        return documents
    
    def save_processed_data(self, documents: List[Dict[str, Any]], 
                           output_file: str = 'processed_articles.json') -> str:
        """
        Save processed documents to a file.
        
        Args:
            documents: List of document dictionaries
            output_file: Name of the output file
            
        Returns:
            Path to the saved file
        """
        if not os.path.exists(self.processed_dir):
            os.makedirs(self.processed_dir)
            
        output_path = os.path.join(self.processed_dir, output_file)
        
        with open(output_path, 'w', encoding='utf-8') as file:
            json.dump(documents, file, ensure_ascii=False, indent=2)
            
        return output_path
    
    def save_as_csv(self, documents: List[Dict[str, Any]], 
                   output_file: str = 'processed_articles.csv') -> str:
        """
        Save processed documents to a CSV file.
        
        Args:
            documents: List of document dictionaries
            output_file: Name of the output file
            
        Returns:
            Path to the saved file
        """
        if not os.path.exists(self.processed_dir):
            os.makedirs(self.processed_dir)
            
        output_path = os.path.join(self.processed_dir, output_file)
        
        df = pd.DataFrame(documents)
        df.to_csv(output_path, index=False)
            
        return output_path
