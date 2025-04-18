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

    def list_files(self, directory: str, file_extension: Optional[str] = None) -> List[str]:
        """
        List all files in a directory with optional extension filter.
        
        Args:
            directory: Directory to search
            file_extension: Optional file extension to filter by (e.g., '.txt', '.json')
            
        Returns:
            List of file paths
        """
        if not os.path.exists(directory):
            raise FileNotFoundError(f"Directory {directory} does not exist")
            
        files = []
        for root, _, filenames in os.walk(directory):
            for filename in filenames:
                if file_extension is None or filename.endswith(file_extension):
                    files.append(os.path.join(root, filename))
        
        return files
    
    def load_text_file(self, file_path: str) -> str:
        """
        Load text from a file.
        
        Args:
            file_path: Path to the text file
            
        Returns:
            Text content
        """
        with open(file_path, 'r', encoding='utf-8') as file:
            return file.read()
    
    def load_json_file(self, file_path: str) -> Dict[str, Any]:
        """
        Load JSON data from a file.
        
        Args:
            file_path: Path to the JSON file
            
        Returns:
            Dictionary containing JSON data
        """
        with open(file_path, 'r', encoding='utf-8') as file:
            return json.load(file)
    
    def load_csv_file(self, file_path: str, **kwargs) -> pd.DataFrame:
        """
        Load CSV data from a file.
        
        Args:
            file_path: Path to the CSV file
            **kwargs: Additional arguments to pass to pandas.read_csv
            
        Returns:
            DataFrame containing CSV data
        """
        return pd.read_csv(file_path, **kwargs)
    
    def load_multiple_files(self, file_paths: List[str], 
                           loader_func: callable) -> List[Any]:
        """
        Load multiple files using the specified loader function.
        
        Args:
            file_paths: List of file paths
            loader_func: Function to use for loading each file
            
        Returns:
            List of loaded file contents
        """
        return [loader_func(file_path) for file_path in file_paths]
    
    def load_dataset(self, directory: str = None, 
                    file_extension: str = '.txt') -> List[Dict[str, Any]]:
        """
        Load a dataset of newspaper articles from files.
        
        Args:
            directory: Directory containing the files (defaults to raw_dir from config)
            file_extension: File extension to filter by
            
        Returns:
            List of document dictionaries
        """
        if directory is None:
            directory = self.raw_dir
            
        file_paths = self.list_files(directory, file_extension)
        
        documents = []
        for file_path in file_paths:
            doc_id = os.path.basename(file_path).split('.')[0]
            
            if file_extension == '.json':
                data = self.load_json_file(file_path)
                # Ensure the document has a text field
                if 'text' not in data:
                    data['text'] = data.get('content', '')
                data['doc_id'] = doc_id
                documents.append(data)
            elif file_extension == '.csv':
                # Assume the CSV contains multiple documents
                df = self.load_csv_file(file_path)
                for _, row in df.iterrows():
                    doc = row.to_dict()
                    if 'text' not in doc and 'content' in doc:
                        doc['text'] = doc['content']
                    if 'doc_id' not in doc:
                        doc['doc_id'] = f"{doc_id}_{_}"
                    documents.append(doc)
            else:
                # Default to text files
                text = self.load_text_file(file_path)
                documents.append({
                    'doc_id': doc_id,
                    'text': text,
                    'file_path': file_path
                })
        
        return documents
    
    def load_from_mongodb(self, query: Optional[Dict[str, Any]] = None, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Load documents from MongoDB.
        
        Args:
            query: MongoDB query dictionary
            limit: Optional limit on number of documents
            
        Returns:
            List of document dictionaries
        """
        if not self.mongo_collection:
            raise RuntimeError("MongoDB collection is not initialized.")
        cursor = self.mongo_collection.find(query or {})
        if limit:
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
