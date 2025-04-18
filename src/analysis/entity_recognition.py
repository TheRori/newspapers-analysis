"""
Named Entity Recognition for newspaper articles.
"""

import os
import json
from collections import Counter
from typing import List, Dict, Any, Optional, Tuple, Set

import spacy
import pandas as pd


class EntityRecognizer:
    """Class for named entity recognition on newspaper articles."""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the EntityRecognizer with configuration settings.
        
        Args:
            config: Dictionary containing NER configuration
        """
        self.config = config
        self.model_name = config.get('model', 'en_core_web_lg')
        self.target_entities = set(config.get('entities', 
                                             ['PERSON', 'ORG', 'GPE', 'LOC', 'DATE']))
        
        # Initialize spaCy model
        self.nlp = None
        self._load_spacy_model()
    
    def _load_spacy_model(self):
        """Load the spaCy model."""
        try:
            self.nlp = spacy.load(self.model_name)
        except OSError:
            print(f"SpaCy model {self.model_name} not found. Please install it with:")
            print(f"python -m spacy download {self.model_name}")
            raise
    
    def extract_entities(self, text: str) -> List[Dict[str, Any]]:
        """
        Extract named entities from text.
        
        Args:
            text: Input text
            
        Returns:
            List of entity dictionaries
        """
        if not text or len(text.strip()) == 0:
            return []
        
        # Process text with spaCy
        doc = self.nlp(text)
        
        # Extract entities
        entities = []
        for ent in doc.ents:
            if ent.label_ in self.target_entities:
                entities.append({
                    'text': ent.text,
                    'label': ent.label_,
                    'start': ent.start_char,
                    'end': ent.end_char
                })
        
        return entities
    
    def process_document(self, document: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process a document to extract named entities.
        
        Args:
            document: Document dictionary
            
        Returns:
            Document with added entities
        """
        # Use cleaned_text if available, otherwise use text
        if 'cleaned_text' in document:
            text = document['cleaned_text']
        elif 'text' in document:
            text = document['text']
        else:
            raise KeyError("Document must contain either 'cleaned_text' or 'text' key")
        
        entities = self.extract_entities(text)
        document['entities'] = entities
        
        # Add entity counts by type
        entity_counts = {}
        for ent in entities:
            label = ent['label']
            if label not in entity_counts:
                entity_counts[label] = 0
            entity_counts[label] += 1
        
        document['entity_counts'] = entity_counts
        
        return document
    
    def process_documents(self, documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Process multiple documents to extract named entities.
        
        Args:
            documents: List of document dictionaries
            
        Returns:
            List of documents with added entities
        """
        return [self.process_document(doc) for doc in documents]
    
    def get_entity_frequency(self, documents: List[Dict[str, Any]], 
                            entity_type: Optional[str] = None) -> Dict[str, int]:
        """
        Get frequency of entities across documents.
        
        Args:
            documents: List of document dictionaries with entities
            entity_type: Optional entity type to filter by
            
        Returns:
            Dictionary mapping entity text to frequency
        """
        # Ensure documents have entities
        documents_with_entities = []
        for doc in documents:
            if 'entities' not in doc:
                doc = self.process_document(doc)
            documents_with_entities.append(doc)
        
        # Count entity occurrences
        entity_counter = Counter()
        for doc in documents_with_entities:
            for entity in doc['entities']:
                if entity_type is None or entity['label'] == entity_type:
                    entity_counter[entity['text']] += 1
        
        return dict(entity_counter)
    
    def get_entity_co_occurrence(self, documents: List[Dict[str, Any]], 
                               entity_types: Optional[List[str]] = None) -> Dict[Tuple[str, str], int]:
        """
        Get co-occurrence of entities within documents.
        
        Args:
            documents: List of document dictionaries with entities
            entity_types: Optional list of entity types to include
            
        Returns:
            Dictionary mapping entity pairs to co-occurrence count
        """
        # Ensure documents have entities
        documents_with_entities = []
        for doc in documents:
            if 'entities' not in doc:
                doc = self.process_document(doc)
            documents_with_entities.append(doc)
        
        # Count co-occurrences
        co_occurrence = Counter()
        for doc in documents_with_entities:
            # Get unique entities in document
            unique_entities = set()
            for entity in doc['entities']:
                if entity_types is None or entity['label'] in entity_types:
                    unique_entities.add((entity['text'], entity['label']))
            
            # Count co-occurrences
            entities_list = list(unique_entities)
            for i in range(len(entities_list)):
                for j in range(i+1, len(entities_list)):
                    # Sort to ensure consistent ordering
                    pair = tuple(sorted([entities_list[i][0], entities_list[j][0]]))
                    co_occurrence[pair] += 1
        
        return dict(co_occurrence)
    
    def get_entity_summary(self, documents: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Get summary statistics of entities across documents.
        
        Args:
            documents: List of document dictionaries with entities
            
        Returns:
            Dictionary with entity summary statistics
        """
        # Ensure documents have entities
        documents_with_entities = []
        for doc in documents:
            if 'entities' not in doc:
                doc = self.process_document(doc)
            documents_with_entities.append(doc)
        
        # Count entities by type
        entity_type_counts = Counter()
        for doc in documents_with_entities:
            for entity in doc['entities']:
                entity_type_counts[entity['label']] += 1
        
        # Get top entities by type
        top_entities_by_type = {}
        for entity_type in self.target_entities:
            entity_counter = Counter()
            for doc in documents_with_entities:
                for entity in doc['entities']:
                    if entity['label'] == entity_type:
                        entity_counter[entity['text']] += 1
            
            top_entities_by_type[entity_type] = dict(entity_counter.most_common(10))
        
        # Calculate percentage of documents with each entity type
        total_docs = len(documents_with_entities)
        entity_type_doc_counts = {entity_type: 0 for entity_type in self.target_entities}
        for doc in documents_with_entities:
            entity_types_in_doc = set(entity['label'] for entity in doc['entities'])
            for entity_type in entity_types_in_doc:
                if entity_type in entity_type_doc_counts:
                    entity_type_doc_counts[entity_type] += 1
        
        entity_type_doc_percentages = {
            entity_type: (count / total_docs * 100) 
            for entity_type, count in entity_type_doc_counts.items()
        }
        
        return {
            'entity_type_counts': dict(entity_type_counts),
            'top_entities_by_type': top_entities_by_type,
            'entity_type_doc_counts': entity_type_doc_counts,
            'entity_type_doc_percentages': entity_type_doc_percentages
        }
    
    def save_results(self, entity_summary: Dict[str, Any], 
                    output_dir: str, filename: str = 'entity_summary.json') -> str:
        """
        Save entity analysis results.
        
        Args:
            entity_summary: Entity summary dictionary
            output_dir: Directory to save results
            filename: Filename for saved results
            
        Returns:
            Path to the saved results
        """
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        output_path = os.path.join(output_dir, filename)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(entity_summary, f, ensure_ascii=False, indent=2)
        
        return output_path
