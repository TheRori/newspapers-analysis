"""
Data provider module for the Dash web application.
This module handles loading and processing data for visualizations.
"""

import os
import json
from pathlib import Path
from typing import Dict, List, Any, Optional, Union, Tuple

import pandas as pd
import numpy as np

# Add the project root to the path to allow imports from other modules
project_root = Path(__file__).parent.parent.parent
import sys
sys.path.append(str(project_root))

from src.utils.config_loader import load_config
from src.preprocessing.data_loader import DataLoader


class DashDataProvider:
    """
    Class for providing data to the Dash web application.
    This class handles loading and processing data from analysis results.
    """
    
    def __init__(self):
        """Initialize the data provider with configuration settings."""
        self.config = load_config()
        self.results_dir = Path(self.config['data']['results_dir'])
        self.data_loader = DataLoader(self.config)
        
    def get_available_analyses(self) -> List[Dict[str, str]]:
        """
        Get a list of available analyses for the dropdown.
        
        Returns:
            List of dictionaries with label and value for each analysis
        """
        # Check what analysis results are available in the results directory
        analyses = []
        
        # Topic modeling results
        topic_modeling_path = self.results_dir / "topic_modeling"
        if topic_modeling_path.exists():
            analyses.append({"label": "Topic Modeling", "value": "topic_modeling"})
        
        # NER results
        ner_path = self.results_dir / "ner"
        if ner_path.exists():
            analyses.append({"label": "Named Entity Recognition", "value": "ner"})
        
        # Sentiment analysis results
        sentiment_path = self.results_dir / "sentiment"
        if sentiment_path.exists():
            analyses.append({"label": "Sentiment Analysis", "value": "sentiment"})
        
        # Classification results
        classification_path = self.results_dir / "classification"
        if classification_path.exists():
            analyses.append({"label": "Text Classification", "value": "classification"})
        
        # If no analyses are available, return default options
        if not analyses:
            analyses = [
                {"label": "Topic Modeling", "value": "topic_modeling"},
                {"label": "Named Entity Recognition", "value": "ner"},
                {"label": "Sentiment Analysis", "value": "sentiment"},
                {"label": "Text Classification", "value": "classification"},
            ]
        
        return analyses
    
    def get_topic_modeling_data(self, num_topics: int = 5) -> Dict[str, Any]:
        """
        Get topic modeling data for visualization.
        
        Args:
            num_topics: Number of topics to include
            
        Returns:
            Dictionary containing topic modeling data
        """
        # Path to topic modeling results
        topic_path = self.results_dir / "topic_modeling"
        
        # Check if results exist
        if not topic_path.exists():
            # Return placeholder data
            return self._get_placeholder_topic_data(num_topics)
        
        try:
            # Load topic model results
            with open(topic_path / "topic_model_results.json", "r") as f:
                topic_data = json.load(f)
            
            # Process the data for visualization
            processed_data = {
                "topic_keywords": topic_data.get("topic_keywords", {}),
                "document_topics": topic_data.get("document_topics", {}),
                "topic_coherence": topic_data.get("coherence_score", 0.0),
                "model_info": topic_data.get("model_info", {})
            }
            
            return processed_data
        
        except (FileNotFoundError, json.JSONDecodeError):
            # Return placeholder data if there's an error
            return self._get_placeholder_topic_data(num_topics)
    
    def get_ner_data(self, entity_types: List[str] = None, top_n: int = 15) -> Dict[str, Any]:
        """
        Get named entity recognition data for visualization.
        
        Args:
            entity_types: Types of entities to include
            top_n: Number of top entities to include
            
        Returns:
            Dictionary containing NER data
        """
        if entity_types is None:
            entity_types = ["PERSON", "ORG", "GPE"]
        
        # Path to NER results
        ner_path = self.results_dir / "ner"
        
        # Check if results exist
        if not ner_path.exists():
            # Return placeholder data
            return self._get_placeholder_ner_data(entity_types, top_n)
        
        try:
            # Load NER results
            with open(ner_path / "entity_counts.json", "r") as f:
                entity_data = json.load(f)
            
            # Filter by entity type and get top N
            filtered_entities = {}
            for entity_type in entity_types:
                if entity_type in entity_data:
                    # Sort by count and take top N
                    sorted_entities = sorted(
                        entity_data[entity_type].items(),
                        key=lambda x: x[1],
                        reverse=True
                    )[:top_n]
                    filtered_entities[entity_type] = dict(sorted_entities)
            
            return {
                "entity_counts": filtered_entities,
                "entity_types": entity_types
            }
        
        except (FileNotFoundError, json.JSONDecodeError):
            # Return placeholder data if there's an error
            return self._get_placeholder_ner_data(entity_types, top_n)
    
    def get_sentiment_data(self) -> Dict[str, Any]:
        """
        Get sentiment analysis data for visualization.
        
        Returns:
            Dictionary containing sentiment data
        """
        # Path to sentiment results
        sentiment_path = self.results_dir / "sentiment"
        
        # Check if results exist
        if not sentiment_path.exists():
            # Return placeholder data
            return self._get_placeholder_sentiment_data()
        
        try:
            # Load sentiment results
            with open(sentiment_path / "sentiment_scores.json", "r") as f:
                sentiment_data = json.load(f)
            
            # Process the data for visualization
            scores = sentiment_data.get("document_scores", {})
            
            # Convert to DataFrame for easier processing
            df = pd.DataFrame([
                {"document_id": doc_id, "score": score}
                for doc_id, score in scores.items()
            ])
            
            # Add metadata if available
            metadata_path = self.results_dir / "metadata.json"
            if metadata_path.exists():
                with open(metadata_path, "r") as f:
                    metadata = json.load(f)
                
                # Create a DataFrame from metadata
                meta_df = pd.DataFrame([
                    {
                        "document_id": doc_id,
                        "date": meta.get("date", ""),
                        "source": meta.get("source", "")
                    }
                    for doc_id, meta in metadata.items()
                ])
                
                # Merge with sentiment data
                if not meta_df.empty:
                    df = pd.merge(df, meta_df, on="document_id", how="left")
            
            return {
                "sentiment_data": df.to_dict(orient="records"),
                "average_score": df["score"].mean(),
                "model_info": sentiment_data.get("model_info", {})
            }
        
        except (FileNotFoundError, json.JSONDecodeError):
            # Return placeholder data if there's an error
            return self._get_placeholder_sentiment_data()
    
    def get_classification_data(self) -> Dict[str, Any]:
        """
        Get text classification data for visualization.
        
        Returns:
            Dictionary containing classification data
        """
        # Path to classification results
        classification_path = self.results_dir / "classification"
        
        # Check if results exist
        if not classification_path.exists():
            # Return placeholder data
            return self._get_placeholder_classification_data()
        
        try:
            # Load classification results
            with open(classification_path / "classification_results.json", "r") as f:
                classification_data = json.load(f)
            
            # Process the data for visualization
            categories = classification_data.get("categories", {})
            confusion_matrix = classification_data.get("confusion_matrix", [])
            
            return {
                "category_counts": categories,
                "confusion_matrix": confusion_matrix,
                "accuracy": classification_data.get("accuracy", 0.0),
                "model_info": classification_data.get("model_info", {})
            }
        
        except (FileNotFoundError, json.JSONDecodeError):
            # Return placeholder data if there's an error
            return self._get_placeholder_classification_data()
    
    def _get_placeholder_topic_data(self, num_topics: int = 5) -> Dict[str, Any]:
        """Generate placeholder data for topic modeling visualization."""
        # Generate random topic keywords
        words = [
            "government", "economy", "health", "education", "technology",
            "science", "politics", "business", "sports", "entertainment",
            "environment", "society", "culture", "international", "local",
            "finance", "policy", "research", "development", "industry"
        ]
        
        topic_keywords = {}
        for i in range(num_topics):
            # Randomly select 10 words for each topic
            selected_words = np.random.choice(words, 10, replace=False)
            # Assign random weights
            weights = np.random.rand(10)
            weights = weights / weights.sum()
            topic_keywords[f"Topic {i+1}"] = {
                word: float(weight) for word, weight in zip(selected_words, weights)
            }
        
        # Generate random document-topic distributions
        document_topics = {}
        for i in range(20):  # 20 documents
            # Random topic distribution
            topic_dist = np.random.rand(num_topics)
            topic_dist = topic_dist / topic_dist.sum()
            document_topics[f"doc_{i}"] = {
                f"Topic {j+1}": float(weight)
                for j, weight in enumerate(topic_dist)
            }
        
        return {
            "topic_keywords": topic_keywords,
            "document_topics": document_topics,
            "topic_coherence": 0.42,  # Placeholder coherence score
            "model_info": {
                "algorithm": "LDA",
                "num_topics": num_topics,
                "max_df": 0.7,
                "min_df": 5
            }
        }
    
    def _get_placeholder_ner_data(self, entity_types: List[str], top_n: int) -> Dict[str, Any]:
        """Generate placeholder data for NER visualization."""
        entity_counts = {}
        
        # Common entity names for each type
        entity_names = {
            "PERSON": [
                "John Smith", "Jane Doe", "Michael Johnson", "Sarah Williams",
                "David Brown", "Emily Davis", "Robert Wilson", "Jennifer Jones",
                "William Taylor", "Elizabeth Anderson", "Richard Thomas", "Mary Jackson",
                "Joseph White", "Patricia Harris", "Charles Martin", "Linda Thompson"
            ],
            "ORG": [
                "Google", "Microsoft", "Apple", "Amazon", "Facebook", "Twitter",
                "IBM", "Intel", "Samsung", "Sony", "Toyota", "Honda", "Ford",
                "Walmart", "Target", "Coca-Cola", "Pepsi", "McDonald's", "Starbucks"
            ],
            "GPE": [
                "United States", "China", "Russia", "United Kingdom", "France",
                "Germany", "Japan", "Canada", "Australia", "India", "Brazil",
                "Mexico", "Italy", "Spain", "South Korea", "New York", "London",
                "Paris", "Tokyo", "Berlin", "Moscow", "Beijing", "Washington"
            ],
            "DATE": [
                "January", "February", "March", "April", "May", "June",
                "July", "August", "September", "October", "November", "December",
                "2020", "2021", "2022", "2023", "Monday", "Tuesday", "Wednesday"
            ]
        }
        
        for entity_type in entity_types:
            if entity_type in entity_names:
                # Select random entities and assign random counts
                selected_entities = np.random.choice(
                    entity_names[entity_type],
                    min(top_n, len(entity_names[entity_type])),
                    replace=False
                )
                
                # Generate random counts (higher for the first entities)
                base_counts = np.linspace(50, 10, len(selected_entities))
                random_variation = np.random.randint(-5, 6, len(selected_entities))
                counts = base_counts + random_variation
                
                entity_counts[entity_type] = {
                    entity: int(count)
                    for entity, count in zip(selected_entities, counts)
                }
        
        return {
            "entity_counts": entity_counts,
            "entity_types": entity_types
        }
    
    def _get_placeholder_sentiment_data(self) -> Dict[str, Any]:
        """Generate placeholder data for sentiment visualization."""
        # Generate random sentiment scores
        num_docs = 100
        # Normal distribution with slight positive bias
        scores = np.random.normal(0.1, 0.3, num_docs)
        # Clip to [-1, 1] range
        scores = np.clip(scores, -1, 1)
        
        # Generate random dates over the past year
        dates = pd.date_range(
            start=pd.Timestamp.now() - pd.Timedelta(days=365),
            end=pd.Timestamp.now(),
            periods=num_docs
        )
        
        # Generate random sources
        sources = np.random.choice(
            ["New York Times", "Washington Post", "CNN", "Fox News", "BBC"],
            num_docs
        )
        
        # Create data records
        data = [
            {
                "document_id": f"doc_{i}",
                "score": float(score),
                "date": date.strftime("%Y-%m-%d"),
                "source": source
            }
            for i, (score, date, source) in enumerate(zip(scores, dates, sources))
        ]
        
        return {
            "sentiment_data": data,
            "average_score": float(scores.mean()),
            "model_info": {
                "model": "VADER",
                "range": "[-1, 1]"
            }
        }
    
    def _get_placeholder_classification_data(self) -> Dict[str, Any]:
        """Generate placeholder data for classification visualization."""
        # Define categories
        categories = ["Politics", "Business", "Sports", "Technology", "Entertainment"]
        
        # Generate random counts for each category
        counts = np.random.randint(30, 150, len(categories))
        category_counts = {
            category: int(count)
            for category, count in zip(categories, counts)
        }
        
        # Generate a random confusion matrix
        confusion_matrix = []
        for i in range(len(categories)):
            row = []
            for j in range(len(categories)):
                if i == j:
                    # Higher values on the diagonal (correct predictions)
                    row.append(int(np.random.randint(15, 30)))
                else:
                    # Lower values off the diagonal (incorrect predictions)
                    row.append(int(np.random.randint(0, 10)))
            confusion_matrix.append(row)
        
        return {
            "category_counts": category_counts,
            "confusion_matrix": confusion_matrix,
            "accuracy": 0.78,  # Placeholder accuracy
            "model_info": {
                "model": "DistilBERT",
                "num_categories": len(categories)
            }
        }
