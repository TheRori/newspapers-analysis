#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Analyze topic distribution in doc_topic_matrix.json to verify it matches the logs.
"""

import json
import os
import numpy as np
from pathlib import Path
import argparse

def analyze_file(file_path):
    print(f"\nAnalyzing topic distribution in {file_path}")
    
    # Load the doc_topic_matrix.json file
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception as e:
        print(f"Error loading file: {e}")
        return
    
    # Print the structure of the data to understand it better
    print(f"Data structure: {type(data)}")
    if isinstance(data, dict):
        print(f"Keys in data: {list(data.keys())}")
    
    # Handle different possible structures
    documents = []
    if isinstance(data, list):
        documents = data
    elif isinstance(data, dict):
        if 'documents' in data:
            documents = data['documents']
        elif 'results' in data:
            documents = data['results']
        elif 'doc_topics' in data:
            # Convert doc_topics dictionary to list format
            documents = [{'doc_id': doc_id, 'topic_distribution': topic_data.get('topic_distribution', [])} 
                         for doc_id, topic_data in data['doc_topics'].items()]
        elif 'doc_topic_matrix' in data:
            documents = data['doc_topic_matrix']
        else:
            # Try to find any key that might contain a list of documents
            for key, value in data.items():
                if isinstance(value, list) and len(value) > 0 and isinstance(value[0], dict):
                    print(f"Using data from key: {key}")
                    documents = value
                    break
    
    if not documents:
        print("Could not find document data in the file")
        return
    
    print(f"Total documents: {len(documents)}")
    
    # Count documents by dominant topic
    topic_counts = {}
    
    # Check the structure of each document
    for doc in documents:
        if 'topic_distribution' not in doc:
            print(f"Document missing topic_distribution: {doc.get('doc_id', 'unknown')}")
            continue
        
        # Get the topic distribution
        topic_dist = doc['topic_distribution']
        
        # Find the dominant topic (index of max value)
        dominant_topic = np.argmax(topic_dist)
        
        # Count this document for its dominant topic
        topic_counts[dominant_topic] = topic_counts.get(dominant_topic, 0) + 1
        
        # Check for documents with very low probabilities across all topics
        max_prob = max(topic_dist)
        if max_prob < 0.01:  # Threshold for considering a document as an outlier
            topic_counts.get(-1, 0) + 1  # Count as outlier (topic -1)
    
    # Print the results
    print("\n--- Topic Distribution Analysis ---")
    num_topics = len(set(topic_counts.keys()) - {-1}) if -1 in topic_counts else len(topic_counts)
    print(f"Number of topics found: {num_topics}")
    
    # Sort topics by ID
    for topic_id in sorted(topic_counts.keys()):
        count = topic_counts[topic_id]
        percentage = (count / len(documents)) * 100
        print(f"Topic #{topic_id}: {count} documents ({percentage:.2f}%)")
    
    # Compare with the logs
    print("\n--- Comparison with Logs ---")
    log_counts = {
        0: 9153,
        1: 778,
        2: 265,
        3: 86,
        4: 14,
        5: 0
    }
    
    for topic_id, log_count in log_counts.items():
        actual_count = topic_counts.get(topic_id, 0)
        difference = actual_count - log_count
        print(f"Topic #{topic_id}: Log shows {log_count}, Found {actual_count}, Difference: {difference}")
    
    # Check if there's a topic 5 in the matrix
    if 5 not in topic_counts:
        print("Note: Topic #5 is not present in the doc_topic_matrix.json file")
    
    # Check the dimensions of topic distributions
    if documents:
        sample_dist = documents[0]['topic_distribution']
        print(f"\nTopic distribution dimensions: {len(sample_dist)} values per document")
        if len(sample_dist) == 5 and len(log_counts) == 6:
            print("Warning: Log shows 6 topics (including topic #5), but doc_topic_matrix.json only has 5 values per document")
            print("This suggests the doc_topic_matrix.json file might be outdated or from a different run")

def main():
    # Get the project root directory
    project_root = Path(__file__).resolve().parent.parent.parent
    
    # Path to the latest doc_topic_matrix.json file
    latest_path = project_root / "data" / "results" / "doc_topic_matrix.json"
    
    # Path to the versioned doc_topic_matrix file (from the logs)
    versioned_path = project_root / "data" / "results" / "doc_topic_matrix.json"
    
    # Analyze both files
    print("=== Analyzing Latest File ===")
    analyze_file(latest_path)
    
    print("\n=== Analyzing Versioned File ===")
    analyze_file(versioned_path)
    

    


if __name__ == "__main__":
    main()
