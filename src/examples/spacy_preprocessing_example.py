"""
Example script demonstrating the SpaCy preprocessing for topic modeling.
"""

import os
import sys
import json
import yaml
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from src.preprocessing import preprocess_with_spacy, SpacyPreprocessor
from src.preprocessing.data_loader import DataLoader


def load_config():
    """Load configuration from config file."""
    config_path = project_root / "config" / "config.yaml"
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main():
    """Run the example."""
    # Load configuration
    config = load_config()
    
    # Load articles data
    data_path = Path(config["data"]["processed"])
    articles_path = data_path / "articles.json"
    
    print(f"Loading articles from {articles_path}")
    with open(articles_path, "r", encoding="utf-8") as f:
        articles = json.load(f)
    
    # Take a sample article for demonstration
    sample_article = articles[0]
    print(f"\nSample article ID: {sample_article.get('id', 'Unknown')}")
    print(f"Title: {sample_article.get('title', 'No title')}")
    
    # Get the text to process
    text = sample_article.get('cleaned_text', sample_article.get('text', ''))
    if not text:
        print("No text found in the article")
        return
    
    print(f"\nOriginal text sample: {text[:200]}...\n")
    
    # Simple example with direct function call
    print("EXAMPLE 1: Direct function call")
    tokens = preprocess_with_spacy(text)
    print(f"Tokens with default settings (NOUN, PROPN, ADJ): {tokens[:20]}")
    
    # Example with different POS tags
    print("\nEXAMPLE 2: Including verbs")
    tokens_with_verbs = preprocess_with_spacy(text, allowed_pos={"NOUN", "PROPN", "ADJ", "VERB"})
    print(f"Tokens including VERB: {tokens_with_verbs[:20]}")
    
    # Example with SpacyPreprocessor class
    print("\nEXAMPLE 3: Using SpacyPreprocessor class")
    preprocessor_config = {
        'spacy_model': 'fr_core_news_md',
        'allowed_pos': ["NOUN", "PROPN", "ADJ", "VERB"],
        'min_token_length': 3
    }
    preprocessor = SpacyPreprocessor(preprocessor_config)
    
    # Process a single document
    processed_doc = preprocessor.process_document(
        sample_article.copy(), 
        text_key="cleaned_text" if "cleaned_text" in sample_article else "text"
    )
    print(f"Tokens from preprocessor: {processed_doc['tokens'][:20]}")
    
    # Process a batch of documents
    print("\nEXAMPLE 4: Batch processing")
    sample_articles = articles[:5]
    processed_articles = preprocessor.process_documents(
        sample_articles, 
        text_key="cleaned_text" if "cleaned_text" in sample_article else "text"
    )
    
    print(f"Processed {len(processed_articles)} articles")
    for i, article in enumerate(processed_articles):
        print(f"Article {i+1} - Token count: {len(article['tokens'])}")
    
    print("\nIntegration with topic modeling:")
    print("1. Import the SpacyPreprocessor in your topic modeling script")
    print("2. Preprocess your documents to get tokens")
    print("3. Use these tokens directly in your topic modeling pipeline")
    print("""
Example integration code:
```python
from src.preprocessing import SpacyPreprocessor

# Initialize preprocessor
preprocessor = SpacyPreprocessor(config['preprocessing'])

# Preprocess documents
processed_docs = preprocessor.process_documents(documents)

# Now documents have 'tokens' field that can be used for topic modeling
# Use these tokens in your topic modeling pipeline
```
""")


if __name__ == "__main__":
    main()
