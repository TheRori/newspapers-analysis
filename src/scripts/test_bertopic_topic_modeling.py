"""
Script de test pour le topic modeling avec BERTopic
"""
from bertopic import BERTopic
import json
import os

# Charger les articles depuis un fichier JSON
DATA_PATH = os.path.join(os.path.dirname(__file__), '../../data/processed/articles.json')

def load_articles(path, n=10):
    with open(path, encoding='utf-8') as f:
        data = json.load(f)
    # Extraire les contenus textuels ("content")
    docs = [article["content"] for article in data if "content" in article]
    return docs[:n]

def main():
    docs = load_articles(DATA_PATH, n=10)
    print(f"Loaded {len(docs)} articles.")
    topic_model = BERTopic(language="multilingual")
    topics, probs = topic_model.fit_transform(docs)

    print("Topics trouvés:")
    for i, doc in enumerate(docs):
        print(f"Doc: {doc[:100]}...\n  Topic: {topics[i]}")
    print("\nTop 5 topics:")
    print(topic_model.get_topic_info().head())

if __name__ == "__main__":
    main()
