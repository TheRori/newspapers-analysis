"""
Script pour extraire les 10 premiers articles de chaque cluster/topic
et les exporter au format JSON pour le débogage.
"""

import os
import json
import sys
from pathlib import Path

# Ajouter le répertoire parent au path pour pouvoir importer les modules du projet
sys.path.append(str(Path(__file__).parent.parent))
from utils.config_loader import load_config

def main():
    # Charger la configuration
    project_root = Path(__file__).resolve().parents[1].parent
    config_path = os.path.join(project_root, "config", "config.yaml")
    config = load_config(config_path)
    
    # Chemins des fichiers
    results_dir = os.path.join(project_root, config['data']['results_dir'])
    doc_topic_matrix_path = os.path.join(results_dir, 'doc_topic_matrix.json')
    articles_path = os.path.join(project_root, 'data', 'processed', 'articles.json')
    output_path = os.path.join(results_dir, 'topic_clusters_debug.json')
    
    # Vérifier si les fichiers existent
    if not os.path.exists(doc_topic_matrix_path):
        print(f"Erreur: Le fichier doc_topic_matrix.json n'existe pas à l'emplacement {doc_topic_matrix_path}")
        return
    
    if not os.path.exists(articles_path):
        print(f"Erreur: Le fichier articles.json n'existe pas à l'emplacement {articles_path}")
        return
    
    # Charger les données
    print(f"Chargement de la matrice doc-topic depuis {doc_topic_matrix_path}...")
    with open(doc_topic_matrix_path, 'r', encoding='utf-8') as f:
        doc_topic_data = json.load(f)
    
    # Vérifier la structure du fichier
    if isinstance(doc_topic_data, dict) and 'doc_topic_matrix' in doc_topic_data:
        doc_topic_matrix = doc_topic_data['doc_topic_matrix']
    else:
        doc_topic_matrix = doc_topic_data
    
    print(f"Chargement des articles depuis {articles_path}...")
    with open(articles_path, 'r', encoding='utf-8') as f:
        articles = json.load(f)
    
    # Créer un dictionnaire pour accéder rapidement aux articles par ID
    article_dict = {}
    for article in articles:
        article_id = article.get('doc_id', article.get('id', ''))
        if article_id:
            article_dict[str(article_id)] = article
    
    # Déterminer le nombre de topics/clusters
    num_topics = 0
    if doc_topic_matrix and len(doc_topic_matrix) > 0:
        first_item = doc_topic_matrix[0]
        if 'topic_distribution' in first_item:
            num_topics = len(first_item['topic_distribution'])
    
    print(f"Nombre de topics/clusters détectés: {num_topics}")
    
    # Organiser les articles par cluster dominant
    clusters = {i: [] for i in range(num_topics)}
    
    for item in doc_topic_matrix:
        doc_id = item.get('doc_id', '')
        topic_distribution = item.get('topic_distribution', [])
        
        if not topic_distribution:
            continue
        
        # Déterminer le topic dominant
        dominant_topic = topic_distribution.index(max(topic_distribution))
        
        # Récupérer les informations de l'article
        article = article_dict.get(str(doc_id), {})
        
        # Ajouter l'article au cluster correspondant
        if article:
            clusters[dominant_topic].append({
                'doc_id': doc_id,
                'content': article.get('content', article.get('original_content', 'Contenu non disponible'))[:500] + '...',  # Limiter la taille
                'title': article.get('title', 'Sans titre'),
                'date': article.get('date', ''),
                'newspaper': article.get('newspaper', ''),
                'topic_value': max(topic_distribution)
            })
    
    # Préparer les données de sortie
    output_data = {}
    
    # Charger les noms de topics s'ils existent
    topic_names = {}
    advanced_topic_json = os.path.join(results_dir, 'advanced_topic', 'advanced_topic_analysis.json')
    if os.path.exists(advanced_topic_json):
        try:
            with open(advanced_topic_json, encoding='utf-8') as f:
                stats = json.load(f)
            
            if stats.get('topic_names_llm'):
                # Peut être string ou dict
                if isinstance(stats['topic_names_llm'], dict):
                    topic_names = stats['topic_names_llm']
                else:
                    try:
                        import ast
                        topic_names = ast.literal_eval(stats['topic_names_llm'])
                    except Exception:
                        topic_names = {}
            
            # Ajouter les mots-clés des topics s'ils existent
            if 'weighted_words' in stats:
                for topic_id, words in stats['weighted_words'].items():
                    topic_key = f'topic_{topic_id}'
                    if topic_key not in output_data:
                        output_data[topic_key] = {}
                    output_data[topic_key]['top_words'] = [word[0] for word in words[:10]]
        except Exception as e:
            print(f"Erreur lors du chargement des noms de topics: {e}")
    
    # Ajouter les articles pour chaque cluster
    for topic_id, articles in clusters.items():
        topic_key = f'topic_{topic_id}'
        
        if topic_key not in output_data:
            output_data[topic_key] = {}
        
        # Trier les articles par valeur de topic (du plus élevé au plus bas)
        sorted_articles = sorted(articles, key=lambda x: x['topic_value'], reverse=True)
        
        # Prendre les 10 premiers articles
        output_data[topic_key]['name'] = topic_names.get(topic_key, f"Topic {topic_id}")
        output_data[topic_key]['article_count'] = len(articles)
        output_data[topic_key]['top_articles'] = sorted_articles[:10]
    
    # Sauvegarder les données au format JSON
    print(f"Sauvegarde des données de débogage dans {output_path}...")
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)
    
    print(f"Terminé! Les données ont été sauvegardées dans {output_path}")
    
    # Afficher un résumé
    print("\nRésumé des clusters:")
    for topic_id in range(num_topics):
        topic_key = f'topic_{topic_id}'
        if topic_key in output_data:
            name = output_data[topic_key].get('name', f"Topic {topic_id}")
            count = output_data[topic_key].get('article_count', 0)
            top_words = output_data[topic_key].get('top_words', [])
            top_words_str = ", ".join(top_words) if top_words else "Non disponible"
            print(f"{name}: {count} articles - Mots-clés: {top_words_str}")

if __name__ == "__main__":
    main()
