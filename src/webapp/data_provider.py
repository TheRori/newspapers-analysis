import os
import json
from pathlib import Path
from typing import Dict, List, Any, Optional

import pandas as pd
import numpy as np

# Assurer que le chemin racine du projet est dans sys.path
project_root = Path(__file__).parent.parent.parent
import sys
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from src.utils.config_loader import load_config
from src.preprocessing.data_loader import DataLoader

class DashDataProvider:
    """
    Fournit des données à l'application web Dash.
    Gère le chargement et le traitement des données à partir des résultats d'analyse.
    """

    def __init__(self):
        """Initialise le fournisseur de données."""
        config_path = project_root / "config" / "config.yaml"
        self.config = load_config(str(config_path))
        self.results_dir = Path(self.config['data']['results_dir'])
        self.data_loader = DataLoader(self.config)
        self.project_root = project_root
        self.custom_sources = self._load_custom_sources()

    def _load_custom_sources(self) -> Dict:
        """Charge la configuration des sources personnalisées depuis un fichier JSON."""
        config_file = self.project_root / "config" / "custom_sources.json"
        if config_file.exists():
            try:
                with open(config_file, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception as e:
                print(f"Erreur lors du chargement de custom_sources.json : {e}")
        return {}

    def update_custom_sources(self, sources_config: Dict):
        """Met à jour et sauvegarde les chemins des fichiers sources personnalisés."""
        config_file = self.project_root / "config" / "custom_sources.json"
        try:
            with open(config_file, "w", encoding="utf-8") as f:
                json.dump(sources_config, f, indent=4)
            self.custom_sources = sources_config
            print(f"Sources personnalisées mises à jour et sauvegardées : {self.custom_sources}")
        except Exception as e:
            print(f"Erreur lors de la sauvegarde de custom_sources.json : {e}")

    def get_current_source_path(self) -> str:
        """Retourne le chemin du fichier source principal (articles de base)."""
        if self.custom_sources and self.custom_sources.get("main_source"):
            return self.custom_sources["main_source"]
        articles_filename = self.config.get('data', {}).get('files', {}).get('articles', "articles.json")
        return str(self.project_root / self.config['data']['processed_dir'] / articles_filename)

    def load_base_articles(self) -> List[Dict]:
        """Charge le corpus de base des articles."""
        articles_path = self.get_current_source_path()
        try:
            print(f"Chargement des articles depuis : {articles_path}")
            with open(articles_path, 'r', encoding='utf-8') as f:
                articles = json.load(f)
            print(f"Chargé {len(articles)} articles.")
            return articles
        except Exception as e:
            print(f"Erreur lors du chargement des articles depuis {articles_path} : {e}")
            return []

    def get_enriched_articles(self) -> List[Dict]:
        """
        Retourne la liste des articles enrichis avec tous les résultats d'analyse disponibles
        (topics, clusters, sentiment, entités).
        """
        articles = self.load_base_articles()
        if not articles:
            return []

        # Indexer les articles par ID pour une fusion efficace
        article_map = {str(article.get('id', article.get('base_id'))): article for article in articles}

        # --- Enrichissement avec Topic Modeling ---
        try:
            topic_model_dir = self.results_dir / "doc_topic_matrix"
            if topic_model_dir.exists():
                topic_files = sorted(topic_model_dir.glob("doc_topic_matrix_*.json"), key=lambda x: x.stat().st_mtime, reverse=True)
                if topic_files:
                    with open(topic_files[0], 'r', encoding='utf-8') as f:
                        topic_data = json.load(f).get('doc_topics', {})
                    
                    topic_names = {}
                    
                    # Extraire l'identifiant unique de la matrice de topics (format: doc_topic_matrix_gensim_lda_20250623-231354_5c684c3b.json)
                    topic_matrix_file = topic_files[0]
                    matrix_filename = topic_matrix_file.name
                    
                    # Chercher un fichier topic_names avec le même identifiant
                    matrix_id = None
                    if "_" in matrix_filename:
                        # Extraire l'identifiant unique (date_hash) à la fin du nom de fichier
                        parts = matrix_filename.split('_')
                        if len(parts) >= 5:  # Au moins 5 parties pour avoir l'identifiant
                            # Les deux dernières parties sont généralement date_hash.json
                            matrix_id = f"{parts[-2]}_{parts[-1].replace('.json', '')}"
                    
                    if matrix_id:
                        # Chercher un fichier topic_names avec le même identifiant
                        topic_names_pattern = f"topic_names*{matrix_id}*.json"
                        matching_files = list(self.results_dir.glob(topic_names_pattern))
                        
                        if matching_files:
                            topic_names_path = matching_files[0]
                            print(f"[TOPICS] Chargement automatique des noms de topics depuis : {topic_names_path}")
                            try:
                                with open(topic_names_path, "r", encoding="utf-8") as f:
                                    topic_names = json.load(f).get("topic_names", {})
                            except Exception as e:
                                print(f"[TOPICS] Erreur lors du chargement des noms de topics : {e}")
                    
                    # Si aucun fichier correspondant n'est trouvé, essayer avec le fichier générique
                    if not topic_names:
                        topic_names_path = self.results_dir / "topic_names_llm.json"
                        if topic_names_path.exists():
                            print(f"[TOPICS] Utilisation du fichier générique : {topic_names_path}")
                            with open(topic_names_path, "r", encoding="utf-8") as f:
                                topic_names = json.load(f).get("topic_names", {})

                    for doc_id, data in topic_data.items():
                        if doc_id in article_map:
                            dist = data.get("topic_distribution")
                            dom_topic = data.get("dominant_topic")
                            if dom_topic is not None:
                                article_map[doc_id]['dominant_topic'] = dom_topic
                                article_map[doc_id]['topic_distribution'] = dist
                                article_map[doc_id]['nom_du_topic'] = topic_names.get(str(dom_topic), f"Topic {dom_topic}")
                                if isinstance(dist, list) and 0 <= dom_topic < len(dist):
                                    article_map[doc_id]['score_topic'] = dist[dom_topic]
        except Exception as e:
            print(f"Erreur lors de l'enrichissement avec les topics : {e}")

        # --- Enrichissement avec Clusters ---
        # Charger le fichier de clusters si disponible
        clusters_data = None
        if "clusters_source" in self.custom_sources and self.custom_sources["clusters_source"]:
            try:
                clusters_path = self.custom_sources["clusters_source"]
                print(f"[CLUSTERS] Chargement du fichier de clusters depuis : {clusters_path}")
                with open(clusters_path, "r", encoding="utf-8") as f:
                    clusters_data = json.load(f)
            except Exception as e:
                print(f"Erreur lors du chargement du fichier de clusters: {e}")
                clusters_data = None
        
        # Appliquer le mapping cluster selon le format du fichier
        if clusters_data:
            cluster_map = {}
            # Format 1 : mapping {cluster_id: [list d'articles]}
            if all(isinstance(v, list) for v in clusters_data.values()):
                for cluster_id, articles_in_cluster in clusters_data.items():
                    for aid in articles_in_cluster:
                        cluster_map[aid] = cluster_id
                print(f"[CLUSTERS] Format mapping détecté. Nb clusters : {len(clusters_data)}")
            # Format 2 : doc_ids + labels
            elif "doc_ids" in clusters_data and "labels" in clusters_data:
                doc_ids = clusters_data["doc_ids"]
                labels = clusters_data["labels"]
                if len(doc_ids) == len(labels):
                    for aid, cluster_id in zip(doc_ids, labels):
                        cluster_map[aid] = cluster_id
                    print(f"[CLUSTERS] Format doc_ids/labels détecté. Nb clusters distincts : {len(set(labels))}")
                else:
                    print("[CLUSTERS] Erreur : doc_ids et labels de longueur différente !")
            else:
                print("[CLUSTERS] Format de fichier clusters non reconnu.")
            # Application du mapping au DataFrame
            for doc_id, article in article_map.items():
                if doc_id in cluster_map:
                    article_map[doc_id]['cluster'] = cluster_map[doc_id]
        # --- Enrichissement avec Sentiment ---
        try:
            sentiment_dir = self.results_dir / "sentiment"
            if sentiment_dir.exists():
                sentiment_files = sorted(sentiment_dir.glob("sentiment_scores_*.json"), key=lambda x: x.stat().st_mtime, reverse=True)
                if sentiment_files:
                    with open(sentiment_files[0], 'r', encoding='utf-8') as f:
                        sentiment_data = json.load(f).get('scores', {})
                    for doc_id, sentiment in sentiment_data.items():
                        if doc_id in article_map:
                            article_map[doc_id]['sentiment'] = sentiment
        except Exception as e:
            print(f"Erreur lors de l'enrichissement avec le sentiment : {e}")

        # --- Enrichissement avec Entités (NER) ---
        try:
            ner_dir = self.results_dir / "ner"
            if ner_dir.exists():
                ner_files = sorted(ner_dir.glob("ner_results_*.json"), key=lambda x: x.stat().st_mtime, reverse=True)
                if ner_files:
                    with open(ner_files[0], 'r', encoding='utf-8') as f:
                        ner_data = json.load(f).get('entities', {})
                    for doc_id, entities in ner_data.items():
                        if doc_id in article_map:
                            article_map[doc_id]['entities'] = entities
        except Exception as e:
            print(f"Erreur lors de l'enrichissement avec les entités NER : {e}")

        return list(article_map.values())

    def export_biblio_csv(self, csv_path: Optional[str] = None):
        """Exporte les articles enrichis dans un fichier CSV."""
        if csv_path is None:
            csv_path = self.project_root / "data" / "biblio_enriched.csv"

        articles_enriched = self.get_enriched_articles()
        if not articles_enriched:
            print("Aucun article enrichi à exporter.")
            return

        df = pd.DataFrame(articles_enriched)

        export_rows = []
        # --- Charger les noms de topics depuis le fichier correspondant à la matrice de topics ---
        topic_names = {}
        topic_matrix_source = None
        if "topic_matrix_source" in self.custom_sources and self.custom_sources["topic_matrix_source"]:
            topic_matrix_source = self.custom_sources["topic_matrix_source"]
            # Extraire l'identifiant unique de la matrice de topics
            matrix_path = Path(topic_matrix_source)
            matrix_filename = matrix_path.name
            
            # Chercher un fichier topic_names avec le même identifiant
            matrix_id = None
            if "_" in matrix_filename:
                # Extraire l'identifiant unique (date_hash) à la fin du nom de fichier
                parts = matrix_filename.split('_')
                if len(parts) >= 5:  # Au moins 5 parties pour avoir l'identifiant
                    # Les deux dernières parties sont généralement date_hash.json
                    matrix_id = f"{parts[-2]}_{parts[-1].replace('.json', '')}"
            
            if matrix_id:
                # Chercher un fichier topic_names avec le même identifiant
                topic_names_pattern = f"topic_names*{matrix_id}*.json"
                matching_files = list(self.results_dir.glob(topic_names_pattern))
                
                if matching_files:
                    topic_names_path = matching_files[0]
                    print(f"[TOPICS] Chargement automatique des noms de topics depuis : {topic_names_path}")
                    try:
                        with open(topic_names_path, "r", encoding="utf-8") as f:
                            topic_names = json.load(f).get("topic_names", {})
                    except Exception as e:
                        print(f"[TOPICS] Erreur lors du chargement des noms de topics : {e}")
            
            # Si aucun fichier correspondant n'est trouvé, essayer avec le fichier générique
            if not topic_names:
                topic_names_path = self.results_dir / "topic_names_llm.json"
                if topic_names_path.exists():
                    print(f"[TOPICS] Utilisation du fichier générique : {topic_names_path}")
                    with open(topic_names_path, "r", encoding="utf-8") as f:
                        topic_names = json.load(f).get("topic_names", {})
        
        # --- Charger le sentiment depuis le fichier sélectionné ou le plus récent ---
        sentiment_map = {}
        sentiment_source = None
        if "sentiment_source" in self.custom_sources and self.custom_sources["sentiment_source"]:
            sentiment_source = self.custom_sources["sentiment_source"]
        else:
            sentiment_dir = self.project_root / "data" / "results" / "sentiment_analysis"
            sentiment_files = sorted(sentiment_dir.glob("*.json"), key=lambda x: x.stat().st_mtime, reverse=True) if sentiment_dir.exists() else []
            if sentiment_files:
                sentiment_source = str(sentiment_files[0])
        
        if sentiment_source:
            try:
                print(f"[SENTIMENT] Chargement du fichier de sentiment depuis : {sentiment_source}")
                with open(sentiment_source, "r", encoding="utf-8") as f:
                    sentiments_json = json.load(f)
                # Index par id ou base_id
                for art in sentiments_json:
                    aid = art.get("id") or art.get("base_id")
                    sent = art.get("sentiment", {})
                    if isinstance(sent, dict) and "compound" in sent:
                        sentiment_map[aid] = round(float(sent["compound"]), 2)
            except Exception as e:
                print(f"[SENTIMENT] Erreur chargement sentiments : {e}")
        
        # --- Charger les entités depuis le fichier sélectionné ou le plus récent ---
        entity_map = {}
        entity_source = None
        if "entity_source" in self.custom_sources and self.custom_sources["entity_source"]:
            entity_source = self.custom_sources["entity_source"]
        else:
            entity_dir = self.project_root / "data" / "results" / "entity_recognition"
            entity_files = sorted(entity_dir.glob("*.json"), key=lambda x: x.stat().st_mtime, reverse=True) if entity_dir.exists() else []
            if entity_files:
                entity_source = str(entity_files[0])
        
        if entity_source:
            try:
                print(f"[ENTITIES] Chargement du fichier d'entités depuis : {entity_source}")
                with open(entity_source, "r", encoding="utf-8") as f:
                    entities_json = json.load(f)
                # Traiter selon le format du fichier d'entités
                if isinstance(entities_json, list):
                    # Format liste d'articles avec entités
                    for art in entities_json:
                        aid = art.get("id") or art.get("base_id")
                        if aid and "entities" in art and isinstance(art["entities"], list):
                            entity_map[aid] = art["entities"]
                elif isinstance(entities_json, dict) and "entities" in entities_json:
                    # Format {"entities": {"doc_id": [entities]}}  
                    entity_map = entities_json["entities"]
            except Exception as e:
                print(f"[ENTITIES] Erreur chargement entités : {e}")
        
        for _, row in df.iterrows():
            score_topic = row.get("score_topic")
            score_topic_pct = f"{float(score_topic)*100:.2f}%" if pd.notna(score_topic) else None

            # Récupérer l'ID de l'article
            article_id = row.get("id") or row.get("base_id")
            
            # --- Traitement des entités ---
            entities = None
            entities_org = []
            entities_loc = []
            entities_str = ""
            
            # Priorité aux entités du fichier sélectionné
            if article_id in entity_map:
                entities = entity_map[article_id]
            else:
                entities = row.get("entities")
                
            if isinstance(entities, list):
                # Extraire toutes les entités
                entities_str = ", ".join(sorted(set(e.get('text', '') for e in entities if isinstance(e, dict))))
                
                # Extraire spécifiquement les entités ORG et LOC
                for e in entities:
                    if isinstance(e, dict) and 'text' in e and 'label' in e:
                        if e['label'] == 'ORG':
                            entities_org.append(e['text'])
                        elif e['label'] == 'LOC':
                            entities_loc.append(e['text'])
            else:
                entities_str = str(entities) if pd.notna(entities) else None

            # Convertir les listes en chaînes de caractères
            entities_org_str = ", ".join(sorted(set(entities_org))) if entities_org else None
            entities_loc_str = ", ".join(sorted(set(entities_loc))) if entities_loc else None

            # --- Sentiment : priorité au mapping, sinon colonne existante ---
            sentiment_val = sentiment_map.get(article_id)
            
            # Normaliser le nom du journal pour le regroupement
            journal_name = row.get("newspaper")
            journal_base = None
            if journal_name:
                import re
                # Fonction de normalisation des noms de journaux
                def normalize_journal_name(name):
                    # D'abord enlever les chiffres à la fin
                    name = re.sub(r'\s*\d+\s*$', '', name)
                    
                    # Normalisation spécifique pour certains journaux
                    if re.search(r'(?i)l[\'’]\s*impartial', name):
                        return "L'Impartial"
                    elif re.search(r'(?i)solidarit[eé]', name):
                        return "Solidarité"
                    
                    # Retourner le nom normalisé
                    return name
                
                journal_base = normalize_journal_name(journal_name)
                
            # Récupérer le topic dominant et utiliser le nom du topic chargé depuis le fichier correspondant
            dominant_topic = row.get("dominant_topic")
            topic_name = "Non défini"
            if dominant_topic is not None:
                # Utiliser le nom du topic depuis le fichier topic_names correspondant
                topic_name = topic_names.get(str(dominant_topic), f"Topic {dominant_topic}")
            else:
                # Fallback sur la valeur existante si disponible
                topic_name = row.get("nom_du_topic", "Non défini")
            
            export_rows.append({
                "id_article": article_id,
                "titre": row.get("title"),
                "date": row.get("date"),
                "journal": row.get("newspaper"),
                "journal_base": journal_base,  # Ajouter le nom de base du journal
                "nom_du_topic": topic_name,
                "score_topic": score_topic_pct,
                "cluster": row.get("cluster"),
                "sentiment": sentiment_val,
                "entities": entities_str,
                "entities_org": entities_org_str,
                "entities_loc": entities_loc_str,
            })
        
        export_df = pd.DataFrame(export_rows)
        
        final_columns = ["id_article", "titre", "date", "journal", "journal_base", "nom_du_topic", "score_topic", "cluster", "sentiment", "entities", "entities_org", "entities_loc"]
        for col in final_columns:
            if col not in export_df.columns:
                export_df[col] = None
        
        export_df = export_df[final_columns]
        
        export_df.to_csv(csv_path, index=False, encoding="utf-8")
        print(f"Bibliothèque CSV exportée avec succès vers : {csv_path}")
        return str(csv_path)

    def get_latest_doc_clusters(self) -> pd.DataFrame:
        """Charge le dernier fichier de clusters et retourne un DataFrame (doc_id, cluster)."""
        clusters_dir = self.results_dir / "clusters"
        if not clusters_dir.exists():
            return pd.DataFrame(columns=["doc_id", "cluster"])
            
        cluster_files = sorted(clusters_dir.glob("doc_clusters_k*.json"), key=lambda x: x.stat().st_mtime, reverse=True)
        if not cluster_files:
            return pd.DataFrame(columns=["doc_id", "cluster"])
            
        with open(cluster_files[0], "r", encoding="utf-8") as f:
            cluster_data = json.load(f)
            
        doc_ids = cluster_data.get("doc_ids") or cluster_data.get("documents")
        labels = cluster_data.get("labels")
        
        if doc_ids and labels and len(doc_ids) == len(labels):
            return pd.DataFrame({"doc_id": doc_ids, "cluster": labels})
        
        return pd.DataFrame(columns=["doc_id", "cluster"])
        
    def get_available_analyses(self) -> List[Dict[str, str]]:
        """Get a list of available analyses for the dropdown."""
        analyses = []
        if (self.results_dir / "doc_topic_matrix").exists():
            analyses.append({"label": "Topic Modeling", "value": "topic_modeling"})
        if (self.results_dir / "ner").exists():
            analyses.append({"label": "Named Entity Recognition", "value": "ner"})
        if (self.results_dir / "sentiment").exists():
            analyses.append({"label": "Sentiment Analysis", "value": "sentiment"})
        if (self.results_dir / "classification").exists():
            analyses.append({"label": "Text Classification", "value": "classification"})
        return analyses

    def get_topic_modeling_data(self, num_topics: int = 5) -> Dict[str, Any]:
        """Get topic modeling data for visualization."""
        topic_path = self.results_dir / "doc_topic_matrix" / "topic_model_results.json"
        if not topic_path.exists():
            return self._get_placeholder_topic_data(num_topics)
        try:
            with open(topic_path, "r", encoding="utf-8") as f:
                topic_data = json.load(f)
            return {
                "topic_keywords": topic_data.get("topic_keywords", {}),
                "document_topics": topic_data.get("document_topics", {}),
                "topic_coherence": topic_data.get("coherence_score", 0.0),
                "model_info": topic_data.get("model_info", {})
            }
        except (FileNotFoundError, json.JSONDecodeError):
            return self._get_placeholder_topic_data(num_topics)

    def get_ner_data(self, entity_types: List[str] = None, top_n: int = 15) -> Dict[str, Any]:
        """Get named entity recognition data for visualization."""
        if entity_types is None:
            entity_types = ["PERSON", "ORG", "GPE"]
        ner_path = self.results_dir / "ner" / "entity_counts.json"
        if not ner_path.exists():
            return self._get_placeholder_ner_data(entity_types, top_n)
        try:
            with open(ner_path, "r", encoding="utf-8") as f:
                entity_data = json.load(f)
            filtered_entities = {}
            for entity_type in entity_types:
                if entity_type in entity_data:
                    sorted_entities = sorted(entity_data[entity_type].items(), key=lambda x: x[1], reverse=True)[:top_n]
                    filtered_entities[entity_type] = dict(sorted_entities)
            return {"entity_counts": filtered_entities, "entity_types": entity_types}
        except (FileNotFoundError, json.JSONDecodeError):
            return self._get_placeholder_ner_data(entity_types, top_n)

    def get_sentiment_data(self) -> Dict[str, Any]:
        """Get sentiment analysis data for visualization."""
        sentiment_path = self.results_dir / "sentiment" / "sentiment_scores.json"
        if not sentiment_path.exists():
            return self._get_placeholder_sentiment_data()
        try:
            with open(sentiment_path, "r", encoding="utf-8") as f:
                sentiment_data = json.load(f)
            scores = sentiment_data.get("document_scores", {})
            df = pd.DataFrame([{"document_id": doc_id, "score": score} for doc_id, score in scores.items()])
            return {
                "sentiment_data": df.to_dict(orient="records"),
                "average_score": df["score"].mean(),
                "model_info": sentiment_data.get("model_info", {})
            }
        except (FileNotFoundError, json.JSONDecodeError):
            return self._get_placeholder_sentiment_data()

    def get_classification_data(self) -> Dict[str, Any]:
        """Get text classification data for visualization."""
        classification_path = self.results_dir / "classification" / "classification_results.json"
        if not classification_path.exists():
            return self._get_placeholder_classification_data()
        try:
            with open(classification_path, "r", encoding="utf-8") as f:
                classification_data = json.load(f)
            return {
                "category_counts": classification_data.get("categories", {}),
                "confusion_matrix": classification_data.get("confusion_matrix", []),
                "accuracy": classification_data.get("accuracy", 0.0),
                "model_info": classification_data.get("model_info", {})
            }
        except (FileNotFoundError, json.JSONDecodeError):
            return self._get_placeholder_classification_data()

    def _get_placeholder_topic_data(self, num_topics: int = 5) -> Dict[str, Any]:
        """Generate placeholder data for topic modeling visualization."""
        words = ["government", "economy", "health", "education", "technology", "science", "politics", "business", "sports"]
        topic_keywords = {}
        for i in range(num_topics):
            selected_words = np.random.choice(words, 5, replace=False)
            weights = np.random.rand(5)
            topic_keywords[f"Topic {i+1}"] = {word: float(weight) for word, weight in zip(selected_words, weights)}
        return {"topic_keywords": topic_keywords, "document_topics": {}, "topic_coherence": 0.42, "model_info": {"algorithm": "Placeholder"}}

    def _get_placeholder_ner_data(self, entity_types: List[str], top_n: int) -> Dict[str, Any]:
        """Generate placeholder data for NER visualization."""
        entity_counts = {}
        entity_names = {"PERSON": ["John Smith", "Jane Doe"], "ORG": ["Google", "Microsoft"], "GPE": ["United States", "France"]}
        for entity_type in entity_types:
            if entity_type in entity_names:
                selected_entities = np.random.choice(entity_names[entity_type], min(top_n, len(entity_names[entity_type])), replace=False)
                counts = np.random.randint(10, 50, len(selected_entities))
                entity_counts[entity_type] = {entity: int(count) for entity, count in zip(selected_entities, counts)}
        return {"entity_counts": entity_counts, "entity_types": entity_types}

    def _get_placeholder_sentiment_data(self) -> Dict[str, Any]:
        """Generate placeholder data for sentiment visualization."""
        scores = np.random.normal(0.1, 0.3, 100)
        return {"sentiment_data": [{"score": float(s)} for s in scores], "average_score": float(scores.mean()), "model_info": {"model": "Placeholder"}}

    def _get_placeholder_classification_data(self) -> Dict[str, Any]:
        """Generate placeholder data for classification visualization."""
        categories = ["Politics", "Business", "Sports"]
        return {"category_counts": {cat: np.random.randint(10, 100) for cat in categories}, "confusion_matrix": [], "accuracy": 0.85, "model_info": {"model": "Placeholder"}}