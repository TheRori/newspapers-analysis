"""
Fonctions de visualisation pour le module de suivi de termes.
"""

import re
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from dash import html, dcc, dash_table
from pathlib import Path
from src.webapp.term_tracking.utils import clean_file_path

def create_term_tracking_visualizations(results_file, viz_type="bar"):
    """
    Crée les visualisations pour le suivi de termes, en reprenant la logique et les IDs dynamiques de la version originale,
    mais en gardant la structure modulaire.
    """
    import pandas as pd
    import plotly.express as px
    from dash import html, dcc
    
    try:
        # Remove cache-busting parameter if present
        if isinstance(results_file, str) and '?' in results_file:
            clean_file_path = results_file.split('?')[0]
        else:
            clean_file_path = results_file
            
        df = pd.read_csv(clean_file_path)
        if df.empty:
            return html.Div("Aucun résultat à afficher.")

        key_column = df.columns[0]
        # Gestion du renommage de la colonne clé comme dans la version originale
        if key_column == 'key':
            first_key = df['key'].iloc[0]
            if isinstance(first_key, str) and len(first_key) > 20:
                key_type = "Article ID"
                df = df.rename(columns={'key': 'Article ID'})
            elif isinstance(first_key, (int, float)) or (isinstance(first_key, str) and first_key.isdigit()):
                key_type = "Année"
                df = df.rename(columns={'key': 'Année'})
            else:
                key_type = "Journal"
                df = df.rename(columns={'key': 'Journal'})
        else:
            key_type = key_column
        term_columns = df.columns[1:].tolist()

        # Remplacer l'extraction par regex : récupérer le vrai nom du journal depuis articles_v1.json
        import json
        import os
        articles_json_path = os.path.join("data", "processed", "articles_v1.json")
        if os.path.exists(articles_json_path):
            with open(articles_json_path, "r", encoding="utf-8") as f:
                articles = json.load(f)
            # mapping id/base_id -> newspaper
            id_to_newspaper = {}
            for article in articles:
                article_id = str(article.get("id") or article.get("base_id"))
                id_to_newspaper[article_id] = article.get("newspaper", "Inconnu")
            def get_newspaper(article_id):
                return id_to_newspaper.get(str(article_id), "Inconnu")
            if "Article ID" in df.columns:
                df["Journal"] = df["Article ID"].apply(get_newspaper)
        # Extraction automatique de la date si possible (conserve la logique précédente)
        if key_type == "Article ID" or (df.columns[0].startswith('article_')):
            if df.columns[0].startswith('article_'):
                df = df.rename(columns={df.columns[0]: 'Article ID'})
                key_type = "Article ID"
            df['Date'] = df['Article ID'].str.extract(r'article_(\d{4}-\d{2}-\d{2})_')
        # NB: la colonne Journal est maintenant issue du JSON, plus du parsing d'ID

        # === LOGIQUE STRICTEMENT IDENTIQUE À term_tracking_viz copy.py POUR LE GRAPHIQUE CHRONOLOGIQUE ===
        # Créer colonne Année à partir de Date si possible
        if 'Date' in df.columns:
            try:
                df['Date'] = pd.to_datetime(df['Date'])
                df['Année'] = df['Date'].dt.year
            except Exception:
                pass
        elif key_type == 'Article ID' and 'Article ID' in df.columns:
            # Extraire la date depuis l'ID et convertir en datetime
            df['Date'] = df['Article ID'].str.extract(r'article_(\d{4}-\d{2}-\d{2})_')[0]
            df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
            df['Année'] = df['Date'].dt.year
        # Groupby année et graphique chronologique
        year_graph = None
        if 'Année' in df.columns:
            df_by_year = df.groupby('Année')[term_columns].sum().reset_index()
            # === Affichage selon viz_type (identique à term_tracking_viz copy.py) ===
            if viz_type == "bar":
                fig_year = px.bar(
                    df_by_year,
                    x='Année',
                    y=term_columns,
                    title="Fréquence des termes par année",
                    labels={'value': 'Fréquence', 'variable': 'Terme'},
                    barmode='group'
                )
                year_graph = dcc.Graph(
                    id={'type': 'term-tracking-graph', 'subtype': 'year-bar'},
                    figure=fig_year
                )
            elif viz_type == "line":
                fig_year = px.line(
                    df_by_year,
                    x='Année',
                    y=term_columns,
                    title="Évolution des termes par année",
                    labels={'value': 'Fréquence', 'variable': 'Terme'}
                )
                year_graph = dcc.Graph(
                    id={'type': 'term-tracking-graph', 'subtype': 'year-line'},
                    figure=fig_year
                )
            elif viz_type == "heatmap":
                import numpy as np
                heatmap_data = df_by_year.set_index('Année')[term_columns].T
                fig_year = px.imshow(
                    heatmap_data,
                    labels=dict(x="Année", y="Terme", color="Occurrences"),
                    title="Carte de chaleur des occurrences par terme et par année",
                    aspect="auto"
                )
                year_graph = dcc.Graph(
                    id={'type': 'term-tracking-graph', 'subtype': 'year-heatmap'},
                    figure=fig_year
                )
            # On peut ajouter d'autres types ici si besoin

        # Visualisation par journal
        if 'Journal' in df.columns or key_type == 'Journal':
            if 'Journal' not in df.columns:
                df['Journal'] = df[key_column]
            journal_counts = df.groupby('Journal')[term_columns].sum().reset_index()
            fig_journal = px.bar(
                journal_counts.melt(id_vars=['Journal'], value_vars=term_columns),
                x='Journal', y='value', color='variable',
                title="Occurrences par journal et par terme",
                labels={'value': "Nombre d'occurrences", 'variable': 'Terme'}
            )
            journal_graph = dcc.Graph(
                id={'type': 'term-tracking-graph', 'subtype': 'journal-bar'},
                figure=fig_journal
            )
        else:
            journal_graph = None

        # Top articles (si Article ID)
        if 'Article ID' in df.columns:
            top_df = df.copy()
            top_df['Total'] = top_df[term_columns].sum(axis=1)
            top_df = top_df.nlargest(20, 'Total')
            top_df = top_df.drop(columns=['Total'])
            fig_top = px.bar(
                top_df.melt(id_vars=['Article ID'], value_vars=term_columns),
                x='Article ID', y='value', color='variable',
                title="Top 20 articles par fréquence totale de termes",
                labels={'value': 'Fréquence', 'variable': 'Terme'}
            )
            top_graph = dcc.Graph(
                id={'type': 'term-tracking-graph', 'subtype': 'article-bar'},
                figure=fig_top
            )
        else:
            top_graph = None

        # Répartition des termes (camembert)
        occurrences_by_term = df[term_columns].sum().reset_index()
        occurrences_by_term.columns = ['Terme', 'Occurrences']
        fig_pie = px.pie(
            occurrences_by_term,
            values='Occurrences', names='Terme',
            title="Répartition des occurrences par terme",
            hover_name='Terme'
        )
        pie_graph = dcc.Graph(
            id={'type': 'term-tracking-graph', 'subtype': 'term-pie'},
            figure=fig_pie
        )

        # Statistiques globales
        summary_table = html.Div([
            html.H4("Statistiques globales"),
            html.P(f"Nombre total d'occurrences: {occurrences_by_term['Occurrences'].sum()}", className="mb-0"),
            html.P(f"Nombre d'articles avec au moins un terme: {len(df)}", className="mb-0"),
            html.P(f"Nombre de journaux: {df['Journal'].nunique() if 'Journal' in df.columns else '-'}", className="mb-0"),
            html.P(f"Période couverte: {df['Date'].min()} - {df['Date'].max()}" if 'Date' in df.columns else "")
        ])

        # Assemblage du layout
        children = [
            comp for comp in [year_graph, journal_graph, top_graph, pie_graph, summary_table] if comp is not None
        ]
        return html.Div(children)

    except Exception as e:
        return html.Div([
            html.P(f"Erreur lors de la création des visualisations : {str(e)}", className="text-danger")
        ])

def create_filtered_term_tracking_visualizations(results_file, topic_results_path, topic_id, cluster_id, exclude_topic_id, exclude_cluster_id, viz_type="bar"):
    """
    Crée les visualisations pour le suivi de termes avec filtrage par topic/cluster.
    
    Args:
        results_file: Chemin vers le fichier de résultats de suivi de termes
        topic_results_path: Chemin vers le fichier de résultats de topic modeling
        topic_id: ID du topic à inclure
        cluster_id: ID du cluster à inclure
        exclude_topic_id: ID du topic à exclure
        exclude_cluster_id: ID du cluster à exclure
        viz_type: Type de visualisation (bar, line, heatmap, table)
        
    Returns:
        Composants Dash pour les visualisations
    """
    import pandas as pd
    import json
    import plotly.express as px
    from dash import html, dcc
    from pathlib import Path
    
    try:
        # Charger les données de suivi de termes
        # Remove cache-busting parameter if present
        if isinstance(results_file, str) and '?' in results_file:
            clean_file_path = results_file.split('?')[0]
        else:
            clean_file_path = results_file
            
        df = pd.read_csv(clean_file_path)
        if df.empty:
            return html.Div("Aucun résultat à afficher.")

        key_column = df.columns[0]
        # Gestion du renommage de la colonne clé comme dans la version originale
        if key_column == 'key':
            first_key = df['key'].iloc[0]
            if isinstance(first_key, str) and len(first_key) > 20:
                # C'est probablement un ID d'article, renommons la colonne
                df = df.rename(columns={'key': 'Article ID'})
                key_column = 'Article ID'
                
        # Charger les données de topic/cluster
        try:
            with open(topic_results_path, 'r', encoding='utf-8') as f:
                topic_data = json.load(f)
                
            # Extraire les articles par topic/cluster
            filtered_articles = set()
            
            # Filtrage par topic
            if topic_id and topic_id != "all":
                for topic in topic_data.get('topics', []):
                    if str(topic.get('id')) == str(topic_id):
                        filtered_articles.update(topic.get('articles', []))
            
            # Filtrage par cluster
            if cluster_id and cluster_id != "all":
                for cluster in topic_data.get('clusters', []):
                    if str(cluster.get('id')) == str(cluster_id):
                        filtered_articles.update(cluster.get('articles', []))
            
            # Exclusion par topic
            if exclude_topic_id and exclude_topic_id != "none":
                exclude_articles = set()
                for topic in topic_data.get('topics', []):
                    if str(topic.get('id')) == str(exclude_topic_id):
                        exclude_articles.update(topic.get('articles', []))
                if filtered_articles:
                    filtered_articles = filtered_articles - exclude_articles
                else:
                    # Si aucun filtre d'inclusion n'est actif, exclure de tous les articles
                    all_articles = set()
                    for topic in topic_data.get('topics', []):
                        all_articles.update(topic.get('articles', []))
                    filtered_articles = all_articles - exclude_articles
            
            # Exclusion par cluster
            if exclude_cluster_id and exclude_cluster_id != "none":
                exclude_articles = set()
                for cluster in topic_data.get('clusters', []):
                    if str(cluster.get('id')) == str(exclude_cluster_id):
                        exclude_articles.update(cluster.get('articles', []))
                if filtered_articles:
                    filtered_articles = filtered_articles - exclude_articles
                else:
                    # Si aucun filtre d'inclusion n'est actif, exclure de tous les articles
                    all_articles = set()
                    for cluster in topic_data.get('clusters', []):
                        all_articles.update(cluster.get('articles', []))
                    filtered_articles = all_articles - exclude_articles
            
            # Si aucun filtre n'est actif, utiliser tous les articles
            if not filtered_articles and (topic_id == "all" or not topic_id) and (cluster_id == "all" or not cluster_id) and (exclude_topic_id == "none" or not exclude_topic_id) and (exclude_cluster_id == "none" or not exclude_cluster_id):
                # Aucun filtre actif, utiliser tous les articles
                return create_term_tracking_visualizations(results_file, viz_type)
            
            # Filtrer le DataFrame par les articles sélectionnés
            if filtered_articles:
                df = df[df[key_column].isin(filtered_articles)]
            
            if df.empty:
                return html.Div("Aucun résultat à afficher avec les filtres sélectionnés.")
                
        except Exception as e:
            print(f"Erreur lors du chargement des données de topic/cluster: {e}")
            # En cas d'erreur, continuer sans filtrage
            pass
        
        # Utiliser la fonction existante pour créer les visualisations
        visualizations = create_term_tracking_visualizations(results_file, viz_type)
        
        # Ajouter une indication que les données sont filtrées
        filter_info = html.Div([
            html.H5("Données filtrées", className="mt-3 mb-2"),
            html.P([
                "Topic: ", html.Span(f"Inclure Topic {topic_id}" if topic_id and topic_id != "all" else "Tous", className="badge bg-primary me-2"),
                "Cluster: ", html.Span(f"Inclure Cluster {cluster_id}" if cluster_id and cluster_id != "all" else "Tous", className="badge bg-primary me-2"),
                "Exclure Topic: ", html.Span(f"Topic {exclude_topic_id}" if exclude_topic_id and exclude_topic_id != "none" else "Aucun", className="badge bg-danger me-2"),
                "Exclure Cluster: ", html.Span(f"Cluster {exclude_cluster_id}" if exclude_cluster_id and exclude_cluster_id != "none" else "Aucun", className="badge bg-danger")
            ])
        ], className="alert alert-info")
        
        # Insérer l'information de filtrage au début des visualisations
        if isinstance(visualizations, html.Div) and visualizations.children:
            if isinstance(visualizations.children, list):
                visualizations.children.insert(0, filter_info)
            else:
                visualizations.children = [filter_info, visualizations.children]
        
        return visualizations
        
    except Exception as e:
        return html.Div(f"Erreur lors de la création des visualisations filtrées: {str(e)}")

def extract_year_from_id(article_id):
    """
    Extrait l'année à partir de l'ID d'un article.
    
    Args:
        article_id: ID de l'article (format: article_YYYY-MM-DD_journal_XXXX_source)
        
    Returns:
        Année extraite ou 'Inconnu' si le format ne correspond pas
    """
    try:
        if isinstance(article_id, str) and article_id.startswith('article_'):
            parts = article_id.split('_')
            if len(parts) >= 2:
                date_part = parts[1]
                if '-' in date_part:
                    return date_part.split('-')[0]  # Prendre la première partie (année) de YYYY-MM-DD
        return 'Inconnu'
    except Exception:
        return 'Inconnu'

def extract_journal_from_id(article_id):
    """
    Extrait le nom du journal à partir de l'ID d'un article.
    
    Args:
        article_id: ID de l'article (format: article_YYYY-MM-DD_journal_XXXX_source)
        
    Returns:
        Nom du journal extrait ou 'Inconnu' si le format ne correspond pas
    """
    try:
        if isinstance(article_id, str) and article_id.startswith('article_'):
            parts = article_id.split('_')
            if len(parts) >= 3:
                return parts[2]  # Le journal est généralement la troisième partie
        return 'Inconnu'
    except Exception:
        return 'Inconnu'

# Fonctions utilitaires
