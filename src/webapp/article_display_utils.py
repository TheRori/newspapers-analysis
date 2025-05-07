"""
Fonctions utilitaires pour l'affichage des articles à partir des graphiques.
Ce module permet d'ajouter une interactivité commune à toutes les visualisations
pour afficher les articles correspondants lorsqu'on clique sur un élément graphique.
"""

import json
import re
import os
from pathlib import Path
from dash import html, dcc, Input, Output, State, ctx, ALL, no_update
import dash_bootstrap_components as dbc
import pandas as pd

def load_config():
    """
    Charge la configuration depuis le fichier config.yaml.
    
    Returns:
        Dictionnaire de configuration
    """
    import yaml
    
    project_root = Path(__file__).resolve().parents[2]
    config_path = project_root / 'config' / 'config.yaml'
    
    with open(config_path, encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    return config

def get_articles_data(results_file=None):
    """
    Charge les données des articles depuis le fichier articles.json ou un fichier de résultats spécifique.
    
    Args:
        results_file: Chemin vers un fichier de résultats spécifique (optionnel)
        
    Returns:
        Liste des articles
    """
    project_root = Path(__file__).resolve().parents[2]
    config = load_config()
    
    # Si un fichier de résultats est spécifié, essayer de le charger en premier
    if results_file:
        try:
            print(f"Tentative de chargement du fichier de résultats: {results_file}")
            with open(results_file, 'r', encoding='utf-8') as f:
                articles = json.load(f)
            print(f"Fichier de résultats chargé avec succès: {len(articles)} articles")
            return articles
        except Exception as e:
            print(f"Erreur lors du chargement du fichier de résultats: {e}")
            # Continuer avec le chargement du fichier articles.json par défaut
    
    # Vérifier si le fichier de résultats est un fichier de résumé d'entités
    if results_file and 'entity_summary' in str(results_file):
        # Essayer de charger le fichier d'articles avec entités correspondant
        entity_file = str(results_file).replace('entity_summary', 'articles_with_entities')
        try:
            print(f"Tentative de chargement du fichier d'entités correspondant: {entity_file}")
            with open(entity_file, 'r', encoding='utf-8') as f:
                articles = json.load(f)
            print(f"Fichier d'entités chargé avec succès: {len(articles)} articles")
            return articles
        except Exception as e:
            print(f"Erreur lors du chargement du fichier d'entités: {e}")
            # Continuer avec les autres options
    
    # Essayer de charger les articles avec entités
    entity_dir = project_root / "data" / "results" / "entity_recognition"
    if entity_dir.exists():
        entity_files = list(entity_dir.glob("articles_with_entities_*.json"))
        if entity_files:
            # Trier par date de modification (le plus récent en premier)
            entity_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
            latest_file = entity_files[0]
            try:
                print(f"Tentative de chargement du fichier d'entités le plus récent: {latest_file}")
                with open(latest_file, 'r', encoding='utf-8') as f:
                    articles = json.load(f)
                print(f"Fichier d'entités chargé avec succès: {len(articles)} articles")
                return articles
            except Exception as e:
                print(f"Erreur lors du chargement du fichier d'entités: {e}")
                # Continuer avec le chargement des articles avec sentiment
    
    # Essayer de charger les articles avec sentiment
    sentiment_dir = project_root / "data" / "results" / "sentiment_analysis"
    if sentiment_dir.exists():
        sentiment_files = list(sentiment_dir.glob("articles_with_sentiment_*.json"))
        if sentiment_files:
            # Trier par date de modification (le plus récent en premier)
            sentiment_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
            latest_file = sentiment_files[0]
            try:
                print(f"Tentative de chargement du fichier de sentiment le plus récent: {latest_file}")
                with open(latest_file, 'r', encoding='utf-8') as f:
                    articles = json.load(f)
                print(f"Fichier de sentiment chargé avec succès: {len(articles)} articles")
                return articles
            except Exception as e:
                print(f"Erreur lors du chargement du fichier de sentiment: {e}")
                # Continuer avec le chargement du fichier articles.json par défaut
    
    # Charger le fichier articles.json par défaut
    articles_path = project_root / config['data']['processed_dir'] / "articles.json"
    try:
        print(f"Tentative de chargement du fichier articles.json par défaut: {articles_path}")
        with open(articles_path, 'r', encoding='utf-8') as f:
            articles = json.load(f)
        print(f"Fichier articles.json chargé avec succès: {len(articles)} articles")
        return articles
    except Exception as e:
        print(f"Erreur lors du chargement des articles: {e}")
        return []

def extract_article_metadata(article_id):
    """
    Extrait les métadonnées d'un article à partir de son ID.
    
    Args:
        article_id: ID de l'article
        
    Returns:
        Dictionnaire avec date, journal, et année
    """
    metadata = {
        'date': 'Date inconnue',
        'journal': 'Journal inconnu',
        'year': 'Année inconnue'
    }
    
    if not article_id:
        return metadata
    
    # Convertir en chaîne de caractères
    article_id = str(article_id)
    
    # Extraire les informations de l'ID
    if article_id.startswith('article_'):
        parts = article_id.split('_')
        if len(parts) >= 3:
            date_part = parts[1]
            journal_part = parts[2]
            
            metadata['date'] = date_part
            metadata['journal'] = journal_part
            
            # Extraire l'année
            if '-' in date_part:
                metadata['year'] = date_part.split('-')[0]
            else:
                metadata['year'] = date_part
    
    return metadata

def extract_excerpt(text, term=None, context_size=100):
    """
    Extrait un extrait de texte contenant le terme recherché.
    
    Args:
        text: Le texte complet
        term: Le terme à rechercher (optionnel)
        context_size: Nombre de caractères à inclure avant et après le terme
        
    Returns:
        Extrait de texte avec le terme mis en évidence
    """
    if not text:
        return ""
    
    if not term:
        # Si aucun terme n'est spécifié, retourner le début du texte
        return text[:300] + "..."
    
    # Créer un pattern insensible à la casse
    pattern = re.compile(re.escape(term), re.IGNORECASE)
    
    # Trouver la première occurrence du terme
    match = pattern.search(text)
    if not match:
        return text[:300] + "..."  # Retourner le début du texte si le terme n'est pas trouvé
    
    # Déterminer les indices de début et de fin de l'extrait
    start = max(0, match.start() - context_size)
    end = min(len(text), match.end() + context_size)
    
    # Extraire l'extrait
    excerpt = text[start:end]
    
    # Ajouter des ellipses si nécessaire
    if start > 0:
        excerpt = "..." + excerpt
    if end < len(text):
        excerpt = excerpt + "..."
    
    # Mettre en évidence le terme dans l'extrait
    highlighted_excerpt = pattern.sub(lambda m: f"**{m.group(0)}**", excerpt)
    
    return highlighted_excerpt

def filter_articles_by_criteria(articles, filter_type=None, filter_value=None, term=None):
    """
    Filtre les articles selon des critères spécifiques.
    
    Args:
        articles: Liste des articles à filtrer
        filter_type: Type de filtre (année, journal, sentiment, etc.)
        filter_value: Valeur du filtre
        term: Terme à rechercher dans le texte (optionnel)
        
    Returns:
        Liste des articles filtrés
    """
    print(f"Début du filtrage: {len(articles)} articles, filter_type={filter_type}, filter_value={filter_value}, term={term}")
    
    if not articles:
        print("Aucun article à filtrer")
        return []
    
    if not filter_type and not filter_value and not term:
        print("Aucun filtre appliqué, retour des 20 premiers articles")
        return articles[:20]  # Limiter à 20 articles si aucun filtre n'est appliqué
    
    filtered_articles = []
    match_count = 0
    
    # Vérifier si nous avons des articles avec sentiment
    has_sentiment = False
    for article in articles[:10]:  # Vérifier les 10 premiers articles
        if 'sentiment' in article:
            has_sentiment = True
            print(f"Exemple de sentiment: {article['sentiment']}")
            break
    
    if not has_sentiment and (filter_type == 'sentiment' or filter_type == 'sentiment_category' or filter_type == 'sentiment_score'):
        print("ATTENTION: Les articles ne semblent pas avoir d'attribut 'sentiment'")
    
    for i, article in enumerate(articles):
        if i < 5 or i % 100 == 0:  # Log pour les 5 premiers articles et tous les 100 articles
            print(f"Traitement de l'article {i}/{len(articles)}: {article.get('id', article.get('base_id', ''))}")
        
        article_id = str(article.get('id', article.get('base_id', '')))
        
        # Extraire les informations de l'article
        metadata = extract_article_metadata(article_id)
        
        # Appliquer les filtres
        match = True
        match_reason = ""
        
        # Filtre par type
        if filter_type == 'année' and filter_value and str(filter_value) != metadata['year']:
            match = False
            match_reason = f"Année: {metadata['year']} != {filter_value}"
        elif filter_type in ['journal', 'newspaper'] and filter_value and filter_value.lower() != metadata['journal'].lower():
            match = False
            match_reason = f"Journal: {metadata['journal']} != {filter_value}"
        elif filter_type == 'date' and filter_value:
            # Filtre par date complète
            if str(filter_value) != metadata['date']:
                match = False
                match_reason = f"Date: {metadata['date']} != {filter_value}"
        elif filter_type == 'sentiment' or filter_type == 'sentiment_category':
            # Filtre par catégorie de sentiment (positif, négatif, neutre)
            sentiment = article.get('sentiment', {})
            if isinstance(sentiment, dict):
                compound = sentiment.get('compound', 0)
                if filter_value == 'positif' and compound < 0.05:
                    match = False
                    match_reason = f"Sentiment: {compound} < 0.05 (positif)"
                elif filter_value == 'négatif' and compound > -0.05:
                    match = False
                    match_reason = f"Sentiment: {compound} > -0.05 (négatif)"
                elif filter_value == 'neutre' and (compound <= -0.05 or compound >= 0.05):
                    match = False
                    match_reason = f"Sentiment: {compound} n'est pas neutre"
            else:
                # Si sentiment n'est pas un dictionnaire, on ne peut pas filtrer
                match = False
                match_reason = "Sentiment n'est pas un dictionnaire"
        elif filter_type == 'sentiment_score':
            # Filtre par score de sentiment (valeur numérique)
            sentiment = article.get('sentiment', {})
            if isinstance(sentiment, dict):
                try:
                    score = float(filter_value)
                    compound = sentiment.get('compound', 0)
                    # Considérer une marge d'erreur plus large de 0.1 pour le score
                    if abs(compound - score) > 0.1:
                        match = False
                        match_reason = f"Score: {compound} != {score} (diff > 0.1)"
                except (ValueError, TypeError):
                    match = False
                    match_reason = "Erreur de conversion du score"
            else:
                match = False
                match_reason = "Sentiment n'est pas un dictionnaire"
        elif filter_type == 'article_id' and filter_value:
            # Filtre par ID d'article
            if str(article_id) != str(filter_value):
                match = False
                match_reason = f"ID: {article_id} != {filter_value}"
        elif filter_type == 'entity_type' and filter_value:
            # Filtre par type d'entité (ORG, PER, LOC, etc.)
            entities = article.get('entities', [])
            entity_type_found = False
            for entity in entities:
                # Vérifier le champ 'label' qui contient le type d'entité
                if entity.get('label', '').upper() == filter_value.upper():
                    entity_type_found = True
                    break
            if not entity_type_found:
                match = False
                match_reason = f"Type d'entité {filter_value} non trouvé"
        elif filter_type == 'entity' and filter_value:
            # Filtre par entité nommée spécifique
            entities = article.get('entities', [])
            entity_found = False
            for entity in entities:
                if entity.get('text', '').lower() == filter_value.lower():
                    entity_found = True
                    break
            if not entity_found:
                match = False
                match_reason = f"Entité {filter_value} non trouvée"
        elif filter_type == 'topic' and filter_value:
            # Filtre par topic
            topics = article.get('topics', [])
            if isinstance(topics, list):
                if filter_value not in topics:
                    match = False
                    match_reason = f"Topic {filter_value} non trouvé dans {topics}"
            else:
                # Si topics est un dictionnaire
                if not topics or filter_value not in topics:
                    match = False
                    match_reason = f"Topic {filter_value} non trouvé"
        
        # Filtre par terme (sauf pour les filtres de sentiment et d'entités)
        if term and match:
            # Pour les filtres de sentiment ou d'entités, ne pas chercher le terme dans le texte
            if filter_type and ('sentiment' in filter_type.lower() or 'entity' in filter_type.lower()):
                # Ne rien faire, le filtre a déjà été appliqué
                pass
            else:
                # Pour les autres types de filtres, chercher le terme dans le texte
                text = article.get('text', article.get('content', article.get('cleaned_text', '')))
                if not text or not re.search(r'\b' + re.escape(term) + r'\b', text, re.IGNORECASE):
                    match = False
                    match_reason = f"Terme {term} non trouvé dans le texte"
        
        # Si tous les filtres sont passés, ajouter l'article
        if match:
            filtered_articles.append(article)
            match_count += 1
            if match_count <= 5:  # Log pour les 5 premiers articles correspondants
                print(f"Article correspondant: {article_id}")
        elif i < 10:  # Log pour les 10 premiers articles non correspondants
            print(f"Article non correspondant: {article_id} - Raison: {match_reason}")
    
    print(f"Fin du filtrage: {match_count}/{len(articles)} articles correspondants")
    return filtered_articles

def create_article_card(article, index, term=None):
    """
    Crée une carte pour afficher un résumé d'article.
    
    Args:
        article: Données de l'article
        index: Indice de l'article dans la liste
        term: Terme à mettre en évidence (optionnel)
        
    Returns:
        Composant dbc.Card avec le résumé de l'article
    """
    article_id = article.get('id', article.get('base_id', 'Inconnu'))
    title = article.get('title', 'Sans titre')
    
    # Extraire les métadonnées
    metadata = extract_article_metadata(article_id)
    date = metadata['date']
    journal = metadata['journal']
    
    # Récupérer le texte et l'URL
    text = article.get('text', article.get('content', article.get('cleaned_text', '')))
    url = article.get('url', '')
    
    # Récupérer les informations de sentiment si disponibles
    sentiment_info = None
    sentiment = article.get('sentiment', {})
    if isinstance(sentiment, dict) and 'compound' in sentiment:
        # Déterminer la catégorie de sentiment
        compound = sentiment['compound']
        if compound >= 0.05:
            sentiment_category = "positif"
            badge_color = "success"
        elif compound <= -0.05:
            sentiment_category = "négatif"
            badge_color = "danger"
        else:
            sentiment_category = "neutre"
            badge_color = "secondary"
            
        # Créer le badge de sentiment
        sentiment_info = html.Div([
            html.Span("Sentiment: ", className="mr-1"),
            dbc.Badge(f"{sentiment_category} ({compound:.3f})", color=badge_color, className="mr-1"),
        ], className="mt-2 mb-2")
    
    # Récupérer les entités nommées si disponibles
    entities_info = None
    entities = article.get('entities', [])
    if entities:
        # Regrouper les entités par type
        entities_by_type = {}
        for entity in entities:
            entity_type = entity.get('label', '')
            entity_text = entity.get('text', '')
            if entity_type and entity_text:
                if entity_type not in entities_by_type:
                    entities_by_type[entity_type] = []
                if entity_text not in entities_by_type[entity_type]:
                    entities_by_type[entity_type].append(entity_text)
        
        # Créer les badges pour chaque type d'entité
        entity_badges = []
        colors = {
            'LOC': 'primary',   # Lieux en bleu
            'ORG': 'warning',   # Organisations en orange
            'PER': 'success',   # Personnes en vert
            'MISC': 'info'      # Divers en bleu clair
        }
        
        for entity_type, entity_texts in entities_by_type.items():
            color = colors.get(entity_type, 'secondary')
            for entity_text in entity_texts[:5]:  # Limiter à 5 entités par type pour éviter de surcharger
                entity_badges.append(
                    dbc.Badge(f"{entity_text} ({entity_type})", 
                             color=color, 
                             className="mr-1 mb-1")
                )
            
            # Ajouter un badge +X si plus de 5 entités
            if len(entity_texts) > 5:
                entity_badges.append(
                    dbc.Badge(f"+{len(entity_texts) - 5} autres", 
                             color="secondary", 
                             className="mr-1 mb-1")
                )
        
        if entity_badges:
            entities_info = html.Div([
                html.Span("Entités détectées: ", className="mr-1"),
                html.Div(entity_badges, className="d-flex flex-wrap")
            ], className="mt-2 mb-2")
    
    # Créer un extrait du texte avec le terme mis en évidence
    excerpt = extract_excerpt(text, term)
    
    # Créer un lien vers l'article original si disponible
    article_link = None
    if url:
        article_link = html.A("Voir l'article original", href=url, target="_blank", className="btn btn-sm btn-primary mt-2 mb-2")
    
    # Créer la carte
    card = dbc.Card([
        dbc.CardHeader([
            html.H5(title, className="card-title"),
            html.H6(f"{date} - {journal}", className="card-subtitle text-muted")
        ]),
        dbc.CardBody([
            # Afficher le badge de sentiment s'il est disponible
            sentiment_info if sentiment_info else html.Div(),
            # Afficher les entités détectées si disponibles
            entities_info if entities_info else html.Div(),
            html.P(dcc.Markdown(excerpt), className="card-text"),
            html.Div([
                dbc.Button(
                    "Afficher l'article complet", 
                    id={'type': 'show-full-article', 'index': index},
                    color="link", 
                    className="mt-2"
                ),
                article_link if article_link else html.Div(),
            ], className="d-flex justify-content-between")
        ])
    ], className="mb-3")
    
    return card

def create_articles_modal(id_prefix=""):
    """
    Crée un modal pour afficher les articles correspondant à un critère.
    
    Args:
        id_prefix: Préfixe pour les IDs des composants (pour éviter les conflits)
        
    Returns:
        dbc.Modal pour afficher les articles
    """
    modal_id = f"{id_prefix}-articles-modal" if id_prefix else "articles-modal"
    body_id = f"{id_prefix}-articles-modal-body" if id_prefix else "articles-modal-body"
    close_id = f"{id_prefix}-close-articles-modal" if id_prefix else "close-articles-modal"
    
    return dbc.Modal(
        [
            dbc.ModalHeader([
                html.H4("Articles correspondants", className="modal-title"),
                dbc.Button("×", id=close_id, className="close")
            ]),
            dbc.ModalBody(id=body_id),
            dbc.ModalFooter([
                dbc.Button("Fermer", id=close_id, className="ml-auto")
            ]),
        ],
        id=modal_id,
        size="xl",
        scrollable=True,
    )

def create_full_article_modal(id_prefix=""):
    """
    Crée un modal pour afficher le contenu complet d'un article.
    
    Args:
        id_prefix: Préfixe pour les IDs des composants (pour éviter les conflits)
        
    Returns:
        dbc.Modal pour afficher l'article complet
    """
    modal_id = f"{id_prefix}-full-article-modal" if id_prefix else "full-article-modal"
    body_id = f"{id_prefix}-full-article-modal-body" if id_prefix else "full-article-modal-body"
    close_id = f"{id_prefix}-close-full-article-modal" if id_prefix else "close-full-article-modal"
    
    return dbc.Modal(
        [
            dbc.ModalHeader([
                html.H4("Article complet", className="modal-title"),
                dbc.Button("×", id=close_id, className="close")
            ]),
            dbc.ModalBody(id=body_id),
            dbc.ModalFooter([
                dbc.Button("Fermer", id=close_id, className="ml-auto")
            ]),
        ],
        id=modal_id,
        size="xl",
        scrollable=True,
    )

def register_articles_modal_callback(app, graph_id_pattern, id_prefix="", data_extraction_func=None):
    """
    Enregistre un callback pour afficher les articles lorsqu'on clique sur un graphique.
    
    Args:
        app: L'application Dash
        graph_id_pattern: Pattern pour les IDs des graphiques (ex: {'type': 'sentiment-graph', 'subtype': ALL})
        id_prefix: Préfixe pour les IDs des composants (pour éviter les conflits)
        data_extraction_func: Fonction personnalisée pour extraire les données du clic (optionnel)
    """
    modal_body_id = f"{id_prefix}-articles-modal-body" if id_prefix else "articles-modal-body"
    modal_id = f"{id_prefix}-articles-modal" if id_prefix else "articles-modal"
    close_id = f"{id_prefix}-close-articles-modal" if id_prefix else "close-articles-modal"
    
    @app.callback(
        Output(modal_body_id, "children"),
        Output(modal_id, "is_open"),
        [
            Input(graph_id_pattern, 'clickData'),
            Input(close_id, "n_clicks"),
        ],
        prevent_initial_call=True
    )
    def handle_articles_modal(click_data, close_clicks):
        # Vérifier si le callback a été déclenché
        if not ctx.triggered:
            return "", False
        
        trigger = ctx.triggered[0]
        prop_id = trigger['prop_id']
        
        # Si fermeture du modal
        if close_id in prop_id:
            return "", False
        
        # Récupérer les données du clic
        click_value = trigger['value']
        
        if click_value is None or 'points' not in click_value or not click_value['points']:
            return "", False
        
        try:
            # Extraire les informations du clic
            point = click_value['points'][0]
            
            # Utiliser la fonction personnalisée si fournie
            if data_extraction_func:
                filter_type, filter_value, term = data_extraction_func(point, prop_id)
            else:
                # Extraction par défaut
                filter_type = None
                filter_value = None
                term = None
                
                # Essayer d'extraire l'ID du graphique (sous forme de dictionnaire)
                try:
                    graph_id = json.loads(prop_id.split('.')[0])
                    subtype = graph_id.get('subtype', '')
                    
                    # Analyser le sous-type du graphique pour déterminer le type de filtre
                    if 'year' in subtype:
                        filter_type = "année"
                        filter_value = point.get('x')
                        # Si c'est un graphique à barres ou ligne, récupérer l'indice de la courbe pour le terme
                        if 'curveNumber' in point:
                            curve_index = point.get('curveNumber')
                            # Le terme sera déterminé plus tard
                            term = curve_index
                    elif 'journal' in subtype:
                        filter_type = "journal"
                        filter_value = point.get('x')
                        if 'curveNumber' in point:
                            curve_index = point.get('curveNumber')
                            term = curve_index
                    elif 'term' in subtype or 'pie' in subtype:
                        filter_type = "terme"
                        filter_value = point.get('label')
                        term = filter_value
                    elif 'sentiment' in subtype:
                        filter_type = "sentiment"
                        filter_value = point.get('x') or point.get('label')
                    elif 'entity' in subtype:
                        filter_type = "entité"
                        filter_value = point.get('x') or point.get('label')
                    elif 'topic' in subtype:
                        filter_type = "topic"
                        filter_value = point.get('x') or point.get('label')
                except:
                    pass
            
            # Déterminer le fichier de résultats à utiliser
            results_file = None
            if 'sentiment' in prop_id.lower():
                # Pour les graphiques de sentiment, utiliser le fichier de résultats de sentiment
                project_root = Path(__file__).resolve().parents[2]
                sentiment_dir = project_root / "data" / "results" / "sentiment_analysis"
                if sentiment_dir.exists():
                    sentiment_files = list(sentiment_dir.glob("articles_with_sentiment_*.json"))
                    if sentiment_files:
                        # Trier par date de modification (le plus récent en premier)
                        sentiment_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
                        results_file = str(sentiment_files[0])
                        print(f"Utilisation du fichier de résultats de sentiment: {results_file}")
            
            # Charger les articles
            articles = get_articles_data(results_file)
            
            # Filtrer les articles
            filtered_articles = filter_articles_by_criteria(
                articles=articles,
                filter_type=filter_type,
                filter_value=filter_value,
                term=term if isinstance(term, str) else None
            )
            
            # Limiter le nombre d'articles
            max_articles = 20
            show_limit_message = False
            if len(filtered_articles) > max_articles:
                filtered_articles = filtered_articles[:max_articles]
                show_limit_message = True
            
            if not filtered_articles:
                return html.P("Aucun article trouvé correspondant aux critères."), True
            
            # Générer le contenu
            article_cards = []
            stored_articles = []  # Pour stocker les articles pour le modal complet
            
            for i, article in enumerate(filtered_articles):
                # Stocker l'article pour le modal complet
                article_id = article.get('id', article.get('base_id', 'Inconnu'))
                title = article.get('title', 'Sans titre')
                metadata = extract_article_metadata(article_id)
                text = article.get('text', article.get('content', article.get('cleaned_text', '')))
                url = article.get('url', '')
                
                stored_articles.append({
                    'id': article_id,
                    'title': title,
                    'date': metadata['date'],
                    'journal': metadata['journal'],
                    'text': text,
                    'url': url
                })
                
                # Créer la carte de l'article
                card = create_article_card(article, i, term if isinstance(term, str) else None)
                article_cards.append(card)
            
            # Stocker les articles dans un composant Store pour les récupérer dans le callback du modal complet
            store = dcc.Store(id=f"{id_prefix}-stored-articles" if id_prefix else "stored-articles", data=stored_articles)
            
            limit_message = html.Div([
                html.Hr(),
                html.P(f"Affichage limité aux {max_articles} premiers articles.", className="text-muted")
            ]) if show_limit_message else html.Div()
            
            term_display = term if isinstance(term, str) else ""
            
            modal_content = html.Div([
                store,
                html.H4(f"Articles contenant '{term_display}'" if term_display else "Articles correspondants"),
                html.P(f"Filtre: {filter_type} = {filter_value}" if filter_type else ""),
                html.Hr(),
                html.Div(article_cards),
                limit_message
            ])
            
            return modal_content, True
        
        except Exception as e:
            print(f"Exception dans handle_articles_modal: {e}")
            return html.P(f"Erreur lors de la récupération des articles: {str(e)}"), True

def register_full_article_modal_callback(app, id_prefix=""):
    """
    Enregistre un callback pour afficher l'article complet dans un modal séparé.
    
    Args:
        app: L'application Dash
        id_prefix: Préfixe pour les IDs des composants (pour éviter les conflits)
    """
    modal_body_id = f"{id_prefix}-full-article-modal-body" if id_prefix else "full-article-modal-body"
    modal_id = f"{id_prefix}-full-article-modal" if id_prefix else "full-article-modal"
    close_id = f"{id_prefix}-close-full-article-modal" if id_prefix else "close-full-article-modal"
    stored_articles_id = f"{id_prefix}-stored-articles" if id_prefix else "stored-articles"
    
    @app.callback(
        Output(modal_body_id, "children"),
        Output(modal_id, "is_open"),
        [
            Input({"type": "show-full-article", "index": ALL}, "n_clicks"),
            Input(close_id, "n_clicks"),
        ],
        [
            State(stored_articles_id, "data")
        ],
        prevent_initial_call=True
    )
    def handle_full_article_modal(show_clicks, close_clicks, stored_articles):
        if not ctx.triggered:
            return "", False
            
        trigger = ctx.triggered[0]
        prop_id = trigger['prop_id']
        
        if close_id in prop_id:
            return "", False
        
        if not stored_articles:
            return html.P("Aucun article disponible."), True
        
        try:
            # Identifier l'index de l'article à afficher
            button_id = json.loads(prop_id.split('.')[0])
            article_index = button_id.get('index')
            
            if article_index is None or article_index >= len(stored_articles):
                return html.P("Article non trouvé."), True
            
            # Récupérer l'article
            article = stored_articles[article_index]
            
            article_id = article.get('id', 'Inconnu')
            title = article.get('title', 'Sans titre')
            date = article.get('date', 'Date inconnue')
            journal = article.get('journal', 'Journal inconnu')
            text = article.get('text', 'Contenu non disponible')
            url = article.get('url', '')
            
            # Récupérer les informations de sentiment si disponibles
            sentiment_info = None
            sentiment = article.get('sentiment', {})
            if isinstance(sentiment, dict) and 'compound' in sentiment:
                # Déterminer la catégorie de sentiment
                compound = sentiment['compound']
                if compound >= 0.05:
                    sentiment_category = "positif"
                    badge_color = "success"
                elif compound <= -0.05:
                    sentiment_category = "négatif"
                    badge_color = "danger"
                else:
                    sentiment_category = "neutre"
                    badge_color = "secondary"
                    
                # Créer le badge de sentiment
                sentiment_info = html.Div([
                    html.H6("Analyse de sentiment:", className="mt-3"),
                    html.Div([
                        dbc.Badge(f"{sentiment_category}", color=badge_color, className="mr-2"),
                        html.Span(f"Score: {compound:.3f}", className="mr-2"),
                        html.Span(f"Positif: {sentiment.get('positive', 0):.3f}", className="mr-2"),
                        html.Span(f"Négatif: {sentiment.get('negative', 0):.3f}", className="mr-2"),
                        html.Span(f"Neutre: {sentiment.get('neutral', 0):.3f}"),
                    ], className="d-flex align-items-center mb-3")
                ])
            
            # Créer un lien vers l'article original si disponible
            article_link = None
            if url:
                article_link = html.A("Voir l'article original", href=url, target="_blank", className="btn btn-primary mt-3")
            
            # Mettre en forme le contenu
            content = html.Div([
                html.H3(title, className="mb-2"),
                html.H5(f"{date} - {journal}", className="text-muted mb-4"),
                # Afficher le badge de sentiment s'il est disponible
                sentiment_info if sentiment_info else html.Div(),
                html.Hr(),
                dcc.Markdown(text, className="article-text"),
                article_link if article_link else html.Div()
            ])
            
            return content, True
            
        except Exception as e:
            print(f"Erreur lors de l'affichage de l'article complet : {str(e)}")
            return html.P(f"Erreur lors de l'affichage de l'article : {str(e)}"), True
