"""
Composants de stockage (Store) pour le module de suivi de termes.
"""

from dash import html, dcc

def create_global_stores():
    """
    Crée les composants Store globaux pour différents contextes.
    
    Returns:
        html.Div: Un conteneur invisible contenant tous les composants Store.
    """
    return html.Div([
        # Store pour les articles du suivi de termes principal
        dcc.Store(id="stored-articles-term-tracking", storage_type="memory"),
        
        # Store pour les articles des termes similaires
        dcc.Store(id="stored-articles-similar-terms", storage_type="memory"),
        
        # Store pour les articles de la dérive sémantique
        dcc.Store(id="stored-articles-semantic-drift", storage_type="memory")
    ], style={"display": "none"})  # Rendre le conteneur invisible
