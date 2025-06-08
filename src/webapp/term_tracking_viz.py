"""
Term Tracking Visualization Page for Dash app
"""

from dash import html, dcc, Input, Output, State
import dash_bootstrap_components as dbc
import sys
import pathlib
import pandas as pd
import plotly.graph_objects as go
import traceback

# Add the project root to the path to allow imports from other modules
project_root = pathlib.Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from src.webapp.export_component import create_export_button, create_export_modal, create_feedback_toast, register_export_callbacks
from src.webapp.term_tracking.layout import get_term_tracking_layout as get_layout
from src.webapp.term_tracking.callbacks import register_term_tracking_callbacks as register_callbacks

# Fonctions pour obtenir le layout et enregistrer les callbacks
def get_term_tracking_layout():
    """
    Obtient le layout pour la page de suivi de termes.
    """
    return get_layout()

def register_term_tracking_callbacks(app):
    """
    Enregistre les callbacks pour la page de suivi de termes.
    """
    register_callbacks(app)
    
    # Callback pour mettre à jour le graphique du réseau sémantique lorsque l'utilisateur change la période
    @app.callback(
        Output("similar-terms-network-graph", "figure"),
        Input("similar-terms-period-selector", "value"),
        Input("similar-terms-results-dropdown", "value")
    )
    def update_similar_terms_network(selected_period, results_file):
        """
        Met à jour le graphique du réseau sémantique en fonction de la période sélectionnée.
        
        Args:
            selected_period: La période sélectionnée dans le dropdown
            results_file: Le fichier de résultats sélectionné
            
        Returns:
            La figure mise à jour du graphique réseau
        """
        print(f"DEBUG - update_similar_terms_network called with period={selected_period}, file={results_file}")
        
        if not selected_period or not results_file:
            print("DEBUG - Pas de période ou fichier sélectionné")
            # Retourner un graphique vide si aucune période ou fichier n'est sélectionné
            return go.Figure()
        
        try:
            # Nettoyer le chemin du fichier (supprimer les paramètres d'URL s'ils existent)
            if '?' in results_file:
                clean_file_path = results_file.split('?')[0]
                print(f"DEBUG - Nettoyage du chemin: {results_file} -> {clean_file_path}")
                results_file = clean_file_path
            
            # Charger les données du fichier de résultats
            print(f"DEBUG - Chargement du fichier {results_file}")
            df = pd.read_csv(results_file)
            print(f"DEBUG - Données chargées: {len(df)} lignes, colonnes: {df.columns.tolist()}")
            print(f"DEBUG - Périodes disponibles: {df['period'].unique().tolist()}")
            
            # Vérifier si la période sélectionnée existe dans les données
            if selected_period not in df['period'].unique():
                print(f"DEBUG - ERREUR: Période {selected_period} non trouvée dans les données")
                fig = go.Figure()
                fig.update_layout(
                    title=f"Période {selected_period} non trouvée dans les données",
                    xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                    yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
                )
                return fig
            
            # Filtrer les données pour la période sélectionnée
            period_df = df[df['period'] == selected_period]
            print(f"DEBUG - Données filtrées pour période {selected_period}: {len(period_df)} lignes")
            
            # Créer le graphique réseau avancé pour la période sélectionnée
            print("DEBUG - Création du graphique réseau avancé")
            from src.webapp.term_tracking.semantic_visualizations import create_advanced_network_graph
            fig = create_advanced_network_graph(df, selected_period)
            print("DEBUG - Graphique réseau créé avec succès")
            return fig
            
        except Exception as e:
            print(f"DEBUG - ERREUR lors de la mise à jour du graphique réseau: {str(e)}")
            import traceback
            print(f"DEBUG - Traceback: {traceback.format_exc()}")
            # Retourner un graphique vide en cas d'erreur
            fig = go.Figure()
            fig.update_layout(
                title="Erreur lors du chargement des données",
                annotations=[
                    dict(
                        text=str(e),
                        showarrow=False,
                        xref="paper",
                        yref="paper",
                        x=0.5,
                        y=0.5
                    )
                ]
            )
            return fig