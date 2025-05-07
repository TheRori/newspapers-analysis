"""
Term Tracking Visualization Page for Dash app
"""

from dash import html, dcc
import dash_bootstrap_components as dbc
import sys
import pathlib

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