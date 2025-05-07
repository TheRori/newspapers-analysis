"""
Fonctions de mise en page pour le module de suivi de termes.
"""

from dash import html, dcc
import dash_bootstrap_components as dbc
from pathlib import Path

from src.webapp.term_tracking.utils import (
    get_term_tracking_results,
    get_semantic_drift_results,
    get_similar_terms_results,
    get_term_tracking_args,
    get_term_files
)
from src.webapp.topic_filter_component import get_topic_filter_component

def get_term_tracking_layout():
    """
    Crée la mise en page pour la page de suivi de termes.
    
    Returns:
        Div HTML avec la mise en page
    """
    # Obtenir les résultats disponibles
    term_tracking_results = get_term_tracking_results()
    semantic_drift_results = get_semantic_drift_results()
    similar_terms_results = get_similar_terms_results()
    term_files = get_term_files()
    parser_args = get_term_tracking_args()
    
    # Créer les onglets
    tabs = dcc.Tabs([
        # Onglet pour lancer une analyse
        dcc.Tab(
            label="Lancer une analyse",
            children=[
                html.Div([
                    html.H4("Paramètres d'analyse"),
                    html.P("Configurez les paramètres pour l'analyse de suivi des termes."),
                    
                    # Afficher explicitement le sélecteur de fichier de termes en premier
                    html.Div([
                        html.H5("Fichier de termes"),
                        dbc.Row([
                            dbc.Col([
                                dcc.Dropdown(
                                    id="term-tracking-term-file-input",
                                    options=term_files,
                                    value=term_files[0]['value'] if term_files else None,
                                    placeholder="Sélectionnez un fichier de termes"
                                )
                            ], width=12),
                        ], className="mb-3"),
                        
                        html.H5("Type d'analyse"),
                        dbc.Row([
                            dbc.Col([
                                dbc.Checkbox(
                                    id="term-tracking-by-year-input",
                                    label="Agréger par année",
                                    value=False
                                )
                            ], width=4),
                            dbc.Col([
                                dbc.Checkbox(
                                    id="term-tracking-by-newspaper-input",
                                    label="Agréger par journal",
                                    value=False
                                )
                            ], width=4),
                            dbc.Col([
                                dbc.Checkbox(
                                    id="term-tracking-semantic-drift-input",
                                    label="Analyse de drift sémantique",
                                    value=False
                                )
                            ], width=4),
                        ], className="mb-3"),
                        
                        # Options pour l'analyse de drift sémantique (conditionnelles)
                        html.Div([
                            html.H5("Options d'analyse sémantique", className="mt-3"),
                            dbc.Row([
                                dbc.Col([
                                    dbc.Label("Type de période"),
                                    dbc.RadioItems(
                                        id="term-tracking-period-type-input",
                                        options=[
                                            {"label": "Année", "value": "year"},
                                            {"label": "Décennie", "value": "decade"},
                                            {"label": "Personnalisé", "value": "custom"}
                                        ],
                                        value="decade",
                                        inline=True
                                    )
                                ], width=6),
                                dbc.Col([
                                    dbc.Label("Périodes personnalisées (format JSON)"),
                                    dbc.Textarea(
                                        id="term-tracking-custom-periods-input",
                                        placeholder="[[1800, 1850], [1851, 1900], [1901, 1950], [1951, 2000]]",
                                        rows=2
                                    )
                                ], width=6),
                            ], className="mb-3"),
                            dbc.Row([
                                dbc.Col([
                                    dbc.Label("Taille des vecteurs"),
                                    dbc.Input(
                                        id="term-tracking-vector-size-input",
                                        type="number",
                                        value=100
                                    )
                                ], width=4),
                                dbc.Col([
                                    dbc.Label("Taille de fenêtre"),
                                    dbc.Input(
                                        id="term-tracking-window-input",
                                        type="number",
                                        value=5
                                    )
                                ], width=4),
                                dbc.Col([
                                    dbc.Label("Nombre min. d'occurrences"),
                                    dbc.Input(
                                        id="term-tracking-min-count-input",
                                        type="number",
                                        value=5
                                    )
                                ], width=4),
                            ], className="mb-3"),
                            dbc.Row([
                                dbc.Col([
                                    dbc.Checkbox(
                                        id="term-tracking-filter-redundant-input",
                                        label="Filtrer les termes redondants (ordi/ordinateur/lordinateur...)",
                                        value=True
                                    )
                                ], width=12),
                            ], className="mb-3"),
                        ], id="semantic-drift-options", style={"display": "none"}),
                        
                        html.H5("Fichier source", className="mt-3"),
                        dbc.Row([
                            dbc.Col([
                                dbc.InputGroup([
                                    dbc.Input(
                                        id="term-tracking-source-file-input",
                                        type="text",
                                        placeholder="Chemin vers le fichier JSON d'articles"
                                    ),
                                    dbc.Button("Parcourir", id="term-tracking-source-file-browse", color="secondary")
                                ]),
                                html.Small("Laissez vide pour utiliser le fichier par défaut de la configuration.", className="text-muted")
                            ], width=12),
                        ], className="mb-3"),
                        
                        html.H5("Autres options", className="mt-3"),
                        dbc.Row([
                            dbc.Col([
                                dbc.Label("Nom de l'analyse"),
                                dbc.Input(
                                    id="term-tracking-analysis-name",
                                    type="text",
                                    placeholder="Nom de l'analyse",
                                    value=""
                                )
                            ], width=6),
                            dbc.Col([
                                dbc.Label("Limite d'articles (0 = pas de limite)"),
                                dbc.Input(
                                    id="term-tracking-limit-input",
                                    type="number",
                                    placeholder="Limite d'articles",
                                    value=0
                                )
                            ], width=6),
                        ], className="mb-3"),
                        
                        # Composant de filtrage par cluster
                        html.Div([
                            html.H5("Filtrage par cluster", className="mt-4 mb-3"),
                            html.P("Filtrez les articles par cluster pour une analyse plus ciblée."),
                            
                            # Sélection du fichier de cluster
                            dbc.Row([
                                dbc.Col([
                                    dbc.Label("Fichier de clusters"),
                                    dcc.Dropdown(
                                        id="term-tracking-cluster-file-dropdown",
                                        options=[],  # Sera rempli dynamiquement
                                        placeholder="Sélectionnez un fichier de clusters",
                                        clearable=True
                                    )
                                ], width=12),
                            ], className="mb-3"),
                            
                            # Sélection du cluster
                            dbc.Row([
                                dbc.Col([
                                    dbc.Label("Cluster"),
                                    dcc.Dropdown(
                                        id="term-tracking-cluster-id-dropdown",
                                        options=[],  # Sera rempli dynamiquement
                                        placeholder="Sélectionnez un cluster",
                                        clearable=True
                                    )
                                ], width=12),
                            ], className="mb-3"),
                        ], style={"marginBottom": "20px"}),
                        
                        dbc.Button(
                            "Lancer l'analyse",
                            id="run-term-tracking-button",
                            color="primary",
                            className="mt-3"
                        ),
                        html.Div(id="term-tracking-run-output", className="mt-3")
                    ])
                ], style={"padding": "20px"})
            ]
        ),
        
        # Onglet de suivi de termes
        dcc.Tab(
            label="Suivi de termes",
            children=[
                html.Div([
                    html.H4("Suivi de termes"),
                    html.P("Visualisez les résultats du suivi de termes dans les articles."),
                    html.Hr(),
                    
                    # Sélection du fichier de résultats
                    html.Div([
                        html.Label("Sélectionner un fichier de résultats:"),
                        dcc.Dropdown(
                            id="term-tracking-results-dropdown",
                            options=term_tracking_results if term_tracking_results else [],
                            value=term_tracking_results[0]['value'] if term_tracking_results else None,
                            clearable=False,
                            style={"width": "100%"}
                        )
                    ], style={"marginBottom": "20px"}),
                    
                    # Sélection du type de visualisation
                    html.Div([
                        html.Label("Type de visualisation:"),
                        dcc.RadioItems(
                            id="term-tracking-viz-type",
                            options=[
                                {"label": "Graphique à barres", "value": "bar"},
                                {"label": "Graphique linéaire", "value": "line"},
                                {"label": "Carte de chaleur", "value": "heatmap"},
                                {"label": "Tableau", "value": "table"}
                            ],
                            value="bar",
                            labelStyle={"display": "block", "marginBottom": "5px"}
                        )
                    ], style={"marginBottom": "20px"}),
                    
                    # Conteneur pour les visualisations
                    html.Div(id="term-tracking-visualizations-container")
                ], style={"padding": "20px"})
            ]
        ),
        
        # Onglet de dérive sémantique
        dcc.Tab(
            label="Dérive sémantique",
            children=[
                html.Div([
                    html.H4("Dérive sémantique"),
                    html.P("Visualisez l'évolution du sens des termes au fil du temps."),
                    html.Hr(),
                    
                    # Sélection du fichier de résultats
                    html.Div([
                        html.Label("Sélectionner un fichier de résultats:"),
                        dcc.Dropdown(
                            id="semantic-drift-results-dropdown",
                            options=semantic_drift_results if semantic_drift_results else [],
                            value=semantic_drift_results[0]['value'] if semantic_drift_results else None,
                            clearable=False,
                            style={"width": "100%"}
                        )
                    ], style={"marginBottom": "20px"}),
                    
                    # Sélection du type de visualisation
                    html.Div([
                        html.Label("Type de visualisation:"),
                        dcc.RadioItems(
                            id="semantic-drift-viz-type",
                            options=[
                                {"label": "Graphique linéaire", "value": "line"},
                                {"label": "Carte de chaleur", "value": "heatmap"},
                                {"label": "Tableau", "value": "table"},
                                {"label": "Comparaison", "value": "comparison"}
                            ],
                            value="line",
                            labelStyle={"display": "block", "marginBottom": "5px"}
                        )
                    ], style={"marginBottom": "20px"}),
                    
                    # Conteneur pour les visualisations
                    html.Div(id="semantic-drift-visualizations-container")
                ], style={"padding": "20px"})
            ]
        ),
        
        # Onglet de termes similaires
        dcc.Tab(
            label="Termes similaires",
            children=[
                html.Div([
                    html.H4("Termes similaires"),
                    html.P("Explorez les mots les plus proches vectoriellement des termes analysés."),
                    html.Hr(),
                    
                    # Sélection du fichier de résultats
                    html.Div([
                        html.Label("Sélectionner un fichier de résultats:"),
                        dcc.Dropdown(
                            id="similar-terms-results-dropdown",
                            options=similar_terms_results if similar_terms_results else [],
                            value=similar_terms_results[0]['value'] if similar_terms_results else None,
                            clearable=False,
                            style={"width": "100%"}
                        )
                    ], style={"marginBottom": "20px"}),
                    
                    # Sélection du type de visualisation
                    html.Div([
                        html.Label("Type de visualisation:"),
                        dcc.RadioItems(
                            id="similar-terms-viz-type",
                            options=[
                                {"label": "Tableau", "value": "table"},
                                {"label": "Carte de chaleur", "value": "heatmap"},
                                {"label": "Réseau", "value": "network"}
                            ],
                            value="network",
                            labelStyle={"display": "block", "marginBottom": "5px"}
                        )
                    ], style={"marginBottom": "20px"}),
                    
                    # Conteneur pour les visualisations
                    html.Div(id="similar-terms-visualizations-container")
                ], style={"padding": "20px"})
            ]
        )
    ], style={"marginTop": "20px"})
    
    # Retourner la mise en page complète
    return html.Div([
        # Contenu principal
        html.Div([
            html.H2("Suivi de termes", className="mb-4"),
            tabs
        ]),
        
        # Modals pour afficher les articles et l'article complet
        dbc.Modal(
            [
                dbc.ModalHeader(dbc.ModalTitle("Articles"), close_button=True),
                dbc.ModalBody(id="articles-modal-body"),
                dbc.ModalFooter(
                    dbc.Button("Fermer", id="close-articles-modal", className="ms-auto")
                ),
            ],
            id="articles-modal",
            size="lg",
            is_open=False,
        ),
        
        dbc.Modal(
            [
                dbc.ModalHeader(dbc.ModalTitle("Article complet"), close_button=True),
                dbc.ModalBody(id="full-article-modal-body"),
                dbc.ModalFooter(
                    dbc.Button("Fermer", id="close-full-article-modal", className="ms-auto")
                ),
            ],
            id="full-article-modal",
            size="xl",
            is_open=False,
        )
    ], className="container-fluid")
