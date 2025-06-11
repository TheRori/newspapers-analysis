from dash import html
import dash_bootstrap_components as dbc

# Pour utiliser les icônes, assurez-vous d'inclure une feuille de style Font Awesome
# dans votre fichier principal app.py, par exemple :
# app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP, dbc.icons.FONT_AWESOME])

def get_enhanced_home_layout():
    return dbc.Container([
        # 1. Section "Héro" - Plus engageante et visuelle
        dbc.Container(
            [
                html.H1("Votre tableau de bord pour l'analyse de presse", className="display-4 text-white"),
                html.P(
                    "Explorez, analysez et comparez de grands corpus d'articles de presse grâce à des outils puissants d'analyse textuelle.",
                    className="lead text-white-50",
                ),
                html.Hr(className="my-2 bg-light"),
                html.P(
                    "Conçu pour les chercheurs, journalistes et toute personne curieuse des tendances médiatiques."
                ),
                dbc.Button("Commencer l'exploration", color="primary", size="lg", href="/bibliotheque-articles"), # Lien vers le module le plus pertinent pour commencer
            ],
            fluid=True,
            className="py-5 px-4 my-4 rounded-3",
            style={'backgroundColor': '#2c3e50'} # Une couleur de fond pour faire ressortir la section
        ),

        # 2. Modules d'Analyse présentés sous forme de Cartes avec Icônes
        html.H2("Boîte à outils d'analyse", className="text-center mb-4"),
        dbc.Row([
            # Carte Suivi de termes
            dbc.Col(dbc.Card([
                dbc.CardHeader(html.H4([
    html.I(className="fas fa-chart-line me-2"),
    "Suivi de termes"
])),
                dbc.CardBody("Visualisez l'évolution de mots ou expressions dans le temps et comparez leur usage.")
            ]), md=4, className="mb-4"),

            # Carte Modélisation de topics
            dbc.Col(dbc.Card([
                dbc.CardHeader(html.H4([
    html.I(className="fas fa-sitemap me-2"),
    "Modélisation de topics"
])),
                dbc.CardBody("Identifiez automatiquement les grands thèmes abordés dans le corpus et leur évolution.")
            ]), md=4, className="mb-4"),

            # Carte Analyse de sentiment
            dbc.Col(dbc.Card([
                dbc.CardHeader(html.H4([
    html.I(className="fas fa-smile-beam me-2"),
    "Analyse de sentiment"
])),
                dbc.CardBody("Mesurez le ton général (positif, négatif, neutre) des articles et son évolution.")
            ]), md=4, className="mb-4"),

            # Carte Entités nommées
            dbc.Col(dbc.Card([
                dbc.CardHeader(html.H4([
    html.I(className="fas fa-tags me-2"),
    "Entités nommées"
])),
                dbc.CardBody("Repérez les personnes, organisations, et lieux mentionnés dans les articles.")
            ]), md=4, className="mb-4"),

            # Carte Clustering
            dbc.Col(dbc.Card([
                dbc.CardHeader(html.H4([
    html.I(className="fas fa-project-diagram me-2"),
    "Clustering thématique"
])),
                dbc.CardBody("Regroupez les articles similaires pour explorer des familles de sujets.")
            ]), md=4, className="mb-4"),

            # Carte Analyse lexicale
            dbc.Col(dbc.Card([
                dbc.CardHeader(html.H4([
    html.I(className="fas fa-book-open me-2"),
    "Analyse lexicale"
])),
                dbc.CardBody("Statistiques sur le vocabulaire, la richesse lexicale, la longueur des textes, etc.")
            ]), md=4, className="mb-4"),
        ]),

        html.Hr(),

        # 3. Parcours Utilisateur - Guide de démarrage rapide et interactif
        html.H2("Guide de démarrage rapide", className="text-center mb-4"),
        dbc.Row([
            dbc.Col(html.Div(className="text-center", children=[
                html.Div(className="p-3 mb-2 bg-primary text-white rounded-circle d-inline-block", children=html.H3("1")),
                html.H5("Importez vos sources"),
                html.P("Ajoutez, vérifiez et gérez les articles qui constituent votre corpus d'analyse."),
                dbc.Button("Gérer les sources", href="/gestion-sources", color="secondary")
            ]), md=4),

            dbc.Col(html.Div(className="text-center", children=[
                html.Div(className="p-3 mb-2 bg-primary text-white rounded-circle d-inline-block", children=html.H3("2")),
                html.H5("Lancez une analyse"),
                html.P("Choisissez un ou plusieurs modules (topics, sentiment, entités...) pour traiter le corpus."),
                dbc.Button("Voir les analyses", href="/analyse-integre", color="secondary") # Exemple de lien
            ]), md=4),

            dbc.Col(html.Div(className="text-center", children=[
                html.Div(className="p-3 mb-2 bg-primary text-white rounded-circle d-inline-block", children=html.H3("3")),
                html.H5("Exportez les résultats"),
                html.P("Sauvegardez vos visualisations et données brutes pour les partager ou les réutiliser."),
                dbc.Button("Gestion des exports", href="/gestion-exports", color="secondary")
            ]), md=4),
        ]),

        html.Hr(className="my-5"),

        # 4. Footer
        html.Div("Pour toute aide ou question, consultez la documentation ou contactez l'équipe de développement.", className="text-center text-muted mb-4")

    ], fluid=True)