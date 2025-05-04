"""
Application Dash minimale pour tester si le serveur peut démarrer.
"""

import dash
from dash import html, dcc
import dash_bootstrap_components as dbc

# Créer une application Dash minimale
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.DARKLY])

# Définir un layout simple
app.layout = dbc.Container([
    html.H1("Application Dash Minimale", className="text-center my-4"),
    html.P("Cette application est un test pour vérifier si Dash peut démarrer correctement.", className="text-center"),
    html.Div([
        dbc.Button("Bouton de test", id="test-button", color="primary", className="me-2"),
        html.Div(id="output-div")
    ], className="text-center mt-4")
], fluid=True)

# Définir un callback simple
@app.callback(
    dash.Output("output-div", "children"),
    [dash.Input("test-button", "n_clicks")]
)
def update_output(n_clicks):
    if n_clicks is None:
        return "Cliquez sur le bouton pour tester le callback."
    return f"Le bouton a été cliqué {n_clicks} fois."

# Lancer l'application
if __name__ == "__main__":
    print("Démarrage de l'application minimale...")
    app.run(debug=False, port=8050)
    print("Application terminée.")
