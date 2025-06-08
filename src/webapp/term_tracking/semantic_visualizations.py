"""
Fonctions de visualisation pour la dérive sémantique et les termes similaires.
"""

import re
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import networkx as nx
from dash import html, dcc, dash_table
from pathlib import Path
from src.webapp.term_tracking.utils import clean_file_path


def create_advanced_network_graph(df: pd.DataFrame, period: str, similarity_threshold: float = 0.5):
    """
    Creates an advanced semantic network visualization for a given period using
    a force-directed layout from networkx.

    Args:
        df: DataFrame containing the similar terms data.
            Expected columns: 'term', 'period', 'similar_word', 'similarity'.
        period: The specific period to visualize.
        similarity_threshold: Minimum similarity score to create edges between similar words.

    Returns:
        A Plotly Figure object representing the network graph.
    """
    print(f"DEBUG - create_advanced_network_graph called with period={period}")
    print(f"DEBUG - DataFrame shape: {df.shape}, columns: {df.columns.tolist()}")
    print(f"DEBUG - Unique periods in DataFrame: {df['period'].unique().tolist()}")
    
    # Filter data for the selected period
    period_df = df[df['period'] == period].copy()
    print(f"DEBUG - Filtered DataFrame for period {period}: {len(period_df)} rows")

    if period_df.empty:
        print(f"DEBUG - No data available for period {period}")
        fig = go.Figure()
        fig.update_layout(
            title=f"Aucune donnée disponible pour la période : {period}",
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        )
        return fig

    # 1. Build the graph with networkx
    G = nx.Graph()
    
    # First, add all nodes and primary edges
    for _, row in period_df.iterrows():
        # Add an edge between the primary term and its similar word
        # The 'weight' attribute is the similarity score, which spring_layout can use
        G.add_edge(
            row['term'],
            row['similar_word'],
            weight=row['similarity']
        )
    
    # 2. Add edges between similar words themselves
    # Get all unique words (both primary terms and similar words)
    all_words = set(period_df['term'].unique()) | set(period_df['similar_word'].unique())
    
    # Create a dictionary to store the words most similar to each term
    term_similar_words = {}
    for term in period_df['term'].unique():
        term_similar_words[term] = set(period_df[period_df['term'] == term]['similar_word'])
    
    # For each pair of similar words, check if they share a primary term
    # or if they appear together in similar words lists
    edges_added = set()  # To avoid duplicate edges
    
    # Method 1: Connect words that are similar to the same primary term
    for term, similar_words in term_similar_words.items():
        # Create edges between words that are similar to the same term
        similar_words_list = list(similar_words)
        for i in range(len(similar_words_list)):
            for j in range(i+1, len(similar_words_list)):
                word1, word2 = similar_words_list[i], similar_words_list[j]
                edge_key = tuple(sorted([word1, word2]))
                
                if edge_key not in edges_added:
                    # Get similarity scores for both words to the primary term
                    sim1 = period_df[(period_df['term'] == term) & (period_df['similar_word'] == word1)]['similarity'].values[0]
                    sim2 = period_df[(period_df['term'] == term) & (period_df['similar_word'] == word2)]['similarity'].values[0]
                    
                    # Calculate a derived similarity between the two similar words
                    # Words that are both highly similar to the same term are likely similar to each other
                    derived_similarity = (sim1 + sim2) / 2
                    
                    if derived_similarity >= similarity_threshold:
                        G.add_edge(word1, word2, weight=derived_similarity)
                        edges_added.add(edge_key)

    # 2. Calculate node positions using a physics-based layout
    # The spring_layout algorithm positions nodes using a force-directed model.
    # 'k' controls the optimal distance between nodes.
    # 'weight' tells the layout to make nodes with higher similarity closer.
    # 'seed' ensures the layout is reproducible.
    try:
        pos = nx.spring_layout(G, k=0.6, iterations=70, weight='weight', seed=42)
    except Exception:
        # Fallback for small or disconnected graphs that might cause errors
        pos = nx.spring_layout(G, k=0.6, iterations=70, seed=42)

    # 3. Create the Plotly Edge Trace
    edge_x, edge_y = [], []
    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])

    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=0.7, color='#888'),
        hoverinfo='none',
        mode='lines')

    # 4. Create the Plotly Node Trace
    node_x, node_y, node_text, node_color, node_size = [], [], [], [], []
    primary_terms = set(period_df['term'].unique())

    for node in G.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        node_text.append(node)

        # Customize node appearance based on type (primary term or similar word)
        if node in primary_terms:
            node_color.append('crimson') # Use a distinct color for primary terms
            # Make primary term size dependent on its number of connections (degree)
            node_size.append(15 + G.degree(node) * 2.5)
        else:
            node_color.append('royalblue')
            node_size.append(10)

    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers+text',  # Ajout de 'text' pour afficher les étiquettes
        hoverinfo='text',
        text=node_text,  # Afficher le texte des nœuds
        textposition="bottom center",
        marker=dict(
            color=node_color,
            size=node_size,
            line_width=1.5,
            line_color='white'
        )
    )
    # Set hover text for nodes
    node_trace.hovertext = [f"<b>{text}</b><br>Connections: {G.degree(text)}<br>Cliquez pour voir les articles" for text in node_text]

    # 5. Combine traces and style the final figure
    fig = go.Figure(data=[edge_trace, node_trace],
                 layout=go.Layout(
                    title=dict(
                        text=f"Réseau sémantique pour la période : {period}",
                        font=dict(size=16)
                    ),
                    showlegend=False,
                    hovermode='closest',
                    margin=dict(b=20, l=5, r=5, t=40),
                    xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                    yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                    plot_bgcolor="rgba(240, 240, 240, 0.95)",
                    clickmode="event+select"  # Activer les événements de clic
                    )
                )

    return fig


def create_semantic_drift_visualizations(results_file, viz_type="line"):
    """
    Crée des visualisations pour les résultats de dérive sémantique.
    
    Args:
        results_file: Chemin vers le fichier de résultats
        viz_type: Type de visualisation (line, heatmap, table, comparison)
        
    Returns:
        Div HTML avec les visualisations
    """
    try:
        # Charger les résultats
        df = pd.read_csv(results_file)
        
        if df.empty:
            return html.Div("Aucun résultat à afficher.")
        
        # Vérifier si nous avons les colonnes attendues
        if not all(col in df.columns for col in ['term', 'period', 'semantic_distance']):
            return html.Div("Format de fichier de résultats non reconnu. Attendu: 'term', 'period', 'semantic_distance'.")
        
        # Obtenir les termes et périodes uniques
        terms = df['term'].unique()
        periods = sorted(df['period'].unique())
        
        # Créer les visualisations en fonction du type demandé
        if viz_type == "line":
            # Créer un graphique linéaire montrant l'évolution de la dérive sémantique dans le temps pour chaque terme
            fig = px.line(
                df, 
                x="period", 
                y="semantic_distance", 
                color="term",
                title="Évolution du drift sémantique dans le temps",
                labels={"period": "Période", "semantic_distance": "Distance sémantique", "term": "Terme"},
                markers=True
            )
            
            # Ajouter une ligne horizontale à y=0 pour référence
            fig.add_shape(
                type="line",
                x0=periods[0],
                y0=0,
                x1=periods[-1],
                y1=0,
                line=dict(color="gray", width=1, dash="dash")
            )
            
            # Améliorer la mise en page
            fig.update_layout(
                xaxis_title="Période",
                yaxis_title="Distance sémantique",
                legend_title="Terme",
                hovermode="closest"
            )
            
            return html.Div([
                html.H5("Évolution du drift sémantique dans le temps"),
                html.P("Ce graphique montre comment le sens des termes évolue au fil du temps. Une distance plus élevée indique un changement sémantique plus important."),
                dcc.Graph(figure=fig),
                html.Hr(),
                html.Div([
                    html.H5("Statistiques globales"),
                    html.Ul([
                        html.Li(f"Nombre de termes analysés: {len(terms)}"),
                        html.Li(f"Périodes couvertes: {', '.join(periods)}"),
                        html.Li(f"Distance sémantique moyenne: {df['semantic_distance'].mean():.4f}"),
                        html.Li(f"Distance sémantique maximale: {df['semantic_distance'].max():.4f} (terme: {df.loc[df['semantic_distance'].idxmax(), 'term']}, période: {df.loc[df['semantic_distance'].idxmax(), 'period']})")
                    ])
                ])
            ])
            
        elif viz_type == "heatmap":
            # Créer une table pivot pour la heatmap
            pivot_df = df.pivot(index="term", columns="period", values="semantic_distance")
            
            # Créer la heatmap
            fig = px.imshow(
                pivot_df,
                labels=dict(x="Période", y="Terme", color="Distance sémantique"),
                x=pivot_df.columns,
                y=pivot_df.index,
                color_continuous_scale="RdBu_r",
                title="Carte de chaleur du drift sémantique"
            )
            
            # Améliorer la mise en page
            fig.update_layout(
                xaxis_title="Période",
                yaxis_title="Terme",
                coloraxis_colorbar=dict(title="Distance sémantique")
            )
            
            return html.Div([
                html.H5("Carte de chaleur du drift sémantique"),
                html.P("Cette carte de chaleur montre l'intensité du drift sémantique pour chaque terme et période. Les couleurs plus intenses indiquent un changement sémantique plus important."),
                dcc.Graph(figure=fig)
            ])
            
        elif viz_type == "table":
            # Créer un tableau avec les résultats
            table = dash_table.DataTable(
                id="semantic-drift-table",
                columns=[{"name": col, "id": col} for col in df.columns],
                data=df.to_dict("records"),
                filter_action="native",
                sort_action="native",
                sort_mode="multi",
                page_size=20,
                style_table={"overflowX": "auto"},
                style_cell={
                    "textAlign": "left",
                    "padding": "10px",
                    "whiteSpace": "normal",
                    "height": "auto"
                },
                style_header={
                    "backgroundColor": "rgb(230, 230, 230)",
                    "fontWeight": "bold"
                },
                style_data_conditional=[
                    {
                        "if": {"row_index": "odd"},
                        "backgroundColor": "rgb(248, 248, 248)"
                    }
                ]
            )
            
            return html.Div([
                html.H5("Tableau des résultats de drift sémantique"),
                html.P("Ce tableau présente les distances sémantiques pour chaque terme et période. Utilisez les filtres et le tri pour explorer les données."),
                table
            ])
            
        elif viz_type == "comparison":
            # Créer un graphique à barres comparant la dérive sémantique entre les termes
            term_avg_drift = df.groupby("term")["semantic_distance"].mean().reset_index()
            term_avg_drift = term_avg_drift.sort_values("semantic_distance", ascending=False)
            
            fig = px.bar(
                term_avg_drift,
                x="term",
                y="semantic_distance",
                title="Comparaison de la dérive sémantique moyenne par terme",
                labels={"term": "Terme", "semantic_distance": "Distance sémantique moyenne"}
            )
            
            # Améliorer la mise en page
            fig.update_layout(
                xaxis_title="Terme",
                yaxis_title="Distance sémantique moyenne"
            )
            
            # Créer un graphique à barres comparant la dérive sémantique entre les périodes
            period_avg_drift = df.groupby("period")["semantic_distance"].mean().reset_index()
            
            period_fig = px.bar(
                period_avg_drift,
                x="period",
                y="semantic_distance",
                title="Comparaison de la dérive sémantique moyenne par période",
                labels={"period": "Période", "semantic_distance": "Distance sémantique moyenne"}
            )
            
            # Améliorer la mise en page
            period_fig.update_layout(
                xaxis_title="Période",
                yaxis_title="Distance sémantique moyenne"
            )
            
            return html.Div([
                html.H5("Comparaison de la dérive sémantique"),
                html.P("Ces graphiques comparent la dérive sémantique moyenne entre les différents termes et périodes."),
                dcc.Graph(figure=fig),
                html.Hr(),
                dcc.Graph(figure=period_fig)
            ])
        
        else:
            return html.Div([
                html.H5("Type de visualisation non pris en charge"),
                html.P(f"Le type de visualisation '{viz_type}' n'est pas pris en charge pour la dérive sémantique.")
            ])
            
    except Exception as e:
        return html.Div([
            html.P(f"Erreur lors de la création des visualisations : {str(e)}", className="text-danger")
        ])

def create_similar_terms_visualizations(results_file, viz_type="table"):
    """
    Crée des visualisations pour les résultats de termes similaires.
    
    Args:
        results_file: Chemin vers le fichier de résultats
        viz_type: Type de visualisation (table, heatmap, network)
        
    Returns:
        Div HTML avec les visualisations
    """
    try:
        # Charger les résultats
        df = pd.read_csv(results_file)
        
        if df.empty:
            return html.Div("Aucun résultat à afficher.")
        
        # Vérifier si nous avons les colonnes attendues
        if not all(col in df.columns for col in ['term', 'period', 'rank', 'similar_word', 'similarity']):
            return html.Div("Format de fichier de résultats non reconnu. Attendu: 'term', 'period', 'rank', 'similar_word', 'similarity'.")
        
        # Obtenir les termes et périodes uniques
        terms = df['term'].unique()
        periods = sorted(df['period'].unique())
        
        # Créer les visualisations en fonction du type demandé
        if viz_type == "table":
            # Créer un tableau avec les résultats
            table = dash_table.DataTable(
                id="similar-terms-table",
                columns=[{"name": col, "id": col} for col in df.columns],
                data=df.to_dict("records"),
                filter_action="native",
                sort_action="native",
                sort_mode="multi",
                page_size=20,
                style_table={"overflowX": "auto"},
                style_cell={
                    "textAlign": "left",
                    "padding": "10px",
                    "whiteSpace": "normal",
                    "height": "auto"
                },
                style_header={
                    "backgroundColor": "rgb(230, 230, 230)",
                    "fontWeight": "bold"
                },
                style_data_conditional=[
                    {
                        "if": {"row_index": "odd"},
                        "backgroundColor": "rgb(248, 248, 248)"
                    }
                ]
            )
            
            return html.Div([
                html.H5("Tableau des termes similaires"),
                html.P("Ce tableau présente les mots les plus proches vectoriellement pour chaque terme et période. Utilisez les filtres et le tri pour explorer les données."),
                table
            ])
            
        elif viz_type == "heatmap":
            # Créer une table pivot pour la heatmap
            # Nous utiliserons les 5 mots les plus similaires pour chaque terme et période
            top_words = df[df['rank'] <= 5].copy()
            
            # Créer une colonne composite pour la heatmap
            top_words['term_period'] = top_words['term'] + ' (' + top_words['period'] + ')'
            top_words['similar_rank'] = top_words['similar_word'] + ' (#' + top_words['rank'].astype(str) + ')'
            
            # Créer la table pivot
            pivot_df = top_words.pivot(index="term_period", columns="similar_rank", values="similarity")
            
            # Créer la heatmap
            fig = px.imshow(
                pivot_df,
                labels=dict(x="Mot similaire (rang)", y="Terme (période)", color="Similarité"),
                x=pivot_df.columns,
                y=pivot_df.index,
                color_continuous_scale="Viridis",
                title="Carte de chaleur des termes similaires"
            )
            
            # Améliorer la mise en page
            fig.update_layout(
                xaxis_title="Mot similaire (rang)",
                yaxis_title="Terme (période)",
                coloraxis_colorbar=dict(title="Similarité")
            )
            
            return html.Div([
                html.H5("Carte de chaleur des termes similaires"),
                html.P("Cette carte de chaleur montre la similarité entre les termes analysés et leurs mots les plus proches vectoriellement."),
                dcc.Graph(figure=fig)
            ])
            
        elif viz_type == "network":
            # --- START: MODIFIED BLOCK ---
            print(f"DEBUG - Creating network visualization from file: {results_file}")
            
            # Obtenir les périodes uniques
            periods = sorted(df['period'].unique())
            print(f"DEBUG - Unique periods found: {periods}")
            
            # Fonction pour créer un graphique en réseau pour une période spécifique
            def create_network_for_period(period_to_show):
                print(f"DEBUG - Creating network for period: {period_to_show}")
                # Appeler la nouvelle fonction de graphique avancé
                return create_advanced_network_graph(df, period_to_show)

            # Create the graph for the most recent period initially
            initial_period = periods[-1] if periods else None
            print(f"DEBUG - Initial period selected: {initial_period}")
            initial_graph = create_network_for_period(initial_period) if initial_period else go.Figure()

            # Create a dropdown to select the period
            period_selector = html.Div([
                html.Label("Sélectionner une période:"),
                dcc.Dropdown(
                    id="similar-terms-period-selector",
                    options=[{"label": p, "value": p} for p in periods],
                    value=initial_period,
                    clearable=False,
                    style={"width": "100%", "marginBottom": "15px"}
                )
            ])

            # Return the layout with the selector and the graph
            return html.Div([
                html.H5("Réseau de termes similaires (Layout dynamique)"),
                html.P("Ce graphique montre les relations sémantiques. Les termes sont positionnés selon la force de leurs liens. Les termes principaux sont en rouge. Cliquez sur un terme pour voir les articles correspondants."),
                period_selector,
                dcc.Graph(
                    id="similar-terms-network-graph", # Keep the ID for the callback
                    figure=initial_graph,
                    config={"displayModeBar": True, "scrollZoom": True},
                    style={"height": "70vh"} # Give it more vertical space
                ),
                # Ajouter un div pour afficher les articles
                html.Div(id="similar-terms-articles-container", style={"marginTop": "20px"})
            ])
            # --- END: MODIFIED BLOCK ---
        
        else:
            return html.Div([
                html.H5("Type de visualisation non pris en charge"),
                html.P(f"Le type de visualisation '{viz_type}' n'est pas pris en charge pour les termes similaires.")
            ])
            
    except Exception as e:
        return html.Div([
            html.P(f"Erreur lors de la création des visualisations : {str(e)}", className="text-danger")
        ])
