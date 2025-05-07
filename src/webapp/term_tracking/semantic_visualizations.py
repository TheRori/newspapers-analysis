"""
Fonctions de visualisation pour la dérive sémantique et les termes similaires.
"""

import re
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from dash import html, dcc, dash_table
from pathlib import Path
from src.webapp.term_tracking.utils import clean_file_path

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
            # Créer une visualisation en réseau montrant les relations entre les termes
            
            # Obtenir les périodes uniques
            periods = sorted(df['period'].unique())
            
            # Fonction pour créer un graphique en réseau pour une période spécifique
            def create_period_network(period):
                # Filtrer les données pour la période sélectionnée
                period_df = df[df['period'] == period]
                
                # Obtenir les termes uniques pour cette période
                period_terms = period_df['term'].unique()
                
                # Filtrer pour les 5 mots les plus similaires pour une meilleure visualisation
                top_words = period_df[period_df['rank'] <= 5].copy()
                
                # Créer le graphique en réseau
                fig = go.Figure()
                
                # Calculer les positions pour les termes principaux (dans un cercle)
                radius = 3
                term_positions = {}
                
                for i, term in enumerate(period_terms):
                    angle = 2 * np.pi * i / len(period_terms)
                    x = radius * np.cos(angle)
                    y = radius * np.sin(angle)
                    term_positions[term] = (x, y)
                    
                    # Ajouter un nœud pour le terme principal
                    fig.add_trace(go.Scatter(
                        x=[x],
                        y=[y],
                        mode="markers+text",
                        marker=dict(size=25, color="red"),
                        text=[term],
                        name=term,
                        textposition="middle center",
                        textfont=dict(color="white", size=12),
                        hoverinfo="text",
                        hovertext=f"<b>{term}</b><br>Cliquez pour voir les articles",
                        showlegend=False
                    ))
                
                # Ajouter des nœuds et des arêtes pour les mots similaires
                for term in period_terms:
                    term_x, term_y = term_positions[term]
                    term_similar = top_words[top_words['term'] == term]
                    
                    for _, row in term_similar.iterrows():
                        # Calculer la position (dans un cercle autour du terme principal)
                        angle = (row['rank'] - 1) * (2 * np.pi / 5)
                        distance = 1.5  # Distance du terme principal
                        x = term_x + distance * np.cos(angle)
                        y = term_y + distance * np.sin(angle)
                        
                        # Taille et opacité basées sur la similarité
                        node_size = 15 + (row['similarity'] * 10)
                        
                        # Ajouter un nœud pour le mot similaire
                        fig.add_trace(go.Scatter(
                            x=[x],
                            y=[y],
                            mode="markers+text",
                            marker=dict(
                                size=node_size, 
                                color="blue",
                                opacity=0.7 + (row['similarity'] * 0.3)
                            ),
                            text=[row['similar_word']],
                            name=f"{row['similar_word']} ({row['similarity']:.2f})",
                            textposition="bottom center",
                            hoverinfo="text",
                            hovertext=f"<b>{row['similar_word']}</b><br>Similarité: {row['similarity']:.3f}<br>Cliquez pour voir les articles",
                            showlegend=False
                        ))
                        
                        # Ajouter une arête avec une largeur basée sur la similarité
                        fig.add_trace(go.Scatter(
                            x=[term_x, x],
                            y=[term_y, y],
                            mode="lines",
                            line=dict(
                                width=row['similarity'] * 5, 
                                color="rgba(100, 100, 100, 0.6)"
                            ),
                            hoverinfo="text",
                            hovertext=f"Similarité: {row['similarity']:.3f}",
                            showlegend=False
                        ))
                
                # Améliorer la mise en page
                fig.update_layout(
                    title=f"Réseau de termes similaires - Période: {period}",
                    xaxis=dict(
                        showgrid=False, 
                        zeroline=False, 
                        showticklabels=False,
                        range=[-5, 5]
                    ),
                    yaxis=dict(
                        showgrid=False, 
                        zeroline=False, 
                        showticklabels=False,
                        range=[-5, 5],
                        scaleanchor="x",
                        scaleratio=1
                    ),
                    hovermode="closest",
                    plot_bgcolor="rgba(240, 240, 240, 0.8)",
                    clickmode="event+select"  # Activer les événements de clic
                )
                
                return fig
            
            # Créer le graphique initial avec la période la plus récente
            initial_period = periods[-1]
            initial_graph = create_period_network(initial_period)
            
            # Créer le sélecteur de période
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
            
            # Retourner la mise en page avec le sélecteur de période et le graphique
            return html.Div([
                html.H5("Réseau de termes similaires"),
                html.P("Ce graphique montre les relations entre les termes analysés et leurs mots les plus proches vectoriellement. Les termes principaux sont en rouge, et les mots similaires en bleu. Cliquez sur un terme pour voir les articles correspondants."),
                period_selector,
                dcc.Graph(
                    id="similar-terms-network-graph",
                    figure=initial_graph,
                    config={"displayModeBar": True, "scrollZoom": True}
                ),
                # Ajouter un div pour afficher les articles
                html.Div(id="similar-terms-articles-container", style={"marginTop": "20px"})
            ])
        
        else:
            return html.Div([
                html.H5("Type de visualisation non pris en charge"),
                html.P(f"Le type de visualisation '{viz_type}' n'est pas pris en charge pour les termes similaires.")
            ])
            
    except Exception as e:
        return html.Div([
            html.P(f"Erreur lors de la création des visualisations : {str(e)}", className="text-danger")
        ])
