"""
Dash web application for newspaper article analysis visualizations.
"""

import os
import sys
import json
from pathlib import Path

# Add the project root to the path to allow imports from other modules
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

import dash
from dash import dcc, html, Input, Output, State
import dash_bootstrap_components as dbc
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np

from src.visualization.visualizer import Visualizer
from src.utils.config_loader import load_config
from src.webapp.lexical_analysis_viz import get_lexical_analysis_layout, register_lexical_analysis_callbacks
from src.webapp.topic_modeling_viz import get_topic_modeling_layout, register_topic_modeling_callbacks
from src.webapp.topic_clustering_viz import get_clustering_layout, get_clustering_args, register_clustering_callbacks
from src.webapp.cluster_map_viz import get_cluster_map_layout, register_cluster_map_callbacks
from src.webapp.sentiment_analysis_viz import get_sentiment_analysis_layout, register_sentiment_analysis_callbacks
from src.webapp.entity_recognition_viz import get_entity_recognition_layout, register_entity_recognition_callbacks
from src.webapp.integrated_analysis_viz import get_integrated_analysis_layout, register_integrated_analysis_callbacks

# Load configuration
config_path = str(project_root / "config" / "config.yaml")
config = load_config(config_path)

# Initialize the Dash app with a Bootstrap theme
app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.DARKLY],
    meta_tags=[{"name": "viewport", "content": "width=device-width, initial-scale=1"}],
    suppress_callback_exceptions=True,
)
app.title = "Newspaper Articles Analysis"
server = app.server  # Expose the server for deployment platforms

# Register callbacks for lexical analysis form
register_lexical_analysis_callbacks(app)
# Register callbacks for topic modeling page
register_topic_modeling_callbacks(app)
# Register callbacks for clustering page
register_clustering_callbacks(app)
# Register callbacks for cluster map page
register_cluster_map_callbacks(app)
# Register callbacks for sentiment analysis page
register_sentiment_analysis_callbacks(app)
# Register callbacks for entity recognition page
register_entity_recognition_callbacks(app)
# Register callbacks for integrated analysis page
register_integrated_analysis_callbacks(app)

# Define the app layout
app.layout = dbc.Container([
    dbc.Row([
        dbc.Col([
            html.H1("Newspaper Articles Analysis Dashboard", className="text-center mb-4"),
            html.Div([
                dbc.Button("Lexical Analysis", id="btn-lexical", color="primary", className="me-2", n_clicks=0),
                dbc.Button("Topic Modeling", id="btn-topic", color="secondary", className="me-2", n_clicks=0),
                dbc.Button("Clustering", id="btn-clustering", color="info", className="me-2", n_clicks=0),
                dbc.Button("Carte des Clusters", id="btn-cluster-map", color="success", className="me-2", n_clicks=0),
                dbc.Button("Analyse de Sentiment", id="btn-sentiment", color="warning", className="me-2", n_clicks=0),
                dbc.Button("Entités Nommées", id="btn-entity", color="danger", className="me-2", n_clicks=0),
                dbc.Button("Analyse Intégrée", id="btn-integrated", color="primary", n_clicks=0),
            ], className="text-center mb-4"),
        ], width=12)
    ], className="mt-4"),
    dbc.Row([
        dbc.Col(html.Div(id="page-content", className="p-4 rounded-3", style={"backgroundColor": "#23272b"}), width=12)
    ]),
    dbc.Row([
        dbc.Col(html.Footer("Newspaper Articles Analysis Dashboard - Created with Dash", className="text-center text-muted mb-4"), width=12)
    ]),
], fluid=True, className="px-4")

@app.callback(
    dash.dependencies.Output("page-content", "children"),
    [dash.dependencies.Input("btn-lexical", "n_clicks"),
     dash.dependencies.Input("btn-topic", "n_clicks"),
     dash.dependencies.Input("btn-clustering", "n_clicks"),
     dash.dependencies.Input("btn-cluster-map", "n_clicks"),
     dash.dependencies.Input("btn-sentiment", "n_clicks"),
     dash.dependencies.Input("btn-entity", "n_clicks"),
     dash.dependencies.Input("btn-integrated", "n_clicks")],
)
def display_page(btn_lexical, btn_topic, btn_clustering, btn_cluster_map, btn_sentiment, btn_entity, btn_integrated):
    ctx = dash.callback_context
    if not ctx.triggered:
        return get_lexical_analysis_layout()
    button_id = ctx.triggered[0]["prop_id"].split(".")[0]
    if button_id == "btn-lexical":
        return get_lexical_analysis_layout()
    elif button_id == "btn-topic":
        return get_topic_modeling_layout()
    elif button_id == "btn-clustering":
        return get_clustering_layout()
    elif button_id == "btn-cluster-map":
        return get_cluster_map_layout()
    elif button_id == "btn-sentiment":
        return get_sentiment_analysis_layout()
    elif button_id == "btn-entity":
        return get_entity_recognition_layout()
    elif button_id == "btn-integrated":
        return get_integrated_analysis_layout()
    return get_lexical_analysis_layout()

# Callback to update data selection controls based on analysis type
@app.callback(
    Output("data-selection-controls", "children"),
    Input("analysis-dropdown", "value"),
)
def update_data_selection_controls(analysis_type):
    """Update the data selection controls based on the selected analysis type."""
    if analysis_type == "topic_modeling":
        return [
            html.P("Number of topics to display:", className="mt-3"),
            dcc.Slider(
                id="num-topics-slider",
                min=2,
                max=10,
                step=1,
                value=5,
                marks={i: str(i) for i in range(2, 11)},
            ),
            html.P("Visualization type:", className="mt-3"),
            dcc.RadioItems(
                id="topic-viz-type",
                options=[
                    {"label": "Topic Distribution", "value": "distribution"},
                    {"label": "Topic Keywords", "value": "keywords"},
                    {"label": "Topic Over Time", "value": "time"},
                ],
                value="distribution",
                className="mb-2",
            ),
        ]
    elif analysis_type == "lexical_analysis":
        return [
            html.P("Visualization type:", className="mt-3"),
            dcc.RadioItems(
                id="lexical-viz-type",
                options=[
                    {"label": "Word Frequency", "value": "word_frequency"},
                    {"label": "Word Cloud", "value": "word_cloud"},
                ],
                value="word_frequency",
                className="mb-2",
            ),
        ]
    elif analysis_type == "clustering":
        return [
            html.P("Number of clusters to display:", className="mt-3"),
            dcc.Slider(
                id="num-clusters-slider",
                min=2,
                max=10,
                step=1,
                value=5,
                marks={i: str(i) for i in range(2, 11)},
            ),
            html.P("Visualization type:", className="mt-3"),
            dcc.RadioItems(
                id="clustering-viz-type",
                options=[
                    {"label": "Cluster Distribution", "value": "distribution"},
                    {"label": "Cluster Over Time", "value": "time"},
                ],
                value="distribution",
                className="mb-2",
            ),
        ]
    
    return html.P("Select an analysis type to view controls")

# Callback to update visualization content
@app.callback(
    Output("visualization-content", "children"),
    [
        Input("analysis-dropdown", "value"),
        Input("data-selection-controls", "children"),
    ],
)
def update_visualization(analysis_type, _):
    """Update the visualization based on the selected analysis type and controls."""
    # This is a placeholder callback that will be expanded as you add real data
    # For now, we'll just show placeholder visualizations
    
    if analysis_type == "topic_modeling":
        # Placeholder data for topic modeling
        topics = [f"Topic {i}" for i in range(1, 6)]
        topic_weights = np.random.rand(5, 10)
        topic_weights = topic_weights / topic_weights.sum(axis=0)
        df = pd.DataFrame(topic_weights.T, columns=topics)
        
        fig = px.bar(
            df, 
            barmode="stack",
            title="Topic Distribution (Placeholder)",
        )
        return dcc.Graph(figure=fig)
    
    elif analysis_type == "lexical_analysis":
        # Placeholder data for lexical analysis
        words = ["word1", "word2", "word3", "word4", "word5"]
        frequencies = [10, 8, 6, 4, 2]
        
        fig = px.bar(
            x=words, 
            y=frequencies,
            title="Word Frequency (Placeholder)",
            labels={"x": "Word", "y": "Frequency"},
        )
        return dcc.Graph(figure=fig)
    
    elif analysis_type == "clustering":
        # Placeholder data for clustering
        clusters = [f"Cluster {i}" for i in range(1, 6)]
        cluster_sizes = [100, 80, 60, 40, 20]
        
        fig = px.pie(
            values=cluster_sizes,
            names=clusters,
            title="Cluster Distribution (Placeholder)",
        )
        return dcc.Graph(figure=fig)
    
    return html.P("Select an analysis type to view visualizations")

# Callback to update analysis details
@app.callback(
    Output("analysis-details", "children"),
    Input("analysis-dropdown", "value"),
)
def update_analysis_details(analysis_type):
    """Update the analysis details based on the selected analysis type."""
    if analysis_type == "topic_modeling":
        return html.Div([
            html.H5("Topic Modeling Analysis"),
            html.P("This visualization shows the distribution of topics across articles."),
            html.P("The data shown is currently placeholder data. Connect to your actual analysis results to see real data."),
            html.Ul([
                html.Li("Topic modeling algorithm: LDA"),
                html.Li("Number of topics: 5"),
                html.Li("Coherence score: 0.42 (placeholder)"),
            ]),
        ])
    
    elif analysis_type == "lexical_analysis":
        return html.Div([
            html.H5("Lexical Analysis"),
            html.P("This visualization shows the frequency of words in the articles."),
            html.P("The data shown is currently placeholder data. Connect to your actual analysis results to see real data."),
            html.Ul([
                html.Li("Lexical analysis algorithm: Word Frequency"),
                html.Li("Number of words: 100"),
                html.Li("Average word length: 5 (placeholder)"),
            ]),
        ])
    
    elif analysis_type == "clustering":
        return html.Div([
            html.H5("Clustering Analysis"),
            html.P("This visualization shows the distribution of clusters across articles."),
            html.P("The data shown is currently placeholder data. Connect to your actual analysis results to see real data."),
            html.Ul([
                html.Li("Clustering algorithm: K-Means"),
                html.Li("Number of clusters: 5"),
                html.Li("Silhouette score: 0.6 (placeholder)"),
            ]),
        ])
    
    return html.P("Select an analysis type to view details")
