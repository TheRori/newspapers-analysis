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

# Load configuration
config = load_config()

# Initialize the Dash app with a Bootstrap theme
app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.BOOTSTRAP],
    meta_tags=[{"name": "viewport", "content": "width=device-width, initial-scale=1"}],
)
app.title = "Newspaper Articles Analysis"
server = app.server  # Expose the server for deployment platforms

# Define the app layout
app.layout = dbc.Container(
    [
        dbc.Row(
            [
                dbc.Col(
                    html.H1("Newspaper Articles Analysis Dashboard", className="text-center mb-4"),
                    width=12,
                )
            ],
            className="mt-4",
        ),
        
        dbc.Row(
            [
                dbc.Col(
                    dbc.Card(
                        [
                            dbc.CardHeader("Data Selection"),
                            dbc.CardBody(
                                [
                                    html.P("Select analysis results to visualize:"),
                                    dcc.Dropdown(
                                        id="analysis-dropdown",
                                        options=[
                                            {"label": "Topic Modeling", "value": "topic_modeling"},
                                            {"label": "Named Entity Recognition", "value": "ner"},
                                            {"label": "Sentiment Analysis", "value": "sentiment"},
                                            {"label": "Text Classification", "value": "classification"},
                                        ],
                                        value="topic_modeling",
                                        clearable=False,
                                    ),
                                    html.Div(id="data-selection-controls", className="mt-3"),
                                ]
                            ),
                        ],
                        className="mb-4",
                    ),
                    width=12,
                    lg=3,
                ),
                
                dbc.Col(
                    [
                        dbc.Card(
                            [
                                dbc.CardHeader("Visualization"),
                                dbc.CardBody(
                                    [
                                        dcc.Loading(
                                            id="loading-visualization",
                                            type="circle",
                                            children=html.Div(id="visualization-content"),
                                        )
                                    ]
                                ),
                            ],
                            className="mb-4",
                        ),
                    ],
                    width=12,
                    lg=9,
                ),
            ]
        ),
        
        dbc.Row(
            [
                dbc.Col(
                    dbc.Card(
                        [
                            dbc.CardHeader("Analysis Details"),
                            dbc.CardBody(
                                [
                                    dcc.Loading(
                                        id="loading-details",
                                        type="circle",
                                        children=html.Div(id="analysis-details"),
                                    )
                                ]
                            ),
                        ],
                        className="mb-4",
                    ),
                    width=12,
                ),
            ]
        ),
        
        dbc.Row(
            [
                dbc.Col(
                    html.Footer(
                        "Newspaper Articles Analysis Dashboard - Created with Dash",
                        className="text-center text-muted mb-4",
                    ),
                    width=12,
                ),
            ]
        ),
    ],
    fluid=True,
    className="px-4",
)

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
    elif analysis_type == "ner":
        return [
            html.P("Entity types to display:", className="mt-3"),
            dcc.Checklist(
                id="entity-types",
                options=[
                    {"label": "Person", "value": "PERSON"},
                    {"label": "Organization", "value": "ORG"},
                    {"label": "Location", "value": "GPE"},
                    {"label": "Date", "value": "DATE"},
                ],
                value=["PERSON", "ORG", "GPE"],
                className="mb-2",
            ),
            html.P("Top N entities:", className="mt-3"),
            dcc.Slider(
                id="top-n-entities",
                min=5,
                max=30,
                step=5,
                value=15,
                marks={i: str(i) for i in range(5, 31, 5)},
            ),
        ]
    elif analysis_type == "sentiment":
        return [
            html.P("Visualization type:", className="mt-3"),
            dcc.RadioItems(
                id="sentiment-viz-type",
                options=[
                    {"label": "Distribution", "value": "distribution"},
                    {"label": "Over Time", "value": "time"},
                    {"label": "By Source", "value": "source"},
                ],
                value="distribution",
                className="mb-2",
            ),
        ]
    elif analysis_type == "classification":
        return [
            html.P("Classification type:", className="mt-3"),
            dcc.RadioItems(
                id="classification-type",
                options=[
                    {"label": "Category Distribution", "value": "distribution"},
                    {"label": "Confusion Matrix", "value": "confusion"},
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
    
    elif analysis_type == "ner":
        # Placeholder data for NER
        entities = ["Person A", "Organization B", "Location C", "Person D", "Organization E"]
        counts = [45, 32, 28, 22, 18]
        
        fig = px.bar(
            x=counts, 
            y=entities,
            orientation="h",
            title="Top Entities (Placeholder)",
            labels={"x": "Count", "y": "Entity"},
        )
        return dcc.Graph(figure=fig)
    
    elif analysis_type == "sentiment":
        # Placeholder data for sentiment analysis
        sentiment_scores = np.random.normal(0.1, 0.3, 100)
        
        fig = px.histogram(
            sentiment_scores,
            nbins=20,
            title="Sentiment Distribution (Placeholder)",
            labels={"value": "Sentiment Score", "count": "Frequency"},
        )
        
        # Add vertical lines for sentiment categories
        fig.add_vline(x=-0.05, line_dash="dash", line_color="red")
        fig.add_vline(x=0.05, line_dash="dash", line_color="green")
        
        return dcc.Graph(figure=fig)
    
    elif analysis_type == "classification":
        # Placeholder data for classification
        categories = ["Politics", "Business", "Sports", "Technology", "Entertainment"]
        counts = [120, 85, 65, 45, 30]
        
        fig = px.pie(
            values=counts,
            names=categories,
            title="Article Categories (Placeholder)",
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
    
    elif analysis_type == "ner":
        return html.Div([
            html.H5("Named Entity Recognition Analysis"),
            html.P("This visualization shows the most frequent entities mentioned in the articles."),
            html.P("The data shown is currently placeholder data. Connect to your actual analysis results to see real data."),
            html.Ul([
                html.Li("NER model: spaCy en_core_web_lg"),
                html.Li("Entity types: PERSON, ORG, GPE"),
                html.Li("Total entities extracted: 1,245 (placeholder)"),
            ]),
        ])
    
    elif analysis_type == "sentiment":
        return html.Div([
            html.H5("Sentiment Analysis"),
            html.P("This visualization shows the distribution of sentiment scores across articles."),
            html.P("The data shown is currently placeholder data. Connect to your actual analysis results to see real data."),
            html.Ul([
                html.Li("Sentiment model: VADER"),
                html.Li("Sentiment range: -1 (negative) to 1 (positive)"),
                html.Li("Average sentiment: 0.12 (placeholder)"),
            ]),
        ])
    
    elif analysis_type == "classification":
        return html.Div([
            html.H5("Text Classification Analysis"),
            html.P("This visualization shows the distribution of article categories."),
            html.P("The data shown is currently placeholder data. Connect to your actual analysis results to see real data."),
            html.Ul([
                html.Li("Classification model: DistilBERT"),
                html.Li("Number of categories: 5"),
                html.Li("Accuracy: 0.78 (placeholder)"),
            ]),
        ])
    
    return html.P("Select an analysis type to view details")

# Run the app
if __name__ == "__main__":
    app.run_server(debug=True, port=8050)
