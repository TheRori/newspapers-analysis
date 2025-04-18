"""
Visualization utilities for newspaper article analysis.
"""

import os
import json
from typing import List, Dict, Any, Optional, Tuple, Union

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.figure import Figure
import seaborn as sns
from wordcloud import WordCloud


class Visualizer:
    """Class for creating visualizations of newspaper article analysis."""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the Visualizer with configuration settings.
        
        Args:
            config: Dictionary containing visualization configuration
        """
        self.config = config
        self.figsize = tuple(config.get('default_figsize', (12, 8)))
        self.dpi = config.get('dpi', 300)
        self.save_format = config.get('save_format', 'png')
        self.color_palette = config.get('color_palette', 'viridis')
        
        # Set default style
        sns.set_style('whitegrid')
        plt.rcParams['figure.figsize'] = self.figsize
        plt.rcParams['figure.dpi'] = self.dpi
        
        # Set color palette
        sns.set_palette(self.color_palette)
    
    def create_wordcloud(self, text_data: Union[str, Dict[str, int]], 
                        title: str = 'Word Cloud') -> Figure:
        """
        Create a word cloud visualization.
        
        Args:
            text_data: Either a string of text or a dictionary mapping words to frequencies
            title: Title for the visualization
            
        Returns:
            Matplotlib figure
        """
        # Create word cloud
        wordcloud = WordCloud(
            width=800, 
            height=400, 
            background_color='white',
            max_words=200,
            contour_width=3
        ).generate_from_text(text_data) if isinstance(text_data, str) else \
          WordCloud(
            width=800, 
            height=400, 
            background_color='white',
            max_words=200,
            contour_width=3
        ).generate_from_frequencies(text_data)
        
        # Create figure
        fig, ax = plt.subplots(figsize=self.figsize)
        ax.imshow(wordcloud, interpolation='bilinear')
        ax.set_axis_off()
        ax.set_title(title, fontsize=16)
        
        return fig
    
    def plot_top_entities(self, entity_counts: Dict[str, int], 
                         title: str = 'Top Entities', 
                         top_n: int = 20) -> Figure:
        """
        Create a bar chart of top entities.
        
        Args:
            entity_counts: Dictionary mapping entity names to counts
            title: Title for the visualization
            top_n: Number of top entities to show
            
        Returns:
            Matplotlib figure
        """
        # Sort entities by count and take top N
        sorted_entities = sorted(entity_counts.items(), key=lambda x: x[1], reverse=True)[:top_n]
        entities, counts = zip(*sorted_entities) if sorted_entities else ([], [])
        
        # Create figure
        fig, ax = plt.subplots(figsize=self.figsize)
        
        # Create horizontal bar chart
        y_pos = np.arange(len(entities))
        ax.barh(y_pos, counts)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(entities)
        ax.invert_yaxis()  # Labels read top-to-bottom
        ax.set_xlabel('Count')
        ax.set_title(title, fontsize=16)
        
        plt.tight_layout()
        return fig
    
    def plot_sentiment_distribution(self, sentiment_scores: List[float], 
                                  title: str = 'Sentiment Distribution') -> Figure:
        """
        Create a histogram of sentiment scores.
        
        Args:
            sentiment_scores: List of sentiment scores
            title: Title for the visualization
            
        Returns:
            Matplotlib figure
        """
        # Create figure
        fig, ax = plt.subplots(figsize=self.figsize)
        
        # Create histogram
        sns.histplot(sentiment_scores, bins=20, kde=True, ax=ax)
        
        # Add vertical lines for sentiment categories
        ax.axvline(x=-0.05, color='r', linestyle='--', alpha=0.7, label='Negative Threshold')
        ax.axvline(x=0.05, color='g', linestyle='--', alpha=0.7, label='Positive Threshold')
        
        # Add shaded regions
        ax.axvspan(-1, -0.05, alpha=0.2, color='r', label='Negative')
        ax.axvspan(-0.05, 0.05, alpha=0.2, color='gray', label='Neutral')
        ax.axvspan(0.05, 1, alpha=0.2, color='g', label='Positive')
        
        ax.set_xlabel('Sentiment Score')
        ax.set_ylabel('Frequency')
        ax.set_title(title, fontsize=16)
        ax.legend()
        
        plt.tight_layout()
        return fig
    
    def plot_topic_distribution(self, topic_data: Dict[str, List[float]], 
                              topic_labels: Optional[List[str]] = None,
                              title: str = 'Topic Distribution') -> Figure:
        """
        Create a stacked bar chart of topic distributions.
        
        Args:
            topic_data: Dictionary mapping document IDs to topic distributions
            topic_labels: Optional list of topic labels
            title: Title for the visualization
            
        Returns:
            Matplotlib figure
        """
        # Convert to DataFrame
        df = pd.DataFrame.from_dict(topic_data, orient='index')
        
        # Set topic labels if provided
        if topic_labels:
            df.columns = topic_labels
        else:
            df.columns = [f'Topic {i}' for i in range(df.shape[1])]
        
        # Create figure
        fig, ax = plt.subplots(figsize=self.figsize)
        
        # Create stacked bar chart
        df.plot(kind='bar', stacked=True, ax=ax, colormap=self.color_palette)
        
        ax.set_xlabel('Document')
        ax.set_ylabel('Topic Proportion')
        ax.set_title(title, fontsize=16)
        ax.legend(title='Topics', bbox_to_anchor=(1.05, 1), loc='upper left')
        
        plt.tight_layout()
        return fig
    
    def plot_entity_network(self, co_occurrences: Dict[Tuple[str, str], int], 
                          min_weight: int = 2,
                          title: str = 'Entity Co-occurrence Network') -> Figure:
        """
        Create a network visualization of entity co-occurrences.
        
        Args:
            co_occurrences: Dictionary mapping entity pairs to co-occurrence counts
            min_weight: Minimum co-occurrence count to include
            title: Title for the visualization
            
        Returns:
            Matplotlib figure
        """
        try:
            import networkx as nx
        except ImportError:
            print("networkx is required for network visualization. Install with: pip install networkx")
            raise
        
        # Create graph
        G = nx.Graph()
        
        # Add edges with weights
        for (entity1, entity2), weight in co_occurrences.items():
            if weight >= min_weight:
                G.add_edge(entity1, entity2, weight=weight)
        
        # Create figure
        fig, ax = plt.subplots(figsize=self.figsize)
        
        # Get positions
        pos = nx.spring_layout(G, seed=42)
        
        # Get edge weights for width
        edge_weights = [G[u][v]['weight'] for u, v in G.edges()]
        
        # Normalize weights for width
        max_weight = max(edge_weights) if edge_weights else 1
        edge_widths = [2 * w / max_weight for w in edge_weights]
        
        # Draw network
        nx.draw_networkx_nodes(G, pos, node_size=100, alpha=0.8, ax=ax)
        nx.draw_networkx_edges(G, pos, width=edge_widths, alpha=0.5, ax=ax)
        nx.draw_networkx_labels(G, pos, font_size=8, ax=ax)
        
        ax.set_title(title, fontsize=16)
        ax.axis('off')
        
        plt.tight_layout()
        return fig
    
    def plot_time_series(self, dates: List[str], values: List[float], 
                       title: str = 'Time Series', 
                       ylabel: str = 'Value') -> Figure:
        """
        Create a time series plot.
        
        Args:
            dates: List of date strings
            values: List of values
            title: Title for the visualization
            ylabel: Label for y-axis
            
        Returns:
            Matplotlib figure
        """
        # Convert to DataFrame
        df = pd.DataFrame({'date': pd.to_datetime(dates), 'value': values})
        df.set_index('date', inplace=True)
        
        # Create figure
        fig, ax = plt.subplots(figsize=self.figsize)
        
        # Create time series plot
        df.plot(ax=ax)
        
        ax.set_xlabel('Date')
        ax.set_ylabel(ylabel)
        ax.set_title(title, fontsize=16)
        
        plt.tight_layout()
        return fig
    
    def plot_heatmap(self, data: np.ndarray, 
                   row_labels: Optional[List[str]] = None,
                   col_labels: Optional[List[str]] = None,
                   title: str = 'Heatmap') -> Figure:
        """
        Create a heatmap visualization.
        
        Args:
            data: 2D array of values
            row_labels: Optional list of row labels
            col_labels: Optional list of column labels
            title: Title for the visualization
            
        Returns:
            Matplotlib figure
        """
        # Create figure
        fig, ax = plt.subplots(figsize=self.figsize)
        
        # Create heatmap
        sns.heatmap(data, annot=True, cmap=self.color_palette, 
                   xticklabels=col_labels, yticklabels=row_labels, ax=ax)
        
        ax.set_title(title, fontsize=16)
        
        plt.tight_layout()
        return fig
    
    def save_figure(self, fig: Figure, output_dir: str, filename: str) -> str:
        """
        Save a figure to file.
        
        Args:
            fig: Matplotlib figure
            output_dir: Directory to save the figure
            filename: Filename for the figure (without extension)
            
        Returns:
            Path to the saved figure
        """
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # Add extension if not present
        if not filename.endswith(f'.{self.save_format}'):
            filename = f"{filename}.{self.save_format}"
        
        output_path = os.path.join(output_dir, filename)
        
        fig.savefig(output_path, dpi=self.dpi, bbox_inches='tight')
        plt.close(fig)
        
        return output_path
