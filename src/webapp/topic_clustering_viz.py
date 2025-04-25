"""
Dash page for Topic Clustering visualization, inspired by topic_modeling_viz.py.
"""
import json
import os
import argparse
import sys
from dash import dcc, html, Input, Output, State, dash_table, callback_context, no_update, ALL
from dash.dependencies import MATCH
import dash
import dash_bootstrap_components as dbc
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
from pathlib import Path
import yaml
import pathlib
import threading
import time
import subprocess

# Helper to load clustering results (assume similar structure to topic modeling)
def load_clustering_stats(path):
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    
    # Traitement des données pour la visualisation
    stats = {}
    
    # Conserver les données brutes pour l'explorateur d'articles
    stats['doc_ids'] = data.get('doc_ids', [])
    stats['labels'] = data.get('labels', [])
    
    # 1. Informations de base
    stats['n_clusters'] = len(set(data.get('labels', [])))
    
    # 2. Distribution des clusters
    if 'labels' in data:
        labels = data['labels']
        cluster_counts = {}
        for label in labels:
            if label not in cluster_counts:
                cluster_counts[label] = 0
            cluster_counts[label] += 1
        
        # Calculer les proportions
        total = len(labels)
        cluster_distribution = [cluster_counts.get(i, 0) / total for i in range(stats['n_clusters'])]
        stats['cluster_distribution'] = cluster_distribution
        stats['cluster_sizes'] = cluster_counts
    
    # 3. Documents représentatifs par cluster
    if 'labels' in data and 'doc_ids' in data:
        doc_ids = data['doc_ids']
        labels = data['labels']
        representative_docs = {}
        
        # Pour chaque cluster, prendre les 3 premiers documents
        for cluster in range(stats['n_clusters']):
            cluster_docs = [doc_id for doc_id, label in zip(doc_ids, labels) if label == cluster]
            representative_docs[str(cluster)] = cluster_docs[:3]  # Prendre les 3 premiers
        
        stats['representative_docs'] = representative_docs
        
        # 4. Extraire les informations temporelles des articles
        temporal_info = {}
        journal_info = {}
        for cluster in range(stats['n_clusters']):
            cluster_docs = [doc_id for doc_id, label in zip(doc_ids, labels) if label == cluster]
            dates = []
            journals = {}
            
            for doc in cluster_docs:
                if isinstance(doc, str) and '_' in doc:
                    # Format attendu: article_YYYY-MM-DD_journal_id
                    parts = doc.split('_')
                    if len(parts) >= 3 and '-' in parts[1]:
                        try:
                            # Extraire la date
                            date_str = parts[1]
                            year = int(date_str.split('-')[0])
                            month = int(date_str.split('-')[1])
                            dates.append((year, month))
                            
                            # Extraire le journal
                            journal = parts[2]
                            if journal not in journals:
                                journals[journal] = 0
                            journals[journal] += 1
                        except (ValueError, IndexError):
                            pass
            
            if dates:
                # Calculer la moyenne des années/mois
                avg_year = sum(d[0] for d in dates) / len(dates)
                avg_month = sum(d[1] for d in dates) / len(dates)
                temporal_info[cluster] = {'year': avg_year, 'month': avg_month}
            
            if journals:
                # Trouver le journal le plus fréquent
                main_journal = max(journals.items(), key=lambda x: x[1])[0]
                journal_info[cluster] = {'main_journal': main_journal, 'distribution': journals}
        
        if temporal_info:
            stats['temporal_info'] = temporal_info
        if journal_info:
            stats['journal_info'] = journal_info
    
    # 5. Centres des clusters (si disponibles)
    if 'cluster_centers' in data:
        stats['cluster_centers'] = data['cluster_centers']
        
        # Générer des "top words" fictifs pour chaque cluster basés sur les centres
        # (dans un vrai cas, il faudrait utiliser les vrais mots)
        top_words = {}
        for i, center in enumerate(data['cluster_centers']):
            # Prendre les 10 dimensions les plus importantes
            top_indices = sorted(range(len(center)), key=lambda i: center[i], reverse=True)[:10]
            words = [("Topic_" + str(idx), center[idx]) for idx in top_indices]
            top_words[str(i)] = words
        
        stats['top_words'] = top_words
    
    # 6. Essayer de charger les statistiques avancées du topic modeling
    try:
        project_root = Path(__file__).resolve().parents[2]
        advanced_topic_path = project_root / 'data' / 'results' / 'advanced_topic' / 'advanced_topic_analysis.json'
        if os.path.exists(advanced_topic_path):
            with open(advanced_topic_path, encoding="utf-8") as f:
                advanced_stats = json.load(f)
            
            # Fusionner les informations pertinentes
            if 'topic_names_llm' in advanced_stats:
                stats['topic_names_llm'] = advanced_stats['topic_names_llm']
            if 'coherence_score' in advanced_stats:
                stats['coherence_score'] = advanced_stats['coherence_score']
    except Exception as e:
        print(f"Erreur lors du chargement des stats avancées: {e}")
    
    return stats

# Helper to render clustering stats
def render_clustering_stats(stats):
    children = []
    # 1. Score de clustering (silhouette, etc.)
    if 'silhouette_score' in stats:
        children.append(dbc.Alert(f"Score de silhouette : {stats['silhouette_score']:.3f}", color="info", className="mb-3"))
    if 'n_clusters' in stats:
        children.append(dbc.Alert(f"Nombre de clusters : {stats['n_clusters']}", color="secondary", className="mb-3"))
    if 'coherence_score' in stats:
        children.append(dbc.Alert(f"Score de cohérence des topics : {stats['coherence_score']:.3f}", color="info", className="mb-3"))
    
    # Bouton pour accéder au browser d'articles - TOUJOURS AFFICHER
    children.append(
        dbc.Button(
            "Explorer les articles par cluster", 
            id="btn-explore-articles",
            color="primary", 
            className="mb-4 mt-2"
        )
    )
    
    # Visualisation temporelle des clusters (année/mois) - PRIORITAIRE
    if 'temporal_info' in stats:
        # Créer un DataFrame pour la visualisation temporelle
        temporal_df = pd.DataFrame([
            {
                'Cluster': f"Cluster {cluster}",
                'Année': info['year'],
                'Mois': info['month'],
                'Taille': stats.get('cluster_sizes', {}).get(int(cluster), 10)
            }
            for cluster, info in stats['temporal_info'].items()
        ])
        
        # Visualisation temporelle avec noms de topics
        time_fig = px.scatter(temporal_df, x='Année', y='Mois', 
                             size='Taille', color='Cluster', hover_name='Cluster',
                             size_max=60, title="Distribution temporelle des clusters")
        time_fig.update_layout(height=600)
        children.append(dcc.Graph(figure=time_fig, id='cluster-temporal-plot'))
    
    # 2. Répartition des clusters
    if 'cluster_distribution' in stats:
        dist = stats['cluster_distribution']
        df_dist = pd.DataFrame({
            'Cluster': [f"Cluster {i}" for i in range(len(dist))],
            'Proportion': dist
        })
        children.append(dcc.Graph(
            figure=px.bar(df_dist, x='Cluster', y='Proportion', title='Distribution des clusters', text_auto='.2f'),
            id='cluster-distribution-plot'
        ))
    
    # Visualisation des journaux par cluster
    if 'journal_info' in stats:
        # Créer un DataFrame pour la visualisation des journaux
        journal_data = []
        for cluster, info in stats['journal_info'].items():
            for journal, count in info['distribution'].items():
                journal_data.append({
                    'Cluster': f"Cluster {cluster}",
                    'Journal': journal,
                    'Articles': count
                })
        
        if journal_data:
            journal_df = pd.DataFrame(journal_data)
            journal_fig = px.bar(journal_df, x='Cluster', y='Articles', 
                                color='Journal', barmode='stack',
                                title="Distribution des journaux par cluster")
            journal_fig.update_layout(height=500)
            children.append(dcc.Graph(figure=journal_fig, id='cluster-journal-plot'))
    
    # Visualisation radar des clusters avec noms de topics
    if 'cluster_centers' in stats and len(stats['cluster_centers'][0]) <= 10 and 'topic_names_llm' in stats:
        centers = stats['cluster_centers']
        # Prendre les 6-10 premières dimensions pour le radar
        dims = min(10, len(centers[0]))
        
        # Utiliser les noms de topics LLM pour les axes du radar
        categories = []
        for i in range(dims):
            if f"topic_{i}" in stats['topic_names_llm']:
                categories.append(stats['topic_names_llm'][f"topic_{i}"])
            else:
                categories.append(f"Topic {i}")
        
        fig = go.Figure()
        for i, center in enumerate(centers):
            fig.add_trace(go.Scatterpolar(
                r=center[:dims],
                theta=categories,
                fill='toself',
                name=f'Cluster {i}'
            ))
        
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                )
            ),
            title="Poids des topics dans chaque cluster (diagramme radar)",
            height=600
        )
        children.append(dcc.Graph(figure=fig, id='cluster-radar-plot'))
    
    # 3. Poids des topics dans chaque cluster (heatmap)
    if 'cluster_centers' in stats and 'topic_names_llm' in stats:
        centers = stats['cluster_centers']
        # Créer une heatmap pour visualiser le poids de chaque topic dans les clusters
        heatmap_data = []
        for i, center in enumerate(centers):
            for j, weight in enumerate(center[:min(10, len(center))]):
                topic_name = stats['topic_names_llm'].get(f"topic_{j}", f"Topic {j}")
                heatmap_data.append({
                    'Cluster': f"Cluster {i}",
                    'Topic': topic_name,
                    'Poids': weight
                })
        
        if heatmap_data:
            heatmap_df = pd.DataFrame(heatmap_data)
            heatmap_fig = px.density_heatmap(
                heatmap_df, x='Cluster', y='Topic', z='Poids',
                title="Poids des topics dans chaque cluster",
                color_continuous_scale='Viridis'
            )
            heatmap_fig.update_layout(height=500)
            children.append(dcc.Graph(figure=heatmap_fig, id='cluster-heatmap-plot'))
    
    # Store pour stocker les données des clusters
    children.append(dcc.Store(id='cluster-data-store', data=stats))
    
    return html.Div(children)

# Page de navigation des articles par cluster
def get_article_browser_layout(cluster_data=None):
    if not cluster_data or 'labels' not in cluster_data or 'doc_ids' not in cluster_data:
        return html.Div([
            html.H3("Explorateur d'articles par cluster"),
            dbc.Alert("Aucune donnée de clustering disponible. Veuillez d'abord lancer un clustering.", color="warning"),
            dbc.Button("Retour aux visualisations", id="btn-back-to-viz", color="secondary")
        ])
    
    # Créer un dictionnaire cluster -> liste d'articles
    cluster_articles = {}
    if 'labels' in cluster_data and 'doc_ids' in cluster_data:
        for doc_id, label in zip(cluster_data['doc_ids'], cluster_data['labels']):
            if label not in cluster_articles:
                cluster_articles[label] = []
            cluster_articles[label].append(doc_id)
    
    # Vérifier si nous avons des clusters
    if not cluster_articles:
        return html.Div([
            html.H3("Explorateur d'articles par cluster"),
            dbc.Alert("Aucun cluster trouvé dans les données. Veuillez relancer le clustering.", color="warning"),
            dbc.Button("Retour aux visualisations", id="btn-back-to-viz", color="secondary")
        ])
    
    # Créer les onglets pour chaque cluster
    tabs = []
    article_modal = dbc.Modal(
        [
            dbc.ModalHeader(dbc.ModalTitle("Contenu de l'article")),
            dbc.ModalBody(id="article-details-body"),
            dbc.ModalFooter(
                dbc.Button("Fermer", id="close-article-modal", className="ms-auto", n_clicks=0)
            ),
        ],
        id="article-modal",
        is_open=False,
        size="xl",
    )
    for cluster in sorted(cluster_articles.keys()):
        # Préparer les données pour la table
        table_data = []
        for article_id in cluster_articles[cluster]:
            # Extraire les informations de l'ID
            date = "N/A"
            journal = "N/A"
            parts = article_id.split("_")
            if len(parts) > 1:
                date = parts[1]
            if len(parts) > 2:
                journal = parts[2]
            
            table_data.append({
                'article_id': article_id,
                'date': date,
                'journal': journal
            })
        
        # Créer un ID unique pour chaque table de cluster
        cluster_id = f"cluster-{cluster}"
        
        # Limiter le nombre d'articles affichés à la fois (pagination)
        items_per_page = 20
        
        # Créer la table avec des boutons explicites pour chaque article
        table = html.Div([
            # Pagination controls
            html.Div([
                dbc.Pagination(
                    id={'type': 'cluster-pagination', 'index': cluster_id},
                    max_value=max(1, (len(table_data) + items_per_page - 1) // items_per_page),
                    first_last=True,
                    previous_next=True,
                    active_page=1,
                    className="mt-3 mb-3 justify-content-center"
                )
            ]),
            # Table container with fixed height for better performance
            html.Div([
                html.Table([
                    # En-tête de la table
                    html.Thead(
                        html.Tr([
                            html.Th("ID Article", style={'width': '40%', 'textAlign': 'left', 'padding': '10px', 'backgroundColor': '#e6e6e6', 'fontWeight': 'bold'}),
                            html.Th("Date", style={'width': '20%', 'textAlign': 'left', 'padding': '10px', 'backgroundColor': '#e6e6e6', 'fontWeight': 'bold'}),
                            html.Th("Journal", style={'width': '20%', 'textAlign': 'left', 'padding': '10px', 'backgroundColor': '#e6e6e6', 'fontWeight': 'bold'}),
                            html.Th("Action", style={'width': '20%', 'textAlign': 'center', 'padding': '10px', 'backgroundColor': '#e6e6e6', 'fontWeight': 'bold'})
                        ])
                    ),
                    # Corps de la table (limité aux premiers articles)
                    html.Tbody([
                        html.Tr([
                            html.Td(row['article_id'], style={'padding': '10px', 'borderBottom': '1px solid #ddd'}),
                            html.Td(row['date'], style={'padding': '10px', 'borderBottom': '1px solid #ddd'}),
                            html.Td(row['journal'], style={'padding': '10px', 'borderBottom': '1px solid #ddd'}),
                            html.Td(
                                dbc.Button(
                                    "Voir", 
                                    id={'type': 'view-article-btn', 'index': row['article_id']},
                                    color="primary",
                                    size="sm",
                                    className="me-1",
                                    n_clicks=0
                                ),
                                style={'textAlign': 'center', 'padding': '10px', 'borderBottom': '1px solid #ddd'}
                            )
                        ]) for row in table_data[:items_per_page]  # Afficher seulement la première page
                    ], id={'type': 'cluster-table-body', 'index': cluster_id})
                ], className="table table-hover", style={'width': '100%', 'borderCollapse': 'collapse'})
            ], style={'overflowX': 'auto', 'maxHeight': '600px', 'overflowY': 'auto'}),
            # Store pour les données de la table complète
            dcc.Store(id={'type': 'cluster-table-data', 'index': cluster_id}, data=table_data)
        ])
        
        # Créer l'onglet pour ce cluster
        tabs.append(
            dbc.Tab(
                [
                    html.H4(f"Cluster {cluster} - {len(cluster_articles[cluster])} articles", className="mt-3"),
                    table,
                    html.Div(id=f'cluster-{cluster}-article-content', className="mt-3")
                ],
                label=f"Cluster {cluster}",
                tab_id=f"tab-cluster-{cluster}"
            )
        )
    
    # Déterminer l'onglet actif (avec gestion d'erreur)
    active_tab = f"tab-cluster-{min(cluster_articles.keys())}" if cluster_articles else None
    
    # Créer un Store pour stocker l'ID de l'article sélectionné
    article_id_store = dcc.Store(id='selected-article-id-store')
    
    # Ajouter le Store et la modale à la mise en page
    return html.Div([
        html.H3("Explorateur d'articles par cluster"),
        dbc.Button("Retour aux visualisations", id="btn-back-to-viz", color="secondary", className="mb-3"),
        dbc.Tabs(tabs, id="cluster-tabs", active_tab=active_tab) if tabs else html.Div(),
        article_modal,
        # Store pour stocker les données des clusters
        dcc.Store(id='browser-cluster-data-store', data=cluster_data),
        # Store pour l'ID de l'article sélectionné
        article_id_store
    ])

# Dash parser extraction (comme topic_modeling_viz)
def get_parser():
    parser = argparse.ArgumentParser(description="Clustering de documents à partir d'une matrice doc-topic.")
    parser.add_argument('--input', type=str, required=True, help='Chemin du fichier JSON doc_topic_matrix ou advanced_topic_analysis')
    parser.add_argument('--n-clusters', type=int, default=6, help='Nombre de clusters KMeans')
    parser.add_argument('--output', type=str, default=None, help='Fichier de sortie pour les labels (JSON)')
    return parser

def get_clustering_args():
    parser = get_parser()
    parser_args = []
    for action in parser._actions:
        if action.dest == 'help':
            continue
        # Set default for --input to doc_topic_matrix.json in data/results
        default = action.default
        if action.dest == 'input':
            project_root = Path(__file__).resolve().parents[2]
            default = str(project_root / 'data' / 'results' / 'doc_topic_matrix.json')
        arg = {
            'name': action.dest,
            'help': action.help,
            'default': default,
            'type': action.type.__name__ if hasattr(action.type, '__name__') else str(action.type),
            'required': action.required,
        }
        parser_args.append(arg)
    return parser_args

from dash import html as _html

def generate_dash_controls_for_clustering(args_list):
    controls = []
    for arg in args_list:
        name = arg['name']
        label = arg['help'] or name
        default = arg['default']
        required = arg['required']
        typ = arg['type']
        # String, int, bool
        if typ == 'int':
            control = dbc.Input(id=f'input-{name}', type='number', value=default, placeholder=label, required=required, min=1)
        elif typ == 'bool':
            control = dbc.Checkbox(id=f'input-{name}', value=default, className='ms-2')
        else:
            control = dbc.Input(id=f'input-{name}', type='text', value=default if default is not None else '', placeholder=label, required=required)
        
        # Utiliser la nouvelle structure recommandée (Row + Col) au lieu de FormGroup
        controls.append(
            dbc.Row([
                dbc.Col(dbc.Label(label, html_for=f'input-{name}'), width=4),
                dbc.Col(control, width=8)
            ], className="mb-3")
        )
    return controls

# Layout for clustering page (dynamic controls + results)
def get_clustering_layout():
    args_list = get_clustering_args()
    controls = generate_dash_controls_for_clustering(args_list)
    return dbc.Container([
        html.H2("Analyse de Clustering des Articles"),
        dbc.Form(controls, id="clustering-form", className="mb-3"),
        dbc.Button("Lancer le clustering", id="btn-run-clustering", color="info", className="mb-3"),
        html.Div(id="clustering-stats-output"),
        
        # Store pour les données de clustering
        dcc.Store(id="cluster-data-store", storage_type="memory"),
        
        # Page d'exploration des articles (initialement cachée)
        html.Div(id="article-browser-container", style={"display": "none"})
    ], fluid=True)

# Callback registration for clustering (to be called from app.py)
def register_clustering_callbacks(app):
    import threading
    import time
    import json
    import pathlib
    import yaml
    import os
    from dash.dependencies import Input, Output, State, ALL
    from dash import callback_context
    import subprocess
    
    @app.callback(
        [Output("clustering-stats-output", "children"),
         Output("cluster-data-store", "data")],
        [Input("btn-run-clustering", "n_clicks")],
        [State(f"input-{arg['name']}", "value") for arg in get_clustering_args()]
    )
    def run_clustering(n_clicks, *args):
        if not n_clicks:
            return [], None
        
        # Récupérer les arguments
        arg_list = get_clustering_args()
        arg_values = {arg['name']: val for arg, val in zip(arg_list, args)}
        
        # Construire la commande
        script_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "scripts", "run_topic_clustering.py")
        cmd = [sys.executable, script_path]
        for arg, val in arg_values.items():
            if val is None or val == '':
                continue
            if arg_list[list(map(lambda x: x['name'], arg_list)).index(arg)]['type'] == 'bool':
                if val:
                    cmd.append(f"--{arg.replace('_','-')}")
            else:
                cmd.append(f"--{arg.replace('_','-')}")
                cmd.append(str(val))
        # Threaded execution to avoid blocking Dash
        result_holder = {'output': None, 'data': None}
        def run_subprocess():
            try:
                # Afficher la commande qui va être exécutée
                print(f"Exécution du clustering: {sys.executable} {script_path} {' '.join(str(c) for c in cmd[2:])}")
                result = subprocess.run(cmd, capture_output=True, text=True, check=True)
                print(f"Résultat du clustering: {result.stdout}")
                
                # Charger les résultats du clustering
                project_root = Path(__file__).resolve().parents[2]
                
                # Récupérer le chemin de sortie du script de clustering
                output_arg = arg_values.get('output')
                n_clusters = arg_values.get('n_clusters', 5)
                
                if output_arg:
                    # Si un chemin de sortie est spécifié explicitement
                    if not os.path.isabs(output_arg):
                        output_path = os.path.join(project_root, output_arg)
                    else:
                        output_path = output_arg
                else:
                    # Utiliser le chemin par défaut du script de clustering
                    output_path = os.path.join(project_root, 'data', 'results', 'clusters', f'doc_clusters_k{n_clusters}.json')
                
                print(f"Recherche du fichier de résultats: {output_path}")
                if os.path.exists(output_path):
                    print(f"Fichier trouvé, chargement des stats...")
                    stats = load_clustering_stats(output_path)
                    if stats:
                        result_holder['output'] = render_clustering_stats(stats)
                        result_holder['data'] = stats
                    else:
                        result_holder['output'] = dbc.Alert("Fichier de clustering trouvé mais impossible de charger les données.", color="warning")
                        result_holder['data'] = None
                else:
                    # Si le chemin spécifié n'existe pas, essayons de trouver le fichier dans le dossier clusters
                    clusters_dir = os.path.join(project_root, 'data', 'results', 'clusters')
                    print(f"Fichier non trouvé, recherche dans: {clusters_dir}")
                    if os.path.exists(clusters_dir):
                        # Chercher le fichier le plus récent
                        cluster_files = [f for f in os.listdir(clusters_dir) if f.startswith('doc_clusters_k')]
                        if cluster_files:
                            # Trier par date de modification (le plus récent d'abord)
                            cluster_files.sort(key=lambda x: os.path.getmtime(os.path.join(clusters_dir, x)), reverse=True)
                            latest_file = os.path.join(clusters_dir, cluster_files[0])
                            print(f"Fichier le plus récent trouvé: {latest_file}")
                            stats = load_clustering_stats(latest_file)
                            if stats:
                                result_holder['output'] = render_clustering_stats(stats)
                                result_holder['data'] = stats
                            else:
                                result_holder['output'] = dbc.Alert(f"Impossible de charger les données depuis {latest_file}", color="warning")
                                result_holder['data'] = None
                        else:
                            result_holder['output'] = dbc.Alert("Aucun fichier de clustering trouvé dans le dossier clusters.", color="warning")
                            result_holder['data'] = None
                    else:
                        result_holder['output'] = dbc.Alert(f"Dossier de clustering non trouvé: {clusters_dir}", color="warning")
                        result_holder['data'] = None
            except subprocess.CalledProcessError as e:
                print(f"Erreur lors de l'exécution du clustering: {e}")
                result_holder['output'] = dbc.Alert(f"Erreur lors de l'exécution : {e.stderr}", color="danger")
                result_holder['data'] = None
            except Exception as e:
                print(f"Exception lors du clustering: {e}")
                result_holder['output'] = dbc.Alert(f"Exception: {str(e)}", color="danger")
                result_holder['data'] = None
        thread = threading.Thread(target=run_subprocess)
        thread.start()
        # Attendre la fin du thread (ou afficher un loading...)
        while thread.is_alive():
            time.sleep(0.2)
        return result_holder['output'], result_holder['data']
    
    # Callback pour détecter les clics sur les boutons "Voir" des articles
    @app.callback(
        Output('selected-article-id-store', 'data', allow_duplicate=True),
        [Input({'type': 'view-article-btn', 'index': ALL}, 'n_clicks')],
        prevent_initial_call=True
    )
    def handle_article_button_click(n_clicks_list):
        ctx = callback_context
        if not ctx.triggered:
            return no_update
        
        # Identifier quel bouton a été cliqué
        button_id = ctx.triggered[0]['prop_id'].split('.')[0]
        if button_id:
            # Extraire l'ID de l'article du bouton
            try:
                button_dict = json.loads(button_id)
                article_id = button_dict.get('index')
                if article_id:
                    print(f"Bouton cliqué pour l'article: {article_id}")
                    return article_id
            except Exception as e:
                print(f"Erreur lors de l'extraction de l'ID de l'article: {e}")
        
        return no_update
    
    # Callback pour ouvrir la modale lorsqu'un article est sélectionné
    @app.callback(
        [Output('article-modal', 'is_open', allow_duplicate=True),
         Output('article-details-body', 'children', allow_duplicate=True)],
        [Input('selected-article-id-store', 'data')],
        [State('browser-cluster-data-store', 'data'),
         State('article-modal', 'is_open')],
        prevent_initial_call=True
    )
    def open_article_modal(selected_article_id, stored_data, is_open):
        if not selected_article_id:
            return no_update, no_update
        
        print(f"Ouverture de l'article: {selected_article_id}")
        article_details = get_article_details(selected_article_id, stored_data)
        if article_details:
            return True, article_details
        
        return no_update, no_update
    
    # Callback pour fermer la modale
    @app.callback(
        Output("article-modal", "is_open", allow_duplicate=True),
        [Input("close-article-modal", "n_clicks")],
        [State("article-modal", "is_open")],
        prevent_initial_call=True
    )
    def close_modal(n_clicks, is_open):
        if n_clicks:
            return False
        return is_open
    
    # Callback pour fermer automatiquement la modale lors d'un changement d'onglet ou d'un retour aux visualisations
    @app.callback(
        Output('article-modal', 'is_open', allow_duplicate=True),
        [Input('cluster-tabs', 'active_tab'),
         Input('btn-back-to-viz', 'n_clicks')],
        prevent_initial_call=True
    )
    def close_modal_on_tab_change(active_tab, back_clicks):
        # Fermer la modale lors d'un changement d'onglet ou retour aux visualisations
        return False

    # Callback pour la pagination des tables de cluster
    @app.callback(
        Output({'type': 'cluster-table-body', 'index': MATCH}, 'children'),
        [Input({'type': 'cluster-pagination', 'index': MATCH}, 'active_page')],
        [State({'type': 'cluster-table-data', 'index': MATCH}, 'data')]
    )
    def update_table_page(page, table_data):
        if not page or not table_data:
            return no_update
        
        # Calculer les indices de début et de fin pour la pagination
        items_per_page = 20
        start_idx = (page - 1) * items_per_page
        end_idx = start_idx + items_per_page
        
        # Extraire les données pour la page actuelle
        page_data = table_data[start_idx:end_idx]
        
        # Générer les lignes de la table pour la page actuelle
        rows = [
            html.Tr([
                html.Td(row['article_id'], style={'padding': '10px', 'borderBottom': '1px solid #ddd'}),
                html.Td(row['date'], style={'padding': '10px', 'borderBottom': '1px solid #ddd'}),
                html.Td(row['journal'], style={'padding': '10px', 'borderBottom': '1px solid #ddd'}),
                html.Td(
                    dbc.Button(
                        "Voir", 
                        id={'type': 'view-article-btn', 'index': row['article_id']},
                        color="primary",
                        size="sm",
                        className="me-1",
                        n_clicks=0
                    ),
                    style={'textAlign': 'center', 'padding': '10px', 'borderBottom': '1px solid #ddd'}
                )
            ]) for row in page_data
        ]
        
        return rows
    
    # Fonction pour récupérer les détails d'un article (optimisée)
    def get_article_details(article_id, stored_data):
        try:
            # Si article_id est un dictionnaire (cellRendererData), extraire la valeur
            if isinstance(article_id, dict) and 'value' in article_id:
                article_id = article_id['value']
            
            # Chemin vers le fichier articles.json en utilisant config.yaml
            project_root = pathlib.Path(__file__).resolve().parents[2]
            
            # Utiliser un cache pour éviter de recharger config.yaml à chaque fois
            if not hasattr(get_article_details, 'config_cache'):
                with open(project_root / 'config' / 'config.yaml', encoding='utf-8') as f:
                    get_article_details.config_cache = yaml.safe_load(f)
            
            config = get_article_details.config_cache
            
            def resolve_path_from_config(path_from_config):
                if os.path.isabs(path_from_config):
                    return path_from_config
                return str(project_root / path_from_config)
            
            processed_dir = resolve_path_from_config(config['data']['processed_dir'])
            articles_json_path = os.path.join(processed_dir, 'articles.json')
            
            print(f"Recherche de l'article {article_id} dans: {articles_json_path}")
            
            # Utiliser un cache pour éviter de recharger articles.json à chaque fois
            if not hasattr(get_article_details, 'articles_cache'):
                if os.path.exists(articles_json_path):
                    with open(articles_json_path, 'r', encoding='utf-8') as f:
                        get_article_details.articles_cache = json.load(f)
                else:
                    get_article_details.articles_cache = []
            
            articles_data = get_article_details.articles_cache
            
            # Extraire l'ID de base du document (sans le suffixe éventuel)
            base_id = article_id
            
            # S'assurer que article_id est une chaîne de caractères
            article_id = str(article_id)
            base_id = str(base_id)
            
            # Extraire l'ID de base sans le suffixe "_mistral" ou similaire
            if '_' in article_id:
                parts = article_id.split("_")
                # Si le dernier segment est court (comme "mistral"), l'ignorer
                if len(parts) > 1 and len(parts[-1]) < 10:
                    base_id = '_'.join(parts[:-1])
            
            print(f"Recherche avec article_id={article_id}, base_id={base_id}")
            
            # Chercher l'article correspondant
            article = None
            for art in articles_data:
                # Convertir les identifiants en chaînes pour la comparaison
                art_id = str(art.get('id', ''))
                art_base_id = str(art.get('base_id', ''))
                
                # Vérifier correspondance exacte avec id ou base_id
                if art_id == article_id or art_base_id == article_id:
                    article = art
                    break
                
                # Vérifier avec base_id calculé
                elif art_id == base_id or art_base_id == base_id:
                    article = art
                    break
                
                # Vérifier correspondance partielle (si l'ID contient l'article_id)
                elif article_id in art_id or article_id in art_base_id:
                    article = art
                    break
                
                # Vérifier correspondance partielle inverse (si l'article_id contient l'ID)
                elif art_id in article_id or (art_base_id and art_base_id in article_id):
                    article = art
                    break
            
            if article:
                print(f"Article trouvé: {article.get('title', 'Sans titre')}")
                # Récupérer le contenu et le titre
                content = article.get('content', 'Contenu non disponible')
                title = article.get('title', 'Sans titre')
                source = article.get('url', 'Source inconnue')
                date = article.get('date', 'Date inconnue')
                
                # Trouver le cluster de l'article
                cluster = None
                if stored_data and 'labels' in stored_data and 'doc_ids' in stored_data:
                    for i, doc_id in enumerate(stored_data['doc_ids']):
                        doc_id_str = str(doc_id)
                        if doc_id_str == article_id or doc_id_str == base_id or article_id in doc_id_str or base_id in doc_id_str:
                            cluster = stored_data['labels'][i]
                            break
                
                # Construire l'affichage des détails
                details = [
                    html.H4(title, className="mb-3"),
                    html.Div([
                        html.Strong("Date: "), 
                        html.Span(f"{date}"),
                        html.Br(),
                        html.Strong("Source: "), 
                        html.A(source, href=source, target="_blank"),
                        html.Br(),
                        html.Strong("Cluster: "), 
                        html.Span(f"Cluster {cluster}" if cluster is not None else "Non classifié"),
                    ], className="mb-3"),
                    html.Hr(),
                    html.Div([
                        html.H5("Contenu:"),
                        html.P(content.replace('\n', '\n\n'), style={"whiteSpace": "pre-wrap"})
                    ])
                ]
                return details
            else:
                print(f"Article non trouvé: {article_id}")
                return [
                    html.H4(f"Article {article_id}"),
                    html.P("Cet article n'a pas été trouvé dans la base de données.")
                ]
        except Exception as e:
            print(f"Erreur lors de la récupération de l'article: {e}")
            return [
                html.H4(f"Article {article_id}"),
                html.P(f"Erreur lors de la récupération des détails: {str(e)}")
            ]
    
    # Callback pour afficher la page d'exploration des articles
    @app.callback(
        [Output("clustering-stats-output", "style"),
         Output("article-browser-container", "style"),
         Output("article-browser-container", "children")],
        [Input("btn-explore-articles", "n_clicks")],
        [State("cluster-data-store", "data")],
        prevent_initial_call=True
    )
    def toggle_article_browser(explore_clicks, cluster_data):
        if not explore_clicks:
            return {"display": "block"}, {"display": "none"}, []
        
        # Afficher la page d'exploration des articles
        return {"display": "none"}, {"display": "block"}, get_article_browser_layout(cluster_data)
    
    # Callback pour revenir à la visualisation depuis la page d'exploration
    @app.callback(
        [Output("clustering-stats-output", "style", allow_duplicate=True),
         Output("article-browser-container", "style", allow_duplicate=True)],
        [Input("btn-back-to-viz", "n_clicks")],
        prevent_initial_call=True
    )
    def back_to_visualization(back_clicks):
        if back_clicks:
            # Revenir à la page de visualisation
            return {"display": "block"}, {"display": "none"}
        return no_update, no_update
