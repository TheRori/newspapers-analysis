"""
Lexical Analysis Visualization Page for Dash app
"""

from dash import html, dcc, Input, Output, State, ctx
import dash_bootstrap_components as dbc
import plotly.express as px
import plotly.graph_objects as go
import subprocess
import importlib.util
import sys
import os
import inspect
import argparse
import glob
import yaml
import pathlib
import pandas as pd
import numpy as np

# Extract parser arguments from run_lexical_analysis.py
def get_lexical_analysis_args():
    spec = importlib.util.spec_from_file_location("run_lexical_analysis", os.path.join(os.path.dirname(__file__), "..", "scripts", "run_lexical_analysis.py"))
    run_lexical_analysis = importlib.util.module_from_spec(spec)
    sys.modules["run_lexical_analysis"] = run_lexical_analysis
    spec.loader.exec_module(run_lexical_analysis)
    parser = run_lexical_analysis.argparse.ArgumentParser(description="Analyse lexicale d'un corpus")
    run_lexical_analysis.main.__globals__["parser"] = parser
    # Parse only the arguments, not the main logic
    parser_args = [
        {"name": action.dest,
         "flags": action.option_strings,
         "help": action.help,
         "required": getattr(action, "required", False),
         "default": action.default,
         "type": getattr(action, "type", str).__name__ if hasattr(action, "type") else "str",
         "choices": getattr(action, "choices", None)
        }
        for action in parser._actions if action.dest != 'help'
    ]
    return parser_args

# Helper to get available files for --input and --techlist
def get_input_file_options():
    import yaml
    import pathlib
    import os
    project_root = pathlib.Path(__file__).resolve().parents[2]
    config_path = project_root / 'config' / 'config.yaml'
    with open(config_path, encoding='utf-8') as f:
        config = yaml.safe_load(f)
    def resolve_path_from_config(path_from_config):
        if os.path.isabs(path_from_config):
            return path_from_config
        return str(project_root / path_from_config)
    processed_dir = resolve_path_from_config(config['data']['processed_dir'])
    results_dir = resolve_path_from_config(config['data']['results_dir'])
    files = []
    for base in [processed_dir, results_dir]:
        files.extend(glob.glob(str(pathlib.Path(base) / '*.txt')))
        files.extend(glob.glob(str(pathlib.Path(base) / '*.json')))
        files.extend(glob.glob(str(pathlib.Path(base) / '*.csv')))
    return sorted(files)

def get_techlist_options():
    import yaml
    import pathlib
    import os
    project_root = pathlib.Path(__file__).resolve().parents[2]
    config_path = project_root / 'config' / 'config.yaml'
    with open(config_path, encoding='utf-8') as f:
        config = yaml.safe_load(f)
    def resolve_path_from_config(path_from_config):
        if os.path.isabs(path_from_config):
            return path_from_config
        return str(project_root / path_from_config)
    processed_dir = resolve_path_from_config(config['data']['processed_dir'])
    results_dir = resolve_path_from_config(config['data']['results_dir'])
    files = []
    for base in [processed_dir, results_dir]:
        files.extend(glob.glob(str(pathlib.Path(base) / '*.txt')))
        files.extend(glob.glob(str(pathlib.Path(base) / '*.csv')))
    return sorted(files)

# Layout for the lexical analysis page
def get_lexical_analysis_layout():
    import pandas as pd
    import os
    import plotly.express as px
    import plotly.graph_objects as go
    from dash import dcc, html
    from dash.dependencies import Input, Output, State
    import dash_bootstrap_components as dbc

    # Essayer de charger les résultats d'analyse lexicale s'ils existent
    project_root = pathlib.Path(__file__).resolve().parents[2]
    config_path = project_root / 'config' / 'config.yaml'
    with open(config_path, encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    def resolve_path_from_config(path_from_config):
        if os.path.isabs(path_from_config):
            return path_from_config
        return str(project_root / path_from_config)
    
    lexical_analysis_dir = resolve_path_from_config(config['data']['lexical_analysis_dir'])
    stats_csv_path = os.path.join(lexical_analysis_dir, 'lexical_stats_all.csv')
    cooc_graph_path = os.path.join(lexical_analysis_dir, 'cooc_graph_top50.png')
    
    # Créer les visualisations si le CSV existe
    stats_summary = html.Div([
        html.H4("Résumé de l'analyse lexicale"),
        html.P("Exécutez l'analyse pour voir les résultats")
    ])
    
    stats_graphs = html.Div()
    
    if os.path.exists(stats_csv_path):
        try:
            # Charger les données
            df = pd.read_csv(stats_csv_path)
            
            # Statistiques de base
            stats_text = [
                f"Documents analysés: {len(df)}",
                f"Moyenne de mots par document: {df['num_tokens'].mean():.1f}",
                f"Moyenne de phrases par document: {df['num_sentences'].mean():.1f}",
                f"Richesse lexicale moyenne (TTR): {df['ttr'].mean():.3f}",
                f"Entropie moyenne: {df['entropy'].mean():.3f}"
            ]
            
            # Créer un résumé des statistiques
            stats_summary = html.Div([
                html.H4("Résumé de l'analyse lexicale"),
                html.Ul([html.Li(stat) for stat in stats_text]),
                html.Hr()
            ])
            
            # Créer des graphiques
            # Création manuelle de l'histogramme avec customdata pour chaque doc_id
            df['ttr_bin'] = np.round(df['ttr'], 2)
            # S'assurer que doc_id est traité comme une chaîne de caractères
            df['doc_id'] = df['doc_id'].astype(str)
            ttr_groups = df.groupby('ttr_bin').agg({'doc_id': list, 'ttr': 'count'}).reset_index()
            fig1 = go.Figure()
            fig1.add_trace(go.Bar(
                x=ttr_groups['ttr_bin'],
                y=ttr_groups['ttr'],
                customdata=ttr_groups['doc_id'],
                marker_color='indianred',
                hovertemplate="TTR: %{x}<br>Nombre d'articles: %{y}<extra></extra>"
            ))
            fig1.update_layout(title="Distribution de la richesse lexicale (TTR)",
                              xaxis_title="Type-Token Ratio",
                              yaxis_title="Nombre de documents")
            
            # Ajouter l'ID du document comme info pour les interactions
            fig2 = px.scatter(df, x="num_tokens", y="ttr", 
                             title="Relation entre longueur et richesse lexicale",
                             labels={"num_tokens": "Nombre de mots", "ttr": "Type-Token Ratio"},
                             custom_data=["doc_id", "top_words"])
            # Mise à jour du template de survol pour afficher l'ID du document correctement
            fig2.update_traces(marker=dict(size=8), 
                              hovertemplate="<b>Document: %{customdata[0]}</b><br>Mots: %{x}<br>TTR: %{y:.3f}<br>Top mots: %{customdata[1]}<extra></extra>")
            
            fig3 = px.box(df, y=["pos_NOUN", "pos_VERB", "pos_ADJ", "pos_ADV"], 
                         title="Distribution des catégories grammaticales",
                         labels={"value": "Proportion", "variable": "Catégorie"})
            
            # Extraire et visualiser les mots les plus fréquents
            import re
            from collections import Counter
            
            # Extraire tous les mots des top_words de chaque document
            all_top_words = []
            for top_words_str in df['top_words'].dropna():
                if isinstance(top_words_str, str):
                    # Format attendu: "mot1:4,mot2:3,..."
                    matches = re.findall(r'([^:,]+):(\d+)', top_words_str)
                    for word, count in matches:
                        all_top_words.extend([word.strip()] * int(count))
            
            # Compter les occurrences
            word_counts = Counter(all_top_words).most_common(20)
            top_words_df = pd.DataFrame(word_counts, columns=['word', 'count'])
            
            # Créer le graphique des mots les plus fréquents
            fig5 = px.bar(top_words_df, x='count', y='word', orientation='h',
                         title="Top 20 des mots les plus fréquents",
                         labels={"count": "Fréquence", "word": "Mot"},
                         height=500)
            fig5.update_layout(yaxis={'categoryorder': 'total ascending'})
            
            # Créer une heatmap des corrélations
            corr_cols = ['num_sentences', 'num_tokens', 'num_types', 'ttr', 'entropy', 
                        'avg_word_length', 'avg_sent_length', 'lexical_density',
                        'pos_NOUN', 'pos_VERB', 'pos_ADJ', 'pos_ADV']
            corr_matrix = df[corr_cols].corr()
            
            fig4 = go.Figure(data=go.Heatmap(
                z=corr_matrix.values,
                x=corr_matrix.columns,
                y=corr_matrix.columns,
                colorscale='Viridis',
                zmin=-1, zmax=1
            ))
            fig4.update_layout(title="Matrice de corrélation des métriques lexicales")
            
            # Assembler les graphiques
            stats_graphs = html.Div([
                dbc.Row([
                    dbc.Col(dcc.Graph(figure=fig1, id='ttr-histogram'), width=6),
                    dbc.Col(dcc.Graph(figure=fig2, id='scatter-plot'), width=6)
                ]),
                dbc.Row([
                    dbc.Col(dcc.Graph(figure=fig3), width=6),
                    dbc.Col(dcc.Graph(figure=fig4), width=6)
                ]),
                dbc.Row([
                    dbc.Col(dcc.Graph(figure=fig5), width=12)
                ])
            ])
        except Exception as e:
            stats_summary = html.Div([
                html.H4("Erreur lors du chargement des résultats"),
                html.P(f"Détails: {str(e)}")
            ])
    
    # Afficher le graphe de cooccurrence s'il existe
    cooc_graph = html.Div()
    if os.path.exists(cooc_graph_path):
        cooc_graph = html.Div([
            html.H4("Graphe de cooccurrence (top 50 termes)"),
            html.Img(src=f"/assets/cooc_graph_top50.png", style={"max-width": "100%"}),
            html.P([
                "Graphe complet disponible au format GEXF pour Gephi: ",
                html.Code(os.path.join(lexical_analysis_dir, 'cooc_graph_global.gexf'))
            ])
        ])
    
    # Layout principal
    layout = html.Div([
        # Div invisible pour déclencher le chargement des données
        html.Div(id='_', style={'display': 'none'}),
        
        html.H2("Analyse Lexicale"),
        
        # Formulaire d'analyse
        html.Div([
            html.H3("Lancer une nouvelle analyse"),
            dbc.Row([
                dbc.Col([
                    html.Label("Fichier d'entrée:"),
                    dcc.Dropdown(
                        id='input-file-dropdown',
                        options=[{'label': f, 'value': f} for f in get_input_file_options()],
                        placeholder="Sélectionner un fichier d'entrée"
                    )
                ], width=6),
                dbc.Col([
                    html.Label("Liste technique (optionnel):"),
                    dcc.Dropdown(
                        id='techlist-dropdown',
                        options=[{'label': f, 'value': f} for f in get_techlist_options()],
                        placeholder="Sélectionner une liste technique"
                    )
                ], width=6)
            ]),
            dbc.Row([
                dbc.Col([
                    dbc.Checklist(
                        options=[
                            {"label": "Générer CSV", "value": "csv"},
                            {"label": "Graphe de cooccurrence", "value": "cooc"},
                            {"label": "Visualiser graphe", "value": "graph"},
                            {"label": "Filtrer stopwords", "value": "stopwords"}
                        ],
                        value=["csv", "cooc", "graph"],
                        id="options-checklist",
                        inline=True
                    )
                ], width=8),
                dbc.Col([
                    html.Label("Processus:"),
                    dcc.Input(
                        id='n-process-input',
                        type='number',
                        min=1,
                        max=16,
                        value=4,
                        style={'width': '100%'}
                    )
                ], width=2),
                dbc.Col([
                    html.Label("Poids min:"),
                    dcc.Input(
                        id='min-edge-weight-input',
                        type='number',
                        min=0,
                        value=5,
                        style={'width': '100%'}
                    )
                ], width=2)
            ], className="mb-3"),
            dbc.Button("Lancer l'analyse", id='run-analysis-button', color="primary")
        ], className="mb-4"),
        
        # Résultats
        html.Div([
            html.H3("Résultats"),
            html.Div(id='analysis-output', style={'whiteSpace': 'pre-line', 'maxHeight': '300px', 'overflow': 'auto', 'fontFamily': 'monospace', 'backgroundColor': '#f8f9fa', 'padding': '10px', 'marginBottom': '20px'}),
            
            # Résumé des statistiques
            stats_summary,
            
            # Graphiques des statistiques
            stats_graphs,
            
            # Graphe de cooccurrence
            cooc_graph
            
        ]),
        
        # Modal pour afficher les détails d'un document
        dbc.Modal(
            [
                dbc.ModalHeader(dbc.ModalTitle("Détails du document")),
                dbc.ModalBody(id="document-details-body"),
                dbc.ModalFooter(
                    dbc.Button("Fermer", id="close-document-modal", className="ms-auto", n_clicks=0)
                ),
            ],
            id="document-modal",
            size="lg",
            is_open=False,
        ),
        
        # Store pour stocker les données du CSV
        dcc.Store(id='lexical-data-store')
    ])
    
    return layout

# Callback registration (to be called from app.py)
def register_lexical_analysis_callbacks(app):
    from dash.dependencies import Input, Output, State
    import subprocess
    import os
    import pathlib
    import yaml
    import shutil
    import json
    import random
    
    @app.callback(
        Output('analysis-output', 'children'),
        Input('run-analysis-button', 'n_clicks'),
        State('input-file-dropdown', 'value'),
        State('techlist-dropdown', 'value'),
        State('options-checklist', 'value'),
        State('n-process-input', 'value'),
        State('min-edge-weight-input', 'value'),
        prevent_initial_call=True
    )
    def run_lexical_analysis(n_clicks, input_file, techlist, options, n_process, min_edge_weight):
        if n_clicks is None or input_file is None:
            return "Veuillez sélectionner un fichier d'entrée."
        
        # Construire la commande
        script_path = str(pathlib.Path(__file__).resolve().parents[1] / 'scripts' / 'run_lexical_analysis.py')
        config_path = str(pathlib.Path(__file__).resolve().parents[2] / 'config' / 'config.yaml')
        
        # Utiliser l'interpréteur Python actuel (celui de l'environnement virtuel)
        python_executable = sys.executable
        cmd = [python_executable, script_path, '--input', input_file, '--config', config_path]
        
        if 'csv' in options:
            cmd.append('--csv')
        if 'cooc' in options:
            cmd.append('--cooc')
        if 'graph' in options:
            cmd.append('--graph')
        if 'stopwords' in options:
            cmd.append('--stopwords-cooc')
        if techlist:
            cmd.extend(['--techlist', techlist])
        if n_process:
            cmd.extend(['--n-process', str(n_process)])
        if min_edge_weight:
            cmd.extend(['--min-edge-weight', str(min_edge_weight)])
        
        # Exécuter la commande
        try:
            # Afficher la commande qui va être exécutée
            print(f"Exécution de la commande: {' '.join(cmd)}")
            
            # Exécuter avec stdout et stderr redirigés vers le terminal et capturés
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,  # Line buffered
            )
            
            # Capturer la sortie tout en l'affichant dans le terminal
            stdout_lines = []
            stderr_lines = []
            
            # Fonction pour lire et afficher les lignes
            def read_and_print(pipe, lines_list):
                for line in iter(pipe.readline, ''):
                    print(line, end='')  # Afficher dans le terminal
                    lines_list.append(line)  # Stocker pour l'interface web
            
            # Lire stdout et stderr
            import threading
            stdout_thread = threading.Thread(target=read_and_print, args=(process.stdout, stdout_lines))
            stderr_thread = threading.Thread(target=read_and_print, args=(process.stderr, stderr_lines))
            stdout_thread.daemon = True
            stderr_thread.daemon = True
            stdout_thread.start()
            stderr_thread.start()
            
            # Attendre la fin du processus
            return_code = process.wait()
            stdout_thread.join()
            stderr_thread.join()
            
            # Vérifier si le processus s'est terminé avec succès
            if return_code != 0:
                stderr_output = ''.join(stderr_lines)
                raise subprocess.CalledProcessError(return_code, cmd, output=None, stderr=stderr_output)
            
            # Simuler un objet result comme celui retourné par subprocess.run
            class Result:
                def __init__(self, stdout, stderr, returncode):
                    self.stdout = stdout
                    self.stderr = stderr
                    self.returncode = returncode
            
            result = Result(''.join(stdout_lines), ''.join(stderr_lines), return_code)
            
            # Copier l'image du graphe dans le dossier assets si elle existe
            project_root = pathlib.Path(__file__).resolve().parents[2]
            config_path = project_root / 'config' / 'config.yaml'
            with open(config_path, encoding='utf-8') as f:
                config = yaml.safe_load(f)
            
            def resolve_path_from_config(path_from_config):
                if os.path.isabs(path_from_config):
                    return path_from_config
                return str(project_root / path_from_config)
            
            lexical_analysis_dir = resolve_path_from_config(config['data']['lexical_analysis_dir'])
            cooc_graph_path = os.path.join(lexical_analysis_dir, 'cooc_graph_top50.png')
            
            # Créer le dossier assets s'il n'existe pas
            assets_dir = pathlib.Path(__file__).parent / 'assets'
            os.makedirs(assets_dir, exist_ok=True)
            
            # Copier l'image si elle existe
            if os.path.exists(cooc_graph_path):
                shutil.copy2(cooc_graph_path, assets_dir / 'cooc_graph_top50.png')
            
            # Rafraîchir la page pour afficher les nouveaux résultats
            return result.stdout + "\n\nAnalyse terminée ! Rafraîchissez la page pour voir les graphiques mis à jour."
        except subprocess.CalledProcessError as e:
            return f"Erreur lors de l'exécution de l'analyse: {e.stderr}"
        except Exception as e:
            return f"Erreur: {str(e)}"
    
    # Charger les données du CSV au chargement de la page
    @app.callback(
        Output('lexical-data-store', 'data'),
        Input('_', 'children'),  # Dummy input pour déclencher au chargement
    )
    def load_lexical_data(_):
        try:
            project_root = pathlib.Path(__file__).resolve().parents[2]
            config_path = project_root / 'config' / 'config.yaml'
            with open(config_path, encoding='utf-8') as f:
                config = yaml.safe_load(f)
            
            def resolve_path_from_config(path_from_config):
                if os.path.isabs(path_from_config):
                    return path_from_config
                return str(project_root / path_from_config)
            
            lexical_analysis_dir = resolve_path_from_config(config['data']['lexical_analysis_dir'])
            stats_csv_path = os.path.join(lexical_analysis_dir, 'lexical_stats_all.csv')
            
            if os.path.exists(stats_csv_path):
                import pandas as pd
                df = pd.read_csv(stats_csv_path)
                
                # Récupérer le fichier d'entrée original si possible
                input_files = get_input_file_options()
                input_file_path = input_files[0] if input_files else None
                
                return {
                    'csv_data': df.to_dict('records'),
                    'input_file': input_file_path
                }
            return {}
        except Exception as e:
            print(f"Erreur lors du chargement des données: {e}")
            return {}
    
    # Callback pour ouvrir la modal avec les détails du document
    @app.callback(
        [Output('document-modal', 'is_open'),
         Output('document-details-body', 'children')],
        [Input('scatter-plot', 'clickData'),
         Input('ttr-histogram', 'clickData')],
        [State('lexical-data-store', 'data'),
         State('document-modal', 'is_open')]
    )
    def display_document_details(scatter_click, ttr_click, stored_data, is_open):
        clickData = scatter_click if scatter_click is not None else ttr_click
        if clickData is None or not stored_data:
            return is_open, "Aucune donnée disponible"
        import random
        # Si clic sur histogramme, choisir un doc_id au hasard dans la barre
        if ttr_click is not None and 'points' in ttr_click and len(ttr_click['points']) > 0:
            doc_ids = ttr_click['points'][0]['customdata']
            if isinstance(doc_ids, list) and len(doc_ids) > 0:
                doc_id = random.choice(doc_ids)
                clickData = {'points': [{'customdata': [doc_id]}]}
        
        try:
            # Récupérer l'ID du document cliqué
            doc_id = clickData['points'][0]['customdata'][0]
            
            # Trouver les données du document
            # S'assurer que la comparaison se fait avec des chaînes de caractères
            doc_id_str = str(doc_id)
            doc_data = next((doc for doc in stored_data['csv_data'] if str(doc['doc_id']) == doc_id_str), None)
            
            if doc_data:
                # Créer un tableau avec toutes les métriques du document
                metrics_table = dbc.Table(
                    [
                        html.Thead(html.Tr([html.Th("Métrique"), html.Th("Valeur")])),
                        html.Tbody([
                            html.Tr([html.Td(k), html.Td(str(v))]) 
                            for k, v in doc_data.items() 
                            if k != 'doc_id' and not pd.isna(v)
                        ])
                    ],
                    bordered=True,
                    hover=True,
                    striped=True,
                    size="sm"
                )
                
                # Récupérer le texte original du document depuis articles.json
                original_text = "Texte original non disponible"
                try:
                    # Chemin vers le fichier articles.json en utilisant config.yaml
                    project_root = pathlib.Path(__file__).resolve().parents[2]
                    config_path = project_root / 'config' / 'config.yaml'
                    
                    with open(config_path, encoding='utf-8') as f:
                        config = yaml.safe_load(f)
                    
                    def resolve_path_from_config(path_from_config):
                        if os.path.isabs(path_from_config):
                            return path_from_config
                        return str(project_root / path_from_config)
                    
                    processed_dir = resolve_path_from_config(config['data']['processed_dir'])
                    articles_json_path = os.path.join(processed_dir, 'articles.json')
                    
                    if os.path.exists(articles_json_path):
                        with open(articles_json_path, 'r', encoding='utf-8') as f:
                            articles_data = json.load(f)
                        
                        # Extraire l'ID de base du document (sans le suffixe éventuel)
                        base_id = doc_id
                        
                        # S'assurer que doc_id est une chaîne de caractères
                        doc_id = str(doc_id)
                        base_id = str(base_id)
                        
                        if '_' in doc_id:
                            # Si l'ID contient des underscores, essayer de trouver une correspondance partielle
                            base_parts = doc_id.split('_')
                            # Ignorer le dernier segment s'il semble être un suffixe (comme "mistral")
                            if len(base_parts) > 1 and len(base_parts[-1]) < 10:
                                base_id = '_'.join(base_parts[:-1])
                        
                        # Chercher l'article correspondant
                        article = None
                        for art in articles_data:
                            # Convertir les identifiants en chaînes pour la comparaison
                            art_id = str(art.get('id', ''))
                            art_base_id = str(art.get('base_id', ''))
                            
                            # Vérifier correspondance exacte avec id ou base_id
                            if art_id == doc_id or art_base_id == doc_id:
                                article = art
                                break
                            # Vérifier correspondance partielle
                            elif doc_id in art_id or doc_id in art_base_id:
                                article = art
                                break
                            # Vérifier avec base_id calculé
                            elif art_id == base_id or art_base_id == base_id:
                                article = art
                                break
                        
                        if article:
                            # Récupérer le contenu et le titre
                            content = article.get('content', 'Contenu non disponible')
                            title = article.get('title', 'Sans titre')
                            source = article.get('url', 'Source inconnue')
                            
                            original_text = f"Titre: {title}\n\n{content}\n\nSource: {source}"
                except Exception as e:
                    original_text = f"Erreur lors de la récupération du texte original: {str(e)}"
                
                # Construire le contenu de la modal
                content = [
                    html.H4(f"Document #{doc_id}"),
                    html.Hr(),
                    html.H5("Métriques lexicales"),
                    metrics_table,
                    html.Hr(),
                    html.H5("Texte original"),
                    html.P(original_text, style={"white-space": "pre-wrap"})
                ]
                
                return True, content
            else:
                return True, f"Document #{doc_id} non trouvé dans les données."
        except Exception as e:
            return True, f"Erreur lors de l'affichage des détails: {str(e)}"
    
    # Callback pour fermer la modal
    @app.callback(
        Output('document-modal', 'is_open', allow_duplicate=True),
        Input('close-document-modal', 'n_clicks'),
        State('document-modal', 'is_open'),
        prevent_initial_call=True
    )
    def close_modal(n_clicks, is_open):
        if n_clicks:
            return False
        return is_open

# To be called in app.py: from src.webapp.lexical_analysis_viz import register_lexical_analysis_callbacks
