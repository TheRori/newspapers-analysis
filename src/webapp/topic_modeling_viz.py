"""
Topic Modeling Visualization Page for Dash app
"""

print("[topic_modeling_viz] Début de l'import du module")

import dash
print("[topic_modeling_viz] dash importé")
from dash import html, dcc, Input, Output, State, ctx
print("[topic_modeling_viz] dash.html, dcc, Input, Output, State, ctx importés")
from src.webapp.topic_filter_component import get_topic_filter_component, register_topic_filter_callbacks
import dash_bootstrap_components as dbc
print("[topic_modeling_viz] dash_bootstrap_components importé")
import plotly.express as px
print("[topic_modeling_viz] plotly.express importé")
import plotly.graph_objects as go
print("[topic_modeling_viz] plotly.graph_objects importé")
import subprocess
print("[topic_modeling_viz] subprocess importé")
import pathlib
print("[topic_modeling_viz] pathlib importé")
import yaml
print("[topic_modeling_viz] yaml importé")
import pandas as pd
print("[topic_modeling_viz] pandas importé")
import json
print("[topic_modeling_viz] json importé")
import os
print("[topic_modeling_viz] os importé")
import sys
print("[topic_modeling_viz] sys importé")
import threading
print("[topic_modeling_viz] threading importé")

print("[topic_modeling_viz] Début des définitions de fonctions")

# Helper to get config and paths
def get_config_and_paths():
    project_root = pathlib.Path(__file__).resolve().parents[2]
    config_path = project_root / 'config' / 'config.yaml'
    with open(config_path, encoding='utf-8') as f:
        config = yaml.safe_load(f)
    results_dir = project_root / config['data']['results_dir']
    advanced_topic_json = results_dir / 'advanced_topic' / 'advanced_topic_analysis.json'
    return project_root, config, advanced_topic_json

# Extract parser arguments from run_topic_modeling.py
def get_topic_modeling_args():
    import importlib.util
    import sys as _sys
    import os as _os
    import argparse as _argparse
    spec = importlib.util.spec_from_file_location(
        "run_topic_modeling", _os.path.join(_os.path.dirname(__file__), "..", "scripts", "run_topic_modeling.py")
    )
    run_topic_modeling = importlib.util.module_from_spec(spec)
    _sys.modules["run_topic_modeling"] = run_topic_modeling
    spec.loader.exec_module(run_topic_modeling)
    parser = run_topic_modeling.get_parser()
    parser_args = []
    for action in parser._actions:
        if action.dest == 'help':
            continue
        # Robust: Detect boolean flags via argparse action type
        is_bool = isinstance(action, (_argparse._StoreTrueAction, _argparse._StoreFalseAction))
        arg_type = 'bool' if is_bool else (getattr(action, "type", str).__name__ if hasattr(action, "type") and getattr(action, "type") is not None else "str")
        parser_args.append({
            "name": action.dest,
            "flags": action.option_strings,
            "help": action.help,
            "required": getattr(action, "required", False),
            "default": action.default,
            "type": arg_type,
            "choices": getattr(action, "choices", None)
        })
    return parser_args

# Helper to generate dash controls for each argument
from dash import html as _html

def get_topic_modeling_controls():
    parser_args = get_topic_modeling_args()
    controls = []
    controls.append(_html.Div(f"Nombre d'arguments trouvés: {len(parser_args)}", className="alert alert-info"))
    for arg in parser_args:
        label = arg['help'] or arg['name']
        input_id = f"arg-{arg['name']}"
        row = []
        row.append(dbc.Label(label, html_for=input_id, className="mb-1 fw-bold"))
        if arg['choices']:
            options = [{'label': str(c), 'value': c} for c in arg['choices']]
            if not arg['required']:
                options = [{'label': '-- Non spécifié --', 'value': ''}] + options
            row.append(dcc.Dropdown(
                id=input_id,
                options=options,
                value=str(arg['default']) if arg['default'] is not None else '',
                clearable=not arg['required'],
                className="mb-2"
            ))
        elif arg['type'] == 'int':
            # Set appropriate min/max values based on parameter name
            if arg['name'] == 'k_min':
                min_val = 2  # Allow k_min to be as low as 2
                max_val = 50  # Reasonable upper limit
            elif arg['name'] == 'k_max':
                min_val = 5
                max_val = 100
            elif arg['name'] == 'num_topics':
                min_val = 2
                max_val = 50
            else:
                # Default values for other integer parameters
                min_val = 0
                max_val = 100
                
            row.append(dcc.Input(
                id=input_id, 
                type="number", 
                value=arg['default'], 
                required=arg['required'], 
                className="mb-2", 
                min=min_val, 
                max=max_val
            ))
        elif arg['type'] == 'bool':
            row.append(dbc.Checkbox(id=input_id, value=bool(arg['default']), className="mb-2"))
        else:
            row.append(dcc.Input(id=input_id, type="text", value=arg['default'] if arg['default'] is not None else '', required=arg['required'], className="mb-2"))
        if arg['help']:
            row.append(_html.Div(arg['help'], className="form-text text-secondary mb-2"))
        controls.append(dbc.Row([dbc.Col(c) for c in row], className="mb-2"))
    return controls

# Layout for the topic modeling page
def get_topic_modeling_layout():
    return dbc.Container([
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader(_html.H3("Paramètres du Topic Modeling", className="mb-0")),
                    dbc.CardBody([
                        _html.P("Configurez les paramètres de l'analyse thématique ci-dessous, puis cliquez sur 'Lancer'.", className="text-muted mb-3"),
                        
                        # Fichier source personnalisé
                        _html.H5("Fichier source", className="mb-2"),
                        dbc.Row([
                            dbc.Col([
                                dbc.InputGroup([
                                    dbc.Input(
                                        id="arg-input-file",
                                        type="text",
                                        placeholder="Chemin vers le fichier JSON d'articles"
                                    ),
                                    dbc.Button("Parcourir", id="source-file-browse", color="secondary")
                                ]),
                                _html.Small("Laissez vide pour utiliser le fichier par défaut de la configuration.", className="text-muted")
                            ], width=12),
                        ], className="mb-3"),
                        
                        # Sélection de fichier de cache
                        _html.H5("Fichier de cache Spacy", className="mb-2"),
                        dbc.Row([
                            dbc.Col([
                                dbc.Select(
                                    id="cache-file-select",
                                    options=[{"label": "Aucun (utiliser le plus récent)", "value": ""}],
                                    value="",
                                    className="mb-2"
                                ),

                                _html.Small("Sélectionnez un fichier de cache Spacy existant pour accélérer le traitement.", className="text-muted d-block"),
                                _html.Div(id="cache-info-display", className="mt-2")
                            ], width=12),
                        ], className="mb-3"),
                        

                        
                        dbc.Form(get_topic_modeling_controls()),
                        
                        # Add topic filter component
                        _html.H5("Filtrage par cluster", className="mt-4 mb-3"),
                        get_topic_filter_component(id_prefix="topic-filter"),
                        
                        dbc.Button("Lancer le Topic Modeling", id="btn-run-topic-modeling", color="primary", n_clicks=0, className="mt-3 mb-2"),
                        _html.Div(id="topic-modeling-run-status", className="mb-3"),
                    ]),
                ], className="mb-4 shadow"),
            ], width=12)
        ]),
        dbc.Row([
            dbc.Col([
                _html.H4("Statistiques avancées", className="mt-4 mb-3"),
                dcc.Loading(
                    id="loading-advanced-topic-stats",
                    type="default",
                    children=_html.Div(id="advanced-topic-stats-content")
                )
            ], width=12)
        ]),
        # Le Store pour l'état des filtres a été supprimé
    ], fluid=True)

# Callback registration
# Fonction pour obtenir les informations sur les fichiers de cache
def get_cache_info():
    """
    Récupère les informations sur les fichiers de cache Spacy existants.
    
    Returns:
        dict: Informations sur les fichiers de cache
    """
    project_root = pathlib.Path(__file__).resolve().parents[2]
    cache_dir = project_root / 'data' / 'cache'
    cache_files = list(cache_dir.glob("preprocessed_docs_*.pkl"))
    
    cache_info = {
        "count": len(cache_files),
        "files": []
    }
    
    for cache_file in cache_files:
        try:
            import pickle
            with open(cache_file, 'rb') as f:
                cache_data = pickle.load(f)
                
            # Extraire les informations du cache
            cache_key_data = cache_data.get('cache_key_data', {})
            articles_path = cache_key_data.get('articles_path', 'Inconnu')
            spacy_model = cache_key_data.get('spacy_model', 'Inconnu')
            allowed_pos = cache_key_data.get('allowed_pos', [])
            min_token_length = cache_key_data.get('min_token_length', 0)
            articles_count = cache_key_data.get('articles_count', 0)
            
            # Taille du fichier
            file_size_bytes = cache_file.stat().st_size
            file_size_mb = file_size_bytes / (1024 * 1024)
            
            # Date de création
            from datetime import datetime
            creation_time = datetime.fromtimestamp(cache_file.stat().st_mtime)
            
            cache_info["files"].append({
                "filename": cache_file.name,
                "articles_path": articles_path,
                "spacy_model": spacy_model,
                "allowed_pos": allowed_pos,
                "min_token_length": min_token_length,
                "articles_count": articles_count,
                "file_size_mb": file_size_mb,
                "creation_time": creation_time.strftime("%Y-%m-%d %H:%M:%S")
            })
        except Exception as e:
            print(f"Erreur lors de la lecture du fichier de cache {cache_file}: {e}")
            cache_info["files"].append({
                "filename": cache_file.name,
                "error": str(e),
                "file_size_mb": file_size_bytes / (1024 * 1024) if 'file_size_bytes' in locals() else 0
            })
    
    return cache_info

def register_topic_modeling_callbacks(app):
    # Register the topic filter component callbacks
    register_topic_filter_callbacks(app, id_prefix="topic-filter")
    parser_args = get_topic_modeling_args()
    
    # Callback pour initialiser la liste des fichiers de cache
    @app.callback(
        Output("cache-file-select", "options", allow_duplicate=True),
        Output("cache-info-display", "children"),
        Input("page-content", "children"),
        prevent_initial_call=True
    )
    def refresh_cache_list(page_content):
        cache_info = get_cache_info()
        
        # Préparer les options pour le sélecteur de cache
        cache_options = [{"label": "Aucun (utiliser le plus récent)", "value": ""}]
        
        if cache_info["count"] == 0:
            return cache_options, html.Div("Aucun fichier de cache trouvé.", className="text-muted")
        
        # Ajouter les options de cache
        for cache_file in cache_info["files"]:
            if "error" not in cache_file:
                description = f"{cache_file['filename']} ({cache_file['articles_count']} articles, {cache_file['spacy_model']})"
                cache_options.append({"label": description, "value": cache_file["filename"]})
        
        # Créer un résumé des informations de cache
        cache_summary = html.Div([
            html.P(f"{cache_info['count']} fichiers de cache trouvés", className="text-info"),
        ])
        
        return cache_options, cache_summary
    
    # Callback pour le bouton de parcourir du fichier source
    @app.callback(
        Output("arg-input-file", "value"),
        Input("source-file-browse", "n_clicks"),
        State("arg-input-file", "value"),
        prevent_initial_call=True
    )
    def browse_source_file(n_clicks, current_value):
        if not n_clicks:
            return current_value
        
        # Obtenir le répertoire de départ pour la boîte de dialogue
        project_root, _, _ = get_config_and_paths()
        data_dir = project_root / "data" / "processed"
        
        # Utiliser une commande PowerShell pour afficher une boîte de dialogue de sélection de fichier
        try:
            cmd = [
                "powershell",
                "-Command",
                "Add-Type -AssemblyName System.Windows.Forms; " +
                "$openFileDialog = New-Object System.Windows.Forms.OpenFileDialog; " +
                "$openFileDialog.InitialDirectory = '" + str(data_dir).replace('\\', '\\\\') + "'; " +
                "$openFileDialog.Filter = 'Fichiers JSON (*.json)|*.json|Tous les fichiers (*.*)|*.*'; " +
                "$openFileDialog.ShowDialog() | Out-Null; " +
                "$openFileDialog.FileName"
            ]
            result = subprocess.run(cmd, capture_output=True, text=True)
            file_path = result.stdout.strip()
            
            if file_path and os.path.exists(file_path):
                return file_path
            return current_value
        except Exception as e:
            print(f"Erreur lors de l'ouverture de la boîte de dialogue: {e}")
            return current_value
    
    # Filtres supprimés
    filtered_parser_args = parser_args
    
    # Create input list for all arguments
    input_list = [Input(f"arg-{arg['name']}", "value") for arg in filtered_parser_args]
    
    # Add cluster filter inputs
    input_list.append(Input("topic-filter-cluster-file-dropdown", "value"))
    input_list.append(Input("topic-filter-cluster-id-dropdown", "value"))
    
    # Add input for source file
    input_list.append(Input("arg-input-file", "value"))
    
    # Add input for cache file
    input_list.append(Input("cache-file-select", "value"))
    
    # Add other inputs
    input_list += [Input("btn-run-topic-modeling", "n_clicks"), Input("page-content", "children")]
    
    @app.callback(
        [Output("topic-modeling-run-status", "children"),
         Output("advanced-topic-stats-content", "children")],
        input_list,
        prevent_initial_call=True
    )
    def run_or_load_topic_modeling(*args):
        import json  # Import json at the function level to avoid variable shadowing
        ctx_trigger = ctx.triggered
        status = ""
        stats_content = None
        
        # Split args: filtered parser values, cluster values, source file, cache file, n_clicks, page_content
        filtered_parser_values = args[:len(filtered_parser_args)]
        cluster_file = args[len(filtered_parser_args)]
        cluster_id = args[len(filtered_parser_args)+1]
        source_file = args[len(filtered_parser_args)+2]
        selected_cache = args[len(filtered_parser_args)+3]
        n_clicks = args[len(filtered_parser_args)+4]
        page_content = args[len(filtered_parser_args)+5]
        
        trigger_id = ctx_trigger[0]["prop_id"].split(".")[0] if ctx_trigger else None
        project_root, config, advanced_topic_json = get_config_and_paths()
        
        # Charger les données de clustering si disponibles
        cluster_data = None
        if cluster_file and cluster_id:
            try:
                from src.webapp.topic_filter_component import load_cluster_data
                cluster_data = load_cluster_data(cluster_file)
                print(f"Données de clustering chargées depuis {cluster_file}")
            except Exception as e:
                print(f"Erreur lors du chargement des données de clustering: {e}")
        
        if trigger_id == "btn-run-topic-modeling" and n_clicks:
            # Build argument list for filtered parser args
            arg_list = []
            for arg, val in zip(filtered_parser_args, filtered_parser_values):
                if arg['type'] == 'bool':
                    if val:
                        arg_list.append(f"--{arg['name'].replace('_','-')}")
                elif val is not None and val != "":
                    arg_list.append(f"--{arg['name'].replace('_','-')}")
                    arg_list.append(str(val))
            
            # Vérifier si un fichier source personnalisé est spécifié
            if source_file:
                # Comme l'argument --input-file a été supprimé du script, nous devons créer un fichier temporaire
                # qui sera utilisé comme source d'articles par défaut
                try:
                    # Copier le fichier source personnalisé vers le chemin par défaut attendu par le script
                    import shutil
                    temp_articles_path = project_root / 'data' / 'temp' / 'custom_source.json'
                    os.makedirs(os.path.dirname(temp_articles_path), exist_ok=True)
                    shutil.copy2(source_file, temp_articles_path)
                    print(f"Fichier source personnalisé copié vers {temp_articles_path}")
                    
                    # Modifier le chemin du fichier d'articles dans la configuration
                    os.environ['TOPIC_MODELING_SOURCE_FILE'] = str(temp_articles_path)
                    print(f"Variable d'environnement TOPIC_MODELING_SOURCE_FILE définie sur {temp_articles_path}")
                except Exception as e:
                    print(f"Erreur lors de la copie du fichier source personnalisé: {e}")
            
            # Ajouter le fichier de cache sélectionné si spécifié
            if selected_cache:
                # Créer un fichier de configuration pour indiquer quel cache utiliser
                cache_config_path = project_root / "config" / "cache_config.json"
                
                with open(cache_config_path, 'w', encoding='utf-8') as f:
                    json.dump({"selected_cache": selected_cache}, f, ensure_ascii=False, indent=2)
                    
                print(f"Cache sélectionné configuré: {selected_cache}")
            else:
                # Si aucun cache n'est sélectionné, supprimer le fichier de configuration s'il existe
                cache_config_path = project_root / "config" / "cache_config.json"
                if cache_config_path.exists():
                    os.remove(cache_config_path)
                    print("Configuration de cache supprimée, le cache par défaut sera utilisé s'il existe.")

            
            # Si nous avons des données de clustering et qu'aucun fichier source personnalisé n'est spécifié
            if cluster_data and cluster_id and not source_file:
                # Charger les articles
                from src.utils.filter_utils import filter_articles_by_cluster
                
                articles_path = project_root / 'data' / 'processed' / 'articles.json'
                with open(articles_path, 'r', encoding='utf-8') as f:
                    articles = json.load(f)
                
                # Filtrer les articles par cluster
                filtered_articles = filter_articles_by_cluster(articles, cluster_id, cluster_data)
                
                # Sauvegarder les articles filtrés dans un fichier temporaire
                temp_articles_path = project_root / 'data' / 'temp' / 'filtered_articles.json'
                os.makedirs(os.path.dirname(temp_articles_path), exist_ok=True)
                with open(temp_articles_path, 'w', encoding='utf-8') as f:
                    json.dump(filtered_articles, f, ensure_ascii=False, indent=2)
                
                # Comme l'argument --input-file a été supprimé du script, nous utilisons une variable d'environnement
                os.environ['TOPIC_MODELING_SOURCE_FILE'] = str(temp_articles_path)
                
                print(f"Articles filtrés par cluster {cluster_id}: {len(filtered_articles)} articles")

            script_path = project_root / 'src' / 'scripts' / 'run_topic_modeling.py'
            try:
                # Afficher la commande qui va être exécutée
                print(f"Exécution de la commande: {sys.executable} {script_path} {' '.join(arg_list)}")
                
                # Utiliser subprocess.run pour afficher les logs directement dans le terminal
                print("\n===== DÉBUT DE L'ANALYSE TOPIC MODELING =====\n")
                
                # Exécuter le processus avec stdout et stderr non redirigés
                # Cela permettra d'afficher les logs directement dans le terminal
                process = subprocess.Popen(
                    [sys.executable, str(script_path), *arg_list],
                    # Ne pas rediriger stdout et stderr pour qu'ils s'affichent directement
                )
                
                # Stocker les lignes pour l'interface web (vide pour l'instant)
                stdout_lines = []
                stderr_lines = []
                
                # Attendre la fin du processus
                return_code = process.wait()
                
                print("\n===== FIN DE L'ANALYSE TOPIC MODELING =====\n")
                
                # Vérifier si le processus s'est terminé avec succès
                if return_code != 0:
                    raise subprocess.CalledProcessError(return_code, [sys.executable, str(script_path), *arg_list])
                
                status = dbc.Alert("Topic modeling terminé avec succès !", color="success")
            except subprocess.CalledProcessError as e:
                print("===== [run_topic_modeling.py ERROR] =====")
                print(f"Erreur lors de l'exécution (code {e.returncode})")
                status = dbc.Alert(f"Erreur lors de l'exécution (code {e.returncode})", color="danger")
        
        # Charger les stats avancées si dispo
        if advanced_topic_json.exists():
            with open(advanced_topic_json, encoding='utf-8') as f:
                stats = json.load(f)
            stats_content = render_advanced_topic_stats(stats)
        else:
            stats_content = dbc.Alert("Fichier de statistiques avancées introuvable.", color="warning")
        
        return status, stats_content
    
    # Les callbacks pour les filtres ont été supprimés

# Helper to render advanced stats (adapt to your JSON structure)
def render_advanced_topic_stats(stats):
    import plotly.express as px
    import plotly.graph_objects as go
    import pandas as pd
    from dash import html
    import traceback  # Pour le débogage
    children = []
    # 1. Coherence Score
    if 'coherence_score' in stats:
        score = stats['coherence_score']
        if score is not None:
            children.append(dbc.Alert(f"Score de cohérence : {score:.3f}", color="info", className="mb-3"))
        else:
            children.append(dbc.Alert("Score de cohérence : N/A", color="info", className="mb-3"))
    # 2. Récupérer les noms LLM s'ils existent
    topic_names_llm = None
    if stats.get('topic_names_llm'):
        # Peut être string ou dict
        if isinstance(stats['topic_names_llm'], dict):
            topic_names_llm = stats['topic_names_llm']
        else:
            try:
                import ast
                topic_names_llm = ast.literal_eval(stats['topic_names_llm'])
            except Exception:
                topic_names_llm = None
    # 2. Répartition des topics
    if 'topic_distribution' in stats:
        dist = stats['topic_distribution']
        topics = [str(i) for i in range(len(dist))]
        if topic_names_llm:
            topics = [topic_names_llm.get(f'topic_{i}', f"Topic {i}") for i in range(len(dist))]
        df_dist = pd.DataFrame({
            'Topic': topics,
            'Proportion': dist
        })
        children.append(dcc.Graph(
            figure=px.bar(df_dist, x='Topic', y='Proportion', title='Distribution des topics (proportion)', text_auto='.2f')
        ))
    # Vérifier si topic_article_counts existe
    if 'topic_article_counts' in stats:
        counts = stats['topic_article_counts']
        topics = list(counts.keys())
        if topic_names_llm:
            topics = [topic_names_llm.get(f'topic_{i}', f"Topic {i}") for i in range(len(topics))]
        df_counts = pd.DataFrame({
            'Topic': topics,
            'Articles': list(counts.values())
        })
        children.append(dcc.Graph(
            figure=px.bar(df_counts, x='Topic', y='Articles', title="Nombre d'articles par topic", text_auto=True)
        ))
    # Alternative: utiliser doc_topic_distribution pour calculer le nombre d'articles par topic
    elif 'doc_topic_distribution' in stats:
        try:
            # Récupérer la distribution doc-topic
            doc_topic_dist = stats['doc_topic_distribution']
            # Compter le nombre d'articles par topic dominant
            topic_counts = {}
            for doc_id, topic_dist in doc_topic_dist.items():
                # Trouver le topic dominant pour ce document
                dominant_topic = max(range(len(topic_dist)), key=lambda i: topic_dist[i])
                # Incrémenter le compteur pour ce topic
                topic_counts[str(dominant_topic)] = topic_counts.get(str(dominant_topic), 0) + 1
            
            # Créer un DataFrame pour la visualisation
            topic_ids = sorted([int(t) for t in topic_counts.keys()])
            topics = []
            counts = []
            for topic_id in topic_ids:
                topic_key = str(topic_id)
                if topic_names_llm:
                    topic_label = topic_names_llm.get(f'topic_{topic_id}', f"Topic {topic_id}")
                else:
                    topic_label = f"Topic {topic_id}"
                topics.append(topic_label)
                counts.append(topic_counts.get(topic_key, 0))
            
            df_counts = pd.DataFrame({
                'Topic': topics,
                'Articles': counts
            })
            children.append(dcc.Graph(
                figure=px.bar(df_counts, x='Topic', y='Articles', title="Nombre d'articles par topic", text_auto=True)
            ))
        except Exception as e:
            # En cas d'erreur, ajouter un message d'erreur au lieu de faire planter l'application
            children.append(html.Div(f"Erreur lors de la génération du graphique 'Nombre d'articles par topic': {str(e)}", 
                                    className="alert alert-danger"))
    # Autre alternative: utiliser la distribution des topics si disponible
    elif 'topic_distribution' in stats and len(dist) > 0:
        try:
            # Estimer le nombre d'articles par topic à partir de la distribution
            # Supposons que nous connaissons le nombre total d'articles
            total_docs = stats.get('total_docs', 1000)  # Valeur par défaut si non disponible
            
            topics = [str(i) for i in range(len(dist))]
            if topic_names_llm:
                topics = [topic_names_llm.get(f'topic_{i}', f"Topic {i}") for i in range(len(dist))]
            
            # Calculer le nombre d'articles par topic en fonction de la distribution
            article_counts = [int(p * total_docs) for p in dist]
            
            df_counts = pd.DataFrame({
                'Topic': topics,
                'Articles': article_counts
            })
            children.append(dcc.Graph(
                figure=px.bar(df_counts, x='Topic', y='Articles', title="Nombre d'articles par topic (estimé)", text_auto=True)
            ))
        except Exception as e:
            children.append(html.Div(f"Erreur lors de l'estimation du nombre d'articles par topic: {str(e)}", 
                                    className="alert alert-danger"))
    # 3. Top mots par topic
    if 'weighted_words' in stats:
        children.append(html.H5("Top mots par topic", className="mt-4"))
        for topic, words in stats['weighted_words'].items():
            words_df = pd.DataFrame(words, columns=['Mot', 'Poids'])
            topic_label = topic_names_llm.get(f'topic_{topic}', f"Topic {topic}") if topic_names_llm else f"Topic {topic}"
            fig = px.bar(words_df, x='Poids', y='Mot', orientation='h', title=topic_label, text_auto='.3f')
            fig.update_layout(height=350, margin=dict(l=80, r=20, t=40, b=40))
            children.append(dcc.Graph(figure=fig))
    # 4. Documents représentatifs
    if 'representative_docs' in stats:
        children.append(html.H5("Documents représentatifs par topic", className="mt-4"))
        for topic, doc_ids in stats['representative_docs'].items():
            topic_label = topic_names_llm.get(f'topic_{topic}', f"Topic {topic}") if topic_names_llm else f"Topic {topic}"
            children.append(html.P(f"{topic_label} : {', '.join(str(i) for i in doc_ids)}", className="mb-2"))
    # 5. Noms LLM
    if stats.get('llm_name'):
        children.append(html.P(f"LLM utilisé : {stats['llm_name']}", className="text-muted"))
    if stats.get('topic_names_llm'):
        children.append(html.P(f"Noms de topics LLM : {stats['topic_names_llm']}", className="text-muted"))
    return html.Div(children)

# To be called in app.py: from src.webapp.topic_modeling_viz import register_topic_modeling_callbacks, get_topic_modeling_layout
