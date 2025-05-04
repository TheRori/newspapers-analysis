"""
Topic Modeling Visualization Page for Dash app
"""

print("[topic_modeling_viz] Début de l'import du module")

import dash
print("[topic_modeling_viz] dash importé")
from dash import html, dcc, Input, Output, State, ctx
print("[topic_modeling_viz] dash.html, dcc, Input, Output, State, ctx importés")
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
                value=str(arg['default']) if arg['default'] is not None else (''),
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
                        
                        # Add collapsible filter section
                        dbc.Button(
                            "Filtres d'articles",
                            id="collapse-filter-button",
                            className="mb-3",
                            color="secondary",
                            outline=True,
                        ),
                        dbc.Collapse(
                            dbc.Card(
                                dbc.CardBody([
                                    dbc.Row([
                                        dbc.Col([
                                            _html.H5("Filtres par date", className="mb-2"),
                                            dbc.Row([
                                                dbc.Col([
                                                    dbc.Label("Date de début", html_for="arg-start-date"),
                                                    dcc.Input(
                                                        id="arg-start-date",
                                                        type="text",
                                                        placeholder="YYYY-MM-DD",
                                                        className="form-control mb-2",
                                                    ),
                                                ], width=6),
                                                dbc.Col([
                                                    dbc.Label("Date de fin", html_for="arg-end-date"),
                                                    dcc.Input(
                                                        id="arg-end-date",
                                                        type="text",
                                                        placeholder="YYYY-MM-DD",
                                                        className="form-control mb-2",
                                                    ),
                                                ], width=6),
                                            ]),
                                        ], width=12),
                                    ]),
                                    dbc.Row([
                                        dbc.Col([
                                            _html.H5("Filtres par source", className="mb-2 mt-3"),
                                            dbc.Row([
                                                dbc.Col([
                                                    dbc.Label("Journal", html_for="arg-newspaper"),
                                                    dcc.Input(
                                                        id="arg-newspaper",
                                                        type="text",
                                                        placeholder="Nom du journal",
                                                        className="form-control mb-2",
                                                    ),
                                                ], width=6),
                                                dbc.Col([
                                                    dbc.Label("Canton", html_for="arg-canton"),
                                                    dcc.Input(
                                                        id="arg-canton",
                                                        type="text",
                                                        placeholder="Code canton (ex: FR)",
                                                        className="form-control mb-2",
                                                    ),
                                                ], width=6),
                                            ]),
                                        ], width=12),
                                    ]),
                                    dbc.Row([
                                        dbc.Col([
                                            _html.H5("Filtres par contenu", className="mb-2 mt-3"),
                                            dbc.Row([
                                                dbc.Col([
                                                    dbc.Label("Tag thématique", html_for="arg-topic"),
                                                    dcc.Input(
                                                        id="arg-topic",
                                                        type="text",
                                                        placeholder="Tag thématique",
                                                        className="form-control mb-2",
                                                    ),
                                                ], width=12),
                                            ]),
                                            dbc.Row([
                                                dbc.Col([
                                                    dbc.Label("Nombre min. de mots", html_for="arg-min-words"),
                                                    dcc.Input(
                                                        id="arg-min-words",
                                                        type="number",
                                                        placeholder="Min",
                                                        className="form-control mb-2",
                                                        min=0,
                                                    ),
                                                ], width=6),
                                                dbc.Col([
                                                    dbc.Label("Nombre max. de mots", html_for="arg-max-words"),
                                                    dcc.Input(
                                                        id="arg-max-words",
                                                        type="number",
                                                        placeholder="Max",
                                                        className="form-control mb-2",
                                                        min=0,
                                                    ),
                                                ], width=6),
                                            ]),
                                        ], width=12),
                                    ]),
                                    dbc.Row([
                                        dbc.Col([
                                            dbc.Button(
                                                "Réinitialiser les filtres",
                                                id="reset-filters-button",
                                                color="secondary",
                                                outline=True,
                                                className="mt-3",
                                                size="sm",
                                            ),
                                        ], width=12, className="text-end"),
                                    ]),
                                ]),
                            ),
                            id="collapse-filter",
                            is_open=False,
                        ),
                        
                        dbc.Form(get_topic_modeling_controls()),
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
        # Store for filter state
        dcc.Store(id="filter-state"),
    ], fluid=True)

# Callback registration
def register_topic_modeling_callbacks(app):
    parser_args = get_topic_modeling_args()
    
    # Filter out the filter arguments that we're handling separately
    filter_arg_names = ['start_date', 'end_date', 'newspaper', 'canton', 'topic', 'min_words', 'max_words']
    filtered_parser_args = [arg for arg in parser_args if arg['name'] not in filter_arg_names]
    
    # Create input list for all arguments (filtered + filter arguments)
    input_list = [Input(f"arg-{arg['name']}", "value") for arg in filtered_parser_args]
    
    # Add filter inputs
    for filter_name in filter_arg_names:
        input_list.append(Input(f"arg-{filter_name}", "value"))
    
    # Add cluster filter inputs
    input_list.append(Input("topic-filter-cluster-results-dropdown", "value"))
    input_list.append(Input("topic-filter-cluster-dropdown", "value"))
    
    # Add other inputs
    input_list += [Input("btn-run-topic-modeling", "n_clicks"), Input("page-content", "children")]
    
    @app.callback(
        [Output("topic-modeling-run-status", "children"),
         Output("advanced-topic-stats-content", "children")],
        input_list,
        prevent_initial_call=True
    )
    def run_or_load_topic_modeling(*args):
        ctx_trigger = ctx.triggered
        status = ""
        stats_content = None
        
        # Split args: filtered parser values, filter values, cluster values, n_clicks, page_content
        filtered_parser_values = args[:len(filtered_parser_args)]
        filter_values = args[len(filtered_parser_args):len(filtered_parser_args)+len(filter_arg_names)]
        cluster_file = args[len(filtered_parser_args)+len(filter_arg_names)]
        cluster_id = args[len(filtered_parser_args)+len(filter_arg_names)+1]
        n_clicks = args[len(filtered_parser_args)+len(filter_arg_names)+2]
        page_content = args[len(filtered_parser_args)+len(filter_arg_names)+3]
        
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
            
            # Add filter arguments
            for filter_name, filter_value in zip(filter_arg_names, filter_values):
                if filter_value is not None and filter_value != "":
                    arg_list.append(f"--{filter_name.replace('_','-')}")
                    arg_list.append(str(filter_value))
            
            # Si nous avons des données de clustering, nous devons filtrer les articles avant de lancer le topic modeling
            if cluster_data and cluster_id:
                # Charger les articles
                import json
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
                
                # Ajouter l'argument pour utiliser le fichier temporaire
                arg_list.append("--input-file")
                arg_list.append(str(temp_articles_path))
                
                print(f"Articles filtrés par cluster {cluster_id}: {len(filtered_articles)} articles")

            script_path = project_root / 'src' / 'scripts' / 'run_topic_modeling.py'
            try:
                # Afficher la commande qui va être exécutée
                print(f"Exécution de la commande: {sys.executable} {script_path} {' '.join(arg_list)}")
                
                # Exécuter avec stdout et stderr redirigés vers le terminal et capturés en temps réel
                process = subprocess.Popen(
                    [sys.executable, str(script_path), *arg_list],
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
                    raise subprocess.CalledProcessError(return_code, [sys.executable, str(script_path), *arg_list], output=None, stderr=stderr_output)
                
                status = dbc.Alert("Topic modeling terminé avec succès !", color="success")
            except subprocess.CalledProcessError as e:
                print("===== [run_topic_modeling.py ERROR] =====")
                print(e.stderr)
                status = dbc.Alert(f"Erreur lors de l'exécution : {e.stderr}", color="danger")
        
        # Charger les stats avancées si dispo
        if advanced_topic_json.exists():
            with open(advanced_topic_json, encoding='utf-8') as f:
                stats = json.load(f)
            stats_content = render_advanced_topic_stats(stats)
        else:
            stats_content = dbc.Alert("Fichier de statistiques avancées introuvable.", color="warning")
        
        return status, stats_content
    
    # Add callback for filter collapse
    @app.callback(
        Output("collapse-filter", "is_open"),
        [Input("collapse-filter-button", "n_clicks")],
        [State("collapse-filter", "is_open")],
    )
    def toggle_collapse(n, is_open):
        if n:
            return not is_open
        return is_open
    
    # Add callback for reset filters button
    @app.callback(
        [Output(f"arg-{filter_name}", "value") for filter_name in filter_arg_names],
        [Input("reset-filters-button", "n_clicks")],
        prevent_initial_call=True,
    )
    def reset_filters(n):
        if n:
            return ["", "", "", "", "", None, None]  # Empty values for all filters
        return dash.no_update

# Helper to render advanced stats (adapt to your JSON structure)
def render_advanced_topic_stats(stats):
    import plotly.express as px
    import plotly.graph_objects as go
    import pandas as pd
    from dash import html
    children = []
    # 1. Coherence Score
    if 'coherence_score' in stats:
        children.append(dbc.Alert(f"Score de cohérence : {stats['coherence_score']:.3f}", color="info", className="mb-3"))
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
