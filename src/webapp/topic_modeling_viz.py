"""
Topic Modeling Visualization Page for Dash app
"""

import dash
from dash import html, dcc, Input, Output, State, ctx
import dash_bootstrap_components as dbc
import plotly.express as px
import plotly.graph_objects as go
import subprocess
import pathlib
import yaml
import pandas as pd
import json
import os
import sys
import threading

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
            min_ = arg['default'] if arg['default'] is not None else 0
            max_ = min_ + 20 if min_ is not None else 100
            row.append(dcc.Input(id=input_id, type="number", value=arg['default'], required=arg['required'], className="mb-2", min=min_, max=max_))
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
        ])
    ], fluid=True)

# Callback registration
def register_topic_modeling_callbacks(app):
    parser_args = get_topic_modeling_args()
    input_list = [Input(f"arg-{arg['name']}", "value") for arg in parser_args]
    input_list += [Input("btn-run-topic-modeling", "n_clicks"), Input("page-content", "children")]
    @app.callback(
        [Output("topic-modeling-run-status", "children"),
         Output("advanced-topic-stats-content", "children")],
        input_list,
        prevent_initial_call=True
    )
    def run_or_load_topic_modeling(*args):
        ctx = dash.callback_context
        status = ""
        stats_content = None
        # Split args: parser values, n_clicks, page_content
        parser_values = args[:len(parser_args)]
        n_clicks = args[len(parser_args)]
        page_content = args[len(parser_args)+1]
        trigger_id = ctx.triggered[0]["prop_id"].split(".")[0] if ctx.triggered else None
        project_root, config, advanced_topic_json = get_config_and_paths()
        if trigger_id == "btn-run-topic-modeling" and n_clicks:
            # Build argument list
            arg_list = []
            for arg, val in zip(parser_args, parser_values):
                if arg['type'] == 'bool':
                    if val:
                        arg_list.append(f"--{arg['name'].replace('_','-')}")
                elif val is not None and val != "":
                    arg_list.append(f"--{arg['name'].replace('_','-')}")
                    arg_list.append(str(val))

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
