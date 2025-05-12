"""
Fonctions de callback pour le module de suivi de termes.
"""

from dash import Input, Output, State, callback, html, ctx, ALL, dcc
import dash_bootstrap_components as dbc
import json
import re
import subprocess
import sys
import os
from pathlib import Path
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd

from src.webapp.term_tracking.utils import (
    get_articles_by_filter,
    load_config,
    extract_excerpt,
    get_term_tracking_results,
    get_cluster_files,
    get_cluster_ids,
    clean_file_path
)
from src.utils.export_utils import save_analysis
from src.webapp.term_tracking.visualizations import create_term_tracking_visualizations, create_filtered_term_tracking_visualizations
from src.webapp.term_tracking.semantic_visualizations import (
    create_semantic_drift_visualizations,
    create_similar_terms_visualizations
)
from src.webapp.topic_filter_component import (
    register_topic_filter_callbacks,
    get_filter_parameters,
    get_filter_states,
    are_filters_active
)
from src.webapp.export_component import register_export_callbacks

def normalize_article_id(article_id):
    """
    Normalise un ID d'article pour s'assurer que les comparaisons fonctionnent correctement.
    
    Args:
        article_id: ID d'article à normaliser
        
    Returns:
        ID d'article normalisé
    """
    # Si l'ID est None ou vide, retourner une chaîne vide
    if not article_id:
        return ""
    
    # Convertir en chaîne de caractères
    article_id = str(article_id)
    
    # Supprimer les préfixes courants comme "article_" s'ils sont présents
    if article_id.startswith("article_"):
        article_id = article_id[8:]
    
    # Supprimer les suffixes comme "_mistral" s'ils sont présents
    if "_mistral" in article_id:
        article_id = article_id.split("_mistral")[0]
    
    return article_id

def register_term_tracking_callbacks(app):
    """
    Enregistre les callbacks pour le module de suivi de termes.
    
    Args:
        app: L'application Dash
    """
    
    # Functions for export
    def get_term_tracking_source_data():
        """Obtient les données source pour l'exportation."""
        results_file = ctx.states.get("term-tracking-results-dropdown.value")
        viz_type = ctx.states.get("term-tracking-viz-type.value", "bar")
        
        source_data = {
            "results_file": results_file,
            "visualization_type": viz_type
        }
        
        # Si un fichier de résultats est sélectionné, ajouter des métadonnées
        if results_file:
            try:
                df = pd.read_csv(results_file)
                
                # Déterminer le type de résultats
                key_column = df.columns[0]
                
                # Identifier les colonnes de termes (toutes sauf la première)
                term_columns = df.columns[1:].tolist()
                
                source_data.update({
                    "key_column": key_column,
                    "term_columns": term_columns,
                    "num_terms": len(term_columns),
                    "num_keys": len(df),
                    "file_name": Path(results_file).name
                })
                
                # Ajouter des statistiques de base
                if len(df) > 0:
                    total_occurrences = df[term_columns].sum().sum()
                    most_frequent_term = df[term_columns].sum().idxmax()
                    
                    source_data.update({
                        "total_occurrences": int(total_occurrences),
                        "most_frequent_term": most_frequent_term
                    })
            except Exception as e:
                print(f"Erreur lors de la récupération des données source : {str(e)}")
        
        return source_data
    
    def get_term_tracking_figure():
        """Obtient la figure pour l'exportation."""
        # Récupérer les valeurs des composants
        results_file = ctx.states.get("term-tracking-results-dropdown.value")
        viz_type = ctx.states.get("term-tracking-viz-type.value", "bar")
        
        if not results_file:
            return {}
        
        try:
            # Charger les données
            df = pd.read_csv(results_file)
            
            # Vérifier si le fichier est vide
            if df.empty:
                return {}
            
            # Déterminer le type de résultats
            key_column = df.columns[0]
            
            # Identifier les colonnes de termes (toutes sauf la première)
            term_columns = df.columns[1:].tolist()
            
            # Créer la figure en fonction du type de visualisation
            if viz_type == "bar":
                # Créer un graphique à barres
                melted_df = pd.melt(df, id_vars=[key_column], value_vars=term_columns,
                                    var_name="Terme", value_name="Fréquence")
                
                fig = px.bar(
                    melted_df,
                    x=key_column,
                    y="Fréquence",
                    color="Terme",
                    title=f"Fréquence des termes par {key_column}",
                    barmode="group"
                )
            elif viz_type == "line":
                # Créer un graphique linéaire
                melted_df = pd.melt(df, id_vars=[key_column], value_vars=term_columns,
                                    var_name="Terme", value_name="Fréquence")
                
                fig = px.line(
                    melted_df,
                    x=key_column,
                    y="Fréquence",
                    color="Terme",
                    title=f"Évolution des termes par {key_column}",
                    markers=True
                )
            elif viz_type == "heatmap":
                # Créer une heatmap
                fig = px.imshow(
                    df[term_columns].T,
                    labels=dict(x=key_column, y="Terme", color="Fréquence"),
                    title="Heatmap des termes",
                    color_continuous_scale='Viridis'
                )
            else:
                # Pour le tableau, pas de figure à exporter
                return {}
            
            return fig.to_dict()
            
        except Exception as e:
            print(f"Erreur lors de la création de la figure: {str(e)}")
            return {}
    
    # Register export callbacks
    register_export_callbacks(
        app,
        analysis_type="term_tracking",
        get_source_data_function=get_term_tracking_source_data,
        get_figure_function=get_term_tracking_figure,
        button_id="term-tracking-export-button",
        modal_id="term-tracking-export-modal",
        toast_id="term-tracking-export-feedback"
    )
    
    # Enregistrer les callbacks du composant de filtrage par topic/cluster
    register_topic_filter_callbacks(app, id_prefix="term-tracking-topic-filter")
    
    # Callback pour remplir le dropdown des fichiers de clusters
    @app.callback(
        Output("term-tracking-cluster-file-dropdown", "options"),
        Input("term-tracking-cluster-file-dropdown", "id")  # Déclenché au chargement
    )
    def populate_cluster_files_dropdown(_):
        """
        Remplit le dropdown des fichiers de clusters disponibles.
        """
        return get_cluster_files()
    
    # Callback pour remplir le dropdown des IDs de clusters en fonction du fichier sélectionné
    @app.callback(
        Output("term-tracking-cluster-id-dropdown", "options"),
        Input("term-tracking-cluster-file-dropdown", "value")
    )
    def populate_cluster_ids_dropdown(cluster_file):
        """
        Remplit le dropdown des IDs de clusters disponibles en fonction du fichier sélectionné.
        """
        if not cluster_file:
            return []
        
        return get_cluster_ids(cluster_file)
    
    # Callback pour afficher/masquer les options d'analyse sémantique
    @app.callback(
        Output("semantic-drift-options", "style"),
        Input("term-tracking-semantic-drift-input", "value")
    )
    def toggle_semantic_drift_options(semantic_drift):
        """
        Affiche ou masque les options d'analyse sémantique en fonction de la sélection.
        """
        if semantic_drift:
            return {"display": "block"}
        else:
            return {"display": "none"}
    
    # Callback pour lancer l'analyse de suivi de termes
    @app.callback(
        Output("term-tracking-run-output", "children"),
        Output("term-tracking-results-dropdown", "options"),
        Output("term-tracking-results-dropdown", "value"),
        [
            Input("run-term-tracking-button", "n_clicks"),
        ],
        [
            State("term-tracking-term-file-input", "value"),
            State("term-tracking-analysis-name", "value"),
            State("term-tracking-results-dropdown", "options"),
            # États pour le filtrage par cluster
            State("term-tracking-cluster-file-dropdown", "value"),
            State("term-tracking-cluster-id-dropdown", "value"),
            # État pour le fichier source personnalisé
            State("term-tracking-source-file-input", "value"),
            # États pour l'analyse sémantique
            State("term-tracking-semantic-drift-input", "value"),
            State("term-tracking-period-type-input", "value"),
            State("term-tracking-custom-periods-input", "value"),
            State("term-tracking-vector-size-input", "value"),
            State("term-tracking-window-input", "value"),
            State("term-tracking-min-count-input", "value"),
            State("term-tracking-filter-redundant-input", "value"),
        ],
        prevent_initial_call=True
    )
    def launch_term_tracking_analysis(n_clicks, terms_file, analysis_name, current_options, 
                                     cluster_file, cluster_id, source_file,
                                     semantic_drift, period_type, custom_periods,
                                     vector_size, window, min_count, filter_redundant):
        """
        Lance l'analyse de suivi de termes avec les paramètres spécifiés.
        """
        if not n_clicks:
            return "Cliquez sur le bouton pour lancer l'analyse.", current_options, None
        
        if not terms_file:
            return "Veuillez sélectionner un fichier de termes.", current_options, None
        
        # Déterminer le chemin du fichier de sortie
        output_file = f"term_tracking_results_{analysis_name}.csv" if analysis_name else "term_tracking_results.csv"
        
        # Construire la commande
        project_root = Path(__file__).resolve().parents[3]
        script_path = project_root / "src" / "scripts" / "run_term_tracking.py"
        terms_path = Path(terms_file)
        
        # Construire les arguments de base
        args = [
            sys.executable,
            str(script_path),
            "--term-file", str(terms_path),
            "--output", output_file
        ]
        
        # Ajouter l'argument du fichier source personnalisé si spécifié
        if source_file:
            args.extend(["--source-file", source_file])
            
        # Ajouter les arguments pour l'analyse de drift sémantique si demandée
        if semantic_drift:
            args.append("--semantic-drift")
            
            # Ajouter le type de période
            if period_type:
                args.extend(["--period-type", period_type])
            
            # Ajouter les périodes personnalisées si spécifiées
            if custom_periods and period_type == "custom":
                args.extend(["--custom-periods", custom_periods])
            
            # Ajouter les paramètres du modèle Word2Vec
            if vector_size is not None:
                args.extend(["--vector-size", str(vector_size)])
            
            if window is not None:
                args.extend(["--window", str(window)])
            
            if min_count is not None:
                args.extend(["--min-count", str(min_count)])
            
            # Ajouter l'option de filtrage des redondances
            if filter_redundant is not None:
                if filter_redundant:
                    args.append("--filter-redundant")
                else:
                    args.append("--no-filter-redundant")
            
            print(f"Analyse de drift sémantique activée avec les paramètres: période={period_type}, taille_vecteur={vector_size}, fenêtre={window}, min_count={min_count}, filtrage_redondances={filter_redundant}")
        
        # Ajouter les arguments de filtrage par cluster si nécessaire
        filtered_articles = set()
        
        # Filtrage par fichier de cluster
        if cluster_file and cluster_id:
            print(f"Filtrage par fichier de cluster: {cluster_file}, cluster ID: {cluster_id}")
            try:
                with open(cluster_file, 'r', encoding='utf-8') as f:
                    cluster_data = json.load(f)
                
                # Vérifier la structure du fichier de cluster
                if "doc_ids" in cluster_data and "labels" in cluster_data:
                    # Format de fichier de cluster avec doc_ids et labels
                    print(f"Format de fichier de cluster avec doc_ids et labels détecté")
                    doc_ids = cluster_data.get("doc_ids", [])
                    labels = cluster_data.get("labels", [])
                    
                    # Convertir cluster_id en entier
                    try:
                        cluster_id_int = int(cluster_id)
                        print(f"Recherche des articles avec label {cluster_id_int}")
                        
                        # Filtrer les articles par label
                        if len(doc_ids) == len(labels):
                            for i, label in enumerate(labels):
                                if label == cluster_id_int:
                                    filtered_articles.add(doc_ids[i])
                            
                            print(f"Trouvé {len(filtered_articles)} articles dans le cluster {cluster_id_int}")
                        else:
                            print(f"Erreur: Les listes doc_ids et labels n'ont pas la même longueur")
                    except ValueError:
                        print(f"Erreur: L'ID de cluster {cluster_id} n'est pas un entier valide")
                elif "clusters" in cluster_data:
                    # Format avec une liste de clusters
                    clusters = cluster_data.get("clusters", [])
                    for cluster in clusters:
                        if str(cluster.get("id", "")) == str(cluster_id):
                            cluster_articles = cluster.get("articles", [])
                            filtered_articles.update(cluster_articles)
                            print(f"Cluster {cluster_id} contient {len(cluster_articles)} articles")
                else:
                    # Format inconnu, essayer de détecter les clusters
                    print(f"Format de fichier de cluster inconnu: {list(cluster_data.keys())}")
                    
                    # Si le fichier contient des clés numériques, supposer que ce sont des clusters
                    if str(cluster_id) in cluster_data:
                        articles = cluster_data.get(str(cluster_id), [])
                        if isinstance(articles, list):
                            filtered_articles.update(articles)
                            print(f"Cluster {cluster_id} contient {len(articles)} articles")
            except Exception as e:
                print(f"Erreur lors du chargement du fichier de cluster: {e}")
        
        # Si des articles ont été filtrés, les écrire dans un fichier temporaire
        if filtered_articles:
            import tempfile
            
            # Créer un fichier temporaire avec la liste des articles filtrés
            temp_file = tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt', encoding='utf-8')
            temp_file_path = temp_file.name
            
            try:
                # Écrire les IDs des articles dans le fichier temporaire
                for article_id in filtered_articles:
                    # Écrire l'ID complet de l'article sans normalisation
                    temp_file.write(f"{article_id}\n")
                
                # Fermer le fichier pour s'assurer que les données sont écrites
                temp_file.close()
                
                # Vérifier que le fichier a bien été créé et contient les données
                with open(temp_file_path, 'r', encoding='utf-8') as f:
                    article_count = len([line for line in f if line.strip()])
                
                print(f"Filtrage appliqué: {len(filtered_articles)} articles sélectionnés, {article_count} écrits dans {temp_file_path}")
                
                # Ajouter le fichier d'articles filtrés aux arguments
                args.extend(["--article-list", temp_file_path])
            except Exception as e:
                print(f"Erreur lors de l'écriture du fichier temporaire: {e}")
                # Continuer sans filtrage
                if os.path.exists(temp_file_path):
                    try:
                        os.unlink(temp_file_path)  # Supprimer le fichier temporaire en cas d'erreur
                    except:
                        pass
        else:
            print("Aucun article filtré, l'analyse sera effectuée sur tous les articles.")
        
        # Exécuter le script
        try:
            print(f"Exécution de la commande: {' '.join(args)}")
            
            # Déterminer le chemin de l'environnement virtuel
            venv_path = project_root / "venv"
            if os.name == 'nt':  # Windows
                python_executable = venv_path / "Scripts" / "python.exe"
            else:  # Unix/Linux/Mac
                python_executable = venv_path / "bin" / "python"
                
            if not python_executable.exists():
                # Fallback sur l'exécutable Python standard
                python_executable = sys.executable
                print(f"Environnement virtuel non trouvé, utilisation de l'exécutable Python standard: {python_executable}")
            else:
                print(f"Utilisation de l'environnement virtuel: {python_executable}")
            
            # Remplacer l'exécutable Python par celui de l'environnement virtuel
            args[0] = str(python_executable)
            
            print(f"Commande finale: {' '.join(args)}")
            
            # Exécuter le script en tant que module Python avec le répertoire racine du projet comme répertoire de travail
            result = subprocess.run(
                args,
                capture_output=True,
                text=True,
                check=True,
                cwd=project_root
            )
            
            # Récupérer la liste mise à jour des résultats disponibles
            updated_results = get_term_tracking_results()
            
            # Déterminer le chemin complet du fichier de sortie
            config = load_config()
            results_dir = project_root / config['data']['results_dir'] / "term_tracking"
            output_path = results_dir / output_file
            
            # Trouver la valeur correspondante dans la liste des résultats
            selected_value = None
            for result_option in updated_results:
                # Supprimer le paramètre de cache-busting pour la comparaison
                clean_value = result_option['value'].split('?')[0]
                if output_path.as_posix() in clean_value:
                    selected_value = result_option['value']
                    break
            
            return html.Div([
                html.H5("Analyse terminée avec succès", className="text-success"),
                html.Pre(result.stdout, style={"max-height": "300px", "overflow-y": "auto", "background-color": "#f8f9fa", "padding": "10px", "border-radius": "5px"})
            ]), updated_results, selected_value
        except subprocess.CalledProcessError as e:
            return html.Div([
                html.H5("Erreur lors de l'exécution du script", className="text-danger"),
                html.Pre(e.stdout, style={"max-height": "300px", "overflow-y": "auto", "background-color": "#f8f9fa", "padding": "10px", "border-radius": "5px"}),
                html.Pre(e.stderr, style={"max-height": "300px", "overflow-y": "auto", "background-color": "#f8f9fa", "padding": "10px", "border-radius": "5px"})
            ]), [], None
    
    # Callback pour afficher les visualisations de suivi de termes
    @app.callback(
        Output("term-tracking-visualizations-container", "children"),
        [
            Input("term-tracking-results-dropdown", "value"),
            Input("term-tracking-viz-type", "value")
        ]
    )
    def update_term_tracking_visualizations(results_file, viz_type):
        """
        Met à jour les visualisations de suivi de termes en fonction du fichier de résultats et du type de visualisation sélectionnés.
        """
        if not results_file:
            return html.Div("Aucun fichier de résultats sélectionné.")
            
        # Supprimer le paramètre de cache-busting s'il est présent
        if '?' in results_file:
            results_file = results_file.split('?')[0]
            
        # Utiliser les visualisations standard
        visualizations = create_term_tracking_visualizations(results_file, viz_type)
        
        # Ajouter un bouton pour sauvegarder l'analyse dans l'onglet médiation
        save_button = html.Div([
            html.Hr(),
            html.H5("Sauvegarder cette analyse"),
            html.P("Sauvegardez cette analyse pour la réutiliser dans l'onglet médiation."),
            dbc.Row([
                dbc.Col([
                    dbc.Label("Titre de l'analyse:"),
                    dbc.Input(
                        id="term-tracking-save-title-input",
                        type="text",
                        placeholder="Titre de l'analyse",
                        value=""
                    )
                ], width=6),
                dbc.Col([
                    dbc.Label("Description:"),
                    dbc.Textarea(
                        id="term-tracking-save-description-input",
                        placeholder="Description de l'analyse...",
                        value="",
                        style={"height": "100px"}
                    )
                ], width=6)
            ]),
            html.Br(),
            dbc.Row([
                dbc.Col([
                    dbc.Label("Collection:"),
                    dcc.Dropdown(
                        id="term-tracking-save-collection-dropdown",
                        options=[],  # Sera rempli dynamiquement
                        placeholder="Sélectionnez une collection (optionnel)"
                    )
                ], width=12)
            ]),
            html.Br(),
            dbc.Button(
                "Sauvegarder l'analyse",
                id="term-tracking-save-button",
                color="success",
                className="mt-2"
            ),
            html.Div(id="term-tracking-save-output", className="mt-3")
        ], className="mt-4")
        
        # Retourner les visualisations avec le bouton de sauvegarde
        return html.Div([
            visualizations,
            save_button
        ])
    
    # Callback pour mettre à jour les visualisations de dérive sémantique
    @app.callback(
        Output("semantic-drift-visualizations-container", "children"),
        [
            Input("semantic-drift-results-dropdown", "value"),
            Input("semantic-drift-viz-type", "value")
        ]
    )
    def update_semantic_drift_visualizations(results_file, viz_type):
        """
        Met à jour les visualisations de dérive sémantique en fonction des sélections de l'utilisateur.
        """
        if not results_file:
            return html.Div("Aucun fichier de résultats sélectionné.")
        
        # Utiliser les visualisations standard
        visualizations = create_semantic_drift_visualizations(results_file, viz_type)
        
        # Ajouter un bouton pour sauvegarder l'analyse dans l'onglet médiation
        save_button = html.Div([
            html.Hr(),
            html.H5("Sauvegarder cette analyse"),
            html.P("Sauvegardez cette analyse pour la réutiliser dans l'onglet médiation."),
            dbc.Row([
                dbc.Col([
                    dbc.Label("Titre de l'analyse:"),
                    dbc.Input(
                        id="semantic-drift-save-title-input",
                        type="text",
                        placeholder="Titre de l'analyse",
                        value=""
                    )
                ], width=6),
                dbc.Col([
                    dbc.Label("Description:"),
                    dbc.Textarea(
                        id="semantic-drift-save-description-input",
                        placeholder="Description de l'analyse...",
                        value="",
                        style={"height": "100px"}
                    )
                ], width=6)
            ]),
            html.Br(),
            dbc.Row([
                dbc.Col([
                    dbc.Label("Collection:"),
                    dcc.Dropdown(
                        id="semantic-drift-save-collection-dropdown",
                        options=[],  # Sera rempli dynamiquement
                        placeholder="Sélectionnez une collection (optionnel)"
                    )
                ], width=12)
            ]),
            html.Br(),
            dbc.Button(
                "Sauvegarder l'analyse",
                id="semantic-drift-save-button",
                color="success",
                className="mt-2"
            ),
            html.Div(id="semantic-drift-save-output", className="mt-3")
        ], className="mt-4")
        
        # Retourner les visualisations avec le bouton de sauvegarde
        return html.Div([
            visualizations,
            save_button
        ])
    
    # Callback pour le bouton de parcourir du fichier source
    @app.callback(
        Output("term-tracking-source-file-input", "value"),
        Input("term-tracking-source-file-browse", "n_clicks"),
        State("term-tracking-source-file-input", "value"),
        prevent_initial_call=True
    )
    def browse_source_file(n_clicks, current_value):
        """
        Ouvre une boîte de dialogue pour sélectionner un fichier source JSON.
        """
        if not n_clicks:
            return current_value
        
        # Obtenir le répertoire de départ pour la boîte de dialogue
        project_root = Path(__file__).resolve().parents[3]
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
    
    # Callback pour mettre à jour les visualisations de termes similaires
    @app.callback(
        Output("similar-terms-visualizations-container", "children"),
        [
            Input("similar-terms-results-dropdown", "value"),
            Input("similar-terms-viz-type", "value")
        ]
    )
    def update_similar_terms_visualizations(results_file, viz_type):
        """
        Met à jour les visualisations de termes similaires en fonction des sélections de l'utilisateur.
        """
        if not results_file:
            return html.Div("Aucun fichier de résultats sélectionné.")
        
        # Utiliser les visualisations standard
        visualizations = create_similar_terms_visualizations(results_file, viz_type)
        
        # Ajouter un bouton pour sauvegarder l'analyse dans l'onglet médiation
        save_button = html.Div([
            html.Hr(),
            html.H5("Sauvegarder cette analyse"),
            html.P("Sauvegardez cette analyse pour la réutiliser dans l'onglet médiation."),
            dbc.Row([
                dbc.Col([
                    dbc.Label("Titre de l'analyse:"),
                    dbc.Input(
                        id="similar-terms-save-title-input",
                        type="text",
                        placeholder="Titre de l'analyse",
                        value=""
                    )
                ], width=6),
                dbc.Col([
                    dbc.Label("Description:"),
                    dbc.Textarea(
                        id="similar-terms-save-description-input",
                        placeholder="Description de l'analyse...",
                        value="",
                        style={"height": "100px"}
                    )
                ], width=6)
            ]),
            html.Br(),
            dbc.Row([
                dbc.Col([
                    dbc.Label("Collection:"),
                    dcc.Dropdown(
                        id="similar-terms-save-collection-dropdown",
                        options=[],  # Sera rempli dynamiquement
                        placeholder="Sélectionnez une collection (optionnel)"
                    )
                ], width=12)
            ]),
            html.Br(),
            dbc.Button(
                "Sauvegarder l'analyse",
                id="similar-terms-save-button",
                color="success",
                className="mt-2"
            ),
            html.Div(id="similar-terms-save-output", className="mt-3")
        ], className="mt-4")
        
        # Retourner les visualisations avec le bouton de sauvegarde
        return html.Div([
            visualizations,
            save_button
        ])
    
    def update_similar_terms_visualizations(results_file, viz_type):
        """
        Met à jour les visualisations de termes similaires en fonction des sélections de l'utilisateur.
        """
        if not results_file:
            return html.Div("Aucun fichier de résultats sélectionné.")
        
        return create_similar_terms_visualizations(results_file, viz_type)
    
    # Callback pour mettre à jour le graphique en réseau des termes similaires en fonction de la période sélectionnée
    @app.callback(
        Output("similar-terms-network-graph", "figure"),
        [
            Input("similar-terms-period-selector", "value"),
            Input("similar-terms-results-dropdown", "value")
        ]
    )
    def update_similar_terms_network(selected_period, results_file):
        """
        Met à jour le graphique en réseau des termes similaires en fonction de la période sélectionnée.
        """
        from src.webapp.term_tracking.semantic_visualizations import create_similar_terms_visualizations
        
        if not results_file or not selected_period:
            return {}
        
        # Créer les visualisations pour récupérer le graphique en réseau
        visualizations = create_similar_terms_visualizations(results_file, "network")
        
        # Extraire le graphique du résultat
        graph = None
        for child in visualizations.children:
            if hasattr(child, 'id') and child.id == "similar-terms-network-graph":
                graph = child
                break
        
        if graph is None or not hasattr(graph, 'figure'):
            return {}
        
        return graph.figure
    
    # Callback pour afficher les articles lorsqu'on clique sur un graphique (pattern-matching sur tous les graphes)
    @app.callback(
        Output("articles-modal-body", "children"),
        Output("articles-modal", "is_open"),
        [
            Input({'type': 'term-tracking-graph', 'subtype': ALL}, 'clickData'),
            Input("close-articles-modal", "n_clicks"),
        ],
        State("term-tracking-results-dropdown", "value"),
        prevent_initial_call=True
    )
    def handle_articles_modal(graph_click_list, close_clicks, results_file):
        import json
        print("DEBUG: callback handle_articles_modal triggered", ctx.triggered, flush=True)
        import pathlib
        import pandas as pd
        
        # Vérifier si le callback a été déclenché
        if not ctx.triggered:
            print("DEBUG: Pas de ctx.triggered", flush=True)
            return "", False
        
        trigger = ctx.triggered[0]
        prop_id = trigger['prop_id']
        print(f"DEBUG: prop_id={prop_id}", flush=True)
        
        # Si fermeture du modal
        if "close-articles-modal" in prop_id:
            print("DEBUG: fermeture modal", flush=True)
            return "", False
        
        if not results_file:
            print("DEBUG: Pas de results_file", flush=True)
            return "", False
        
        # Récupérer les données du clic
        click_data = trigger['value']
        print(f"DEBUG: click_data={click_data}", flush=True)
        
        # Vérifier si le clic provient d'un graphique (pattern-matching)
        if 'term-tracking-graph' not in prop_id:
            print("DEBUG: prop_id ne contient pas term-tracking-graph", flush=True)
            return "", False
        
        if click_data is None or 'points' not in click_data or not click_data['points']:
            print("DEBUG: click_data mal formé", flush=True)
            return "", False
        
        try:
            point = click_data['points'][0]
            # Extraire l'ID du graphique (sous forme de dictionnaire)
            graph_id = json.loads(prop_id.split('.')[0])
            subtype = graph_id.get('subtype', '')
            print(f"DEBUG: subtype={subtype}", flush=True)
            filter_type = None
            filter_value = None
            term = None
            
            # Analyser le sous-type du graphique pour déterminer le type de filtre
            if 'year' in subtype:
                filter_type = "année"
                filter_value = point.get('x')
                # Si c'est un graphique à barres ou ligne, récupérer l'indice de la courbe pour le terme
                if 'curveNumber' in point:
                    curve_index = point.get('curveNumber')
                    # Nous récupérerons le nom du terme plus tard
                    term = curve_index
            elif 'journal' in subtype:
                filter_type = "journal"
                filter_value = point.get('x')
                if 'curveNumber' in point:
                    curve_index = point.get('curveNumber')
                    term = curve_index
            elif 'term-pie' in subtype:
                filter_type = "terme"
                filter_value = point.get('label')
                term = filter_value
            elif 'article' in subtype:
                filter_type = "article"
                filter_value = point.get('x')
            else:
                print(f"DEBUG: Type de graphique non géré: {subtype}", flush=True)
                return html.P("Type de graphique non géré."), True
            
            # Charger les articles
            project_root = pathlib.Path(__file__).resolve().parents[3]
            config = load_config()
            articles_path = project_root / config['data']['processed_dir'] / "articles.json"
            with open(articles_path, 'r', encoding='utf-8') as f:
                articles = json.load(f)
            
            # Charger les colonnes de termes
            # Nettoyer le chemin du fichier pour supprimer les paramètres de cache-busting
            clean_results_file = clean_file_path(results_file)
            df_results = pd.read_csv(clean_results_file)
            term_columns = df_results.columns[1:].tolist()
            
            # Si term est un indice de courbe, récupérer le nom du terme
            if term is not None and isinstance(term, int) and term < len(term_columns):
                term_name = term_columns[term]
            elif term is not None and isinstance(term, str):
                term_name = term
            else:
                term_name = None
            
            # Filtrer les articles
            filtered_articles = get_articles_by_filter(
                articles=articles,
                filter_type=filter_type,
                filter_value=filter_value,
                term=term_name
            )
            
            print(f"DEBUG: {len(filtered_articles)} articles filtrés", flush=True)
            max_articles = 20
            show_limit_message = False
            if len(filtered_articles) > max_articles:
                filtered_articles = filtered_articles[:max_articles]
                show_limit_message = True
            
            if not filtered_articles:
                print("DEBUG: Aucun article trouvé", flush=True)
                return html.P("Aucun article trouvé correspondant aux critères."), True
            
            # Générer le contenu
            article_cards = []
            stored_articles = []  # Pour stocker les articles pour le modal complet
            
            for i, article in enumerate(filtered_articles):
                article_id = article.get('id', article.get('base_id', 'Inconnu'))
                title = article.get('title', 'Sans titre')
                date = article_id.split('_')[1] if '_' in article_id else 'Date inconnue'
                journal = article_id.split('_')[2] if len(article_id.split('_')) > 2 else 'Journal inconnu'
                text = article.get('text', article.get('content', ''))
                url = article.get('url', '')
                
                # Stocker l'article pour le modal complet
                stored_articles.append({
                    'id': article_id,
                    'title': title,
                    'date': date,
                    'journal': journal,
                    'text': text,
                    'url': url
                })
                
                # Créer un extrait du texte avec le terme mis en évidence
                excerpt = extract_excerpt(text, term_name) if term_name else text[:300] + "..."
                
                # Créer un lien vers l'article original si disponible
                article_link = None
                if url:
                    article_link = html.A("Voir l'article original", href=url, target="_blank", className="btn btn-sm btn-primary mt-2 mb-2")
                
                card = dbc.Card([
                    dbc.CardHeader([
                        html.H5(title, className="card-title"),
                        html.H6(f"{date} - {journal}", className="card-subtitle text-muted")
                    ]),
                    dbc.CardBody([
                        html.P(dcc.Markdown(excerpt), className="card-text"),
                        html.Div([
                            dbc.Button(
                                "Afficher l'article complet", 
                                id={'type': 'show-full-article', 'index': i},
                                color="link", 
                                className="mt-2"
                            ),
                            article_link if article_link else html.Div(),
                        ], className="d-flex justify-content-between")
                    ])
                ], className="mb-3")
                article_cards.append(card)
            
            # Stocker les articles dans un composant Store pour les récupérer dans le callback du modal complet
            store = dcc.Store(id="stored-articles", data=stored_articles)
            
            limit_message = html.Div([
                html.Hr(),
                html.P(f"Affichage limité aux {max_articles} premiers articles.", className="text-muted")
            ]) if show_limit_message else html.Div()
            
            modal_content = html.Div([
                store,
                html.H4(f"Articles contenant '{term_name}'" if term_name else "Articles correspondants"),
                html.P(f"Filtre: {filter_type} = {filter_value}" if filter_type else ""),
                html.Hr(),
                html.Div(article_cards),
                limit_message
            ])
            
            print("DEBUG: Modal prêt à s'afficher", flush=True)
            return modal_content, True
        
        except Exception as e:
            print(f"DEBUG: Exception {e}", flush=True)
            return html.P(f"Erreur lors de la récupération des articles: {str(e)}"), True

    def extract_excerpt(text, term, context_size=100):
        """
        Extrait un extrait de texte contenant le terme recherché.
        
        Args:
            text: Le texte complet
            term: Le terme à rechercher
            context_size: Nombre de caractères à inclure avant et après le terme
            
        Returns:
            Extrait de texte avec le terme mis en évidence
        """
        if not text or not term:
            return ""
        
        # Créer un pattern insensible à la casse
        pattern = re.compile(re.escape(term), re.IGNORECASE)
        
        # Trouver la première occurrence du terme
        match = pattern.search(text)
        if not match:
            return text[:200] + "..."  # Retourner le début du texte si le terme n'est pas trouvé
        
        # Déterminer les indices de début et de fin de l'extrait
        start = max(0, match.start() - context_size)
        end = min(len(text), match.end() + context_size)
        
        # Extraire l'extrait
        excerpt = text[start:end]
        
        # Ajouter des ellipses si nécessaire
        if start > 0:
            excerpt = "..." + excerpt
        if end < len(text):
            excerpt = excerpt + "..."
        
        # Mettre en évidence le terme dans l'extrait
        highlighted_excerpt = pattern.sub(lambda m: f"**{m.group(0)}**", excerpt)
        
        return highlighted_excerpt

    # Callback pour afficher l'article complet dans un modal séparé
    @app.callback(
        Output("full-article-modal-body", "children"),
        Output("full-article-modal", "is_open"),
        [
            Input({"type": "show-full-article", "index": ALL}, "n_clicks"),
            Input("close-full-article-modal", "n_clicks"),
        ],
        [
            State("stored-articles", "data")
        ],
        prevent_initial_call=True
    )
    def handle_full_article_modal(show_clicks, close_clicks, stored_articles):
        """
        Affiche le contenu complet d'un article dans un modal séparé.
        """
        if not ctx.triggered:
            return "", False
            
        trigger = ctx.triggered[0]
        prop_id = trigger['prop_id']
        
        # Si fermeture du modal
        if "close-full-article-modal" in prop_id:
            return "", False
            
        if not stored_articles:
            return html.P("Aucun article disponible."), True
            
        try:
            # Extraire l'index de l'article à afficher
            article_index = json.loads(prop_id.split('.')[0])['index']
            
            if article_index >= len(stored_articles):
                return html.P("Article non trouvé."), True
                
            article = stored_articles[article_index]
            
            # Créer le contenu du modal
            title = article.get('title', 'Sans titre')
            date = article.get('date', '')
            journal = article.get('journal', '')
            text = article.get('text', '')
            url = article.get('url', '')
            
            # Créer un lien vers l'article original si disponible
            article_link = None
            if url:
                article_link = html.Div([
                    html.A("Voir l'article original", href=url, target="_blank", className="btn btn-primary mt-3")
                ])
            
            # Mettre en forme le contenu
            content = html.Div([
                html.H3(title, className="mb-2"),
                html.H5(f"{date} - {journal}", className="text-muted mb-4"),
                html.Hr(),
                dcc.Markdown(text, className="article-text"),
                article_link if article_link else html.Div()
            ])
            
            return content, True
            
        except Exception as e:
            print(f"Erreur lors de l'affichage de l'article complet : {str(e)}")
            return html.P(f"Erreur lors de l'affichage de l'article : {str(e)}"), True
    
    # Callback pour remplir les dropdowns de collections
    @app.callback(
        [
            Output("term-tracking-save-collection-dropdown", "options"),
            Output("semantic-drift-save-collection-dropdown", "options"),
            Output("similar-terms-save-collection-dropdown", "options")
        ],
        Input("term-tracking-save-collection-dropdown", "id")  # Déclenché au chargement
    )
    def populate_collection_dropdowns(_):
        """
        Remplit les dropdowns de collections pour tous les types d'analyses.
        """
        from src.utils.export_utils import get_collections
        from src.webapp.term_tracking.utils import load_config
        
        try:
            # Charger la configuration
            config = load_config()
            
            # Obtenir les collections disponibles
            collections = get_collections(config=config)
            
            # Formater les options pour les dropdowns
            options = [{'label': coll['name'], 'value': coll['name']} for coll in collections]
            
            # Retourner les mêmes options pour tous les dropdowns
            return options, options, options
        except Exception as e:
            print(f"Erreur lors du chargement des collections: {str(e)}")
            return [], [], []
    
    # Callback pour sauvegarder l'analyse de suivi de termes
    @app.callback(
        Output("term-tracking-save-output", "children"),
        [
            Input("term-tracking-save-button", "n_clicks")
        ],
        [
            State("term-tracking-save-title-input", "value"),
            State("term-tracking-save-description-input", "value"),
            State("term-tracking-save-collection-dropdown", "value"),
            State("term-tracking-results-dropdown", "value"),
            State("term-tracking-viz-type", "value")
        ],
        prevent_initial_call=True
    )
    def save_term_tracking_analysis(n_clicks, title, description, collection, results_file, viz_type):
        """
        Sauvegarde l'analyse de suivi de termes pour la réutiliser dans l'onglet médiation.
        """
        from src.utils.export_utils import save_analysis
        from src.webapp.term_tracking.utils import load_config, clean_file_path
        import plotly.graph_objects as go
        import os
        import pandas as pd
        import json
        from pathlib import Path
        
        if not n_clicks:
            return ""
        
        if not title:
            return html.Div("Veuillez spécifier un titre pour l'analyse.", className="text-danger")
        
        if not results_file:
            return html.Div("Aucun fichier de résultats sélectionné.", className="text-danger")
        
        try:
            # Nettoyer le chemin du fichier pour supprimer les paramètres de cache-busting
            clean_results_file = clean_file_path(results_file)
            
            # Charger la configuration
            config = load_config()
            
            # Obtenir le chemin du fichier d'articles par défaut
            project_root = Path(__file__).resolve().parents[3]
            default_articles_path = project_root / config['data']['processed_dir'] / "articles.json"
            
            # Vérifier s'il existe un fichier de métadonnées associé au fichier de résultats
            meta_file = Path(clean_results_file).with_suffix('.meta.json')
            source_file = str(default_articles_path)
            term_file = None
            analysis_type = None
            statistics = None
            
            if meta_file.exists():
                try:
                    with open(meta_file, 'r', encoding='utf-8') as f:
                        meta_data = json.load(f)
                        if 'source_file' in meta_data:
                            source_file = meta_data['source_file']
                        if 'term_file' in meta_data:
                            term_file = meta_data['term_file']
                        if 'analysis_type' in meta_data:
                            analysis_type = meta_data['analysis_type']
                        if 'statistics' in meta_data:
                            statistics = meta_data['statistics']
                except Exception as e:
                    print(f"Erreur lors de la lecture du fichier de métadonnées: {e}")
            
            # Charger les données pour la visualisation
            df = pd.read_csv(clean_results_file)
            
            # Préparer les données source avec tous les fichiers nécessaires
            source_data = {
                "results_file": clean_results_file,
                "viz_type": viz_type,
                "analysis_parameters": {
                    "articles_file": source_file,
                    "term_file": term_file,
                    "analysis_type": analysis_type
                },
                "statistics": statistics
            }
            
            # Créer une figure factice (sera remplacée par la vraie figure dans le callback)
            figure = go.Figure()
            
            # Sauvegarder l'analyse avec la collection spécifiée
            analysis_id = save_analysis(
                title=title,
                description=description,
                source_data=source_data,
                analysis_type="term_tracking",
                figure=figure,
                collection=collection,
                config=config,
                save_source_files=True
            )
            
            return html.Div([
                html.P("Analyse sauvegardée avec succès!", className="text-success"),
                html.P(f"ID: {analysis_id}")
            ])
        except Exception as e:
            print(f"Erreur lors de la sauvegarde de l'analyse: {str(e)}")
            return html.Div(f"Erreur lors de la sauvegarde: {str(e)}", className="text-danger")
    
    # Callback pour sauvegarder l'analyse de dérive sémantique
    @app.callback(
        Output("semantic-drift-save-output", "children"),
        [
            Input("semantic-drift-save-button", "n_clicks")
        ],
        [
            State("semantic-drift-save-title-input", "value"),
            State("semantic-drift-save-description-input", "value"),
            State("semantic-drift-save-collection-dropdown", "value"),
            State("semantic-drift-results-dropdown", "value"),
            State("semantic-drift-viz-type", "value")
        ],
        prevent_initial_call=True
    )
    def save_semantic_drift_analysis(n_clicks, title, description, collection, results_file, viz_type):
        """
        Sauvegarde l'analyse de dérive sémantique pour la réutiliser dans l'onglet médiation.
        """
        from src.utils.export_utils import save_analysis
        from src.webapp.term_tracking.utils import load_config, clean_file_path
        import plotly.graph_objects as go
        import os
        import pandas as pd
        import json
        from pathlib import Path
        
        if not n_clicks:
            return ""
        
        if not title:
            return html.Div("Veuillez spécifier un titre pour l'analyse.", className="text-danger")
        
        if not results_file:
            return html.Div("Aucun fichier de résultats sélectionné.", className="text-danger")
        
        try:
            # Nettoyer le chemin du fichier pour supprimer les paramètres de cache-busting
            clean_results_file = clean_file_path(results_file)
            
            # Charger la configuration
            config = load_config()
            
            # Obtenir le chemin du fichier d'articles par défaut
            project_root = Path(__file__).resolve().parents[3]
            default_articles_path = project_root / config['data']['processed_dir'] / "articles.json"
            
            # Vérifier s'il existe un fichier de métadonnées associé au fichier de résultats
            meta_file = Path(clean_results_file).with_suffix('.meta.json')
            source_file = str(default_articles_path)
            term_file = None
            analysis_type = None
            statistics = None
            model_path = None
            
            if meta_file.exists():
                try:
                    with open(meta_file, 'r', encoding='utf-8') as f:
                        meta_data = json.load(f)
                        if 'source_file' in meta_data:
                            source_file = meta_data['source_file']
                        if 'term_file' in meta_data:
                            term_file = meta_data['term_file']
                        if 'analysis_type' in meta_data:
                            analysis_type = meta_data['analysis_type']
                        if 'statistics' in meta_data:
                            statistics = meta_data['statistics']
                        if 'model_path' in meta_data:
                            model_path = meta_data['model_path']
                except Exception as e:
                    print(f"Erreur lors de la lecture du fichier de métadonnées: {e}")
            
            # Charger les données pour la visualisation
            df = pd.read_csv(clean_results_file)
            
            # Préparer les données source avec tous les fichiers nécessaires
            source_data = {
                "results_file": clean_results_file,
                "viz_type": viz_type,
                "analysis_parameters": {
                    "articles_file": source_file,
                    "term_file": term_file,
                    "analysis_type": analysis_type
                },
                "statistics": statistics
            }
            
            # Ajouter le chemin du modèle Word2Vec si disponible
            if model_path:
                source_data["model_path"] = model_path
            
            # Créer une figure factice (sera remplacée par la vraie figure dans le callback)
            figure = go.Figure()
            
            # Sauvegarder l'analyse avec la collection spécifiée
            analysis_id = save_analysis(
                title=title,
                description=description,
                source_data=source_data,
                analysis_type="semantic_drift",
                figure=figure,
                collection=collection,
                config=config,
                save_source_files=True
            )
            
            return html.Div([
                html.P("Analyse sauvegardée avec succès!", className="text-success"),
                html.P(f"ID: {analysis_id}")
            ])
        except Exception as e:
            print(f"Erreur lors de la sauvegarde de l'analyse: {str(e)}")
            return html.Div(f"Erreur lors de la sauvegarde: {str(e)}", className="text-danger")
    
    # Callback pour sauvegarder l'analyse de termes similaires
    @app.callback(
        Output("similar-terms-save-output", "children"),
        [
            Input("similar-terms-save-button", "n_clicks")
        ],
        [
            State("similar-terms-save-title-input", "value"),
            State("similar-terms-save-description-input", "value"),
            State("similar-terms-save-collection-dropdown", "value"),
            State("similar-terms-results-dropdown", "value"),
            State("similar-terms-viz-type", "value")
        ],
        prevent_initial_call=True
    )
    def save_similar_terms_analysis(n_clicks, title, description, collection, results_file, viz_type):
        """
        Sauvegarde l'analyse de termes similaires pour la réutiliser dans l'onglet médiation.
        """
        from src.utils.export_utils import save_analysis
        from src.webapp.term_tracking.utils import load_config, clean_file_path
        import plotly.graph_objects as go
        import os
        import pandas as pd
        import json
        from pathlib import Path
        
        if not n_clicks:
            return ""
        
        if not title:
            return html.Div("Veuillez spécifier un titre pour l'analyse.", className="text-danger")
        
        if not results_file:
            return html.Div("Aucun fichier de résultats sélectionné.", className="text-danger")
        
        try:
            # Nettoyer le chemin du fichier pour supprimer les paramètres de cache-busting
            clean_results_file = clean_file_path(results_file)
            
            # Charger la configuration
            config = load_config()
            
            # Obtenir le chemin du fichier d'articles par défaut
            project_root = Path(__file__).resolve().parents[3]
            default_articles_path = project_root / config['data']['processed_dir'] / "articles.json"
            
            # Vérifier s'il existe un fichier de métadonnées associé au fichier de résultats
            meta_file = Path(clean_results_file).with_suffix('.meta.json')
            source_file = str(default_articles_path)
            term_file = None
            analysis_type = None
            statistics = None
            model_path = None
            
            if meta_file.exists():
                try:
                    with open(meta_file, 'r', encoding='utf-8') as f:
                        meta_data = json.load(f)
                        if 'source_file' in meta_data:
                            source_file = meta_data['source_file']
                        if 'term_file' in meta_data:
                            term_file = meta_data['term_file']
                        if 'analysis_type' in meta_data:
                            analysis_type = meta_data['analysis_type']
                        if 'statistics' in meta_data:
                            statistics = meta_data['statistics']
                        if 'model_path' in meta_data:
                            model_path = meta_data['model_path']
                except Exception as e:
                    print(f"Erreur lors de la lecture du fichier de métadonnées: {e}")
            
            # Charger les données pour la visualisation
            df = pd.read_csv(clean_results_file)
            
            # Préparer les données source avec tous les fichiers nécessaires
            source_data = {
                "results_file": clean_results_file,
                "viz_type": viz_type,
                "analysis_parameters": {
                    "articles_file": source_file,
                    "term_file": term_file,
                    "analysis_type": analysis_type
                },
                "statistics": statistics
            }
            
            # Ajouter le chemin du modèle Word2Vec si disponible
            if model_path:
                source_data["model_path"] = model_path
            
            # Créer une figure factice (sera remplacée par la vraie figure dans le callback)
            figure = go.Figure()
            
            # Sauvegarder l'analyse avec la collection spécifiée
            analysis_id = save_analysis(
                title=title,
                description=description,
                source_data=source_data,
                analysis_type="similar_terms",
                figure=figure,
                collection=collection,
                config=config,
                save_source_files=True
            )
            
            return html.Div([
                html.P("Analyse sauvegardée avec succès!", className="text-success"),
                html.P(f"ID: {analysis_id}")
            ])
        except Exception as e:
            print(f"Erreur lors de la sauvegarde de l'analyse: {str(e)}")
            return html.Div(f"Erreur lors de la sauvegarde: {str(e)}", className="text-danger")
