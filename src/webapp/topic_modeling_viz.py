"""
Topic Modeling Visualization Page for Dash app
"""

print("[topic_modeling_viz] Début de l'import du module")

import dash
print("[topic_modeling_viz] dash importé")
from dash import html, dcc, Input, Output, State, ctx, ALL, MATCH, no_update
print("[topic_modeling_viz] dash.html, dcc, Input, Output, State, ctx importés")
from src.webapp.topic_filter_component import get_topic_filter_component, register_topic_filter_callbacks
import dash_bootstrap_components as dbc
print("[topic_modeling_viz] dash_bootstrap_components importé")
import plotly.express as px
print("[topic_modeling_viz] plotly.express importé")
import plotly.graph_objects as go
import subprocess
import pathlib
import yaml
import pandas as pd
import json
import os
import sys
import threading
import re
import ast
import logging
import datetime

# Import des composants d'exportation
from src.webapp.export_component import (
    create_export_button,
    create_export_modal,
    create_feedback_toast,
    register_export_callbacks
)
from src.utils.export_utils import save_analysis

# Variable globale pour stocker le résultat actuellement sélectionné
current_selected_result = None

# Configuration du logging
def setup_logging():
	project_root = pathlib.Path(__file__).resolve().parents[2]
	config_path = project_root / 'config' / 'config.yaml'

	# Charger la configuration
	with open(config_path, encoding='utf-8') as f:
		config = yaml.safe_load(f)

	# Créer le répertoire de logs s'il n'existe pas
	log_dir = project_root / 'logs'
	log_dir.mkdir(exist_ok=True)

	# Configurer le logger principal
	logger = logging.getLogger('newspapers_analysis')
	logger.setLevel(logging.INFO)

	# Formatter pour les logs
	formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

	# Handler pour la console
	console_handler = logging.StreamHandler()
	console_handler.setFormatter(formatter)
	logger.addHandler(console_handler)

	# Handler pour le fichier de log
	log_file = log_dir / 'newspaper_analysis.log'
	file_handler = logging.FileHandler(str(log_file), encoding='utf-8')
	file_handler.setFormatter(formatter)
	logger.addHandler(file_handler)

	# Handler spécifique pour le topic modeling
	topic_modeling_logger = logging.getLogger('newspapers_analysis.topic_modeling')

	# Créer un fichier de log spécifique pour le topic modeling avec timestamp
	timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
	topic_log_file = log_dir / f'topic_modeling_{timestamp}.log'
	topic_file_handler = logging.FileHandler(str(topic_log_file), encoding='utf-8')
	topic_file_handler.setFormatter(formatter)
	topic_modeling_logger.addHandler(topic_file_handler)

	# Handler spécifique pour le topic naming
	topic_naming_logger = logging.getLogger('newspapers_analysis.topic_naming')
	topic_naming_log_file = log_dir / f'topic_naming_{timestamp}.log'
	topic_naming_file_handler = logging.FileHandler(str(topic_naming_log_file), encoding='utf-8')
	topic_naming_file_handler.setFormatter(formatter)
	topic_naming_logger.addHandler(topic_naming_file_handler)

	return logger, topic_modeling_logger, topic_naming_logger

# Initialiser les loggers
main_logger, topic_modeling_logger, topic_naming_logger = setup_logging()

# Logs d'initialisation
main_logger.info("Module topic_modeling_viz chargé")
topic_modeling_logger.info("Logger de topic modeling initialisé")
topic_naming_logger.info("Logger de topic naming initialisé")


print("[topic_modeling_viz] Début des définitions de fonctions")

# Helper to get config and paths
def get_config_and_paths():
	project_root = pathlib.Path(__file__).resolve().parents[2]
	config_path = project_root / 'config' / 'config.yaml'
	with open(config_path, encoding='utf-8') as f:
		config = yaml.safe_load(f)
	results_dir = project_root / config['data']['results_dir']
	advanced_topic_dir = results_dir / 'advanced_analysis'
	doc_topic_dir = results_dir / 'doc_topic_matrix'
	topic_names_dir = results_dir
	return project_root, config, advanced_topic_dir, doc_topic_dir, topic_names_dir

# Helper to get available topic modeling result files
def get_topic_modeling_results():
	print("\n\n==== DÉBUT DEBUG get_topic_modeling_results ====\n")
	print("Démarrage de get_topic_modeling_results()...")

	project_root, config, advanced_analysis_dir, doc_topic_dir, topic_names_dir = get_config_and_paths()
	print(f"Chemins obtenus:\n  - project_root: {project_root}\n  - advanced_analysis_dir: {advanced_analysis_dir}\n  - doc_topic_dir: {doc_topic_dir}\n  - topic_names_dir: {topic_names_dir}")

	if not advanced_analysis_dir.exists():
		print(f"ERREUR: Répertoire {advanced_analysis_dir} non trouvé")
		# Essayer de créer le répertoire
		try:
			advanced_analysis_dir.mkdir(parents=True, exist_ok=True)
			print(f"Répertoire {advanced_analysis_dir} créé avec succès")
		except Exception as e:
			print(f"Impossible de créer le répertoire: {e}")
		return []

	# Get all topic modeling result files
	print(f"Recherche des fichiers dans {advanced_analysis_dir}...")
	result_files = list(advanced_analysis_dir.glob('advanced_analysis_*.json'))
	print(f"Nombre de fichiers trouvés: {len(result_files)}")
	if len(result_files) > 0:
		print(f"Fichiers trouvés:\n{[f.name for f in result_files]}")
	else:
		print("Aucun fichier trouvé avec le pattern 'advanced_analysis_*.json'")
		# Essayer avec un pattern plus large pour voir ce qui est disponible
		all_files = list(advanced_analysis_dir.glob('*.json'))
		print(f"Tous les fichiers JSON dans le répertoire: {[f.name for f in all_files]}")

		# Si aucun fichier n'est trouvé, vérifier l'ancien répertoire
		old_dir = project_root / config['data']['results_dir'] / 'advanced_topic'
		if old_dir.exists():
			old_files = list(old_dir.glob('*.json'))
			print(f"Fichiers dans l'ancien répertoire {old_dir}: {[f.name for f in old_files]}")
			# Utiliser ces fichiers si nécessaire
			if len(old_files) > 0 and len(result_files) == 0:
				print(f"Utilisation des fichiers de l'ancien répertoire")
				result_files = old_files

	# Sort by modification time (newest first)
	result_files.sort(key=lambda x: os.stat(x).st_mtime, reverse=True)

	# Ajout du paramètre de cache-busting pour forcer le rechargement des fichiers
	import time as time_module
	cache_buster = int(time_module.time())

	# Format for dropdown
	options = []
	for file in result_files:
		# Try different regex patterns to extract timestamp
		match = re.search(r'advanced_analysis_\w+_(\d{8}-\d{6})_\w+\.json', file.name)
		if not match:
			# Try alternative pattern for older files
			match = re.search(r'advanced_(?:topic_)?analysis_?(\d{8}_\d{6})?\.json', file.name)

		if match and match.group(1):
			# If there's a timestamp in the filename
			timestamp = match.group(1)
			print(f"Timestamp extrait: {timestamp}")
			# Try to format it nicely if it's a valid timestamp format
			try:
				from datetime import datetime
				# Try different date formats
				try:
					date_str = datetime.strptime(timestamp, '%Y%m%d-%H%M%S').strftime('%d/%m/%Y %H:%M')
				except ValueError:
					try:
						date_str = datetime.strptime(timestamp, '%Y%m%d_%H%M%S').strftime('%d/%m/%Y %H:%M')
					except ValueError:
						date_str = timestamp

				# Add model type to the label if available
				model_type = ""
				if "gensim" in file.name.lower():
					model_type = "Gensim"
				elif "bertopic" in file.name.lower():
					model_type = "BERTopic"

				if model_type:
					label = f"Analyse {model_type} du {date_str}"
				else:
					label = f"Analyse du {date_str}"
			except Exception as e:
				# If not a valid timestamp format, just use the raw string
				print(f"Erreur de format de date: {e}")
				label = f"Analyse {timestamp}"
		else:
			# If no timestamp in filename, use last modified time
			from datetime import datetime
			mod_time = datetime.fromtimestamp(os.path.getmtime(file))
			label = f"Analyse du {mod_time.strftime('%d/%m/%Y %H:%M')}"

		# Add cache buster to force reload when needed
		value = f"{file}?cache={cache_buster}"
		options.append({"label": label, "value": value})
		print(f"Ajout de l'option: {label} -> {file}")

	print("\n==== FIN DEBUG get_topic_modeling_results ====\n\n")

	# Add default option if no files found
	if not options:
		options = [{"label": "Aucun résultat disponible", "value": ""}]

	print(f"Options de fichiers de résultats: {len(options)}")
	return options

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
# Fichier : src/webapp/topic_modeling_viz.py
def get_topic_modeling_layout():
	return dbc.Container([
		# NOUVELLE STRUCTURE : Onglets principaux pour séparer la configuration des résultats
		dbc.Tabs([
			# Onglet 1: Paramètres pour lancer une nouvelle analyse
			dbc.Tab([
				dbc.Card([
					dbc.CardHeader(html.H3("Paramètres du Topic Modeling", className="mb-0")),
					dbc.CardBody([
						html.P("Configurez les paramètres de l'analyse thématique ci-dessous, puis cliquez sur 'Lancer'.", className="text-muted mb-3"),

						# Fichier source personnalisé
						html.H5("Fichier source", className="mb-2"),
						dbc.Row([
							dbc.Col([
								dbc.InputGroup([
									dbc.Input(
										id="arg-source-file",
										type="text",
										placeholder="Chemin vers le fichier JSON d'articles"
									),
									dbc.Button("Parcourir", id="source-file-browse", color="secondary", n_clicks=0)
								]),
								html.Small("Laissez vide pour utiliser le fichier par défaut de la configuration.", className="text-muted")
							], width=12),
						], className="mb-3"),

						# Sélection de fichier de cache
						html.H5("Fichier de cache Spacy", className="mb-2"),
						dbc.Row([
							dbc.Col([
								dbc.Select(
									id="cache-file-select",
									options=get_cache_file_options(),
									value="",
									className="mb-2"
								),
								html.Small("Sélectionnez un fichier de cache Spacy existant pour accélérer le traitement.", className="text-muted d-block"),
								html.Div(id="cache-info-display", className="mt-2")
							], width=12),
						], className="mb-3"),

						dbc.Form(get_topic_modeling_controls()),

						# Add topic filter component
						html.H5("Filtrage par cluster", className="mt-4 mb-3"),
						get_topic_filter_component(id_prefix="topic-filter"),

						dbc.Button("Lancer le Topic Modeling", id="btn-run-topic-modeling", color="primary", n_clicks=0, className="mt-3 mb-2"),
						html.Div(id="topic-modeling-run-status", className="mb-3"),
					]),
				], className="my-4 shadow"),
			], label="Paramètres", tab_id="tab-params"),

			# Onglet 2: Visualisation des résultats existants
			dbc.Tab([
				# Bloc de sélection du fichier de résultats et bouton d'export
				dbc.Row([
					dbc.Col([
						dbc.Card([
							dbc.CardHeader(
								dbc.Row([
									dbc.Col(html.H4("Résultats de Topic Modeling", className="mb-0"), width="auto"),
									dbc.Col(
										create_export_button("topic_modeling", button_id="export-topic-modeling-button"),
										width="auto", className="ms-auto"
									)
								])
							),
							dbc.CardBody([
								dbc.Label("Sélectionner un fichier de résultats:", className="fw-bold"),
								dcc.Dropdown(
									id="topic-modeling-results-dropdown",
									options=get_topic_modeling_results(),
									value=get_topic_modeling_results()[0]['value'] if get_topic_modeling_results() else None,
									clearable=False,
									className="mb-3"
								),
							])
						], className="my-4 shadow")
					], width=12)
				]),

				# Bloc des sous-onglets pour les différentes visualisations
				dbc.Tabs([
					dbc.Tab([
						dbc.Row([
							dbc.Col([
								html.H4("Statistiques avancées", className="mt-4 mb-3"),
								dbc.Button("Charger les résultats", id="load-stats-button", color="primary", className="mb-3"),
								html.Hr(),
								dcc.Loading(
									id="loading-advanced-topic-stats",
									type="circle",
									children=html.Div(id="advanced-topic-stats-content")
								)
							], width=12)
						])
					], label="Statistiques", tab_id="stats-tab"),
					dbc.Tab([
						dbc.Row([
							dbc.Col([
								html.H4("Explorateur d'articles", className="mt-4 mb-3"),
								dbc.Button("Charger les résultats", id="load-articles-button", color="primary", className="mb-3"),
								html.Hr(),
								dcc.Loading(
									id="loading-article-browser",
									type="circle",
									children=html.Div(id="article-browser-content")
								)
							], width=12)
						])
					], label="Explorateur d'articles", tab_id="article-browser-tab"),
					dbc.Tab([
						dbc.Row([
							dbc.Col([
								html.H4("Nommage des topics avec LLM", className="mt-4 mb-3"),
								html.P("Cet outil vous permet de générer automatiquement des noms et des résumés pour vos topics en utilisant un LLM.", className="text-muted"),
								dbc.Card([
									dbc.CardBody([
										# ... (contenu du formulaire de nommage) ...
										dbc.Row([
											dbc.Col([
												dbc.Label("Méthode de génération", html_for="topic-naming-method"),
												dbc.Select(
													id="topic-naming-method",
													options=[
														{"label": "Utiliser les articles représentatifs", "value": "articles"},
														{"label": "Utiliser les mots-clés", "value": "keywords"}
													],
													value="articles",
													className="mb-2"
												),
											], width=6),
											dbc.Col([
												dbc.Label("Nombre d'articles par topic", html_for="topic-naming-num-articles"),
												dbc.Input(
													id="topic-naming-num-articles",
													type="number", min=1, max=20, step=1, value=10, className="mb-2"
												),
											], width=6),
										]),
										dbc.Row([
											dbc.Col([
												dbc.Label("Seuil de probabilité", html_for="topic-naming-threshold"),
												dbc.Input(
													id="topic-naming-threshold",
													type="number", min=0.1, max=0.9, step=0.1, value=0.5, className="mb-2"
												),
											], width=6),
											dbc.Col([
												dbc.Label("Fichier de sortie", html_for="topic-naming-output-path"),
												dbc.Input(
													id="topic-naming-output-path",
													type="text", placeholder="Laissez vide pour générer automatiquement", className="mb-2"
												),
											], width=6),
										]),
										dbc.Button("Générer les noms des topics", id="btn-run-topic-naming", color="primary", className="mt-2"),
									])
								], className="mb-4"),
								html.Div(id="topic-naming-status", className="mt-3"),
								dcc.Loading(id="loading-topic-naming-results", type="default", children=html.Div(id="topic-naming-results"))
							], width=12)
						])
					], label="Nommage des topics", tab_id="topic-naming-tab"),
					dbc.Tab([
						dbc.Row([
							dbc.Col([
								html.H4("Filtrage des publicités par topic", className="mt-4 mb-3"),
								html.P("Cet outil vous permet de détecter et filtrer les publicités d'un topic spécifique en utilisant un LLM.", className="text-muted"),
								dbc.Card([
									dbc.CardBody([
										# ... (contenu du formulaire de filtrage) ...
										dbc.Row([
											dbc.Col([
												dbc.Label("Topic à analyser", html_for="ad-filter-topic-id"),
												dbc.InputGroup([
													dbc.Select(id="ad-filter-topic-id", options=[{"label": "Chargement des topics...", "value": ""}], value=None, className="mb-2"),
													dbc.Button("Rafraîchir", id="btn-refresh-topics", color="secondary", className="mb-2 ms-2")
												]),
											], width=6),
											dbc.Col([
												dbc.Label("Valeur minimale du topic", html_for="ad-filter-min-value"),
												dbc.Input(id="ad-filter-min-value", type="number", min=0.1, max=0.9, step=0.1, value=0.5, className="mb-2"),
											], width=6),
										]),
										dbc.Row([
											dbc.Col([
												dbc.Label("Fichier de sortie", html_for="ad-filter-output-path"),
												dbc.Input(id="ad-filter-output-path", type="text", placeholder="Laissez vide pour générer automatiquement", className="mb-2"),
											], width=12),
										]),
										dbc.Row([
											dbc.Col([dbc.Checkbox(id="ad-filter-dry-run", label="Mode test (ne pas écrire le fichier)", value=False, className="mb-3")], width=12),
										]),
										dbc.Button("Lancer le filtrage des publicités", id="btn-run-ad-filter", color="primary", className="mt-2"),
									])
								], className="mb-4"),
								html.Div(id="ad-filter-status", className="mt-3"),
								dcc.Loading(id="loading-ad-filter-results", type="default", children=html.Div(id="ad-filter-results"))
							], width=12)
						])
					], label="Filtrage des publicités", tab_id="ad-filter-tab")
				], id="topic-modeling-sub-tabs"), # Renommé pour clarté
			], label="Résultats", tab_id="tab-results")
		], id="topic-modeling-main-tabs", active_tab="tab-params"), # ID des onglets principaux

		# Composants globaux (Stores, Modals) qui doivent être en dehors des onglets
		dcc.Store(id="topic-modeling-selected-result-store", data=None),
		create_export_modal("topic_modeling", modal_id="export-topic-modeling-modal"),
		create_feedback_toast("export-topic-modeling-feedback-toast")
	], fluid=True)
# Fonction pour obtenir les informations sur les fichiers de cache
def get_cache_file_options():
	"""
	Génère les options pour le dropdown de sélection des fichiers de cache spaCy.

	Returns:
		list: Liste d'options pour le dropdown
	"""
	cache_info = get_cache_info()
	options = [{"label": "Aucun (utiliser le plus récent)", "value": ""}]

	for cache_file in cache_info["files"]:
		filename = cache_file["filename"]
		articles_count = cache_file.get("articles_count", "?")
		file_size = round(cache_file.get("file_size_mb", 0), 1)
		creation_time = cache_file.get("creation_time", "")
		
		label = f"{filename} ({articles_count} articles, {file_size} MB, {creation_time})"
		project_root = pathlib.Path(__file__).resolve().parents[2]
		cache_dir = project_root / 'data' / 'cache'
		value = str(cache_dir / filename)
		
		options.append({"label": label, "value": value})
	
	return options

def get_cache_info():
	"""
	Récupère les informations sur les fichiers de cache Spacy existants.

	Returns:
		dict: Informations sur les fichiers de cache
	"""
	project_root = pathlib.Path(__file__).resolve().parents[2]
	cache_dir = project_root / 'data' / 'cache'
	cache_files = list(cache_dir.glob("preprocessed_*.pkl"))

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
			
			# Détection du nombre d'articles - plusieurs méthodes possibles selon la structure
			articles_count = 0
			
			# Cas spécial pour les fichiers spacy_tokens
			if 'spacy_tokens' in cache_file.name:
				# Pour les fichiers spacy_tokens, chercher dans la structure spécifique
				if 'tokens' in cache_data:
					if isinstance(cache_data['tokens'], dict):
						articles_count = len(cache_data['tokens'])
					elif isinstance(cache_data['tokens'], list):
						articles_count = len(cache_data['tokens'])
				# Chercher dans d'autres structures possibles pour les fichiers spacy_tokens
				elif 'articles' in cache_data:
					if isinstance(cache_data['articles'], (list, dict)):
						articles_count = len(cache_data['articles'])
				# Chercher dans le dictionnaire original
				elif isinstance(cache_data, dict) and len(cache_data) > 0:
					# Prendre la première clé qui pourrait contenir des articles
					for key, value in cache_data.items():
						if isinstance(value, (list, dict)) and len(value) > 0:
							articles_count = len(value)
							break
							
				# Si on a trouvé des articles dans le fichier spacy_tokens, on peut sortir
				if articles_count > 0:
					print(f"Cache spacy_tokens {cache_file.name}: {articles_count} articles détectés")
					
			# Pour les autres types de fichiers
			else:
				# Méthode 1: Directement dans cache_key_data (ancienne structure)
				if 'articles_count' in cache_key_data:
					articles_count = cache_key_data['articles_count']
				# Méthode 2: Dans les données du cache (nouvelle structure)
				elif 'docs' in cache_data and isinstance(cache_data['docs'], list):
					articles_count = len(cache_data['docs'])
				# Méthode 3: Dans les données du cache sous forme de dictionnaire
				elif 'docs' in cache_data and isinstance(cache_data['docs'], dict):
					articles_count = len(cache_data['docs'])
				# Méthode 4: Chercher dans d'autres structures possibles
				elif 'articles' in cache_data and isinstance(cache_data['articles'], (list, dict)):
					articles_count = len(cache_data['articles'])
				# Méthode 5: Chercher dans les tokens (si disponible)
				elif 'tokens' in cache_data and isinstance(cache_data['tokens'], dict):
					articles_count = len(cache_data['tokens'])
			
			# Log pour debug
			print(f"Cache {cache_file.name}: {articles_count} articles détectés")

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


def load_article_browser(custom_doc_topic_path=None):
    """
    [VERSION DE DÉBOGAGE]
    Charge la matrice doc-topic et les détails des articles pour créer un explorateur interactif.
    """
    print("\n--- [DÉBUT] Exécution de load_article_browser ---")

    # --- Étape 1: Trouver les chemins des fichiers ---
    project_root, config, _, doc_topic_dir, _ = get_config_and_paths()
    results_dir = project_root / config['data']['results_dir']
    
    # Déterminer le chemin du fichier doc_topic
    if custom_doc_topic_path:
        doc_topic_path = custom_doc_topic_path
    else:
        # Logique pour trouver le dernier fichier de topics
        if doc_topic_dir.exists():
            doc_topic_files = list(doc_topic_dir.glob('doc_topic_matrix_*.json'))
            if doc_topic_files:
                doc_topic_files.sort(key=lambda x: os.stat(x).st_mtime, reverse=True)
                doc_topic_path = str(doc_topic_files[0])
            else:
                doc_topic_path = str(results_dir / 'doc_topic_matrix.json')
        else:
            doc_topic_path = str(results_dir / 'doc_topic_matrix.json')

    print(f"  [INFO] Chemin du fichier de topics utilisé : {doc_topic_path}")
    if not os.path.exists(doc_topic_path):
        return html.Div(f"Fichier de topics introuvable : {doc_topic_path}", className="alert alert-danger")

    # --- Étape 2: Charger et normaliser les données des topics ---
    try:
        with open(doc_topic_path, 'r', encoding='utf-8') as f:
            doc_topic_data = json.load(f)
        
        doc_topic_list = []
        if 'doc_topics' in doc_topic_data:
            print("  [INFO] Format 'doc_topics' détecté. Conversion en liste...")
            for doc_id, data in doc_topic_data['doc_topics'].items():
                doc_topic_list.append({"doc_id": doc_id, **data})
        elif isinstance(doc_topic_data, list):
             doc_topic_list = doc_topic_data
        else:
            return html.Div("Format du fichier de topics non reconnu.", className="alert alert-danger")
        
        print(f"  [INFO] Nombre d'articles chargés depuis le fichier de topics : {len(doc_topic_list)}")
        if doc_topic_list:
            print(f"  [INFO] Exemple de premier article du fichier de topics : {doc_topic_list[0]}")

    except Exception as e:
        return html.Div(f"Erreur lors du chargement ou de la normalisation du fichier de topics : {e}", className="alert alert-danger")


    # --- Étape 3: Charger les détails des articles ---
    articles_path = project_root / config['data']['processed_dir'] / 'articles_v1_filtered.json'
    print(f"  [INFO] Chemin du fichier d'articles utilisé : {articles_path}")
    article_info = {}
    if articles_path.exists():
        try:
            with open(articles_path, 'r', encoding='utf-8') as f:
                articles = json.load(f)
            
            for article in articles:
                main_id = article.get('id')
                base_id = article.get('base_id')
                if main_id:
                    article_info[main_id] = article
                if base_id and base_id not in article_info:
                    article_info[base_id] = article
            
            print(f"  [INFO] Nombre d'articles chargés depuis {articles_path.name} : {len(article_info)}")
            if article_info:
                 # Affiche un exemple d'ID chargé pour vérifier le format
                 example_key = next(iter(article_info))
                 print(f"  [INFO] Exemple d'ID dans article_info : '{example_key}'")

        except Exception as e:
            print(f"  [ERREUR] Impossible de charger les détails des articles depuis {articles_path.name}: {e}")
    else:
        print(f"  [AVERTISSEMENT] Fichier d'articles non trouvé : {articles_path}")


    # --- Étape 4: Boucle de correspondance et préparation des données pour la table ---
    print("\n--- [DÉBUT] Correspondance des articles ---")
    table_data = []
    found_count = 0
    for i, item in enumerate(doc_topic_list):
        doc_id = item.get('doc_id', '')
        if not doc_id:
            continue

        # Logique de correspondance améliorée
        article_details = article_info.get(doc_id)
        match_type = "Correspondance exacte"
        
        if not article_details:
            base_doc_id = doc_id.split('_mistral')[0] if '_mistral' in doc_id else doc_id
            article_details = article_info.get(base_doc_id)
            match_type = "Correspondance de base"

        if article_details:
            found_count += 1
            # Affiche seulement les 5 premières correspondances trouvées pour ne pas surcharger les logs
            if found_count <= 5:
                print(f"  [SUCCÈS] ID du topic '{doc_id}' trouvé via '{match_type}'. Titre: {article_details.get('title', 'N/A')}")
        else:
            # Affiche seulement les 5 premiers échecs
            if (i - found_count) < 5:
                 print(f"  [ÉCHEC] ID du topic '{doc_id}' n'a PAS été trouvé dans article_info.")
            # Important : Assurer que article_details est un dictionnaire vide pour éviter les erreurs
            article_details = {} 

        topic_distribution = item.get('topic_distribution', [])
        dominant_topic_idx = item.get('dominant_topic', -1)
        if dominant_topic_idx == -1 and topic_distribution:
             dominant_topic_idx = topic_distribution.index(max(topic_distribution))
        
        # Le contenu vient bien du champ "content"
        table_data.append({
            'doc_id': doc_id,
            'title': article_details.get('title', 'TITRE INTROUVABLE'),
            'date': article_details.get('date', 'Date introuvable'),
            'newspaper': article_details.get('newspaper', 'Journal introuvable'),
            'content': article_details.get('content', 'Contenu introuvable'), # <-- Ajout du contenu complet
            'dominant_topic': dominant_topic_idx,
            'topic_distribution': topic_distribution,
        })
    print(f"--- [FIN] Correspondance terminée. Articles trouvés: {found_count} / {len(doc_topic_list)} ---")


    # --- Étape 5: Création de l'interface Dash ---
    # Charger les noms de topics (cette partie reste inchangée)
    topic_names = {}
    _, _, _, _, topic_names_dir = get_config_and_paths()
    topic_names_files = list(topic_names_dir.glob('topic_names_llm*.json'))
    if topic_names_files:
        topic_names_files.sort(key=lambda x: os.stat(x).st_mtime, reverse=True)
        try:
            with open(topic_names_files[0], 'r', encoding='utf-8') as f:
                topic_names = json.load(f).get('topic_names', {})
        except Exception as e:
            print(f"Avertissement : Erreur lors du chargement des noms de topics: {e}")

    # Compléter les données de la table avec les noms de topics
    for row in table_data:
        row['dominant_topic_name'] = get_topic_name(row['dominant_topic'], topic_names, f"Topic {row['dominant_topic']}")

    # Le reste de la fonction pour créer les contrôles et la table
    num_topics = len(table_data[0]['topic_distribution']) if table_data and table_data[0]['topic_distribution'] else 0
    sort_options = [
        {'label': 'ID du document', 'value': 'doc_id'},
        {'label': 'Date', 'value': 'date'},
        {'label': 'Journal', 'value': 'newspaper'},
        {'label': 'Topic dominant (valeur)', 'value': 'dominant_topic_value'}
    ]
    for i in range(num_topics):
        topic_name = get_topic_name(i, topic_names, f"Topic {i}")
        sort_options.append({'label': f'Poids du {topic_name}', 'value': f'topic_{i}'})
    
    # ... le reste de la création du layout est identique ...
    children = [
        dbc.Row([
            dbc.Col([
                html.H5("Trier les articles par:"),
                dcc.Dropdown(id='article-sort-dropdown', options=sort_options, value='dominant_topic_value', clearable=False),
                dbc.Checkbox(id='sort-descending-checkbox', label="Ordre décroissant", value=True, className="mt-2 mb-3")
            ], width=6),
            dbc.Col([
                html.H5("Filtrer par topic dominant:"),
                dcc.Dropdown(
                    id='dominant-topic-filter',
                    options=[{'label': 'Tous les topics', 'value': 'all'}] + [
                        {'label': get_topic_name(i, topic_names, f"Topic {i}"), 'value': i}
                        for i in range(num_topics)
                    ],
                    value='all',
                    clearable=False
                )
            ], width=6)
        ]),
        dcc.Store(id='article-browser-data', data=table_data),
        html.Div(id='article-browser-table-container', className="mt-4"),
        dbc.Modal([
            dbc.ModalHeader(dbc.ModalTitle("Détails de l'article")),
            dbc.ModalBody(id="article-detail-body"),
            dbc.ModalFooter(dbc.Button("Fermer", id="close-article-modal", className="ms-auto", n_clicks=0)),
        ], id="article-detail-modal", size="xl", is_open=False),
    ]
    print("--- [FIN] Exécution de load_article_browser ---")
    return html.Div(children)

def load_topic_names(model_id=None):
	"""
	Charge les noms de topics pour un model_id spécifique.
	Cherche un fichier topic_names_llm_[model_id].json.
	"""
	if not model_id:
		topic_modeling_logger.warning("load_topic_names a été appelé sans model_id.")
		return {}

	try:
		project_root, config, _, _, results_dir = get_config_and_paths()
		
		# Construire le nom de fichier attendu
		topic_names_path = results_dir / f"topic_names_llm_{model_id}.json"
		topic_modeling_logger.info(f"Tentative de chargement du fichier de noms de topics : {topic_names_path}")
		
		if topic_names_path.exists():
			with open(topic_names_path, 'r', encoding='utf-8') as f:
				data = json.load(f)
				topic_names = data.get('topic_names', {})
				if topic_names:
					topic_modeling_logger.info(f"Noms de topics chargés avec succès depuis {topic_names_path}")
					return topic_names
				else:
					topic_modeling_logger.warning(f"Fichier {topic_names_path} trouvé mais ne contient pas la clé 'topic_names' ou elle est vide.")

		# Fallback vers l'ancien nom de fichier pour la rétrocompatibilité
		legacy_path = results_dir / f"topic_names_{model_id}.json"
		if legacy_path.exists():
			topic_modeling_logger.info(f"Tentative de chargement du fichier legacy : {legacy_path}")
			with open(legacy_path, 'r', encoding='utf-8') as f:
				topic_names = json.load(f)
				topic_modeling_logger.info(f"Noms de topics chargés avec succès depuis le fichier legacy {legacy_path}")
				return topic_names

		topic_modeling_logger.info(f"Aucun fichier de noms de topics trouvé pour model_id: {model_id}")
		return {}

	except Exception as e:
		topic_modeling_logger.error(f"Erreur lors du chargement des noms de topics pour model_id {model_id}: {e}")
		return {}


# NOUVELLE LOGIQUE : Fonction pour obtenir un nom de topic unique
def get_topic_name(topic_id, all_topic_names, default=None):
	"""
	Retourne le nom d'un topic spécifique de manière robuste.
	Si le nom n'est pas disponible dans le dictionnaire fourni, retourne un nom par défaut.
	
	Args:
		topic_id: L'identifiant du topic (peut être int ou str).
		all_topic_names (dict): Le dictionnaire de tous les noms de topics déjà chargés.
		default (str, optional): Le nom par défaut à utiliser si non trouvé.
	"""
	if default is None:
		default = f"Topic {topic_id}"

	if not all_topic_names:
		return default

	try:
		# Essayer de trouver le nom en utilisant différentes variations de la clé
		topic_id_str = str(topic_id).strip()
		topic_id_digits = ''.join(filter(str.isdigit, topic_id_str))

		possible_keys = [
			topic_id_str,          # "0", "1", "topic_0"
			topic_id_digits,       # "0", "1"
		]

		for key in possible_keys:
			if key in all_topic_names:
				name_data = all_topic_names[key]
				# Gérer les cas où le nom est un tuple/liste ('nom', 'résumé') ou un simple str
				if isinstance(name_data, (list, tuple)) and name_data:
					return name_data[0]
				elif isinstance(name_data, str):
					return name_data
		
		# Si aucune clé n'est trouvée
		return default

	except Exception as e:
		topic_modeling_logger.error(f"Erreur dans get_topic_name pour topic_id {topic_id}: {e}")
		return default


# Helper to render the results of the topic naming script.
def render_topic_naming_results(topic_names_data):
	"""
	Renders the topic naming results in a user-friendly format.

	Args:
		topic_names_data (dict): A dictionary containing the generated topic names and summaries, usually under a 'topic_names' key.

	Returns:
		A Dash component to display the results.
	"""
	if not topic_names_data:
		return html.Div("Aucune donnée de nom de topic n'a été générée.", className="alert alert-info")

	# Les noms sont souvent sous une clé 'topic_names'
	topic_names = topic_names_data.get('topic_names', {})
	if not topic_names:
		return html.Div("Le fichier de résultats ne contenait pas de noms de topics.", className="alert alert-warning")

	accordion_items = []
	# Trier les topics par leur numéro pour un affichage ordonné
	sorted_topic_ids = sorted(topic_names.keys(), key=lambda x: int(''.join(filter(str.isdigit, str(x))) or 0))

	for topic_id in sorted_topic_ids:
		data = topic_names[topic_id]
		title = "Titre non disponible"
		summary = "Résumé non disponible"

		if isinstance(data, (list, tuple)):
			title = data[0] if len(data) > 0 else title
			summary = data[1] if len(data) > 1 else summary
		elif isinstance(data, str):
			title = data
		
		item_title = f"{str(topic_id).replace('_', ' ').title()}: {title}"
		
		item = dbc.AccordionItem(
			[html.P(summary, className="mb-0")],
			title=item_title,
		)
		accordion_items.append(item)

	return html.Div([
		html.H5("Noms et résumés des topics générés", className="mt-4 mb-3"),
		dbc.Accordion(
			accordion_items,
			start_collapsed=True,
			always_open=True,
			className="mb-4"
		)
	])


# Note: La fonction get_topic_name est déjà définie plus haut dans le fichier
# Nous utilisons une seule implémentation robuste pour éviter les conflits

def register_topic_modeling_callbacks(app):
	"""Register callbacks for the topic modeling page."""
	from dash import html, dcc, Input, Output, State, ctx, ALL, MATCH, no_update, dash_table
	import dash_bootstrap_components as dbc
	import pandas as pd
	import json
	from pathlib import Path

	# ... (tous les callbacks existants comme run_topic_modeling, update_advanced_topic_stats, etc. restent ici)
    # ... (je ne les remets pas pour la clarté, mais ils doivent rester dans votre fonction)

	# --- NOUVEAU CALLBACK : Affiche et met à jour la table des articles ---
	@app.callback(
		Output('article-browser-table-container', 'children'),
		[Input('article-sort-dropdown', 'value'),
		 Input('sort-descending-checkbox', 'value'),
		 Input('dominant-topic-filter', 'value'),
		 Input('article-browser-data', 'data')]
	)
	def update_article_table(sort_by, sort_desc, topic_filter, data):
		if not data:
			return html.P("Aucune donnée d'article à afficher. Veuillez d'abord charger un résultat.")

		df = pd.DataFrame(data)

		# 1. Filtrer par topic dominant
		if topic_filter != 'all':
			df = df[df['dominant_topic'] == topic_filter]

		# 2. Trier les données
		if sort_by:
			# Gérer le tri par poids de topic (ex: 'topic_5')
			if sort_by.startswith('topic_'):
				parts = sort_by.split('_', 1)
				if len(parts) == 2 and parts[1].isdigit():
					topic_idx = int(parts[1])
					# Extraire le poids pour ce topic et le mettre dans une nouvelle colonne pour le tri
					df['sort_value'] = df['topic_distribution'].apply(lambda dist: dist[topic_idx] if len(dist) > topic_idx else 0)
					df = df.sort_values(by='sort_value', ascending=not sort_desc)
				else:
					# Si ce n'est pas un index numérique, trier normalement si possible
					if sort_by in df.columns:
						df = df.sort_values(by=sort_by, ascending=not sort_desc)
			else:
				if sort_by in df.columns:
					df = df.sort_values(by=sort_by, ascending=not sort_desc)

		# 3. Créer la table
		table = dash_table.DataTable(
			id='article-table',
			columns=[
				{"name": "Titre", "id": "title", "presentation": "markdown"},
				{"name": "Date", "id": "date"},
				{"name": "Journal", "id": "newspaper"},
				{"name": "Topic Dominant", "id": "dominant_topic_name"},
			],
			data=df[["title", "date", "newspaper", "dominant_topic_name", "content"]].to_dict('records'),
			page_size=20,
			style_cell={'textAlign': 'left', 'padding': '10px', 'fontFamily': 'sans-serif'},
			style_header={'fontWeight': 'bold', 'backgroundColor': 'rgb(230, 230, 230)'},
			style_data_conditional=[{'if': {'row_index': 'odd'}, 'backgroundColor': 'rgb(248, 248, 248)'}],
			style_cell_conditional=[{'if': {'column_id': 'title'}, 'width': '50%'}],
			markdown_options={"html": True}
		)
		return table

	# --- NOUVEAU CALLBACK : Affiche le contenu de l'article dans une modale ---
	@app.callback(
		Output("article-detail-modal", "is_open"),
		Output("article-detail-body", "children"),
		Input("article-table", "active_cell"),
		State("article-table", "data"), # Les données actuellement affichées dans la table
		prevent_initial_call=True
	)
	def show_article_details(active_cell, table_rows):
		if not active_cell:
			return False, None

		# Récupérer les données de la ligne cliquée
		row_index = active_cell['row']
		clicked_row_data = table_rows[row_index]
		
		# Récupérer les informations de l'article
		title = clicked_row_data.get('title', 'Titre non disponible')
		date = clicked_row_data.get('date', '')
		newspaper = clicked_row_data.get('newspaper', '')
		content = clicked_row_data.get('content', 'Contenu non disponible.')

		# Mettre en forme le contenu pour l'affichage
		# Remplacer les sauts de ligne par des balises <br> pour un affichage HTML correct
		formatted_content = []
		for paragraph in content.split('\n'):
			if paragraph.strip(): # Ignorer les lignes vides
				formatted_content.append(paragraph)
				formatted_content.append(html.Br())

		modal_body = html.Div([
			html.H4(title),
			html.P(f"{newspaper} - {date}", className="text-muted"),
			html.Hr(),
			html.P(formatted_content)
		])

		return True, modal_body

	# --- CALLBACK EXISTANT : pour fermer la modale ---
	@app.callback(
		Output("article-detail-modal", "is_open", allow_duplicate=True),
		Input("close-article-modal", "n_clicks"),
		prevent_initial_call=True,
	)
	def close_article_modal(n_clicks):
		if n_clicks > 0:
			return False
		return no_update

    # NOTE : Assurez-vous que tous vos autres callbacks (run_topic_modeling, update_advanced_topic_stats, etc.)
    # sont bien présents ici, dans cette même fonction `register_topic_modeling_callbacks`.
	"""Register callbacks for the topic modeling page."""
	from dash import html, dcc, Input, Output, State, ctx, ALL, MATCH, no_update
	import dash_bootstrap_components as dbc
	import plotly.express as px
	import plotly.graph_objects as go
	import subprocess
	import importlib.util
	import sys
	import os
	import re
	import subprocess
	import time
	import uuid
	import glob
	from pathlib import Path
	import pandas as pd
	import numpy as np
	import json
	import time  # Ajout du module time pour mesurer les performances
	from datetime import datetime
	
	# Enregistrer les callbacks d'exportation pour le topic modeling
	register_export_callbacks(
		app,
		analysis_type="topic_modeling",
		get_source_data_function=get_topic_modeling_source_data,
		get_figure_function=get_topic_modeling_figure,
		button_id="export-topic-modeling-button",
		modal_id="export-topic-modeling-modal",
		toast_id="export-topic-modeling-feedback-toast"
	)
	
	# Register the topic filter component callbacks
	register_topic_filter_callbacks(app, id_prefix="topic-filter")
	parser_args = get_topic_modeling_args()
	
	# Callback pour le bouton de parcourir du fichier source
	@app.callback(
		Output("arg-source-file", "value"),
		Input("source-file-browse", "n_clicks"),
		State("arg-source-file", "value"),
		prevent_initial_call=True
	)
	def browse_source_file(n_clicks, current_value):
		if not n_clicks:
			return current_value
		
		# Obtenir le répertoire de départ pour la boîte de dialogue
		project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
		data_dir = os.path.join(project_root, "data", "processed")
		
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

	# Callback pour mettre à jour la variable globale current_selected_result
	@app.callback(
		Output("topic-modeling-selected-result-store", "data"),
		[Input("topic-modeling-results-dropdown", "value")]
	)
	def update_selected_result(selected_file):
		global current_selected_result
		current_selected_result = selected_file
		return selected_file

	# Callback pour lancer le topic modeling
	@app.callback(
		Output("topic-modeling-run-status", "children"),
		Output("topic-modeling-results-dropdown", "options"),
		Input("btn-run-topic-modeling", "n_clicks"),
		State("arg-source-file", "value"),
		State("cache-file-select", "value"),
		# Récupérer tous les arguments du formulaire
		*[State(f"arg-{arg['name']}", "value") for arg in parser_args],
		prevent_initial_call=True
	)
	def run_topic_modeling(n_clicks, input_file, cache_file, *args_values):
		if not n_clicks:
			return dash.no_update, dash.no_update

		try:
			# Récupérer les chemins nécessaires
			project_root, config, advanced_analysis_dir, doc_topic_dir, _ = get_config_and_paths()
			results_dir = project_root / config['data']['results_dir']

			# Construire la commande
			script_path = project_root / "src" / "scripts" / "run_topic_modeling.py"

			# Utiliser sys.executable pour s'assurer d'utiliser le bon interpréteur Python
			python_executable = sys.executable

			# Construire les arguments de la commande
			cmd_args = [python_executable, str(script_path)]

			# Ajouter les arguments du formulaire
			for i, arg in enumerate(parser_args):
				value = args_values[i]
				if value is None or value == "":
					continue

				# Pour les arguments booléens, leur présence suffit
				if arg['type'] == 'bool':
					if value:
						cmd_args.append(arg['flags'][0])
				else:
					cmd_args.append(arg['flags'][0])
					cmd_args.append(str(value))

			# Ajouter le fichier source s'il est spécifié
			if input_file:
				cmd_args.extend(["--source-file", input_file])

			# Ajouter le fichier de cache s'il est spécifié
			if cache_file:
				cmd_args.extend(["--cache-file", cache_file])

			# Créer un message de statut
			status = dbc.Alert(
				[
					html.P("Lancement du script de topic modeling...", className="mb-0"),
					html.P(f"Commande exécutée: {' '.join(cmd_args)}", className="mb-0 small")
				],
				color="info"
			)

			# Logger le début du processus de topic modeling
			topic_modeling_logger.info("==== DÉBUT DU PROCESSUS DE TOPIC MODELING ====")
			topic_modeling_logger.info(f"Commande exécutée: {' '.join(cmd_args)}")

			# Exécuter le processus avec redirection vers des pipes pour logger la sortie
			# Utiliser subprocess.run au lieu de Popen pour une exécution synchrone avec capture de sortie
			print("\n==== LANCEMENT DU PROCESSUS DE TOPIC MODELING ====")
			print(f"Commande exécutée: {' '.join(cmd_args)}\n")
			
			# Exécuter le processus en affichant la sortie en temps réel
			process = subprocess.Popen(
				cmd_args,
				stdout=subprocess.PIPE,
				stderr=subprocess.STDOUT,  # Rediriger stderr vers stdout pour avoir tout dans un seul flux
				text=True,
				bufsize=1,
				universal_newlines=True,
				encoding='utf-8',  # Ensure correct encoding
				errors='replace'  # Remplacer les caractères non décodables au lieu de lever une erreur
			)
			
			# Capturer la sortie ligne par ligne pour l'afficher en temps réel
			output_lines = []
			for line in iter(process.stdout.readline, ''):
				if not line:
					break
				print(line.rstrip())  # Afficher dans le terminal
				topic_modeling_logger.info(line.rstrip())  # Logger dans le fichier de log
				output_lines.append(line)
			
			# Attendre la fin du processus
			return_code = process.wait()
			
			# Afficher le code de retour
			print(f"\nProcessus terminé avec le code de retour: {return_code}")
			topic_modeling_logger.info(f"Processus terminé avec le code de retour: {return_code}")
			
			# Joindre toutes les lignes de sortie pour les logs
			full_output = ''.join(output_lines)
			
			topic_modeling_logger.info("==== FIN DU PROCESSUS DE TOPIC MODELING ====")

			# Vérifier si le processus s'est terminé avec succès
			if return_code == 0:
				success_message = dbc.Alert(
					[
						html.P("Le topic modeling s'est terminé avec succès!", className="mb-0"),
						html.P("Les résultats sont disponibles dans le menu déroulant ci-dessous.", className="mb-0"),
						html.Details([
							html.Summary("Afficher les logs", className="text-primary"),
							html.Pre(full_output, className="mt-2 p-2 bg-light border small overflow-auto", style={"max-height": "300px"})
						], className="mt-2")
					],
					color="success"
				)

				# Mettre à jour les options du dropdown
				options = get_topic_modeling_results()
				
				return success_message, options

			else:
				error_message = f"Le script de topic modeling a échoué avec le code de retour {return_code}."
				topic_modeling_logger.error(error_message)

				error_div = html.Div([
					dbc.Alert(
						[
							html.P(error_message, className="mb-0"),
							html.Pre(full_output, className="mt-2 p-2 bg-light border small overflow-auto", style={"max-height": "300px"})
						],
						color="danger"
					)
				])
				return error_div, dash.no_update

		except Exception as e:
			topic_modeling_logger.error(f"Erreur lors du lancement du topic modeling: {str(e)}")
			return dbc.Alert(f"Erreur lors du lancement du topic modeling: {str(e)}", color="danger"), dash.no_update

	# Callback pour afficher les résultats de topic modeling lorsqu'on clique sur le bouton "Charger les résultats"
	@app.callback(
		Output("advanced-topic-stats-content", "children"),
		Input("load-stats-button", "n_clicks"),
		State("topic-modeling-results-dropdown", "value"),
		prevent_initial_call=True
	)
	def update_advanced_topic_stats(n_clicks, selected_file):
		print("\n\n==== DÉBUT DEBUG update_advanced_topic_stats ====\n")
		print(f"Bouton cliqué {n_clicks} fois")
		print(f"Fichier sélectionné: {selected_file}")

		if not n_clicks:
			return html.Div("Cliquez sur 'Charger les résultats' pour afficher les statistiques.", className="alert alert-info")

		if not selected_file:
			print("Aucun fichier sélectionné")
			return html.Div("Sélectionnez un fichier de résultats pour afficher les statistiques.", className="alert alert-info")

		# Remove cache busting parameter if present
		if isinstance(selected_file, str) and '?' in selected_file:
			selected_file = selected_file.split('?')[0]
			print(f"Fichier après suppression du cache: {selected_file}")

		print(f"Appel de render_advanced_topic_stats_from_json avec {selected_file}")
		result = render_advanced_topic_stats_from_json(selected_file)
		print("\n==== FIN DEBUG update_advanced_topic_stats ====\n\n")
		return result
		
	# Callback pour afficher l'explorateur d'articles lorsqu'on clique sur le bouton "Charger les résultats"
	@app.callback(
		Output("article-browser-content", "children"),
		Input("load-articles-button", "n_clicks"),
		State("topic-modeling-results-dropdown", "value"),
		prevent_initial_call=True
	)
	def update_article_browser(n_clicks, selected_file):
		print("\n\n==== DÉBUT DEBUG update_article_browser ====\n")
		print(f"Bouton cliqué {n_clicks} fois")
		print(f"Fichier sélectionné: {selected_file}")

		if not n_clicks:
			return html.Div("Cliquez sur 'Charger les résultats' pour afficher l'explorateur d'articles.", className="alert alert-info")

		if not selected_file:
			print("Aucun fichier sélectionné")
			return html.Div("Sélectionnez un fichier de résultats pour afficher l'explorateur d'articles.", className="alert alert-info")

		# Remove cache busting parameter if present
		if isinstance(selected_file, str) and '?' in selected_file:
			selected_file = selected_file.split('?')[0]
			print(f"Fichier après suppression du cache: {selected_file}")

		# Extraire le modèle et la date du nom de fichier pour trouver le fichier doc_topic correspondant
		try:
			# Récupérer les chemins nécessaires
			project_root, config, advanced_analysis_dir, doc_topic_dir, _ = get_config_and_paths()
			
			# Extraire l'ID du modèle à partir du nom du fichier de résultats sélectionné
			model_id_match = re.search(r'advanced_analysis_(.+)\.json', Path(selected_file).name)
			if not model_id_match:
				return html.Div("Impossible d'extraire l'ID du modèle à partir du fichier de résultats sélectionné.", className="alert alert-danger")
			
			model_id = model_id_match.group(1)
			print(f"ID du modèle extrait: {model_id}")
			
			# Chercher le fichier doc_topic correspondant
			doc_topic_file = doc_topic_dir / f"doc_topic_matrix_{model_id}.json"
			if not doc_topic_file.exists():
				return html.Div(f"Fichier doc_topic introuvable: {doc_topic_file}", className="alert alert-danger")
			
			print(f"Chargement de l'explorateur d'articles avec le fichier: {doc_topic_file}")
			result = load_article_browser(str(doc_topic_file))
			print("\n==== FIN DEBUG update_article_browser ====\n\n")
			return result
			
		except Exception as e:
			print(f"Erreur lors du chargement de l'explorateur d'articles: {str(e)}")
			return html.Div(f"Erreur lors du chargement de l'explorateur d'articles: {str(e)}", className="alert alert-danger")
	
	# CORRECTION : Callback pour le bouton de nommage des topics
	@app.callback(
		Output("topic-naming-status", "children"),
		Output("topic-naming-results", "children"),
		Input("btn-run-topic-naming", "n_clicks"), # CORRECTION : ID du bouton
		State("topic-naming-method", "value"),
		State("topic-naming-num-articles", "value"),
		State("topic-naming-threshold", "value"),
		State("topic-naming-output-path", "value"),
		State("topic-modeling-results-dropdown", "value"), # CORRECTION : ID du dropdown
		prevent_initial_call=True
	)
	def run_topic_naming(n_clicks, method, num_articles, threshold, output_path, selected_result_file):
		if not n_clicks or not selected_result_file:
			return no_update, no_update

		try:
			# Récupérer les chemins nécessaires
			project_root, config, advanced_analysis_dir, doc_topic_dir, results_dir = get_config_and_paths()
			articles_path = project_root / 'data' / 'processed' / 'articles.json'

			# Extraire l'ID du modèle à partir du nom du fichier de résultats sélectionné
			# Ex: advanced_analysis_bertopic_20250605-103000_... .json -> bertopic_20250605-103000_...
			model_id_match = re.search(r'advanced_analysis_(.+)\.json', Path(selected_result_file).name)
			if not model_id_match:
				return dbc.Alert("Impossible d'extraire l'ID du modèle à partir du fichier de résultats sélectionné.", color="danger"), no_update
			
			model_id = model_id_match.group(1)
			topic_naming_logger.info(f"ID du modèle extrait pour le nommage : {model_id}")

			# NOUVELLE LOGIQUE : Construire le chemin de sortie avec l'ID du modèle
			if not output_path:
				output_path = str(results_dir / f"topic_names_llm_{model_id}.json")
				topic_naming_logger.info(f"Chemin de sortie généré automatiquement : {output_path}")

			# Construire la commande
			script_path = project_root / "src" / "scripts" / "run_topic_naming.py"
			python_executable = sys.executable
			cmd_args = [
				python_executable, str(script_path),
				"--source-file", str(articles_path),
				"--doc-topic-dir", str(doc_topic_dir),
				"--advanced-analysis-dir", str(advanced_analysis_dir),
				"--method", method,
				"--output-file", output_path,
				"--num-articles", str(num_articles),
				"--threshold", str(threshold),
				"--config", str(project_root / "config" / "config.yaml"),
				"--model-id", model_id # Passer l'ID du modèle au script
			]

			topic_naming_logger.info("==== DÉBUT DU PROCESSUS DE NOMMAGE DES TOPICS AVEC LLM ====")
			topic_naming_logger.info(f"Commande exécutée: {' '.join(cmd_args)}")

			# Exécution du script
			process = subprocess.run(cmd_args, capture_output=True, text=True, encoding='utf-8')
			topic_naming_logger.info(f"Script terminé avec le code {process.returncode}")

			if process.returncode != 0:
				topic_naming_logger.error(f"Erreur du script de nommage: {process.stderr}")
				error_details = html.Details([
					html.Summary("Afficher les détails de l'erreur"),
					html.Pre(process.stdout + "\n" + process.stderr, className="mt-2 p-2 bg-light border small overflow-auto", style={"max-height": "300px"})
				])
				return dbc.Alert(["Le nommage des topics a échoué.", error_details], color="danger"), no_update
			
			# Si succès, charger et afficher les résultats
			with open(output_path, 'r', encoding='utf-8') as f:
				topic_names_data = json.load(f)

			success_alert = dbc.Alert("Nommage des topics terminé avec succès !", color="success")
			results_display = render_topic_naming_results(topic_names_data)

			return success_alert, results_display

		except Exception as e:
			topic_naming_logger.error(f"Erreur lors du lancement du nommage des topics: {e}")
			return dbc.Alert(f"Erreur lors du lancement du nommage des topics: {e}", color="danger"), no_update


	# Callback pour appliquer les noms de topics aux visualisations
	@app.callback(
		Output("topic-names-apply-result", "children"),
		[Input("btn-apply-topic-names", "n_clicks")],
		[State("topic-names-store", "data")],
		prevent_initial_call=True
	)
	def apply_topic_names(n_clicks, topic_names_data):
		if not n_clicks:
			return dash.no_update

		topic_naming_logger.info("==== DÉBUT APPLY_TOPIC_NAMES ====")
		topic_naming_logger.info(f"Bouton cliqué {n_clicks} fois")
		topic_naming_logger.info(f"Type de topic_names_data: {type(topic_names_data)}")

		if topic_names_data is None:
			topic_naming_logger.warning("Aucune donnée de noms de topics disponible")
			return dbc.Alert("Aucune donnée de noms de topics disponible", color="warning")

		topic_naming_logger.info(f"Contenu de topic_names_data: {topic_names_data}")

		try:
			# Récupérer les chemins nécessaires
			project_root, config, advanced_analysis_dir, doc_topic_dir, _ = get_config_and_paths()
			results_dir = project_root / config['data']['results_dir']

			# Trouver le fichier d'analyse avancée le plus récent pour obtenir l'ID du modèle
			advanced_analysis_files = list(advanced_analysis_dir.glob('advanced_analysis_*.json'))
			if not advanced_analysis_files:
				topic_naming_logger.error("Aucun fichier d'analyse avancée trouvé")
				return dbc.Alert("Aucun fichier d'analyse avancée trouvé", color="danger")

			# Trier par date de modification (le plus récent en premier)
			advanced_analysis_files.sort(key=lambda x: os.stat(x).st_mtime, reverse=True)
			advanced_analysis_path = advanced_analysis_files[0]
			topic_naming_logger.info(f"Fichier d'analyse avancée: {advanced_analysis_path}")

			# Extraire l'ID du modèle du nom de fichier
			model_id = None
			filename = advanced_analysis_path.name
			if '_' in filename:
				# Essayer d'extraire l'ID du modèle (dernier segment après le dernier underscore)
				model_id = filename.split('_')[-1].replace('.json', '')
				topic_naming_logger.info(f"ID du modèle extrait: {model_id}")

			# Convertir les clés numériques en chaînes pour assurer la compatibilité
			topic_names_dict = {}
			for topic_id, data in topic_names_data.items():
				# Extraire le numéro de topic si le format est "Topic #X"
				if isinstance(topic_id, str) and topic_id.startswith("Topic #"):
					try:
						topic_num = int(topic_id.replace("Topic #", ""))
						topic_id = str(topic_num)  # Convertir en chaîne
					except ValueError:
						pass  # Garder le format original si la conversion échoue

				# S'assurer que topic_id est une chaîne
				topic_id = str(topic_id)

				# Extraire le titre et le résumé
				if isinstance(data, dict):
					title = data.get('title', "Titre non disponible")
					summary = data.get('summary', "Résumé non disponible")
				elif isinstance(data, (list, tuple)) and len(data) > 0:
					title = data[0]
					summary = data[1] if len(data) > 1 else "Résumé non disponible"
				else:
					title = str(data)
					summary = "Résumé non disponible"

				topic_names_dict[topic_id] = {
					"title": title,
					"summary": summary
				}

			topic_naming_logger.info(f"Noms de topics formatés: {topic_names_dict}")

			# Sauvegarder les noms de topics dans un fichier dédié
			# 1. Fichier spécifique au modèle
			if model_id:
				model_topic_names_path = results_dir / f"topic_names_{model_id}.json"
				with open(model_topic_names_path, 'w', encoding='utf-8') as f:
					json.dump(topic_names_dict, f, ensure_ascii=False, indent=2)
				topic_naming_logger.info(f"Noms de topics sauvegardés pour le modèle {model_id}: {model_topic_names_path}")

			# 2. Fichier global (toujours mis à jour avec les noms les plus récents)
			global_topic_names_path = results_dir / "topic_names.json"
			with open(global_topic_names_path, 'w', encoding='utf-8') as f:
				json.dump(topic_names_dict, f, ensure_ascii=False, indent=2)
			topic_naming_logger.info(f"Noms de topics sauvegardés globalement: {global_topic_names_path}")

			topic_naming_logger.info("==== FIN APPLY_TOPIC_NAMES ====\n")

			return dbc.Alert("Noms de topics appliqués avec succès! Actualisez les visualisations pour voir les changements.", color="success")

		except Exception as e:
			topic_naming_logger.error(f"Erreur lors de l'application des noms de topics: {str(e)}")
			topic_naming_logger.error("==== FIN APPLY_TOPIC_NAMES (AVEC ERREUR) ====\n")
			return dbc.Alert(f"Erreur lors de l'application des noms de topics: {str(e)}", color="danger")

	# Callback pour remplir la liste des topics disponibles pour le filtrage des publicités
	@app.callback(
		Output("ad-filter-topic-id", "options"),
		[Input("topic-modeling-tabs", "active_tab"),
		Input("btn-refresh-topics", "n_clicks")],
		prevent_initial_call=True
	)
	def update_ad_filter_topic_options(active_tab, n_clicks):
		# Déterminer quel élément a déclenché le callback
		trigger = ctx.triggered_id if ctx.triggered else None

		# Only run if the ad filter tab is active or the refresh button is clicked
		if active_tab != "ad-filter-tab" and trigger != "btn-refresh-topics":
			return dash.no_update

		# Récupérer les informations sur les topics
		project_root, config, advanced_analysis_dir, doc_topic_dir, topic_names_dir = get_config_and_paths()
		results_dir = project_root / config['data']['results_dir']

		# Trouver le fichier d'analyse avancée le plus récent
		advanced_analysis_files = list(advanced_analysis_dir.glob('advanced_analysis_*.json'))
		if not advanced_analysis_files:
			print("Aucun fichier d'analyse avancée trouvé.")
			return [{"label": "Aucun topic disponible", "value": ""}]

		# Trier par date de modification (le plus récent en premier)
		advanced_analysis_files.sort(key=lambda x: os.stat(x).st_mtime, reverse=True)
		advanced_analysis_path = advanced_analysis_files[0]

		# Vérifier si le fichier d'analyse existe
		if not advanced_analysis_path.exists():
			print(f"Fichier {advanced_analysis_path} introuvable.")
			return [{"label": "Aucun topic disponible", "value": ""}]

		topic_options = [{"label": "Sélectionnez un topic", "value": ""}]

		try:
			with open(advanced_analysis_path, 'r', encoding='utf-8') as f:
				stats = json.load(f)

			num_topics = len(stats.get('topic_distribution', []))
			if num_topics == 0:
				return [{"label": "Aucun topic disponible", "value": ""}]

			topic_names = {}
			if 'topic_names_llm' in stats:
				llm_names = stats['topic_names_llm']
				if isinstance(llm_names, str):
					llm_names = ast.literal_eval(llm_names)
				if isinstance(llm_names, dict):
					for topic_id, data in llm_names.items():
						topic_num = int(re.search(r'\d+', topic_id).group())
						title = data.get('title', f"Topic {topic_num}") if isinstance(data, dict) else data[0]
						topic_names[topic_num] = title

			topic_keywords = {}
			if 'weighted_words' in stats:
				for topic_id, words_data in stats['weighted_words'].items():
					topic_num = int(topic_id)
					keywords = [word[0] for word in words_data[:5]]
					topic_keywords[topic_num] = ", ".join(keywords)

			for i in range(num_topics):
				name = topic_names.get(i, f"Topic {i}")
				keywords_str = topic_keywords.get(i, "")
				label = f"{name}"
				if keywords_str:
					label += f" - ({keywords_str})"

				topic_options.append({
					"label": label,
					"value": str(i)
				})

			print(f"Chargé {num_topics} topics pour le filtrage des publicités")
			return topic_options

		except Exception as e:
			print(f"Erreur lors du chargement des options de topic pour le filtre pub: {e}")


# Fonctions pour l'exportation des résultats de topic modeling
def get_topic_modeling_source_data():
    """
    Récupère les chemins des fichiers source pour l'exportation du topic modeling.
    Utilise le résultat actuellement sélectionné dans le dropdown pour trouver les fichiers correspondants.
    
    Returns:
        list: Liste des chemins de fichiers à inclure dans l'exportation
    """
    global current_selected_result
    project_root, config, advanced_analysis_dir, doc_topic_dir, topic_names_dir = get_config_and_paths()
    
    source_files = []
    
    # Si aucun résultat n'est sélectionné, retourner une liste vide
    if not current_selected_result:
        return source_files
    
    # Extraire l'ID du modèle à partir du chemin du fichier sélectionné
    # Format typique: advanced_analysis_gensim_lda_20250609-191600_08f53591.json
    try:
        # Supprimer le cache buster s'il est présent
        clean_path = current_selected_result.split('?')[0]
        file_name = os.path.basename(clean_path)
        
        # Extraire le timestamp et l'ID du modèle
        match = re.search(r'advanced_analysis_(\w+)_(\d{8}-\d{6})_(\w+)\.json', file_name)
        if match:
            model_type = match.group(1)
            timestamp = match.group(2)
            model_id = match.group(3)
            
            # Ajouter le fichier d'analyse avancée sélectionné
            source_files.append(clean_path)
            
            # Trouver le fichier de matrice document-topic correspondant (CSV ou JSON)
            doc_topic_path_csv = doc_topic_dir / f"doc_topic_matrix_{model_type}_{timestamp}_{model_id}.csv"
            doc_topic_path_json = doc_topic_dir / f"doc_topic_matrix_{model_type}_{timestamp}_{model_id}.json"
            
            # Vérifier d'abord le format CSV
            if doc_topic_path_csv.exists():
                source_files.append(str(doc_topic_path_csv))
            # Sinon, vérifier le format JSON
            elif doc_topic_path_json.exists():
                source_files.append(str(doc_topic_path_json))
            
            # Trouver le fichier de noms de topics correspondant
            topic_names_path = topic_names_dir / f"topic_names_llm_{model_type}_{timestamp}_{model_id}.json"
            if topic_names_path.exists():
                source_files.append(str(topic_names_path))
            
            # Fallback pour les anciens formats de noms de fichiers
            if not topic_names_path.exists():
                # Chercher avec un pattern plus générique
                pattern = f"topic_names_llm_*_{model_id}.json"
                matching_files = list(topic_names_dir.glob(pattern))
                if matching_files:
                    source_files.append(str(matching_files[0]))
        else:
            # Si le format ne correspond pas, utiliser les fichiers les plus récents
            topic_modeling_logger.warning(f"Format de nom de fichier non reconnu: {file_name}. Utilisation des fichiers les plus récents.")
            
            # Fichier d'analyse avancée
            source_files.append(clean_path)
            
            # Fichier de matrice document-topic (CSV ou JSON)
            doc_topic_files = list(doc_topic_dir.glob('doc_topic_matrix_*.csv')) + list(doc_topic_dir.glob('doc_topic_matrix_*.json'))
            if doc_topic_files:
                doc_topic_files.sort(key=lambda x: os.stat(x).st_mtime, reverse=True)
                source_files.append(str(doc_topic_files[0]))
            
            # Fichier de noms de topics
            topic_names_files = list(topic_names_dir.glob('topic_names_llm_*.json'))
            if topic_names_files:
                topic_names_files.sort(key=lambda x: os.stat(x).st_mtime, reverse=True)
                source_files.append(str(topic_names_files[0]))
    
    except Exception as e:
        topic_modeling_logger.error(f"Erreur lors de la récupération des fichiers source: {e}")
        # En cas d'erreur, retourner le fichier sélectionné uniquement
        if current_selected_result:
            source_files.append(current_selected_result.split('?')[0])
    
    return source_files

def get_topic_modeling_figure():
    """
    Récupère la figure actuelle du topic modeling pour l'exportation.
    
    Returns:
        dict: Figure au format JSON ou None si aucune figure n'est disponible
    """
    # Pour l'instant, nous n'avons pas de figure spécifique à exporter
    # Cette fonction pourrait être améliorée pour récupérer une visualisation
    # des topics si elle est disponible
    return None

# Helper to render advanced stats (adapt to your JSON structure)
def render_advanced_topic_stats_from_json(json_file_path):
	import json
	import pathlib
	from dash import html, dcc
	import pandas as pd
	import plotly.express as px

	topic_modeling_logger.info("==== DÉBUT render_advanced_topic_stats_from_json ====")
	
	if not json_file_path:
		return html.Div("Sélectionnez un fichier de résultats.", className="alert alert-info")

	json_file_path = pathlib.Path(json_file_path.split('?')[0])
	if not json_file_path.exists():
		return html.Div(f"Fichier non trouvé : {json_file_path}", className="alert alert-danger")

	try:
		with open(json_file_path, 'r', encoding='utf-8') as f:
			stats = json.load(f)
		
		# NOUVELLE LOGIQUE : Extraire l'ID du modèle et charger les noms correspondants
		model_id_match = re.search(r'advanced_analysis_(.+)\.json', json_file_path.name)
		model_id = model_id_match.group(1) if model_id_match else None
		topic_names = load_topic_names(model_id) # Charge les noms ou retourne {}
		topic_modeling_logger.info(f"Noms de topics chargés pour {model_id}: {list(topic_names.keys())}")

		children = []
		
		# Affichage du score de cohérence
		if 'coherence_score' in stats and stats['coherence_score'] is not None:
			children.append(dbc.Alert(f"Score de cohérence : {stats['coherence_score']:.3f}", color="info"))

		# Graphique de distribution des topics
		if 'topic_distribution' in stats:
			dist = stats['topic_distribution']
			# Utilise get_topic_name qui gère les noms par défaut
			labels = [get_topic_name(i, topic_names) for i in range(len(dist))]
			df_dist = pd.DataFrame({'Topic': labels, 'Proportion': dist})
			fig = px.bar(df_dist, x='Topic', y='Proportion', title='Distribution des topics', text_auto='.2f')
			children.append(dcc.Graph(figure=fig))
		
		# Graphique du nombre d'articles par topic
		if 'topic_article_counts' in stats:
			counts = stats['topic_article_counts']
			# S'assurer que les clés sont triées numériquement
			sorted_ids = sorted(counts.keys(), key=lambda k: int(k))
			labels = [get_topic_name(i, topic_names) for i in sorted_ids]
			values = [counts[i] for i in sorted_ids]
			df_counts = pd.DataFrame({'Topic': labels, 'Nombre d\'articles': values})
			fig = px.bar(df_counts, x='Topic', y='Nombre d\'articles', title="Nombre d'articles par topic", text_auto=True)
			children.append(dcc.Graph(figure=fig))

		# Graphiques des mots-clés par topic
		if 'weighted_words' in stats:
			children.append(html.H5("Top mots par topic", className="mt-4"))
			sorted_ids = sorted(stats['weighted_words'].keys(), key=lambda k: int(str(k).replace("topic_", "")))
			
			for topic_id in sorted_ids:
				words_data = stats['weighted_words'][topic_id]
				# Gérer le cas où les données sont des listes de [mot, poids]
				if not isinstance(words_data, list) or not all(isinstance(i, list) and len(i) == 2 for i in words_data):
					continue
				
				df_words = pd.DataFrame(words_data, columns=['Mot', 'Poids'])
				df_words['Poids'] = pd.to_numeric(df_words['Poids'], errors='coerce')
				df_words = df_words.sort_values('Poids', ascending=False).head(10)
				
				topic_title = get_topic_name(topic_id, topic_names)
				fig = px.bar(df_words, x='Poids', y='Mot', orientation='h', title=topic_title, text_auto='.3f')
				fig.update_layout(height=350, margin=dict(l=120), yaxis={'categoryorder':'total ascending'})
				children.append(dcc.Graph(figure=fig))
		
		return html.Div(children)

	except Exception as e:
		topic_modeling_logger.error(f"Erreur lors du rendu des stats pour {json_file_path}: {e}")
		import traceback
		traceback.print_exc()
		return html.Div(f"Erreur lors de la lecture ou du rendu du fichier JSON: {e}", className="alert alert-danger")


# To be called in app.py: from src.webapp.topic_modeling_viz import register_topic_modeling_callbacks, get_topic_modeling_layout