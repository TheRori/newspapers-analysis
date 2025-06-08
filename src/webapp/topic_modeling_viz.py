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
		# Sélecteur de fichiers de résultats
		dbc.Row([
			dbc.Col([
				dbc.Card([
					dbc.CardHeader(_html.H4("Résultats de Topic Modeling", className="mb-0")),
					dbc.CardBody([
						_html.Div([
							dbc.Label("Sélectionner un fichier de résultats:", className="fw-bold"),
							dcc.Dropdown(
								id="topic-modeling-results-dropdown",
								options=get_topic_modeling_results(),
								value=get_topic_modeling_results()[0]['value'] if get_topic_modeling_results() else None,
								clearable=False,
								className="mb-3"
							),
						]),
					])
				], className="mb-4 shadow")
			], width=12)
		]),
		# Tabs for different visualizations
		dbc.Tabs([
			dbc.Tab([
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
			], label="Statistiques", tab_id="stats-tab"),
			dbc.Tab([
				dbc.Row([
					dbc.Col([
						_html.H4("Explorateur d'articles", className="mt-4 mb-3"),
						dcc.Loading(
							id="loading-article-browser",
							type="default",
							children=_html.Div(id="article-browser-content")
						)
					], width=12)
				])
			], label="Explorateur d'articles", tab_id="article-browser-tab"),
			dbc.Tab([
				dbc.Row([
					dbc.Col([
						_html.H4("Nommage des topics avec LLM", className="mt-4 mb-3"),
						_html.P("Cet outil vous permet de générer automatiquement des noms et des résumés pour vos topics en utilisant un LLM.", className="text-muted"),
						dbc.Card([
							dbc.CardBody([
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
											type="number",
											min=1,
											max=20,
											step=1,
											value=10,
											className="mb-2"
										),
									], width=6),
								]),
								dbc.Row([
									dbc.Col([
										dbc.Label("Seuil de probabilité", html_for="topic-naming-threshold"),
										dbc.Input(
											id="topic-naming-threshold",
											type="number",
											min=0.1,
											max=0.9,
											step=0.1,
											value=0.5,
											className="mb-2"
										),
									], width=6),
									dbc.Col([
										dbc.Label("Fichier de sortie", html_for="topic-naming-output-path"),
										dbc.Input(
											id="topic-naming-output-path",
											type="text",
											placeholder="Laissez vide pour générer automatiquement",
											className="mb-2"
										),
									], width=6),
								]),
								dbc.Button(
									"Générer les noms des topics",
									id="btn-run-topic-naming",
									color="primary",
									className="mt-2"
								),
							])
						], className="mb-4"),
						_html.Div(id="topic-naming-status", className="mt-3"),
						dcc.Loading(
							id="loading-topic-naming-results",
							type="default",
							children=_html.Div(id="topic-naming-results")
						)
					], width=12)
				])
			], label="Nommage des topics", tab_id="topic-naming-tab"),
			dbc.Tab([
				dbc.Row([
					dbc.Col([
						_html.H4("Filtrage des publicités par topic", className="mt-4 mb-3"),
						_html.P("Cet outil vous permet de détecter et filtrer les publicités d'un topic spécifique en utilisant un LLM.", className="text-muted"),
						dbc.Card([
							dbc.CardBody([
								dbc.Row([
									dbc.Col([
										dbc.Label("Topic à analyser", html_for="ad-filter-topic-id"),
										dbc.InputGroup([
											dbc.Select(
												id="ad-filter-topic-id",
												options=[{"label": "Chargement des topics...", "value": ""}],  # Valeur initiale
												value=None,
												className="mb-2"
											),
											dbc.Button("Rafraîchir", id="btn-refresh-topics", color="secondary", className="mb-2 ms-2")
										]),
									], width=6),
									dbc.Col([
										dbc.Label("Valeur minimale du topic", html_for="ad-filter-min-value"),
										dbc.Input(
											id="ad-filter-min-value",
											type="number",
											min=0.1,
											max=0.9,
											step=0.1,
											value=0.5,
											className="mb-2"
										),
									], width=6),
								]),
								dbc.Row([
									dbc.Col([
										dbc.Label("Fichier de sortie", html_for="ad-filter-output-path"),
										dbc.Input(
											id="ad-filter-output-path",
											type="text",
											placeholder="Laissez vide pour générer automatiquement",
											className="mb-2"
										),
									], width=12),
								]),
								dbc.Row([
									dbc.Col([
										dbc.Checkbox(
											id="ad-filter-dry-run",
											label="Mode test (ne pas écrire le fichier)",
											value=False,
											className="mb-3"
										),
									], width=12),
								]),
								dbc.Button(
									"Lancer le filtrage des publicités",
									id="btn-run-ad-filter",
									color="primary",
									className="mt-2"
								),
							])
						], className="mb-4"),
						_html.Div(id="ad-filter-status", className="mt-3"),
						dcc.Loading(
							id="loading-ad-filter-results",
							type="default",
							children=_html.Div(id="ad-filter-results")
						)
					], width=12)
				])
			], label="Filtrage des publicités", tab_id="ad-filter-tab")
		], id="topic-modeling-tabs", active_tab="stats-tab"),
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

# Function to load and display the article browser with topic distribution
def load_article_browser(custom_doc_topic_path=None):
	"""
	Loads the doc_topic_matrix.json file and creates an interactive table to browse articles
	with their topic distributions.

	Args:
		custom_doc_topic_path: Chemin personnalisé vers un fichier doc_topic_matrix.json

	Returns:
		dash components for the article browser
	"""
	project_root, config, _, doc_topic_dir, _ = get_config_and_paths()
	results_dir = project_root / config['data']['results_dir']

	# Use custom path if provided, otherwise find the latest doc_topic_matrix file
	if custom_doc_topic_path:
		doc_topic_path = custom_doc_topic_path
	else:
		# Find the latest doc_topic_matrix file
		if doc_topic_dir.exists():
			doc_topic_files = list(doc_topic_dir.glob('doc_topic_matrix_*.json'))
			if doc_topic_files:
				# Sort by modification time (newest first)
				doc_topic_files.sort(key=lambda x: os.stat(x).st_mtime, reverse=True)
				doc_topic_path = str(doc_topic_files[0])
			else:
				# Fallback to old location
				doc_topic_path = str(results_dir / 'doc_topic_matrix.json')
		else:
			# Fallback to old location
			doc_topic_path = str(results_dir / 'doc_topic_matrix.json')

	articles_path = project_root / 'data' / 'processed' / 'articles.json'

	if not doc_topic_path:
		return html.Div("Fichier doc_topic_matrix.json introuvable. Exécutez d'abord le topic modeling.",
					className="alert alert-warning")

	# Load doc_topic_matrix.json
	with open(doc_topic_path, 'r', encoding='utf-8') as f:
		doc_topic_data = json.load(f)

	# Check if the file has the expected structure
	if not isinstance(doc_topic_data, list) and 'doc_topic_matrix' in doc_topic_data:
		doc_topic_matrix = doc_topic_data['doc_topic_matrix']
	else:
		doc_topic_matrix = doc_topic_data

	# Load article information if available
	article_info = {}
	if articles_path.exists():
		try:
			with open(articles_path, 'r', encoding='utf-8') as f:
				articles = json.load(f)

			# Create a lookup dictionary for article information
			for article in articles:
				article_id = article.get('doc_id', article.get('id', ''))
				if article_id:
					# Extract date and newspaper from article ID if available
					date = ''
					newspaper = ''
					if isinstance(article_id, str) and 'article_' in article_id:
						parts = article_id.split('_')
						if len(parts) > 2:
							date = parts[1]  # Extract date part
						if len(parts) > 3:
							newspaper = parts[2]  # Extract newspaper part

					article_info[str(article_id)] = {
						'title': article.get('title', 'Sans titre'),
						'date': article.get('date', date),
						'newspaper': article.get('newspaper', newspaper),
						'content': article.get('content', article.get('original_content', 'Contenu non disponible'))[:200] + '...'  # Preview
					}
		except Exception as e:
			print(f"Erreur lors du chargement des articles: {e}")

	# Load topic names if available
	topic_names = {}

	# Try to load from topic_names_llm.json first
	project_root, config, advanced_analysis_dir, _, topic_names_dir = get_config_and_paths()

	# Look for topic_names_llm files
	topic_names_files = list(topic_names_dir.glob('topic_names_llm*.json'))
	if topic_names_files:
		# Sort by modification time (newest first)
		topic_names_files.sort(key=lambda x: os.stat(x).st_mtime, reverse=True)
		topic_names_file = topic_names_files[0]

		try:
			with open(topic_names_file, 'r', encoding='utf-8') as f:
				topic_names_data = json.load(f)
				if 'topic_names' in topic_names_data:
					topic_names = topic_names_data['topic_names']
				# Can be string or dict
				if isinstance(stats['topic_names_llm'], dict):
					topic_names = stats['topic_names_llm']
				else:
					try:
						topic_names = ast.literal_eval(stats['topic_names_llm'])
					except Exception:
						topic_names = {}
		except Exception as e:
			print(f"Erreur lors du chargement des noms de topics: {e}")

	# Prepare data for the table
	table_data = []
	for item in doc_topic_matrix:
		doc_id = item.get('doc_id', '')
		topic_distribution = item.get('topic_distribution', [])

		# Get article info if available
		info = article_info.get(str(doc_id), {})

		# Find dominant topic
		dominant_topic_idx = 0
		if topic_distribution:
			dominant_topic_idx = topic_distribution.index(max(topic_distribution))

		# Format topic distribution for display
		topic_dist_formatted = []
		for i, value in enumerate(topic_distribution):
			topic_name = topic_names.get(f'topic_{i}', f"Topic {i}")
			topic_dist_formatted.append({
				'topic_id': i,
				'topic_name': topic_name,
				'value': value
			})

		row = {
			'doc_id': doc_id,
			'title': info.get('title', 'Sans titre'),
			'date': info.get('date', ''),
			'newspaper': info.get('newspaper', ''),
			'content_preview': info.get('content', 'Contenu non disponible'),
			'dominant_topic': dominant_topic_idx,
			'dominant_topic_name': topic_names.get(f'topic_{dominant_topic_idx}', f"Topic {dominant_topic_idx}"),
			'dominant_topic_value': max(topic_distribution) if topic_distribution else 0,
			'topic_distribution': topic_distribution,
			'topic_dist_formatted': topic_dist_formatted
		}
		table_data.append(row)

	# Create dropdown for sorting options
	num_topics = len(table_data[0]['topic_distribution']) if table_data else 0
	sort_options = [{'label': 'ID du document', 'value': 'doc_id'}]
	sort_options.append({'label': 'Date', 'value': 'date'})
	sort_options.append({'label': 'Journal', 'value': 'newspaper'})
	sort_options.append({'label': 'Topic dominant', 'value': 'dominant_topic'})

	for i in range(num_topics):
		topic_name = topic_names.get(f'topic_{i}', f"Topic {i}")
		sort_options.append({'label': f'Valeur du {topic_name}', 'value': f'topic_{i}'})

	# Create the layout
	children = [
		dbc.Row([
			dbc.Col([
				html.H5("Trier les articles par:"),
				dcc.Dropdown(
					id='article-sort-dropdown',
					options=sort_options,
					value='dominant_topic',
					clearable=False
				),
				dbc.Checkbox(
					id='sort-descending-checkbox',
					label="Ordre décroissant",
					value=True,
					className="mt-2 mb-3"
				)
			], width=6),
			dbc.Col([
				html.H5("Filtrer par topic dominant:"),
				dcc.Dropdown(
					id='dominant-topic-filter',
					options=[{'label': 'Tous les topics', 'value': 'all'}] + [
						{'label': topic_names.get(f'topic_{i}', f"Topic {i}"), 'value': i}
						for i in range(num_topics)
					],
					value='all',
					clearable=False
				)
			], width=6)
		]),

		# Store the data
		dcc.Store(id='article-browser-data', data=table_data),

		# Table to display articles
		html.Div(id='article-browser-table-container', className="mt-4"),

		# Modal for viewing article details
		dbc.Modal([
			dbc.ModalHeader(dbc.ModalTitle("Détails de l'article")),
			dbc.ModalBody(id="article-detail-body"),
			dbc.ModalFooter(
				dbc.Button("Fermer", id="close-article-modal", className="ms-auto", n_clicks=0)
			),
		], id="article-detail-modal", size="lg", is_open=False),
	]

	return html.Div(children)

# Fonction pour charger les noms de topics
def load_topic_names(model_id=None):
	"""Charge les noms de topics depuis un fichier JSON."""
	try:
		# Déterminer le chemin du fichier de noms de topics
		project_root, config, _, _, _ = get_config_and_paths()
		results_dir = project_root / config['data']['results_dir']
		
		topic_names = {}
		
		# 1. Essayer d'abord avec le model_id spécifique si fourni
		if model_id:
			topic_names_path = results_dir / f"topic_names_{model_id}.json"
			topic_modeling_logger.info(f"Tentative de chargement des noms de topics depuis: {topic_names_path}")
			
			if topic_names_path.exists():
				with open(topic_names_path, 'r', encoding='utf-8') as f:
					topic_names_data = json.load(f)
					# Vérifier si les noms sont directement dans le fichier ou sous une clé 'topic_names'
					if 'topic_names' in topic_names_data and isinstance(topic_names_data['topic_names'], dict):
						topic_names = topic_names_data['topic_names']
					else:
						topic_names = topic_names_data
				topic_modeling_logger.info(f"Noms de topics chargés depuis {topic_names_path}")
				return topic_names
		
		# 2. Chercher le fichier topic_names_llm le plus récent (priorité plus élevée que topic_names.json)
		topic_names_llm_files = list(results_dir.glob("topic_names_llm_*.json"))
		if topic_names_llm_files:
			# Trier par date de modification (le plus récent en premier)
			import os
			topic_names_llm_files.sort(key=lambda x: os.stat(x).st_mtime, reverse=True)
			latest_llm_file = topic_names_llm_files[0]
			
			topic_modeling_logger.info(f"Tentative de chargement des noms de topics depuis: {latest_llm_file}")
			with open(latest_llm_file, 'r', encoding='utf-8') as f:
				topic_names_data = json.load(f)
				# Vérifier si les noms sont directement dans le fichier ou sous une clé 'topic_names'
				if 'topic_names' in topic_names_data and isinstance(topic_names_data['topic_names'], dict):
					topic_names = topic_names_data['topic_names']
					topic_modeling_logger.info(f"Noms de topics chargés depuis {latest_llm_file} (clé 'topic_names')")
				else:
					topic_names = topic_names_data
					topic_modeling_logger.info(f"Noms de topics chargés depuis {latest_llm_file}")
				
				# Afficher les clés pour le débogage
				topic_modeling_logger.info(f"Clés disponibles dans le fichier de noms de topics: {list(topic_names.keys())[:10]}...")
			return topic_names
		
		# 3. Essayer avec le fichier global topic_names.json (priorité plus basse)
		topic_names_path = results_dir / "topic_names.json"
		topic_modeling_logger.info(f"Tentative de chargement des noms de topics depuis: {topic_names_path}")
		
		if topic_names_path.exists():
			with open(topic_names_path, 'r', encoding='utf-8') as f:
				topic_names_data = json.load(f)
				# Vérifier si les noms sont directement dans le fichier ou sous une clé 'topic_names'
				if 'topic_names' in topic_names_data and isinstance(topic_names_data['topic_names'], dict):
					topic_names = topic_names_data['topic_names']
					topic_modeling_logger.info(f"Noms de topics chargés depuis {topic_names_path} (clé 'topic_names')")
					return topic_names
				# Vérifier si ce fichier contient réellement des noms de topics ou juste des métadonnées
				elif any(str(i) in topic_names_data for i in range(10)):
					topic_names = topic_names_data
					topic_modeling_logger.info(f"Noms de topics chargés depuis {topic_names_path}")
					return topic_names
				else:
					topic_modeling_logger.info(f"Le fichier {topic_names_path} ne contient pas de noms de topics valides")
		
		topic_modeling_logger.warning("Aucun fichier de noms de topics trouvé")
		return {}

	except Exception as e:
		topic_modeling_logger.error(f"Erreur lors du chargement des noms de topics: {str(e)}")
		return {}

# Fonction pour obtenir le nom d'un topic
def get_topic_name(topic_id, model_id=None, default=None):
	"""
	Retourne le nom d'un topic spécifique de manière robuste.
	Si le nom n'est pas disponible, retourne un nom par défaut.
	"""
	if default is None:
		default = f"Topic {topic_id}"

	try:
		# 1. Charger les noms de topics
		topic_names = load_topic_names(model_id) # Utilise la fonction de chargement existante

		if not topic_names: # Si le dictionnaire est vide
			return default

		# 2. Essayer plusieurs formats de clés pour trouver le nom
		topic_id_str = str(topic_id).strip()
		topic_id_digits = re.sub(r"[^\d]", "", topic_id_str)
		
		possible_keys = [
			topic_id_str,
			topic_id_digits,
			f'topic_{topic_id_digits}',
			f'Topic #{topic_id_digits}',
			f'Topic {topic_id_digits}'
		]
		
		# DEBUG: Afficher les clés générées et le type de topic_id
		print(f"DEBUG get_topic_name: topic_id={topic_id} ({type(topic_id)}), model_id={model_id}, possible_keys={possible_keys}")
		
		found_name = None
		for key in possible_keys:
			if key in topic_names:
				name_data = topic_names[key]
				if isinstance(name_data, dict) and 'title' in name_data:
					found_name = name_data['title']
				elif isinstance(name_data, (list, tuple)) and len(name_data) > 0:
					found_name = name_data[0]
				else:
					found_name = str(name_data)
				break # Arrêter dès que le nom est trouvé

		if found_name:
			topic_modeling_logger.info(f"Nom trouvé pour Topic ID {topic_id_str}: '{found_name}'")
			return found_name
		else:
			topic_modeling_logger.warning(f"Aucun nom trouvé pour le topic {topic_id_str}, utilisation du nom par défaut")
			return default

	except Exception as e:
		topic_modeling_logger.error(f"Erreur lors de la récupération du nom du topic {topic_id}: {str(e)}")
		return default

# Helper to render the results of the topic naming script.
def render_topic_naming_results(topic_names):
	"""
	Renders the topic naming results in a user-friendly format.

	Args:
		topic_names (dict): A dictionary containing the generated topic names and summaries.

	Returns:
		A Dash component to display the results.
	"""
	if not topic_names:
		return html.Div("Aucun nom de topic n'a été généré.", className="alert alert-info")

	# Can be a dict or a string representation of a dict
	if isinstance(topic_names, str):
		try:
			topic_names = ast.literal_eval(topic_names)
		except (ValueError, SyntaxError) as e:
			return dbc.Alert(f"Erreur lors de la lecture des noms de topics: {e}", color="danger")

	accordion_items = []
	for topic_id, data in topic_names.items():
		title = "Titre non disponible"
		summary = "Résumé non disponible"

		if isinstance(data, dict):
			title = data.get('title', title)
			summary = data.get('summary', summary)
		elif isinstance(data, (list, tuple)) and len(data) > 0:
			title = data[0]
			if len(data) > 1:
				summary = data[1]

		item = dbc.AccordionItem(
			[
				html.P(summary, className="mb-0")
			],
			title=f"{topic_id.replace('_', ' ').title()}: {title}",
		)
		accordion_items.append(item)

	children = [
		html.H5("Noms et résumés des topics générés", className="mt-4 mb-3"),
		dbc.Accordion(
			accordion_items,
			start_collapsed=True,
			always_open=True,
			className="mb-4"
		)
	]

	# Add a button to apply the names to visualizations and a store for the names
	children.append(
		html.Div([
			dbc.Button(
				"Appliquer ces noms aux visualisations",
				id="btn-apply-topic-names",
				color="success",
				className="mb-3"
			),
			dcc.Store(id="topic-names-store", data=topic_names),
			html.Div(id="topic-names-apply-result"),  # Div pour afficher le résultat de l'application
			html.P("Note: Les noms générés seront automatiquement appliqués aux visualisations et à l'explorateur d'articles.",
				className="text-muted fst-italic small mt-2")
		])
	)

	return html.Div(children)


# Note: La fonction get_topic_name est déjà définie plus haut dans le fichier
# Nous utilisons une seule implémentation robuste pour éviter les conflits

def register_topic_modeling_callbacks(app):
	"""Register callbacks for the topic modeling page."""
	from dash import Input, Output, State, ctx, ALL, MATCH, no_update, html, dcc
	import dash_bootstrap_components as dbc
	import subprocess
	import json
	import ast
	import os
	import time
	import pandas as pd
	import numpy as np
	import plotly.express as px
	import plotly.graph_objects as go
	from datetime import datetime
	
	# Register the topic filter component callbacks
	register_topic_filter_callbacks(app, id_prefix="topic-filter")
	parser_args = get_topic_modeling_args()

	# Callback pour lancer le topic modeling
	@app.callback(
		Output("topic-modeling-run-status", "children"),
		Output("topic-modeling-results-dropdown", "options"),
		Input("btn-run-topic-modeling", "n_clicks"),
		State("arg-input-file", "value"),
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

			# Ajouter le fichier d'entrée s'il est spécifié
			if input_file:
				cmd_args.extend(["--input-file", input_file])

			# Ajouter le fichier de cache s'il est spécifié
			if cache_file:
				cmd_args.extend(["--cache-file", cache_file])

			# Ajouter le fichier de configuration
			cmd_args.extend(["--config", str(project_root / "config" / "config.yaml")])

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
			process = subprocess.Popen(
				cmd_args,
				stdout=subprocess.PIPE,
				stderr=subprocess.PIPE,
				text=True,
				bufsize=1,
				universal_newlines=True,
				encoding='utf-8' # Ensure correct encoding
			)

			# Utiliser communicate() pour obtenir stdout et stderr, évite les blocages
			stdout, stderr = process.communicate()
			
			topic_modeling_logger.info("==== STDOUT DU SCRIPT ====")
			topic_modeling_logger.info(stdout)
			if stderr:
				topic_modeling_logger.error("==== STDERR DU SCRIPT ====")
				topic_modeling_logger.error(stderr)
			
			topic_modeling_logger.info("==== FIN DU PROCESSUS DE TOPIC MODELING ====")

			# Vérifier si le processus s'est terminé avec succès
			if process.returncode == 0:
				success_message = dbc.Alert(
					[
						html.P("Le topic modeling s'est terminé avec succès!", className="mb-0"),
						html.P("Les résultats sont disponibles dans le menu déroulant ci-dessous.", className="mb-0")
					],
					color="success"
				)

				# Mettre à jour les options du dropdown
				options = get_topic_modeling_results()
				
				return success_message, options

			else:
				error_message = f"Le script de topic modeling a échoué avec le code de retour {process.returncode}."
				topic_modeling_logger.error(error_message)
				topic_modeling_logger.error(f"Stderr: {stderr}")

				error_div = html.Div([
					dbc.Alert(error_message, color="danger"),
					html.H5("Logs d'erreur:", className="mt-3"),
					html.Pre(stderr, style={
						"height": "300px",
						"overflow-y": "auto",
						"background-color": "#f8f9fa",
						"padding": "10px",
						"border-radius": "5px"
					})
				])
				return error_div, dash.no_update

		except Exception as e:
			topic_modeling_logger.error(f"Erreur lors du lancement du topic modeling: {str(e)}")
			return dbc.Alert(f"Erreur lors du lancement du topic modeling: {str(e)}", color="danger"), dash.no_update

	# Callback pour afficher les résultats de topic modeling lorsqu'un fichier est sélectionné
	@app.callback(
		Output("advanced-topic-stats-content", "children"),
		Input("topic-modeling-results-dropdown", "value"),
		prevent_initial_call=True
	)
	def update_advanced_topic_stats(selected_file):
		print("\n\n==== DÉBUT DEBUG update_advanced_topic_stats ====\n")
		print(f"Fichier sélectionné: {selected_file}")

		if not selected_file:
			print("Aucun fichier sélectionné")
			return html.Div("Sélectionnez un fichier de résultats pour afficher les statistiques.", className="alert alert-info")

		# Remove cache busting parameter if present
		if isinstance(selected_file, str) and '?' in selected_file:
			selected_file = selected_file.split('?')[0]
			print(f"Fichier après suppression du cache: {selected_file}")

		print(f"Appel de render_advanced_topic_stats_from_json avec {selected_file}")
		return render_advanced_topic_stats_from_json(selected_file)
		print("\n==== FIN DEBUG update_advanced_topic_stats ====\n\n")
		return result

	# Callback pour le bouton de nommage des topics
	@app.callback(
		Output("topic-naming-status", "children"),
		Output("loading-topic-naming-results", "children"),
		Output("topic-modeling-results-dropdown", "options", allow_duplicate=True),
		Input("btn-run-topic-naming", "n_clicks"),
		State("topic-naming-method", "value"),
		State("topic-naming-num-articles", "value"),
		State("topic-naming-threshold", "value"),
		State("topic-naming-output-path", "value"),
		prevent_initial_call=True
	)
	def run_topic_naming(n_clicks, method, num_articles, threshold, output_path):
		if not n_clicks:
			return dash.no_update, dash.no_update, dash.no_update

		try:
			# Récupérer les chemins nécessaires
			project_root, config, advanced_analysis_dir, doc_topic_dir, topic_names_dir = get_config_and_paths()
			results_dir = project_root / config['data']['results_dir']

			# Vérifier si les répertoires nécessaires existent
			articles_path = project_root / config['data']['processed_dir'] / 'articles.json'

			if not doc_topic_dir.exists():
				return dbc.Alert("Répertoire doc_topic_matrices introuvable. Exécutez d'abord le topic modeling.", color="danger"), dash.no_update, dash.no_update

			if not articles_path.exists():
				return dbc.Alert("Fichier d'articles introuvable.", color="danger"), dash.no_update, dash.no_update
		except Exception as e:
			return dbc.Alert(f"Erreur lors de la récupération des chemins: {str(e)}", color="danger"), dash.no_update, dash.no_update

		# Préparer le chemin de sortie
		if not output_path:
			from datetime import datetime
			timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
			output_path = str(results_dir / f"topic_names_llm_{timestamp}.json")

		# Construire la commande
		script_path = project_root / "src" / "scripts" / "run_topic_naming.py"

		# Utiliser sys.executable pour s'assurer d'utiliser le bon interpréteur Python
		python_executable = sys.executable

		# Construire les arguments de la commande avec les répertoires au lieu des fichiers spécifiques
		cmd_args = [
			python_executable,
			str(script_path),
			"--source-file", str(articles_path),
			"--doc-topic-dir", str(doc_topic_dir),
			"--advanced-analysis-dir", str(advanced_analysis_dir),
			"--method", method,
			"--output-file", output_path,
			"--num-articles", str(num_articles),
			"--threshold", str(threshold),
			"--config", str(project_root / "config" / "config.yaml")
		]

		# Exécuter la commande
		try:
			# Créer un message de statut
			status = dbc.Alert(
				[
					html.P("Lancement du script de nommage des topics...", className="mb-0"),
					html.P(f"Méthode: {method}, Nombre d'articles: {num_articles}, Seuil: {threshold}", className="mb-0 small"),
					html.P(f"Fichier de sortie: {output_path}", className="mb-0 small")
				],
				color="info"
			)

			# Créer un div pour afficher les logs
			log_div = html.Div(
				[
					html.P("Logs du processus de nommage des topics:", className="font-weight-bold"),
					html.Pre(style={"height": "200px", "overflow-y": "auto", "background-color": "#f8f9fa", "padding": "10px", "border-radius": "5px", "white-space": "pre-wrap"})
				]
			)

			# Afficher un message initial avec espace pour les logs
			status_with_logs = html.Div([status, log_div])

			# Logger le début du processus de nommage des topics
			topic_naming_logger.info("==== DÉBUT DU PROCESSUS DE NOMMAGE DES TOPICS AVEC LLM ====")
			topic_naming_logger.info(f"Commande exécutée: {' '.join(cmd_args)}")

			# Créer une fonction pour capturer et logger la sortie du processus
			def log_output(pipe, log_level=logging.INFO):
				for line in iter(pipe.readline, ''):
					if line.strip():
						topic_naming_logger.log(log_level, line.strip())

			# Exécuter le processus avec redirection vers des pipes pour logger la sortie
			process = subprocess.Popen(
				cmd_args,
				stdout=subprocess.PIPE,
				stderr=subprocess.PIPE,
				text=True,
				bufsize=1,
				universal_newlines=True
			)

			# Créer des threads pour lire et logger stdout et stderr
			stdout_thread = threading.Thread(target=log_output, args=(process.stdout, logging.INFO))
			stderr_thread = threading.Thread(target=log_output, args=(process.stderr, logging.ERROR))

			# Démarrer les threads
			stdout_thread.start()
			stderr_thread.start()

			# Attendre que le processus se termine
			process.wait()

			# Attendre que les threads se terminent
			stdout_thread.join()
			stderr_thread.join()

			topic_naming_logger.info("==== FIN DU PROCESSUS DE NOMMAGE DES TOPICS AVEC LLM ====")

			# Créer un fichier temporaire pour stocker les logs pour l'affichage dans l'interface
			import tempfile
			log_file = tempfile.NamedTemporaryFile(mode='w+', delete=False, suffix='.log', encoding='utf-8')
			log_file_path = log_file.name
			log_file.close()

			# Au lieu de réexécuter le processus, utilisons les logs de la console
			# Nous allons simplement enregistrer un message indiquant que les logs sont visibles dans la console
			with open(log_file_path, 'w', encoding='utf-8') as log_output:
				log_output.write("Les logs complets du processus de nommage des topics sont visibles dans la console.\n")
				log_output.write("Veuillez consulter la console pour voir les détails du processus en temps réel.\n")

			# Lire le message du fichier
			with open(log_file_path, 'r', encoding='utf-8') as f:
				logs = f.read()

			# Supprimer le fichier temporaire
			import os
			try:
				os.unlink(log_file_path)
			except:
				pass

			# Stocker les logs pour l'affichage
			stdout = logs
			stderr = ""

			# Vérifier si le processus s'est terminé avec succès
			if process.returncode == 0:
				# Charger les résultats
				try:
					with open(output_path, 'r', encoding='utf-8') as f:
						topic_names = json.load(f)

					# Mettre à jour le fichier d'analyse avancée avec les noms de topics
					# Trouver le fichier d'analyse avancée le plus récent
					advanced_analysis_files = list(advanced_analysis_dir.glob('advanced_analysis_*.json'))

					if advanced_analysis_files:
						# Trier par date de modification (le plus récent en premier)
						advanced_analysis_files.sort(key=lambda x: os.stat(x).st_mtime, reverse=True)
						latest_advanced_file = advanced_analysis_files[0]

						try:
							with open(latest_advanced_file, 'r', encoding='utf-8') as f:
								advanced_stats = json.load(f)

							# Ajouter les noms de topics au fichier d'analyse avancée
							advanced_stats['topic_names_llm'] = topic_names

							with open(latest_advanced_file, 'w', encoding='utf-8') as f:
								json.dump(advanced_stats, f, ensure_ascii=False, indent=2)

							topic_modeling_logger.info(f"Noms de topics ajoutés au fichier d'analyse avancée: {latest_advanced_file}")
						except Exception as e:
							topic_modeling_logger.error(f"Erreur lors de la mise à jour du fichier d'analyse avancée: {str(e)}")
				except Exception as e:
					topic_modeling_logger.error(f"Erreur lors du chargement des résultats: {str(e)}")
					return dbc.Alert(f"Erreur lors du chargement des résultats: {str(e)}", color="danger"), no_update, no_update

				# Afficher les résultats avec les logs
				results = html.Div([
					# Résultats du nommage
					render_topic_naming_results(topic_names),
					# Logs du processus
					html.Div([
						html.H5("Logs d'exécution:", className="mt-3"),
						html.Pre(stdout, style={
							"height": "300px",
							"overflow-y": "auto",
							"background-color": "#f8f9fa",
							"padding": "10px",
							"border-radius": "5px"
						})
					])
				])

				# Mettre à jour les options du dropdown
				options = get_topic_modeling_results()

				return dbc.Alert("Nommage des topics terminé avec succès!", color="success"), results, options
			else:
				# Afficher l'erreur avec les logs
				error_div = html.Div([
					dbc.Alert(f"Erreur lors de l'exécution du script", color="danger"),
					html.Div([
						html.H5("Logs d'erreur:", className="mt-3"),
						html.Pre(stdout, style={
							"height": "300px",
							"overflow-y": "auto",
							"background-color": "#f8f9fa",
							"padding": "10px",
							"border-radius": "5px"
						})
					])
				])
				return error_div, no_update, no_update
		except Exception as e:
			topic_modeling_logger.error(f"Erreur lors du nommage des topics: {str(e)}")
			return dbc.Alert(f"Erreur lors du nommage des topics: {str(e)}", color="danger"), no_update, no_update

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


# Helper to render advanced stats (adapt to your JSON structure)
def render_advanced_topic_stats_from_json(json_file_path):
	"""Render advanced topic statistics from a JSON file."""
	import ast
	import json
	import pathlib
	import dash_bootstrap_components as dbc
	from dash import html, dcc
	import traceback  # Pour le débogage

	# Référence à la fonction globale get_topic_name
	global get_topic_name

	topic_modeling_logger.info("==== DÉBUT render_advanced_topic_stats_from_json ====")
	topic_modeling_logger.info(f"Fichier reçu: {json_file_path} (type: {type(json_file_path)})")

	if not json_file_path:
		topic_modeling_logger.warning("Aucun fichier de résultats sélectionné.")
		return html.Div("Aucun fichier de résultats sélectionné.", className="alert alert-warning")

	# Remove any cache busting parameter
	if isinstance(json_file_path, str) and '?' in json_file_path:
		json_file_path = json_file_path.split('?')[0]
		topic_modeling_logger.info(f"Fichier après suppression du cache: {json_file_path}")

	# Convert to Path object if it's a string
	if isinstance(json_file_path, str):
		json_file_path = pathlib.Path(json_file_path)
		topic_modeling_logger.info(f"Converti en objet Path: {json_file_path}")

	if not json_file_path.exists():
		topic_modeling_logger.error(f"Fichier non trouvé: {json_file_path}")
		return html.Div(f"Fichier de résultats non trouvé: {json_file_path}", className="alert alert-warning")

	topic_modeling_logger.info(f"Chargement du fichier d'analyse avancée: {json_file_path}")
	try:
		with open(json_file_path, encoding='utf-8') as f:
			stats = json.load(f)
			topic_modeling_logger.info(f"Fichier chargé avec succès, clés disponibles: {list(stats.keys())}")

			# Utiliser la fonction globale get_topic_name au lieu d'une implémentation locale
			# Cela assure la cohérence du nommage des topics dans toute l'application

			# Extraire l'ID du modèle du nom de fichier pour charger les noms de topics spécifiques
			model_id = None
			filename = json_file_path.name
			if '_' in filename:
				# Essayer d'extraire l'ID du modèle (dernier segment après le dernier underscore)
				model_id = filename.split('_')[-1].replace('.json', '')
				topic_modeling_logger.info(f"ID du modèle extrait pour le chargement des noms de topics: {model_id}")

			def get_topic_label(topic_id, default_prefix="Topic"):
				return get_topic_name(topic_id, model_id)

			# Debug: Vérifier la structure des weighted_words s'ils existent
			if 'weighted_words' in stats:
				topic_modeling_logger.info("Structure de weighted_words:")
				for topic_id, words in list(stats['weighted_words'].items())[:1]:  # Afficher seulement le premier topic
					# Utiliser le nom du topic s'il existe
					topic_name = get_topic_name(topic_id, model_id)
					topic_modeling_logger.info(f"  Topic {topic_id} ({topic_name}): {words[:3]}")
					topic_modeling_logger.info(f"  Type des poids: {type(words[0][1]) if words and len(words[0]) > 1 else 'N/A'}")

			else:
				topic_modeling_logger.warning("Aucune clé 'weighted_words' trouvée dans le fichier")

			# Debug: Vérifier la structure de topic_distribution s'il existe
			if 'topic_distribution' in stats:
				topic_modeling_logger.info(f"topic_distribution: {stats['topic_distribution'][:5]}...")
			else:
				topic_modeling_logger.warning("Aucune clé 'topic_distribution' trouvée dans le fichier")
	except Exception as e:
		topic_modeling_logger.error(f"ERREUR lors de la lecture du JSON: {e}")
		import traceback
		topic_modeling_logger.error(traceback.format_exc())
		return html.Div(f"Erreur de lecture du JSON: {e}", className="alert alert-danger")

	topic_modeling_logger.info("==== FIN render_advanced_topic_stats_from_json ====")

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
				topic_names_llm = ast.literal_eval(stats['topic_names_llm'])
			except Exception:
				topic_names_llm = None

	# Extraire l'ID du modèle du nom de fichier
	model_id = None
	
	# Chercher le fichier topic_names_llm le plus récent
	project_root, config, _, _, _ = get_config_and_paths()
	results_dir = project_root / config['data']['results_dir']
	topic_names_llm_files = list(results_dir.glob('topic_names_llm_*.json'))
	
	if topic_names_llm_files:
		# Trier par date de modification (le plus récent en premier)
		import os
		topic_names_llm_files.sort(key=lambda x: os.stat(x).st_mtime, reverse=True)
		latest_llm_file = topic_names_llm_files[0]
		topic_modeling_logger.info(f"Utilisation du fichier de noms de topics le plus récent: {latest_llm_file}")
		model_id = latest_llm_file.stem.replace("topic_names_", "")  # ex: llm_20250525_154216
	else:
		# Fallback à l'ancienne logique
		filename = json_file_path.name
		if '_' in filename:
			model_id = filename.split('_')[-1].replace('.json', '')
			topic_modeling_logger.info(f"Aucun fichier topic_names_llm trouvé, utilisation de l'ID du modèle extrait du nom de fichier: {model_id}")

	# 2. Répartition des topics
	if 'topic_distribution' in stats:
		dist = stats['topic_distribution']
		
		# Utiliser la fonction get_topic_name pour obtenir les noms des topics
		# Garantir que les IDs des topics sont des entiers
		topics = [get_topic_name(int(i), model_id) for i in range(len(dist))]
		topic_modeling_logger.info(f"Topic names loaded for distribution graph: {topics[:5]}...")
		
		df_dist = pd.DataFrame({
			'Topic': topics,
			'Proportion': dist
		})
		
		# Créer le graphique avec les noms personnalisés
		fig = px.bar(df_dist, x='Topic', y='Proportion', title='Distribution des topics (proportion)', text_auto='.2f')
		fig.update_layout(
			xaxis_title="",  # Supprimer le titre de l'axe x pour plus de clarté
			title={
				'font': {'size': 16},
				'x': 0.5,
				'xanchor': 'center'
			}
		)
		children.append(dcc.Graph(figure=fig))
		topic_modeling_logger.info("Topic distribution graph created with custom names")

	# Vérifier si topic_article_counts existe
	if 'topic_article_counts' in stats:
		counts = stats['topic_article_counts']
		topic_ids = sorted([int(k) for k in counts.keys()])
		
		# Utiliser la fonction get_topic_name pour obtenir les noms des topics
		# Garantir que les IDs des topics sont des entiers
		topics = [get_topic_name(int(i), model_id) for i in topic_ids]
		topic_modeling_logger.info(f"Topic names loaded for article counts graph: {topics[:5] if len(topics) >= 5 else topics}...")
		
		articles = [counts[str(i)] for i in topic_ids]

		df_counts = pd.DataFrame({
			'Topic': topics,
			'Articles': articles
		})
		
		# Créer le graphique avec les noms personnalisés
		fig = px.bar(df_counts, x='Topic', y='Articles', title="Nombre d'articles par topic", text_auto=True)
		fig.update_layout(
			xaxis_title="",  # Supprimer le titre de l'axe x pour plus de clarté
			title={
				'font': {'size': 16},
				'x': 0.5,
				'xanchor': 'center'
			}
		)
		children.append(dcc.Graph(figure=fig))
		topic_modeling_logger.info("Topic article counts graph created with custom names")
	# Alternative: utiliser doc_topic_distribution pour calculer le nombre d'articles par topic
	elif 'doc_topic_distribution' in stats:
		try:
			doc_topic_dist = stats['doc_topic_distribution']
			topic_counts = {}
			for doc_id, topic_dist_list in doc_topic_dist.items():
				if not topic_dist_list: continue
				dominant_topic = max(range(len(topic_dist_list)), key=lambda i: topic_dist_list[i])
				topic_counts[dominant_topic] = topic_counts.get(dominant_topic, 0) + 1

			if topic_counts:
				topic_ids = sorted(topic_counts.keys())
				
				# Charger les noms de topics
				# Garantir que les IDs des topics sont des entiers
				topics = [get_topic_name(int(i), model_id) for i in topic_ids]
				topic_modeling_logger.info(f"Topic names loaded for doc_topic graph: {topics[:5] if len(topics) >= 5 else topics}...")
				
				counts = [topic_counts[i] for i in topic_ids]

				df_counts = pd.DataFrame({'Topic': topics, 'Articles': counts})
				
				# Créer le graphique avec les noms personnalisés
				fig = px.bar(df_counts, x='Topic', y='Articles', title="Nombre d'articles par topic (dominant)", text_auto=True)
				fig.update_layout(
					xaxis_title="",  # Supprimer le titre de l'axe x pour plus de clarté
					title={
						'font': {'size': 16},
						'x': 0.5,
						'xanchor': 'center'
					}
				)
				children.append(dcc.Graph(figure=fig))
				topic_modeling_logger.info("Doc topic distribution graph created with custom names")
		except Exception as e:
			children.append(html.Div(f"Erreur lors de la génération du graphique 'Nombre d'articles par topic': {str(e)}",
								className="alert alert-danger"))

	# 3. Top mots par topic
	if 'weighted_words' in stats:
		topic_modeling_logger.info("==== DEBUG: Traitement des weighted_words ====")
		topic_modeling_logger.info(f"Type de weighted_words: {type(stats['weighted_words'])}")

		# Vérifier si la clé est corrompue ou mal formée
		if 'wei' in stats and 'ighted_words' in stats:
			topic_modeling_logger.warning("Clé 'weighted_words' corrompue détectée, tentative de correction...")
			# Reconstruire la clé correcte
			stats['weighted_words'] = stats.get('wei', {}) | stats.get('ighted_words', {})
			# Supprimer les clés corrompues
			if 'wei' in stats: del stats['wei']
			if 'ighted_words' in stats: del stats['ighted_words']

		# Si weighted_words est vide ou non valide, afficher un message
		if not stats['weighted_words']:
			topic_modeling_logger.warning("weighted_words est vide")
			children.append(html.Div("Aucun mot-clé disponible pour les topics.", className="alert alert-warning"))
			return html.Div(children)

		children.append(html.H5("Top mots par topic", className="mt-4"))

		# Convertir en liste de topics si c'est un dictionnaire
		topics_to_process = stats['weighted_words'].items() if isinstance(stats['weighted_words'], dict) else enumerate(stats['weighted_words'])

		for topic, words in topics_to_process:
			topic_modeling_logger.info(f"Traitement du topic {topic}")
			topic_modeling_logger.info(f"Type de words: {type(words)}, Longueur: {len(words) if hasattr(words, '__len__') else 'N/A'}")
			if words and hasattr(words, '__getitem__'):
				topic_modeling_logger.info(f"Premier élément: {words[0] if words else 'Aucun mot'}")
			else:
				topic_modeling_logger.warning(f"Format non itérable: {words}")
				continue  # Passer au topic suivant si le format n'est pas itérable

			try:
				# Gérer différents formats possibles de données
				topic_modeling_logger.info(f"Format des données: {type(words)}")

				# Si c'est une liste de listes
				if isinstance(words, list) and all(isinstance(item, list) for item in words):
					topic_modeling_logger.info("Format détecté: liste de listes")

					# Vérifier si chaque sous-liste a 2 éléments (mot, poids)
					if all(len(item) == 2 for item in words):
						topic_modeling_logger.info("Format valide: chaque sous-liste a 2 éléments")
						words_df = pd.DataFrame(words, columns=['Mot', 'Poids'])
					else:
						# Essayer de gérer d'autres formats possibles
						topic_modeling_logger.warning("Format inhabituel, tentative d'adaptation...")
						# Si chaque sous-liste est [mot, poids, ...], prendre seulement les 2 premiers éléments
						processed_words = [[item[0], item[1]] if len(item) >= 2 else [item[0], 0] for item in words]
						words_df = pd.DataFrame(processed_words, columns=['Mot', 'Poids'])
				# Si c'est un dictionnaire
				elif isinstance(words, dict):
					topic_modeling_logger.info("Format détecté: dictionnaire")
					words_df = pd.DataFrame(list(words.items()), columns=['Mot', 'Poids'])
				else:
					topic_modeling_logger.warning(f"Format non reconnu: {type(words)}")
					raise ValueError(f"Format de données non pris en charge: {type(words)}")

				topic_modeling_logger.info(f"DataFrame créé: {words_df.shape}")
				topic_modeling_logger.info(f"Types des colonnes: {words_df.dtypes}")
				topic_modeling_logger.info(f"Premières lignes:\n{words_df.head(3)}")

				# Convertir la colonne 'Poids' en valeurs numériques
				# D'abord, s'assurer que tous les poids sont des chaînes ou des nombres
				if words_df['Poids'].dtype == 'object':
					topic_modeling_logger.info("Conversion des poids en chaînes de caractères...")
					words_df['Poids'] = words_df['Poids'].astype(str)

				# Ensuite, convertir en valeurs numériques
				topic_modeling_logger.info("Conversion des poids en valeurs numériques...")
				words_df['Poids'] = pd.to_numeric(words_df['Poids'], errors='coerce')
				topic_modeling_logger.info(f"Après conversion numérique: {words_df.dtypes}")
				topic_modeling_logger.info(f"Valeurs NaN: {words_df['Poids'].isna().sum()}")

				# Supprimer les lignes avec des valeurs NaN
				if words_df['Poids'].isna().any():
					topic_modeling_logger.info("Suppression des lignes avec des valeurs NaN...")
					words_df = words_df.dropna(subset=['Poids'])
					topic_modeling_logger.info(f"Après suppression des NaN: {words_df.shape}")

				# Trier par poids en ordre décroissant
				words_df = words_df.sort_values('Poids', ascending=False).head(10)
				topic_modeling_logger.info(f"Après tri: {words_df.shape}")
				topic_modeling_logger.info(f"Top 3 mots:\n{words_df.head(3)}")

				# Vérifier si le DataFrame est vide
				if words_df.empty:
					topic_modeling_logger.warning("DataFrame vide après traitement")
					raise ValueError("Aucun mot avec un poids valide n'a été trouvé")

				# Utiliser la fonction get_topic_name pour obtenir le nom du topic
				# Garantir que l'ID du topic est un entier en extrayant les chiffres
				topic_id_int = int(re.sub(r"[^\d]", "", str(topic)))
				topic_title = get_topic_name(topic_id_int, model_id)
				print(f"Label du topic: {topic_title}")
				topic_modeling_logger.info(f"Création du graphique avec le titre: {topic_title}")
				
				# Créer le graphique avec le titre personnalisé
				fig = px.bar(words_df, x='Poids', y='Mot', orientation='h', title=topic_title, text_auto='.3f')
				
				# Mettre à jour la mise en page pour rendre le titre plus visible
				fig.update_layout(
					height=350, 
					margin=dict(l=120, r=20, t=60, b=40),  # Augmenter l'espace pour le titre
					yaxis={'categoryorder':'total ascending'},
					title={
						'text': topic_title,
						'font': {'size': 16, 'color': 'black', 'family': 'Arial, sans-serif'},
						'x': 0.5,  # Centrer le titre
						'xanchor': 'center'
					}
				)
				
				children.append(dcc.Graph(figure=fig))
				print(f"Graphique créé pour le topic {topic} avec le titre '{topic_title}'")
				topic_modeling_logger.info(f"Graphique créé pour le topic {topic} avec le titre '{topic_title}'")
				
			except Exception as e:
				print(f"ERREUR lors du traitement du topic {topic}: {e}")
				import traceback
				traceback.print_exc()
				children.append(html.Div(f"Erreur lors de la création du graphique pour le topic {topic}: {str(e)}",
									className="alert alert-danger"))
		print("\n==== FIN DEBUG: Traitement des weighted_words ====\n")

	# 4. Documents représentatifs
	if 'representative_docs' in stats:
		children.append(html.H5("Documents représentatifs par topic", className="mt-4"))
		for topic, doc_ids in stats['representative_docs'].items():
			topic_label = get_topic_label(topic)
			children.append(html.P(f"{topic_label} : {', '.join(str(i) for i in doc_ids)}", className="mb-2"))

	# 5. Noms LLM
	if stats.get('llm_name'):
		children.append(html.P(f"LLM utilisé : {stats['llm_name']}", className="text-muted"))
	if topic_names_llm:
		children.append(render_topic_naming_results(topic_names_llm))

	return html.Div(children)


# To be called in app.py: from src.webapp.topic_modeling_viz import register_topic_modeling_callbacks, get_topic_modeling_layout
