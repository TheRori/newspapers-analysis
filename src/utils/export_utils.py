"""
export_utils.py
Module pour sauvegarder et exporter les analyses et visualisations pertinentes.

Ce module permet de :
- Sauvegarder le contexte d'une analyse (paramètres, métadonnées)
- Exporter des graphiques Plotly en HTML ou JSON
- Gérer des collections thématiques d'analyses
- Faciliter la réutilisation des analyses dans une interface de médiation

Exemple d'utilisation:
    from src.utils.export_utils import save_analysis, load_saved_analyses
"""

import os
import json
import datetime
import uuid
from pathlib import Path
from typing import Dict, List, Any, Optional, Union
import plotly.graph_objects as go
import plotly.io as pio
import pandas as pd
import yaml
import shutil

# Configuration du chemin d'exportation
def get_export_path(config: Dict[str, Any]) -> Path:
    """
    Obtient le chemin d'exportation à partir de la configuration.
    
    Args:
        config: Configuration du projet
        
    Returns:
        Chemin vers le répertoire d'exportation
    """
    if 'data' in config and 'results_dir' in config['data']:
        base_dir = Path(config['data']['results_dir'])
    else:
        base_dir = Path('data/results')
    
    export_dir = base_dir / 'exports'
    export_dir.mkdir(exist_ok=True, parents=True)
    
    return export_dir

def save_analysis(
    title: str,
    description: str,
    source_data: Dict[str, Any],
    analysis_type: str,
    figure: Optional[go.Figure] = None,
    additional_metadata: Optional[Dict[str, Any]] = None,
    collection: Optional[str] = None,
    config: Optional[Dict[str, Any]] = None,
    save_source_files: bool = True
) -> Dict[str, Any]:
    """
    Sauvegarde une analyse avec son contexte et éventuellement sa visualisation.
    
    Args:
        title: Titre de l'analyse
        description: Description détaillée de l'analyse
        source_data: Données source (paramètres, filtres, etc.)
        analysis_type: Type d'analyse (term_tracking, topic_modeling, etc.)
        figure: Figure Plotly à sauvegarder (optionnel)
        additional_metadata: Métadonnées supplémentaires (optionnel)
        collection: Nom de la collection thématique (optionnel)
        config: Configuration du projet (optionnel)
    
    Returns:
        Métadonnées de l'analyse sauvegardée
    """
    # Charger la configuration si non fournie
    if config is None:
        from src.utils.config_loader import load_config
        project_root = Path(__file__).parent.parent.parent
        config_path = str(project_root / "config" / "config.yaml")
        config = load_config(config_path)
    
    # Obtenir le chemin d'exportation
    export_dir = get_export_path(config)
    
    # Créer un identifiant unique pour l'analyse
    analysis_id = str(uuid.uuid4())
    timestamp = datetime.datetime.now().isoformat()
    
    # Créer le répertoire de collection si spécifié
    if collection:
        collection_dir = export_dir / 'collections' / collection
        collection_dir.mkdir(exist_ok=True, parents=True)
        analysis_dir = collection_dir / analysis_id
    else:
        analysis_dir = export_dir / analysis_id
    
    # Créer le répertoire pour cette analyse
    analysis_dir.mkdir(exist_ok=True, parents=True)
    
    # Préparer les métadonnées
    metadata = {
        "id": analysis_id,
        "title": title,
        "description": description,
        "analysis_type": analysis_type,
        "created_at": timestamp,
        "source_data": source_data,
        "collection": collection
    }
    
    # Ajouter les métadonnées supplémentaires
    if additional_metadata:
        metadata.update(additional_metadata)
    
    # Sauvegarder la figure si fournie
    if figure:
        # Sauvegarder en HTML pour visualisation directe
        html_path = analysis_dir / "figure.html"
        pio.write_html(figure, file=str(html_path), auto_open=False)
        
        # Sauvegarder en JSON pour réutilisation
        json_path = analysis_dir / "figure.json"
        pio.write_json(figure, file=str(json_path))
        
        # Ajouter les chemins aux métadonnées
        metadata["figure_html"] = str(html_path)
        metadata["figure_json"] = str(json_path)
    
    # Sauvegarder les fichiers source si demandé
    if save_source_files:
        # Créer un sous-répertoire pour les fichiers source
        source_dir = analysis_dir / "source_files"
        source_dir.mkdir(exist_ok=True)
        
        # Sauvegarder les fichiers source mentionnés dans source_data
        saved_files = []
        
        # Fichier de résultats (pour term_tracking, topic_modeling, etc.)
        if "results_file" in source_data and source_data["results_file"]:
            results_file = Path(source_data["results_file"])
            if results_file.exists():
                # Copier le fichier de résultats
                dest_path = source_dir / results_file.name
                shutil.copy2(results_file, dest_path)
                saved_files.append({
                    "type": "results_file",
                    "original_path": str(results_file),
                    "saved_path": str(dest_path)
                })
                
                # Si c'est un fichier CSV ou JSON, le sauvegarder également
                if results_file.suffix.lower() in [".csv", ".json"]:
                    try:
                        if results_file.suffix.lower() == ".csv":
                            df = pd.read_csv(results_file)
                            # Sauvegarder en format parquet pour une meilleure compression
                            parquet_path = source_dir / f"{results_file.stem}.parquet"
                            df.to_parquet(parquet_path)
                            saved_files.append({
                                "type": "results_parquet",
                                "original_path": str(results_file),
                                "saved_path": str(parquet_path)
                            })
                        elif results_file.suffix.lower() == ".json":
                            with open(results_file, 'r', encoding='utf-8') as f:
                                json_data = json.load(f)
                            # Sauvegarder une copie du JSON
                            json_copy_path = source_dir / results_file.name
                            with open(json_copy_path, 'w', encoding='utf-8') as f:
                                json.dump(json_data, f, ensure_ascii=False, indent=2)
                    except Exception as e:
                        print(f"Erreur lors de la sauvegarde du fichier de données: {e}")
        
        # Fichier de termes (pour term_tracking ou semantic_drift)
        if (analysis_type in ["term_tracking", "semantic_drift"]) and "analysis_parameters" in source_data:
            params = source_data["analysis_parameters"]
            if "term_file" in params and params["term_file"]:
                term_file = Path(params["term_file"])
                if term_file.exists():
                    dest_path = source_dir / term_file.name
                    shutil.copy2(term_file, dest_path)
                    saved_files.append({
                        "type": "term_file",
                        "original_path": str(term_file),
                        "saved_path": str(dest_path)
                    })
            
            # Fichier principal des articles (articles.json)
            if "articles_file" in params and params["articles_file"]:
                articles_file = Path(params["articles_file"])
                if articles_file.exists():
                    # Créer un sous-répertoire pour le fichier d'articles (qui peut être volumineux)
                    articles_dir = source_dir / "articles"
                    articles_dir.mkdir(exist_ok=True)
                    
                    # Copier le fichier d'articles
                    dest_path = articles_dir / articles_file.name
                    
                    try:
                        # Pour les fichiers volumineux, utiliser une approche par morceaux
                        # ou créer un lien symbolique si possible
                        if os.name == 'nt':  # Windows
                            # Sur Windows, copier le fichier (peut être lent pour les gros fichiers)
                            shutil.copy2(articles_file, dest_path)
                        else:  # Unix/Linux/Mac
                            # Sur Unix, essayer de créer un lien symbolique
                            try:
                                os.symlink(articles_file, dest_path)
                            except:
                                # Si le lien symbolique échoue, copier le fichier
                                shutil.copy2(articles_file, dest_path)
                        
                        saved_files.append({
                            "type": "articles_file",
                            "original_path": str(articles_file),
                            "saved_path": str(dest_path)
                        })
                        
                        # Créer un fichier d'information sur la source des articles
                        info_path = articles_dir / "articles_info.json"
                        articles_info = {
                            "original_path": str(articles_file),
                            "size_bytes": articles_file.stat().st_size,
                            "last_modified": datetime.datetime.fromtimestamp(articles_file.stat().st_mtime).isoformat(),
                            "saved_as": os.path.basename(dest_path)
                        }
                        
                        # Essayer de compter le nombre d'articles
                        try:
                            with open(articles_file, 'r', encoding='utf-8') as f:
                                # Lire juste assez pour compter les articles sans charger tout le fichier
                                data = json.load(f)
                                if isinstance(data, list):
                                    articles_info["article_count"] = len(data)
                                elif isinstance(data, dict) and "articles" in data:
                                    articles_info["article_count"] = len(data["articles"])
                        except Exception as e:
                            articles_info["count_error"] = str(e)
                        
                        with open(info_path, 'w', encoding='utf-8') as f:
                            json.dump(articles_info, f, ensure_ascii=False, indent=2)
                    
                    except Exception as e:
                        print(f"Erreur lors de la sauvegarde du fichier d'articles: {e}")
                        # Ajouter l'erreur aux métadonnées
                        saved_files.append({
                            "type": "articles_file_error",
                            "original_path": str(articles_file),
                            "error": str(e)
                        })
        
        # Fichier de modèle Word2Vec (pour semantic_drift)
        if analysis_type == "semantic_drift" and "model_path" in source_data:
            model_path = Path(source_data["model_path"])
            if model_path.exists():
                # Pour les modèles Word2Vec, créer un sous-répertoire spécifique
                model_dir = source_dir / "model"
                model_dir.mkdir(exist_ok=True)
                
                # Si c'est un répertoire, copier tout son contenu
                if model_path.is_dir():
                    for item in model_path.glob("*"):
                        if item.is_file():
                            dest_path = model_dir / item.name
                            shutil.copy2(item, dest_path)
                    saved_files.append({
                        "type": "word2vec_model_directory",
                        "original_path": str(model_path),
                        "saved_path": str(model_dir)
                    })
                else:  # Fichier unique
                    dest_path = model_dir / model_path.name
                    shutil.copy2(model_path, dest_path)
                    saved_files.append({
                        "type": "word2vec_model_file",
                        "original_path": str(model_path),
                        "saved_path": str(dest_path)
                    })
        
        # Fichier de modèle (pour topic_modeling)
        if analysis_type == "topic_modeling" and "model_path" in source_data:
            model_path = Path(source_data["model_path"])
            if model_path.exists():
                # Pour les modèles, créer un sous-répertoire spécifique
                model_dir = source_dir / "model"
                model_dir.mkdir(exist_ok=True)
                
                # Si c'est un répertoire, copier tout son contenu
                if model_path.is_dir():
                    for item in model_path.glob("*"):
                        if item.is_file():
                            dest_path = model_dir / item.name
                            shutil.copy2(item, dest_path)
                    saved_files.append({
                        "type": "model_directory",
                        "original_path": str(model_path),
                        "saved_path": str(model_dir)
                    })
                else:  # Fichier unique
                    dest_path = model_dir / model_path.name
                    shutil.copy2(model_path, dest_path)
                    saved_files.append({
                        "type": "model_file",
                        "original_path": str(model_path),
                        "saved_path": str(dest_path)
                    })
        
        # Ajouter les fichiers sauvegardés aux métadonnées
        if saved_files:
            metadata["source_files"] = saved_files
    
    # Sauvegarder les métadonnées
    metadata_path = analysis_dir / "metadata.json"
    with open(metadata_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)
    
    # Ajouter à l'index des analyses
    index_path = export_dir / "analyses_index.json"
    
    if index_path.exists():
        with open(index_path, 'r', encoding='utf-8') as f:
            index = json.load(f)
    else:
        index = {"analyses": []}
    
    # Ajouter l'entrée à l'index
    index_entry = {
        "id": analysis_id,
        "title": title,
        "analysis_type": analysis_type,
        "created_at": timestamp,
        "collection": collection,
        "path": str(analysis_dir)
    }
    
    index["analyses"].append(index_entry)
    
    # Sauvegarder l'index mis à jour
    with open(index_path, 'w', encoding='utf-8') as f:
        json.dump(index, f, ensure_ascii=False, indent=2)
    
    return metadata

def load_saved_analyses(
    config: Optional[Dict[str, Any]] = None,
    collection: Optional[str] = None,
    analysis_type: Optional[str] = None
) -> List[Dict[str, Any]]:
    """
    Charge les analyses sauvegardées, avec filtrage optionnel.
    
    Args:
        config: Configuration du projet (optionnel)
        collection: Filtrer par collection (optionnel)
        analysis_type: Filtrer par type d'analyse (optionnel)
    
    Returns:
        Liste des métadonnées des analyses correspondant aux critères
    """
    # Charger la configuration si non fournie
    if config is None:
        from src.utils.config_loader import load_config
        project_root = Path(__file__).parent.parent.parent
        config_path = str(project_root / "config" / "config.yaml")
        config = load_config(config_path)
    
    # Obtenir le chemin d'exportation
    export_dir = get_export_path(config)
    
    # Vérifier si l'index existe
    index_path = export_dir / "analyses_index.json"
    if not index_path.exists():
        return []
    
    # Charger l'index
    with open(index_path, 'r', encoding='utf-8') as f:
        index = json.load(f)
    
    # Filtrer les analyses
    analyses = index.get("analyses", [])
    
    if collection:
        analyses = [a for a in analyses if a.get("collection") == collection]
    
    if analysis_type:
        analyses = [a for a in analyses if a.get("analysis_type") == analysis_type]
    
    # Trier par date de création (plus récent en premier)
    analyses.sort(key=lambda x: x.get("created_at", ""), reverse=True)
    
    return analyses

def get_analysis_details(analysis_id: str, config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Obtient les détails complets d'une analyse sauvegardée.
    
    Args:
        analysis_id: Identifiant de l'analyse
        config: Configuration du projet (optionnel)
    
    Returns:
        Métadonnées complètes de l'analyse
    """
    # Charger la configuration si non fournie
    if config is None:
        from src.utils.config_loader import load_config
        project_root = Path(__file__).parent.parent.parent
        config_path = str(project_root / "config" / "config.yaml")
        config = load_config(config_path)
    
    # Obtenir le chemin d'exportation
    export_dir = get_export_path(config)
    
    # Charger l'index pour trouver le chemin
    index_path = export_dir / "analyses_index.json"
    if not index_path.exists():
        raise FileNotFoundError(f"Index des analyses non trouvé: {index_path}")
    
    with open(index_path, 'r', encoding='utf-8') as f:
        index = json.load(f)
    
    # Trouver l'analyse dans l'index
    analysis_entry = next((a for a in index.get("analyses", []) if a.get("id") == analysis_id), None)
    
    if not analysis_entry:
        raise ValueError(f"Analyse non trouvée avec l'ID: {analysis_id}")
    
    # Charger les métadonnées complètes
    metadata_path = Path(analysis_entry["path"]) / "metadata.json"
    
    if not metadata_path.exists():
        raise FileNotFoundError(f"Métadonnées non trouvées: {metadata_path}")
    
    with open(metadata_path, 'r', encoding='utf-8') as f:
        metadata = json.load(f)
    
    return metadata

def load_figure(analysis_id: str, config: Optional[Dict[str, Any]] = None) -> go.Figure:
    """
    Charge la figure Plotly d'une analyse sauvegardée.
    
    Args:
        analysis_id: Identifiant de l'analyse
        config: Configuration du projet (optionnel)
    
    Returns:
        Figure Plotly
    """
    # Obtenir les détails de l'analyse
    metadata = get_analysis_details(analysis_id, config)
    
    # Vérifier si une figure est disponible
    if "figure_json" not in metadata:
        raise ValueError(f"Aucune figure disponible pour l'analyse: {analysis_id}")
    
    # Charger la figure depuis le JSON
    figure_path = metadata["figure_json"]
    
    if not Path(figure_path).exists():
        raise FileNotFoundError(f"Fichier de figure non trouvé: {figure_path}")
    
    figure = pio.read_json(figure_path)
    
    return figure

def create_collection(
    name: str,
    description: str,
    config: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Crée une nouvelle collection thématique.
    
    Args:
        name: Nom de la collection
        description: Description de la collection
        config: Configuration du projet (optionnel)
    
    Returns:
        Métadonnées de la collection
    """
    # Charger la configuration si non fournie
    if config is None:
        from src.utils.config_loader import load_config
        project_root = Path(__file__).parent.parent.parent
        config_path = str(project_root / "config" / "config.yaml")
        config = load_config(config_path)
    
    # Obtenir le chemin d'exportation
    export_dir = get_export_path(config)
    
    # Créer le répertoire de collections
    collections_dir = export_dir / 'collections'
    collections_dir.mkdir(exist_ok=True, parents=True)
    
    # Créer le répertoire pour cette collection
    collection_dir = collections_dir / name
    collection_dir.mkdir(exist_ok=True)
    
    # Préparer les métadonnées
    metadata = {
        "name": name,
        "description": description,
        "created_at": datetime.datetime.now().isoformat(),
        "analyses_count": 0
    }
    
    # Sauvegarder les métadonnées
    metadata_path = collection_dir / "collection_info.json"
    with open(metadata_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)
    
    # Ajouter à l'index des collections
    index_path = collections_dir / "collections_index.json"
    
    if index_path.exists():
        with open(index_path, 'r', encoding='utf-8') as f:
            index = json.load(f)
    else:
        index = {"collections": []}
    
    # Ajouter l'entrée à l'index
    index_entry = {
        "name": name,
        "description": description,
        "created_at": metadata["created_at"],
        "path": str(collection_dir)
    }
    
    # Vérifier si la collection existe déjà
    existing = next((c for c in index["collections"] if c["name"] == name), None)
    if existing:
        # Mettre à jour la collection existante
        index["collections"].remove(existing)
    
    index["collections"].append(index_entry)
    
    # Sauvegarder l'index mis à jour
    with open(index_path, 'w', encoding='utf-8') as f:
        json.dump(index, f, ensure_ascii=False, indent=2)
    
    return metadata

def get_collections(config: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
    """
    Obtient la liste des collections disponibles.
    
    Args:
        config: Configuration du projet (optionnel)
    
    Returns:
        Liste des métadonnées des collections
    """
    # Charger la configuration si non fournie
    if config is None:
        from src.utils.config_loader import load_config
        project_root = Path(__file__).parent.parent.parent
        config_path = str(project_root / "config" / "config.yaml")
        config = load_config(config_path)
    
    # Obtenir le chemin d'exportation
    export_dir = get_export_path(config)
    
    # Vérifier si l'index des collections existe
    collections_dir = export_dir / 'collections'
    index_path = collections_dir / "collections_index.json"
    
    if not collections_dir.exists() or not index_path.exists():
        return []
    
    # Charger l'index
    with open(index_path, 'r', encoding='utf-8') as f:
        index = json.load(f)
    
    # Obtenir les collections
    collections = index.get("collections", [])
    
    # Pour chaque collection, compter le nombre d'analyses
    for collection in collections:
        collection_dir = Path(collection["path"])
        analyses_count = 0
        
        # Compter les répertoires d'analyses (chaque analyse a son propre répertoire)
        for item in collection_dir.iterdir():
            if item.is_dir() and item.name != "__pycache__":
                # Vérifier si c'est un répertoire d'analyse (contient metadata.json)
                if (item / "metadata.json").exists():
                    analyses_count += 1
        
        collection["analyses_count"] = analyses_count
    
    # Trier par date de création (plus récent en premier)
    collections.sort(key=lambda x: x.get("created_at", ""), reverse=True)
    
    return collections

def export_collection_for_mediation(
    collection_name: str,
    output_dir: Optional[str] = None,
    config: Optional[Dict[str, Any]] = None
) -> str:
    """
    Exporte une collection complète pour la médiation.
    
    Args:
        collection_name: Nom de la collection à exporter
        output_dir: Répertoire de sortie (optionnel)
        config: Configuration du projet (optionnel)
    
    Returns:
        Chemin vers le répertoire d'exportation
    """
    # Charger la configuration si non fournie
    if config is None:
        from src.utils.config_loader import load_config
        project_root = Path(__file__).parent.parent.parent
        config_path = str(project_root / "config" / "config.yaml")
        config = load_config(config_path)
    
    # Obtenir le chemin d'exportation
    export_dir = get_export_path(config)
    collections_dir = export_dir / 'collections'
    
    # Vérifier si la collection existe
    collection_dir = collections_dir / collection_name
    if not collection_dir.exists():
        raise ValueError(f"Collection non trouvée: {collection_name}")
    
    # Déterminer le répertoire de sortie
    if output_dir:
        mediation_dir = Path(output_dir)
    else:
        mediation_dir = export_dir / 'mediation' / collection_name
    
    mediation_dir.mkdir(exist_ok=True, parents=True)
    
    # Charger les analyses de cette collection
    analyses = load_saved_analyses(config, collection=collection_name)
    
    # Préparer les données pour la médiation
    mediation_data = {
        "collection": collection_name,
        "exported_at": datetime.datetime.now().isoformat(),
        "analyses": []
    }
    
    # Traiter chaque analyse
    for analysis in analyses:
        # Obtenir les détails complets
        analysis_id = analysis["id"]
        details = get_analysis_details(analysis_id, config)
        
        # Créer un répertoire pour cette analyse
        analysis_export_dir = mediation_dir / analysis_id
        analysis_export_dir.mkdir(exist_ok=True)
        
        # Copier la figure HTML si disponible
        if "figure_html" in details and Path(details["figure_html"]).exists():
            html_dest = analysis_export_dir / "figure.html"
            shutil.copy2(details["figure_html"], html_dest)
            details["mediation_figure_html"] = str(html_dest.relative_to(mediation_dir))
        
        # Ajouter à la liste des analyses
        mediation_data["analyses"].append(details)
    
    # Sauvegarder les données de médiation
    mediation_path = mediation_dir / "mediation_data.json"
    with open(mediation_path, 'w', encoding='utf-8') as f:
        json.dump(mediation_data, f, ensure_ascii=False, indent=2)
    
    return str(mediation_dir)

def delete_analysis(analysis_id: str, config: Optional[Dict[str, Any]] = None) -> bool:
    """
    Supprime une analyse sauvegardée.
    
    Args:
        analysis_id: Identifiant de l'analyse
        config: Configuration du projet (optionnel)
    
    Returns:
        True si la suppression a réussi, False sinon
    """
    # Charger la configuration si non fournie
    if config is None:
        from src.utils.config_loader import load_config
        project_root = Path(__file__).parent.parent.parent
        config_path = str(project_root / "config" / "config.yaml")
        config = load_config(config_path)
    
    # Obtenir le chemin d'exportation
    export_dir = get_export_path(config)
    
    # Charger l'index pour trouver le chemin
    index_path = export_dir / "analyses_index.json"
    if not index_path.exists():
        return False
    
    with open(index_path, 'r', encoding='utf-8') as f:
        index = json.load(f)
    
    # Trouver l'analyse dans l'index
    analysis_entry = next((a for a in index.get("analyses", []) if a.get("id") == analysis_id), None)
    
    if not analysis_entry:
        return False
    
    # Supprimer le répertoire de l'analyse
    analysis_dir = Path(analysis_entry["path"])
    if analysis_dir.exists():
        shutil.rmtree(analysis_dir)
    
    # Mettre à jour l'index
    index["analyses"] = [a for a in index["analyses"] if a.get("id") != analysis_id]
    
    # Sauvegarder l'index mis à jour
    with open(index_path, 'w', encoding='utf-8') as f:
        json.dump(index, f, ensure_ascii=False, indent=2)
    
    return True
