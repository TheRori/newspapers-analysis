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
    config: Optional[Dict[str, Any]] = None
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
