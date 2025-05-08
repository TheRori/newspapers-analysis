#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script de test pour la conversion des fichiers JSON en Parquet.
Ce script permet de tester la conversion d'un fichier JSON d'articles en format Parquet
et d'afficher des logs détaillés dans le terminal.
"""

import os
import sys
import json
import traceback
from pathlib import Path
import pandas as pd

# Essayer d'importer PyArrow
try:
    import pyarrow as pa
    import pyarrow.parquet as pq
    import pyarrow.json as pj
    PYARROW_AVAILABLE = True
except ImportError:
    PYARROW_AVAILABLE = False
    print("PyArrow n'est pas disponible. Installation recommandée: pip install pyarrow")

# Essayer d'importer ijson
try:
    import ijson
    IJSON_AVAILABLE = True
except ImportError:
    IJSON_AVAILABLE = False
    print("ijson n'est pas disponible. Installation recommandée: pip install ijson")

def log_info(message):
    """Affiche un message d'information dans le terminal."""
    print(f"[INFO] {message}", flush=True)
    sys.stdout.flush()

def log_success(message):
    """Affiche un message de succès dans le terminal."""
    print(f"[SUCCÈS] {message}", flush=True)
    sys.stdout.flush()

def log_warning(message):
    """Affiche un message d'avertissement dans le terminal."""
    print(f"[ATTENTION] {message}", flush=True)
    sys.stdout.flush()

def log_error(message):
    """Affiche un message d'erreur dans le terminal."""
    print(f"[ERREUR] {message}", flush=True)
    sys.stderr.flush()

def convert_json_to_parquet(json_file, parquet_file=None, chunk_size=1000):
    """
    Convertit un fichier JSON en format Parquet.
    
    Args:
        json_file: Chemin vers le fichier JSON à convertir
        parquet_file: Chemin vers le fichier Parquet de sortie (optionnel)
        chunk_size: Taille des morceaux pour la lecture par morceaux (optionnel)
    
    Returns:
        Chemin vers le fichier Parquet créé
    """
    json_file = Path(json_file)
    
    if not json_file.exists():
        log_error(f"Le fichier JSON n'existe pas: {json_file}")
        return None
    
    if parquet_file is None:
        parquet_file = json_file.with_suffix('.parquet')
    else:
        parquet_file = Path(parquet_file)
    
    log_info(f"Début de la conversion du fichier JSON: {json_file}")
    log_info(f"Fichier Parquet de sortie: {parquet_file}")
    
    # Obtenir la taille du fichier pour l'afficher
    file_size_mb = json_file.stat().st_size / (1024 * 1024)
    log_info(f"Taille du fichier JSON: {file_size_mb:.2f} MB")
    
    try:
        # Utiliser PyArrow si disponible
        if PYARROW_AVAILABLE:
            log_info("Utilisation de PyArrow pour la conversion (méthode optimisée)")
            
            # Lire le fichier JSON avec PyArrow
            log_info("Lecture du fichier JSON avec PyArrow...")
            table = pj.read_json(str(json_file))
            log_info(f"Fichier JSON chargé avec succès. Nombre de colonnes: {len(table.column_names)}")
            log_info(f"Nombre d'articles: {table.num_rows}")
            
            # Écrire le fichier Parquet
            log_info("Conversion en Parquet...")
            pq.write_table(table, str(parquet_file))
            
            # Vérifier la taille du fichier Parquet pour comparer avec le JSON
            parquet_size_mb = parquet_file.stat().st_size / (1024 * 1024)
            compression_ratio = file_size_mb / parquet_size_mb if parquet_size_mb > 0 else 0
            log_success(f"Fichier Parquet créé: {parquet_file}")
            log_success(f"Taille du fichier Parquet: {parquet_size_mb:.2f} MB (compression {compression_ratio:.2f}x)")
            
            return parquet_file
        
        # Sinon, utiliser pandas avec une approche par morceaux
        log_warning("PyArrow non disponible, utilisation de pandas avec approche par morceaux")
        
        # Déterminer la structure du fichier JSON
        with open(json_file, 'r', encoding='utf-8') as f:
            log_info("Analyse de la structure du fichier JSON...")
            
            if IJSON_AVAILABLE:
                # Utiliser ijson pour un parsing efficace
                parser = ijson.parse(f)
                # Vérifier si c'est un tableau ou un objet
                for prefix, event, value in parser:
                    if prefix == '' and event == 'start_array':
                        is_array = True
                        log_info("Structure détectée: Tableau JSON")
                        break
                    elif prefix == '' and event == 'start_map':
                        is_array = False
                        log_info("Structure détectée: Objet JSON")
                        break
            else:
                # Si ijson n'est pas disponible, utiliser json standard
                log_warning("Module ijson non disponible, utilisation de la méthode standard")
                f.seek(0)
                first_char = f.read(1).strip()
                is_array = first_char == '['
                log_info(f"Structure détectée: {'Tableau' if is_array else 'Objet'} JSON")
        
        if is_array:
            # Lire par morceaux
            log_info(f"Lecture par morceaux de {chunk_size} articles")
            
            # Compter le nombre total d'articles pour afficher la progression
            with open(json_file, 'r', encoding='utf-8') as f:
                try:
                    # Essayer de compter le nombre d'articles sans charger tout le fichier
                    data = json.load(f)
                    total_articles = len(data)
                    log_info(f"Nombre total d'articles: {total_articles}")
                except:
                    log_warning("Impossible de compter le nombre total d'articles")
                    total_articles = None
            
            # Lire par morceaux
            for i, chunk in enumerate(pd.read_json(json_file, lines=False, chunksize=chunk_size)):
                if i == 0:
                    # Premier chunk, écrire avec mode='w'
                    log_info(f"Traitement du premier morceau ({chunk.shape[0]} articles)")
                    chunk.to_parquet(parquet_file, index=False)
                else:
                    # Chunks suivants, ajouter au fichier existant
                    log_info(f"Traitement du morceau {i+1} ({chunk.shape[0]} articles)")
                    chunk.to_parquet(parquet_file, index=False, mode='a')
                
                # Afficher la progression
                articles_processed = (i+1) * chunk_size
                if total_articles:
                    progress = min(100, articles_processed / total_articles * 100)
                    log_info(f"Progrès: {articles_processed}/{total_articles} articles traités ({progress:.1f}%)")
                else:
                    log_info(f"Progrès: environ {articles_processed} articles traités")
        else:
            # Si c'est un objet JSON, le convertir en DataFrame
            log_info("Conversion directe de l'objet JSON en DataFrame")
            df = pd.read_json(json_file)
            log_info(f"Fichier JSON chargé avec succès. Dimensions: {df.shape}")
            log_info("Conversion en Parquet...")
            df.to_parquet(parquet_file, index=False)
        
        # Vérifier la taille du fichier Parquet pour comparer avec le JSON
        parquet_size_mb = parquet_file.stat().st_size / (1024 * 1024)
        compression_ratio = file_size_mb / parquet_size_mb if parquet_size_mb > 0 else 0
        log_success(f"Fichier Parquet créé: {parquet_file}")
        log_success(f"Taille du fichier Parquet: {parquet_size_mb:.2f} MB (compression {compression_ratio:.2f}x)")
        
        return parquet_file
    
    except Exception as e:
        log_error(f"Erreur lors de la conversion en Parquet: {e}")
        log_error(f"Détails de l'erreur:\n{traceback.format_exc()}")
        return None

def main():
    """Fonction principale du script."""
    # Vérifier les arguments
    if len(sys.argv) < 2:
        print("Usage: python test_parquet_conversion.py <chemin_vers_fichier_json> [chemin_vers_fichier_parquet]")
        print("Exemple: python test_parquet_conversion.py data/processed/articles.json")
        return
    
    # Récupérer les arguments
    json_file = sys.argv[1]
    parquet_file = sys.argv[2] if len(sys.argv) > 2 else None
    
    # Convertir le fichier JSON en Parquet
    convert_json_to_parquet(json_file, parquet_file)

if __name__ == "__main__":
    main()
