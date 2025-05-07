import json
import os
from pathlib import Path

# Chemin vers le fichier JSON à réparer
file_path = Path("data/processed/articles.json")

try:
    # Essayer de charger le fichier JSON pour voir s'il est valide
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    print("Le fichier JSON est valide.")
except json.JSONDecodeError as e:
    print(f"Erreur JSON détectée: {e}")
    
    # Lire le fichier comme texte
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Obtenir les informations sur l'erreur
    error_line = e.lineno
    error_col = e.colno
    error_pos = e.pos
    
    # Afficher le contexte de l'erreur
    start = max(0, error_pos - 50)
    end = min(len(content), error_pos + 50)
    context = content[start:end]
    
    print(f"\nContexte de l'erreur (environ 100 caractères autour de la position {error_pos}):")
    print(context)
    print(" " * (error_pos - start) + "^")  # Pointeur vers l'erreur
    
    # Essayer de corriger automatiquement les erreurs courantes
    # 1. Remplacer les simples quotes par des doubles quotes pour les clés
    # 2. Ajouter des virgules manquantes
    # 3. Supprimer les virgules en trop
    
    # Créer une version corrigée du fichier
    corrected_content = content
    
    # Remplacer les simples quotes par des doubles quotes pour les clés
    # Attention: cela pourrait ne pas fonctionner pour tous les cas
    # corrected_content = corrected_content.replace("'", '"')
    
    # Écrire le contenu corrigé dans un nouveau fichier
    backup_path = file_path.with_suffix('.json.bak')
    with open(backup_path, 'w', encoding='utf-8') as f:
        f.write(content)
    
    print(f"\nUne sauvegarde du fichier original a été créée à {backup_path}")
    print("Veuillez éditer manuellement le fichier JSON pour corriger l'erreur.")
    print("Vous pouvez utiliser un éditeur JSON en ligne ou un IDE avec validation JSON.")
