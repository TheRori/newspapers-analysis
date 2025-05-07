"""
Script pour corriger l'erreur de syntaxe dans le fichier sentiment_analysis_viz.py
"""

import os

# Chemin du fichier à corriger
file_path = "src/webapp/sentiment_analysis_viz.py"

# Lire le contenu du fichier
with open(file_path, 'r', encoding='utf-8') as f:
    lines = f.readlines()

# Corriger la ligne problématique (ligne 92)
# Remplacer la ligne qui contient l'accolade en trop
for i, line in enumerate(lines):
    if "'value': f\"{str(f)}?t={current_time}\"  # Ajouter un paramètre pour éviter le cache}" in line:
        lines[i] = line.replace("cache}", "cache")
        print(f"Ligne corrigée: {i+1}")

# Écrire le contenu corrigé dans le fichier
with open(file_path, 'w', encoding='utf-8') as f:
    f.writelines(lines)

print("Correction terminée. Vérifiez le fichier pour vous assurer que la correction a été appliquée correctement.")
