"""
Script pour corriger directement le fichier sentiment_analysis_viz.py
"""

import os

# Chemin du fichier à corriger
file_path = "src/webapp/sentiment_analysis_viz.py"

# Lire le contenu du fichier
with open(file_path, 'r', encoding='utf-8') as f:
    content = f.read()

# Corriger la ligne problématique
# Remplacer la ligne qui contient l'accolade en trop ou manquante
corrected_content = content.replace(
    "'value': f\"{str(f)}?t={current_time}\"  # Ajouter un paramètre pour éviter le cache}",
    "'value': f\"{str(f)}?t={current_time}\"  # Ajouter un paramètre pour éviter le cache"
)

# Vérifier si une correction a été effectuée
if content == corrected_content:
    print("Aucune correction n'a été effectuée. Essayons une autre approche.")
    
    # Reconstruire la section problématique
    options_section = """    # Format for dropdown
    options = [
        {'label': f"{f.stem} ({pd.to_datetime(os.stat(f).st_mtime, unit='s').strftime('%Y-%m-%d %H:%M')})", 
         'value': f"{str(f)}?t={current_time}"  # Ajouter un paramètre pour éviter le cache}
        for f in summary_files
    ]"""
    
    corrected_options_section = """    # Format for dropdown
    options = [
        {'label': f"{f.stem} ({pd.to_datetime(os.stat(f).st_mtime, unit='s').strftime('%Y-%m-%d %H:%M')})", 
         'value': f"{str(f)}?t={current_time}"  # Ajouter un paramètre pour éviter le cache
        }
        for f in summary_files
    ]"""
    
    corrected_content = content.replace(options_section, corrected_options_section)

# Écrire le contenu corrigé dans le fichier
with open(file_path, 'w', encoding='utf-8') as f:
    f.write(corrected_content)

print("Correction terminée. Vérifiez le fichier pour vous assurer que la correction a été appliquée correctement.")
