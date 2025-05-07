"""
Script pour mettre à jour la détection automatique des fichiers de résultats
dans les modules d'analyse de sentiment et de reconnaissance d'entités.
"""

import os
import re
import fileinput

def update_get_results_function(file_path, function_name, result_type):
    """
    Met à jour la fonction de récupération des résultats pour ajouter un paramètre de cache-busting.
    
    Args:
        file_path: Chemin vers le fichier à mettre à jour
        function_name: Nom de la fonction à mettre à jour
        result_type: Type de résultat (sentiment_analysis ou entity_recognition)
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Rechercher la fonction
    function_pattern = rf"def {function_name}\(\):"
    function_match = re.search(function_pattern, content)
    
    if not function_match:
        print(f"Fonction {function_name} non trouvée dans {file_path}")
        return False
    
    # Mettre à jour la fonction pour utiliser os.stat au lieu de os.path.getmtime
    updated_content = re.sub(
        r"summary_files\.sort\(key=lambda x: os\.path\.getmtime\(x\), reverse=True\)",
        "summary_files.sort(key=lambda x: os.stat(x).st_mtime, reverse=True)",
        content
    )
    
    # Ajouter l'import de time_module si nécessaire
    if "import time as time_module" not in updated_content:
        updated_content = re.sub(
            r"# Sort by modification time \(newest first\)",
            "# Sort by modification time (newest first)\n    # Utiliser os.stat au lieu de os.path.getmtime pour forcer la mise à jour des horodatages\n    import time as time_module\n    current_time = int(time_module.time())  # Timestamp actuel pour éviter les problèmes de cache",
            updated_content
        )
    
    # Mettre à jour la création des options pour ajouter le paramètre de cache-busting
    updated_content = re.sub(
        r"'value': str\(f\)",
        "'value': f\"{str(f)}?t={current_time}\"  # Ajouter un paramètre pour éviter le cache",
        updated_content
    )
    
    # Mettre à jour l'utilisation de os.path.getmtime dans le label
    updated_content = re.sub(
        r"pd\.to_datetime\(os\.path\.getmtime\(f\), unit='s'\)",
        "pd.to_datetime(os.stat(f).st_mtime, unit='s')",
        updated_content
    )
    
    # Écrire le contenu mis à jour dans le fichier
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(updated_content)
    
    print(f"Fonction {function_name} mise à jour dans {file_path}")
    return True

def update_run_analysis_callback(file_path, callback_name, results_function_name):
    """
    Met à jour le callback de lancement d'analyse pour récupérer la liste mise à jour des fichiers de résultats.
    
    Args:
        file_path: Chemin vers le fichier à mettre à jour
        callback_name: Nom du callback à mettre à jour
        results_function_name: Nom de la fonction de récupération des résultats
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Rechercher le callback
    callback_pattern = rf"def {callback_name}\("
    callback_match = re.search(callback_pattern, content)
    
    if not callback_match:
        print(f"Callback {callback_name} non trouvé dans {file_path}")
        return False
    
    # Trouver la fin du bloc try dans le callback
    try_end_pattern = r"return html\.Div\(\[\s+html\.P\(\".*terminée avec succès.*\"\),"
    try_end_match = re.search(try_end_pattern, content)
    
    if not try_end_match:
        print(f"Bloc try non trouvé dans le callback {callback_name}")
        return False
    
    # Ajouter la récupération de la liste mise à jour des fichiers de résultats
    updated_content = content[:try_end_match.start()] + \
        f"            # Récupérer la liste mise à jour des résultats disponibles\n" + \
        f"            updated_results = {results_function_name}()\n\n" + \
        content[try_end_match.start():]
    
    # Écrire le contenu mis à jour dans le fichier
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(updated_content)
    
    print(f"Callback {callback_name} mis à jour dans {file_path}")
    return True

def update_display_results_callback(file_path, callback_name, dropdown_id):
    """
    Met à jour le callback d'affichage des résultats pour extraire le chemin du fichier du paramètre de cache-busting.
    
    Args:
        file_path: Chemin vers le fichier à mettre à jour
        callback_name: Nom du callback à mettre à jour
        dropdown_id: ID du dropdown contenant les fichiers de résultats
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Rechercher le callback
    callback_pattern = rf"def {callback_name}\("
    callback_match = re.search(callback_pattern, content)
    
    if not callback_match:
        print(f"Callback {callback_name} non trouvé dans {file_path}")
        return False
    
    # Trouver le début du corps du callback
    callback_body_pattern = rf"def {callback_name}\([^)]+\):\s+"
    callback_body_match = re.search(callback_body_pattern, content)
    
    if not callback_body_match:
        print(f"Corps du callback {callback_name} non trouvé")
        return False
    
    # Ajouter le code pour extraire le chemin du fichier du paramètre de cache-busting
    updated_content = content[:callback_body_match.end()] + \
        f"    # Extraire le chemin du fichier du paramètre de cache-busting\n" + \
        f"    if results_file and '?' in results_file:\n" + \
        f"        results_file = results_file.split('?')[0]\n\n" + \
        content[callback_body_match.end():]
    
    # Écrire le contenu mis à jour dans le fichier
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(updated_content)
    
    print(f"Callback {callback_name} mis à jour dans {file_path}")
    return True

def main():
    # Mettre à jour le module d'analyse de sentiment
    sentiment_viz_path = "src/webapp/sentiment_analysis_viz.py"
    update_get_results_function(sentiment_viz_path, "get_sentiment_results", "sentiment_analysis")
    update_run_analysis_callback(sentiment_viz_path, "run_sentiment_analysis", "get_sentiment_results")
    update_display_results_callback(sentiment_viz_path, "display_sentiment_results", "sentiment-results-dropdown")
    
    # Mettre à jour le module de reconnaissance d'entités
    entity_viz_path = "src/webapp/entity_recognition_viz.py"
    update_get_results_function(entity_viz_path, "get_entity_results", "entity_recognition")
    update_run_analysis_callback(entity_viz_path, "run_entity_recognition", "get_entity_results")
    update_display_results_callback(entity_viz_path, "display_entity_results", "entity-results-dropdown")
    
    print("Mise à jour terminée.")

if __name__ == "__main__":
    main()
