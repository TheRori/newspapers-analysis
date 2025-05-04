import sys
import os
from pathlib import Path

# Afficher le chemin Python
print("Python path:")
for p in sys.path:
    print(f"  {p}")

print("\nTentative d'importation de dash:")
try:
    import dash
    print(f"Dash importé avec succès. Version: {dash.__version__}")
except ImportError as e:
    print(f"Erreur d'importation de dash: {e}")

print("\nTentative d'importation depuis src:")
try:
    from src.utils.config_loader import load_config
    print("Module src.utils.config_loader importé avec succès")
except ImportError as e:
    print(f"Erreur d'importation de src.utils.config_loader: {e}")

# Vérifier si le répertoire src est dans le chemin
project_root = Path(__file__).parent
sys.path.append(str(project_root))
print(f"\nAjout de {project_root} au chemin Python")

print("\nNouvelle tentative d'importation depuis src:")
try:
    from src.utils.config_loader import load_config
    print("Module src.utils.config_loader importé avec succès")
except ImportError as e:
    print(f"Erreur d'importation de src.utils.config_loader: {e}")
