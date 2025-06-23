"""
Script to run the Dash web application.
"""

import os
import sys
from pathlib import Path

# Add the project root to the path to allow imports from other modules
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from src.webapp.app import app

if __name__ == "__main__":
    # Utiliser le port défini par l'environnement (pour Render) ou 8050 par défaut
    port = int(os.environ.get("PORT", 8050))
    # Utiliser 0.0.0.0 pour être accessible depuis l'extérieur
    app.run(host="0.0.0.0", port=port, debug=False)
