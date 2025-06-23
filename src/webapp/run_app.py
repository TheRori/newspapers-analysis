"""
Script to run the Dash web application with ultra-light boot approach.
"""

import os
import sys
import threading
import time
from pathlib import Path

# Add the project root to the path to allow imports from other modules
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

# Import the lightweight app shell first
from src.webapp.app_shell import create_app_shell

# Create the lightweight app shell that will show a loading screen
app = create_app_shell()

# Function to load heavy components in background after server starts
def load_heavy_components():
    # Wait a moment for the server to fully start
    time.sleep(2)
    print("Starting to load heavy components...")
    # Import the full app initialization (this will register all callbacks)
    from src.webapp.app_initializer import initialize_app
    # Initialize the app with all components
    initialize_app(app)
    print("Heavy components loaded successfully!")

if __name__ == "__main__":
    # Start the background loading thread
    loading_thread = threading.Thread(target=load_heavy_components)
    loading_thread.daemon = True
    loading_thread.start()
    # Utiliser le port défini par l'environnement (pour Render) ou 8050 par défaut
    port = int(os.environ.get("PORT", 8050))
    print(f"Starting server on port {port}...")
    # Utiliser 0.0.0.0 pour être accessible depuis l'extérieur
    app.run(host="0.0.0.0", port=port, debug=False)
