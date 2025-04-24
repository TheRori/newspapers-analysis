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
    app.run(debug=True, port=8050)
