#!/bin/bash
set -e

echo "Lancement de l'application Dash..."
export PYTHONUNBUFFERED=1
PORT=${PORT:-8050}
python src/webapp/run_app.py
