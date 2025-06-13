#!/bin/bash

# Bash script to launch the app and display logs live

set -e  # Exit on error

echo -e "\033[36mActivation de l'environnement virtuel...\033[0m"
if [ -f .venv/bin/activate ]; then
    source .venv/bin/activate
    echo -e "\033[32mEnvironnement virtuel activé avec succès.\033[0m"
else
    echo -e "\033[31mErreur : impossible de trouver .venv/bin/activate\033[0m"
    exit 1
fi

echo -e "\033[36mVérification des modules nécessaires...\033[0m"
missing_modules=()
for module in dash plotly pandas numpy; do
    if python -c "import $module" 2>/dev/null; then
        version=$(python -c "import $module; print(getattr($module, '__version__', ''))")
        echo -e "\033[32mModule $module disponible, version: $version\033[0m"
    else
        echo -e "\033[31mModule $module manquant!\033[0m"
        missing_modules+=($module)
    fi
done

if [ ${#missing_modules[@]} -gt 0 ]; then
    echo -e "\033[33mModules manquants détectés. Installation en cours...\033[0m"
    for module in "${missing_modules[@]}"; do
        echo -e "\033[33mInstallation de $module...\033[0m"
        pip install "$module"
    done
fi

echo -e "\033[36mLancement de l'application...\033[0m"
export PYTHONUNBUFFERED=1
echo -e "\033[32mApplication démarrée. Accédez à http://127.0.0.1:8050/ dans votre navigateur.\033[0m"
echo -e "\033[33mAppuyez sur Ctrl+C pour arrêter l'application.\033[0m"

python src/webapp/run_app.py
