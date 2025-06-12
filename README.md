# Analyse d'Articles de Presse - Guide d'Installation et Fonctionnalités

## Présentation
Cette application permet d'explorer et d'analyser un corpus d'articles de presse numérisés via OCR. Elle offre une suite complète d'outils d'analyse textuelle et de visualisation interactive pour extraire des informations pertinentes à partir de données textuelles.

L'application est composée de deux volets principaux :
- **Application d'analyse** : interface web interactive pour l'exploration des corpus
- **Application de médiation** : interface de restitution des résultats pour un public non-technique

## Installation

### Prérequis
- Python 3.11 ou supérieur
- Git
- PowerShell (pour Windows) ou Terminal (pour Linux/Mac)

### Étapes d'installation

1. **Cloner le dépôt**
   ```bash
   git clone https://github.com/TheRori/newspapers-analysis.git
   cd newspapers-analysis
   ```

2. **Créer et activer un environnement virtuel**
   
   Sous Windows :
   ```powershell
   python -m venv .venv
   .\.venv\Scripts\activate
   ```
   
   Sous Linux/Mac :
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   ```

3. **Installer les dépendances**
   ```bash
   pip install -r requirements.txt
   ```

4. **Installer le modèle spaCy**
   ```bash
   python -m spacy download fr_core_news_md
   ```

5. **Configuration**
   - Vérifier et modifier si nécessaire le fichier `config/config.yaml` :
     - Chemins vers les données
     - Paramètres d'analyse
     - Options de visualisation

6. **Placer les données**
   - Par défaut, les données doivent être placées dans `data/processed/`

## Lancement de l'application

### Méthode recommandée (Windows)
Utiliser le script PowerShell fourni qui vérifie automatiquement l'environnement et lance l'application :
```powershell
.\run_app_direct.ps1
```

Ce script :
- Active l'environnement virtuel
- Vérifie que les modules nécessaires sont installés
- Lance l'application avec affichage des logs en direct

### Méthode alternative (toutes plateformes)
```bash
# Activer l'environnement virtuel (si ce n'est pas déjà fait)
# Windows:
.\.venv\Scripts\activate
# Linux/Mac:
# source .venv/bin/activate

# Lancer l'application
python src/webapp/run_app.py
```

L'application sera accessible via : **http://127.0.0.1:8050/**

## Principales fonctionnalités

### 1. Analyse lexicale
- Fréquences de mots et nuages de mots
- N-grammes (bi-/trigrammes)
- Termes significatifs

### 2. Modélisation thématique (LDA)
- Identification des thèmes principaux et mots-clés
- Distribution et évolution temporelle des thèmes
- Optimisation automatique du nombre de thèmes

### 3. Clustering d'articles
- Regroupement via K-means
- Visualisation 2D des clusters
- Analyse des caractéristiques par cluster

### 4. Analyse de sentiment
- Polarité des articles (positif, négatif, neutre)
- Évolution temporelle et comparaison entre sources

### 5. Reconnaissance d'entités nommées
- Extraction des personnes, lieux, organisations, dates
- Analyse des cooccurrences et suivi temporel

### 6. Suivi de termes
- Évolution temporelle des termes spécifiques
- Comparaison entre différentes sources
- Visualisations dynamiques (courbes, flux)
- Liens directs vers les articles

### 7. Application de médiation
- Visualisation de l'évolution des termes informatiques (1950–1999)
- Filtres par terme, journal, période, canton
- Accès direct aux articles (avec prévisualisation)
- Visualisations variées (lignes, aires, flux)

## Exécution d'analyses individuelles
```bash
# Analyse lexicale
python src/scripts/run_lexical_analysis.py

# Modélisation thématique
python src/scripts/run_topic_modeling.py

# Analyse de sentiment
python src/scripts/run_sentiment_analysis.py

# Reconnaissance d'entités
python src/scripts/run_entity_recognition.py

# Suivi de termes
python src/scripts/run_term_tracking.py --terms "informatique,ordinateur,internet"
```

## Dépannage

| Problème | Solution |
|----------|----------|
| `ImportError` | Vérifiez que vous êtes dans l'environnement virtuel et que le chemin Python est correct |
| `KeyError` | Vérifiez que les champs `content` et `cleaned_text` sont présents dans vos données |
| `spaCy model not found` | Exécutez `python -m spacy download fr_core_news_md` |
| Erreur de démarrage | Vérifiez les logs dans le dossier racine du projet |
| Modules manquants | Exécutez `pip install -r requirements.txt` |

## Structure des données
L'application attend des fichiers JSON avec la structure suivante :
```json
{
  "id": "article_1992-04-12_journal_XYZ",
  "title": "Titre de l'article",
  "date": "1992-04-12",
  "source": "Le Journal",
  "content": "Texte nettoyé",
  "original_content": "Texte OCR original",
  "cleaned_text": "Texte prétraité pour l'analyse"
}
```
