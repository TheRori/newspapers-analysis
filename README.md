# Analyse d'Articles de Presse

## Description
Ce projet propose une suite d'outils pour explorer, analyser et visualiser un corpus d'articles de presse numérisés (OCR) via une interface web interactive et des scripts d'analyse avancée. Il comprend :
- Une application web d'analyse interactive
- Une interface de médiation grand public pour la restitution des résultats
- Des scripts d'analyse textuelle (lexicale, thématique, sentiment, entités, suivi de termes)

## Fonctionnalités principales

- Navigation et exploration interactive des corpus
- Analyse lexicale : fréquences, nuages de mots, n-grammes
- Modélisation thématique (LDA) avec optimisation automatique du nombre de thèmes
- Clustering d'articles (K-means)
- Analyse de sentiment (polarité, évolution temporelle)
- Reconnaissance d'entités nommées (personnes, lieux, organisations, dates)
- Suivi de termes spécifiques, comparaisons temporelles et inter-journaux
- Visualisations dynamiques et accès direct aux articles
- Application de médiation grand public : visualisation filtrable de l'évolution des termes informatiques (1950–1999)

## Aperçu de l'application

*Un aperçu visuel peut être ajouté ici si une capture d'écran est disponible.*

## Installation

### Prérequis
- Python 3.11 ou supérieur
- Git
- PowerShell (Windows) ou Terminal (Linux/Mac)

### Étapes d'installation

1. Cloner le dépôt :
   ```bash
   git clone https://github.com/TheRori/newspapers-analysis.git
   cd newspapers-analysis
   ```
2. Créer et activer un environnement virtuel :
   - Windows :
     ```powershell
     python -m venv .venv
     .\.venv\Scripts\activate
     ```
   - Linux/Mac :
     ```bash
     python -m venv .venv
     source .venv/bin/activate
     ```
3. Installer les dépendances :
   ```bash
   pip install -r requirements.txt
   ```
4. Installer le modèle spaCy :
   ```bash
   python -m spacy download fr_core_news_md
   ```
5. Configurer le fichier `config/config.yaml` selon vos chemins de données et préférences.
6. Placer les données dans `data/processed/` (voir structure attendue ci-dessous).

## Lancement de l'application

### Méthode recommandée (Windows)
- Utiliser le script PowerShell fourni pour activer l'environnement et lancer l'application :
  ```powershell
  .\run_app_direct.ps1
  ```
  Ce script active l'environnement virtuel, vérifie les modules et lance l'application avec logs en direct.

### Méthode alternative (toutes plateformes)
- Activer l'environnement virtuel puis lancer :
  ```bash
  python src/webapp/run_app.py
  ```
- Accéder à l'application sur : http://127.0.0.1:8050/

## Exécution des analyses individuelles

Des scripts sont disponibles pour lancer chaque type d'analyse séparément :
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

## Structure des données attendue

Les fichiers JSON doivent respecter la structure suivante :
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

## Dépannage

| Problème                      | Solution                                                                 |
|-------------------------------|--------------------------------------------------------------------------|
| ImportError                   | Vérifiez l'environnement virtuel et le chemin Python                      |
| KeyError                      | Vérifiez la présence des champs `content` et `cleaned_text` dans les données |
| spaCy model not found         | Exécutez `python -m spacy download fr_core_news_md`                      |
| Erreur de démarrage           | Consultez les logs dans le dossier racine                                 |
| Modules manquants             | Exécutez `pip install -r requirements.txt`                                |

## Modules et librairies utilisés

- Dash, Plotly, dash-bootstrap-components : interface web et visualisations interactives
- spaCy : traitement du langage naturel (modèle français)
- Gensim, BERTopic : modélisation thématique et clustering
- Pandas, NumPy, scikit-learn, NLTK : analyse et manipulation de données
- Wordcloud, Seaborn, Matplotlib : visualisation
- pymongo, requests, transformers, pyyaml, networkx : utilitaires avancés

Voir `requirements.txt` pour la liste complète.

## Structure des dossiers principaux

```
newspapers-analysis/
├── config/           # Fichiers de configuration (YAML, JSON)
├── data/             # Données brutes et traitées
├── src/              # Code source (analyses, webapp, scripts)
│   └── webapp/       # Application Dash principale
├── mediation_app/    # Interface de médiation grand public (HTML/JS)
├── requirements.txt  # Dépendances Python
└── run_app_direct.ps1# Script de lancement rapide (Windows)
```

## Sources et licence

- Ce projet utilise des éléments de code adaptés de la documentation officielle de Dash, spaCy, Gensim, etc.
- Les ressources graphiques et scripts sont originaux sauf mention contraire dans les dossiers concernés.
- Licence : MIT (à préciser dans un fichier LICENSE si besoin)
- Projet développé dans le cadre du cours *Humanités numériques, UNIL* (enseignant : Isaac Pante).

## Contact et support

Pour toute question, suggestion ou bug, merci d’ouvrir une issue sur le dépôt GitHub.

---

*Bonnes analyses !*
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
