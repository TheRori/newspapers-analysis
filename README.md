# Analyse d'Articles de Presse

## Description

Ce projet propose une suite d'outils pour explorer, analyser et visualiser un corpus d'articles de presse numérisés (OCR) via une interface web interactive et des scripts d'analyse avancée. Il comprend :

* Une application web d'analyse interactive
* Une interface de médiation grand public pour la restitution des résultats
* Des scripts d'analyse textuelle (lexicale, thématique, sentiment, entités, suivi de termes)

## Fonctionnalités principales

* Navigation et exploration interactive des corpus
* Analyse lexicale : fréquences, nuages de mots, n-grammes
* Modélisation thématique (LDA) avec optimisation automatique du nombre de thèmes
* Clustering d'articles (K-means)
* Analyse de sentiment (polarité, évolution temporelle)
* Reconnaissance d'entités nommées (personnes, lieux, organisations, dates)
* Suivi de termes spécifiques, comparaisons temporelles et inter-journaux
* Visualisations dynamiques et accès direct aux articles
* Application de médiation grand public : visualisation filtrable de l'évolution des termes informatiques (1950–1999)

## Aperçu de l'application

*Un aperçu visuel peut être ajouté ici si une capture d'écran est disponible.*

---

## Installation

### Prérequis

* **Python 3.11** (recommandé)

  > ⚠️ *Le projet a été testé et validé avec Python 3.11. Les versions plus récentes (ex: 3.12/3.13) peuvent poser problème lors de l'installation de certaines dépendances scientifiques (notamment scipy, numpy).*
  > Pour vérifier votre version :
  >
  > ```bash
  > python --version
  > ```
  >
  > Si besoin, utilisez [pyenv](https://github.com/pyenv/pyenv) ou l'installeur officiel pour disposer de Python 3.11 sur votre machine.

* Git

* PowerShell (Windows) ou Terminal (Linux/Mac)

> **Déploiement cloud**
> Le fichier `runtime.txt` impose Python 3.11 sur Render ou Heroku.
> En local, vérifiez que vous utilisez bien cette version (voir ci-dessus).

#### Vérification rapide

Avant d’installer les dépendances, assurez-vous d’utiliser la bonne version de Python :

```bash
python --version
# Doit afficher : Python 3.11.x

# Si ce n'est pas le cas, installez Python 3.11 et/ou utilisez pyenv :
pyenv install 3.11.9
pyenv local 3.11.9
```

### Étapes d'installation

1. Cloner le dépôt :

   ```bash
   git clone https://github.com/TheRori/newspapers-analysis.git
   cd newspapers-analysis
   ```
2. Créer et activer un environnement virtuel :

   * Windows :

     ```powershell
     python -m venv .venv
     .\.venv\Scripts\activate
     ```
   * Linux/Mac :

     ```bash
     python -m venv .venv
     source .venv/bin/activate
     ```
3. Installer les dépendances :

   ```bash
   pip install -r requirements.txt
   ```
4. Installer le modèle spaCy :

   ```bash
   python -m spacy download fr_core_news_md
   ```
5. Configurer le fichier `config/config.yaml` selon vos chemins de données et préférences.
6. Placer les données dans `data/processed/` (voir structure attendue ci-dessous).

---

## Lancement de l'application

### Windows

* Utiliser le script PowerShell fourni pour activer l'environnement et lancer l'application :

  ```powershell
  .\run_app_direct.ps1
  ```

  Ce script active l'environnement virtuel, vérifie les modules et lance l'application avec logs en direct.

### Linux/Mac (ou Bash sous Windows)

* Utiliser le script Bash fourni :

  ```bash
  ./run_app_direct.sh
  ```

  Ce script active l'environnement virtuel, vérifie les modules et lance l'application avec logs en direct.

### Méthode alternative (toutes plateformes)

* Activer l'environnement virtuel puis lancer :

  ```bash
  python src/webapp/run_app.py
  ```
* Accéder à l'application sur : [http://127.0.0.1:8050/](http://127.0.0.1:8050/)

---

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

---

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

---

## Dépannage

| Problème              | Solution                                                                     |
| --------------------- | ---------------------------------------------------------------------------- |
| ImportError           | Vérifiez l'environnement virtuel et le chemin Python                         |
| KeyError              | Vérifiez la présence des champs `content` et `cleaned_text` dans les données |
| spaCy model not found | Exécutez `python -m spacy download fr_core_news_md`                          |
| Erreur de démarrage   | Consultez les logs dans le dossier racine                                    |
| Modules manquants     | Exécutez `pip install -r requirements.txt`                                   |

---

## Modules et librairies utilisés

* Dash, Plotly, dash-bootstrap-components : interface web et visualisations interactives
* spaCy : traitement du langage naturel (modèle français)
* Gensim, BERTopic : modélisation thématique et clustering
* Pandas, NumPy, scikit-learn, NLTK : analyse et manipulation de données
* Wordcloud, Seaborn, Matplotlib : visualisation
* pymongo, requests, transformers, pyyaml, networkx : utilitaires avancés

Voir `requirements.txt` pour la liste complète.

---

## Structure des dossiers principaux

```
newspapers-analysis/
├── config/           # Fichiers de configuration (YAML, JSON)
├── data/             # Données brutes et traitées
├── src/              # Code source (analyses, webapp, scripts)
│   └── webapp/       # Application Dash principale
├── mediation_app/    # Interface de médiation grand public (HTML/JS)
├── requirements.txt  # Dépendances Python
├── run_app_direct.ps1# Script de lancement rapide (Windows)
└── run_app_direct.sh # Script de lancement rapide (Linux/Mac/Bash)
```

---

## Sources et licence

* Ce projet utilise des éléments de code adaptés de la documentation officielle de Dash, spaCy, Gensim, etc.
* Les ressources graphiques et scripts sont originaux sauf mention contraire dans les dossiers concernés.
* Licence : MIT (à préciser dans un fichier LICENSE si besoin)
* Projet développé dans le cadre d'un mémoire de master à l'université de Lausanne.

---

## Contact et support

Pour toute question, suggestion ou bug, merci d’ouvrir une issue sur le dépôt GitHub.

---

*Bonnes analyses !*