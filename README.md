📰 Newspaper Articles Analysis
Présentation générale
Cette application permet d’explorer un vaste corpus d’articles de presse numérisés via OCR. Elle propose une suite complète d’outils d’analyse textuelle et de visualisation interactive pour extraire des informations pertinentes à partir de données textuelles complexes.

Elle est composée de deux volets :

Application d’analyse : interface web interactive pour l’exploration des corpus.

Application de médiation : interface de restitution des résultats pour un public non-technique.

Fonctionnalités d’analyse
1. Gestion des sources
Importation depuis diverses sources

Filtres par journal, date, etc.

Statistiques de base du corpus

Accès au texte original des articles

2. Analyse lexicale
Fréquences de mots

Nuages de mots

N-grammes (bi-/trigrammes)

Termes significatifs

3. Modélisation thématique (LDA)
Thèmes principaux et mots-clés

Distribution et évolution temporelle

Optimisation automatique du nombre de thèmes

4. Clustering d’articles
Regroupement via K-means

Visualisation 2D des clusters

Analyse des caractéristiques par cluster

5. Carte des clusters
Représentation spatiale interactive

Navigation dans l’espace thématique

6. Analyse de sentiment
Polarité des articles (positif, négatif, neutre)

Évolution temporelle et comparaison entre sources

7. Reconnaissance d'entités nommées
Extraction : personnes, lieux, organisations, dates

Cooccurrences et suivi temporel

8. Analyse intégrée
Corrélation entre thèmes, entités, sentiment, etc.

Tableaux de bord personnalisables

9. Suivi de termes
Évolution temporelle

Comparaison inter-sources

Visualisations dynamiques (courbes, flux)

Liens directs vers les articles

10. Exports
Résultats exportables

Rapports personnalisables

Partage de visualisations

Application de médiation
Interface dédiée à la présentation des résultats au grand public :

Fonctionnalités
Visualisation de l’évolution des termes informatiques (1950–1999)

Filtres par terme, journal, période, canton

Accès direct aux articles (modale + Swiper.js)

Visualisations variées (lignes, aires, flux)

Structure du projet
bash
Copier
Modifier
newspapers-analysis/
├── config/
│   ├── config.yaml               # Configuration principale
│   └── cache_config.json         # Cache
├── data/
│   ├── raw/                      # Données OCR brutes
│   ├── processed/                # Données nettoyées
│   └── results/
│       ├── clusters/             # Résultats de clustering
│       ├── exports/              # Données pour médiation
│       ├── lexical_analysis/
│       └── ...
├── src/
│   ├── analysis/                 # Modules d’analyse
│   ├── preprocessing/           # Prétraitements
│   ├── scripts/                 # Scripts d'exécution
│   ├── utils/                   # Fonctions utilitaires
│   ├── visualization/           # Visualisation (Dash, Plotly)
│   └── webapp/                  # Interface Dash
├── mediation_app.html           # Interface web de médiation
├── mediation_app.css
├── mediation_app.js
├── requirements.txt
└── README.md
Installation
Créer un environnement virtuel

bash
Copier
Modifier
python -m venv venv
source venv/bin/activate  # (ou venv\Scripts\activate sous Windows)
Installer les dépendances

bash
Copier
Modifier
pip install -r requirements.txt
Installer le modèle spaCy

bash
Copier
Modifier
python -m spacy download fr_core_news_md
Configurer l’application
Modifier config/config.yaml :

Chemins vers les données

Paramètres d’analyse

Options de visualisation

Placer les données
Par défaut dans data/processed/

🖥️ Utilisation
Lancer l'application web
bash
Copier
Modifier
python src/webapp/run_app.py
Accessible via : http://127.0.0.1:8050/

Exécuter des analyses individuelles
bash
Copier
Modifier
# Analyse lexicale
python src/scripts/run_lexical_analysis.py

# Modélisation thématique (option : --num-topics ou --auto-num-topics)
python src/scripts/run_topic_modeling.py

# Analyse de sentiment
python src/scripts/run_sentiment_analysis.py

# Reconnaissance d'entités
python src/scripts/run_entity_recognition.py

# Suivi de termes
python src/scripts/run_term_tracking.py --terms "informatique,ordinateur,internet"
Interface de médiation
Ouvrir mediation_app.html dans un navigateur.

Format des données (JSON)
json
Copier
Modifier
{
  "id": "article_1992-04-12_journal_XYZ",
  "title": "Titre de l'article",
  "date": "1992-04-12",
  "source": "Le Journal",
  "content": "Texte nettoyé",
  "original_content": "Texte OCR original",
  "cleaned_text": "Texte prétraité pour l’analyse"
}
Sorties des analyses
Lexicale : fréquences, nuages de mots

Thématique : mots-clés, évolution

Sentiment : scores, évolutions, comparaisons

Entités : extraits, cooccurrences, suivi

Termes : dynamiques temporelles, contextes

Fonctionnalités avancées
Correction automatique d’erreurs OCR

Suppression des doublons par similarité

Optimisation du nombre de thèmes (LDA)

Tableaux de bord interactifs

Dépannage
Problème	Solution
ImportError	Vérifiez le chemin Python
KeyError	Assurez-vous que les champs content, cleaned_text sont présents
spaCy model not found	Installer fr_core_news_md
Erreur Dash	Utilisez run_app_safe.ps1 sur Windows

Contributions
Le projet est en développement actif. 