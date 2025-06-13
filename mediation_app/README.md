# Médiation App – Interface de visualisation grand public

#### Présentation
L'application de médiation est une interface web interactive dédiée à la restitution et à la valorisation des résultats d'analyse textuelle sur la presse suisse (1950–1999). Elle permet à un public non technique d'explorer l'évolution des termes informatiques, leur répartition par journal, par canton et dans le temps, à travers des visualisations dynamiques.

## Fonctionnalités principales
- **Évolution chronologique** : visualisation de la fréquence des termes informatiques par année.
- **Répartition par journal** : comparaison de la présence des termes selon les titres de presse.
- **Timeline historique** : exploration des contextes d'apparition des termes clés.
- **Filtres dynamiques** : sélection de termes, périodes, cantons, journaux.
- **Accès direct aux articles** : affichage des extraits d'articles associés à chaque terme ou période (avec Swiper.js).
- **Visualisations variées** : courbes, aires empilées, streamgraphs, barres, camemberts.
- **Génération d'insights automatiques** : synthèse textuelle des tendances observées.

## Structure du dossier
```
mediation_app/
├── data/                 # Données d'entrée (CSV, JSON)
├── mediation_app.html    # Page principale de l'application
├── mediation_app.js      # Logique et interactions (D3.js, Swiper.js)
├── mediation_app.css     # Styles principaux
├── heatmap_app.html      # Visualisation Heatmap sémantique
├── heatmap_app.js        # Logique pour la heatmap
├── entity_cards.html     # Cartes d'entités nommées
├── entity_cards.js       # Logique des cartes d'entités
├── ...                  # Autres fichiers (timelines, styles, composants)
```

## Installation et lancement
1. **Prérequis** :
   - Un navigateur web moderne (Chrome, Firefox, Edge, ...)
   - Les fichiers de données nécessaires dans `mediation_app/data/`

2. **Aucune installation logicielle requise** :
   - L'application fonctionne en ouvrant directement le fichier `mediation_app.html` dans votre navigateur (double-clic ou clic-droit > "Ouvrir avec...").

3. **Pour une expérience optimale** :
   - Vérifiez que tous les fichiers JS et CSS associés sont présents dans le dossier.
   - Une connexion internet est nécessaire pour charger certaines librairies externes (D3.js, Swiper.js, FontAwesome).

## Personnalisation des données
- Les fichiers de données doivent respecter les formats attendus (CSV pour les évolutions, JSON pour les articles).
- Les chemins des fichiers peuvent être modifiés dans `mediation_app.js` (voir la variable `config`).

## Technologies utilisées
- **D3.js** : visualisations interactives
- **Swiper.js** : carrousels d’articles
- **noUiSlider** : sliders temporels
- **FontAwesome** : icônes

## Exemples d'utilisation
- Explorer l'évolution d'un terme :
  - Sélectionner un ou plusieurs termes dans le panneau de gauche.
  - Ajuster la période avec le slider.
  - Observer la courbe ou l’aire correspondante.
- Afficher les articles associés :
  - Cliquer sur un point ou une zone du graphique pour voir les extraits d’articles.
- Comparer les journaux ou cantons :
  - Utiliser les filtres dédiés et changer de type de visualisation.

## Hébergement
Le site web est hébergé sur Netlify à l'adresse suivante : https://mediation-app.netlify.app/

## Dépannage
| Problème | Solution |
|----------|----------|
| Les graphiques ne s'affichent pas | Vérifiez la présence des fichiers de données et l'accès au JS/CSS |
| Les articles ne s'affichent pas | Vérifiez le chemin des fichiers JSON dans `mediation_app.js` |
| Les librairies externes ne chargent pas | Vérifiez votre connexion internet |

## Contribution
Pour toute suggestion ou amélioration, ouvrez une issue ou contactez l’équipe projet.
