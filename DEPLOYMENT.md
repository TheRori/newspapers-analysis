# Guide de déploiement sur Render

Ce guide explique comment déployer l'application Newspaper Articles Analysis sur Render.

## Prérequis

- Un compte [Render](https://render.com/)
- Votre code source sur GitHub ou GitLab

## Configuration pour Render

L'application a été configurée pour fonctionner sur Render avec les fichiers suivants :

1. **render.yaml** - Configuration du service Render
2. **Procfile** - Instructions pour démarrer l'application
3. **gunicorn_config.py** - Configuration du serveur Gunicorn
4. **app.py** - Modifications pour utiliser le PORT fourni par Render

## Étapes de déploiement

1. **Créez un compte sur Render** si vous n'en avez pas déjà un.

2. **Connectez votre dépôt GitHub ou GitLab** à Render.

3. **Créez un nouveau service Web** et sélectionnez votre dépôt.

4. **Configurez le service** avec les paramètres suivants :
   - **Environment**: Python
   - **Build Command**: `pip install -r requirements.txt && python -m spacy download fr_core_news_md`
   - **Start Command**: `gunicorn -c gunicorn_config.py src.webapp.app:server`

5. **Ajoutez les variables d'environnement** nécessaires dans la section "Environment Variables" de Render.

6. **Cliquez sur "Create Web Service"** pour déployer l'application.

## Vérification du déploiement

Une fois le déploiement terminé, Render fournira une URL pour accéder à votre application. Vérifiez que l'application fonctionne correctement en accédant à cette URL.

## Dépannage

Si vous rencontrez des problèmes lors du déploiement :

1. Vérifiez les logs de build et de runtime sur Render
2. Assurez-vous que toutes les dépendances sont correctement installées
3. Vérifiez que le fichier gunicorn_config.py est correctement configuré
4. Assurez-vous que le serveur est correctement exposé dans app.py

## Notes importantes

- L'application utilise automatiquement le port défini par la variable d'environnement `PORT` fournie par Render
- Les fichiers statiques sont servis par Dash/Flask
- Les performances peuvent varier selon le plan Render que vous utilisez
