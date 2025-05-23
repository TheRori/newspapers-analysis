import os
import requests
import logging
import json

# Configuration du logger
logger = logging.getLogger(__name__)

class LLMClient:
    """
    Client générique pour interagir avec des LLMs (Mistral, OpenAI, etc.)
    La configuration (clé API, endpoint, modèle, etc.) est passée au constructeur.
    """
    def __init__(self, config: dict):
        # Vérifier si nous avons la nouvelle structure de configuration (avec des providers imbriqués)
        if 'default_provider' in config:
            # Nouvelle structure de configuration
            self.provider = config.get("default_provider", "openai")
            provider_config = config.get(self.provider, {})
            
            # Toujours récupérer les valeurs depuis la configuration
            self.api_key = provider_config.get("api_key")
            self.model = provider_config.get("model")
            self.language = provider_config.get("language", "fr")
            self.endpoint = provider_config.get("endpoint")
            
            # Vérifier que l'endpoint est défini
            if not self.endpoint:
                raise ValueError(f"Endpoint manquant pour le provider {self.provider} dans la configuration")
        else:
            # Ancienne structure de configuration (pour rétrocompatibilité)
            self.provider = config.get("provider", "mistral")
            self.api_key = config.get("api_key") or os.environ.get("MISTRAL_API_KEY")
            self.model = config.get("model", "mistral-small")
            self.endpoint = config.get("endpoint")
            self.language = config.get("language", "fr")
            
            # Vérifier que l'endpoint est défini
            if not self.endpoint:
                raise ValueError(f"Endpoint manquant pour le provider {self.provider} dans la configuration")
        
        # Vérifier que nous avons une clé API
        if not self.api_key:
            raise ValueError("Clé API LLM manquante dans la configuration")
            
        logger.info(f"LLMClient initialisé avec provider: {self.provider}, model: {self.model}")

    def get_topic_name(self, words, prompt_template=None, max_tokens=20, temperature=0.3):
        if not prompt_template:
            prompt_template = (
                "Tu es un assistant expert en analyse de texte et en fouille de données. "
                "On t'a extrait les mots les plus représentatifs d'un thème (topic) identifié automatiquement par un algorithme de topic modeling (LDA, NMF, etc.) sur un grand corpus d'articles de presse. "
                "Donne un titre court, explicite et pertinent pour ce groupe de mots-clés, qui résume le thème principal représenté par ces mots. "
                "Les mots-clés du topic sont : {words} "
                "Réponds uniquement par le titre proposé, sans phrase introductive."
            )
        prompt = prompt_template.format(words=", ".join(words))
        # Utiliser la méthode ask qui gère déjà tous les providers
        return self.ask(prompt, max_tokens=max_tokens, temperature=temperature)

    def get_topic_name_from_articles(self, articles, prompt_template=None, max_tokens=50, temperature=0.3):
        """
        Génère un nom de topic en analysant des articles représentatifs au lieu des mots-clés.
        
        Args:
            articles: Liste des articles représentatifs (textes complets ou extraits)
            prompt_template: Template de prompt personnalisé (optionnel)
            max_tokens: Nombre maximum de tokens pour la réponse
            temperature: Température pour contrôler la créativité
            
        Returns:
            Un tuple (titre, résumé) où titre est un nom court pour le topic et résumé est une description plus détaillée
        """
        if not prompt_template:
            prompt_template = (
                "Tu es un assistant expert en analyse de texte et en fouille de données. "
                "On t'a fourni 10 articles représentatifs d'un thème (topic) identifié automatiquement par un algorithme de topic modeling sur un grand corpus d'articles de presse. "
                "Après avoir lu ces articles, donne: "
                "1. Un titre court (5 mots maximum), explicite et pertinent qui résume le thème principal représenté par ces articles. "
                "2. Un bref résumé (3-4 phrases) des thématiques principales abordées dans ces articles. "
                "\n\nVoici les articles:\n{articles} "
                "\n\nRéponds en utilisant exactement ce format:\nTITRE: [ton titre court]\nRÉSUMÉ: [ton résumé de 3-4 phrases]"
            )
        
        # Limiter la taille des articles pour éviter de dépasser les limites de tokens
        processed_articles = []
        for i, article in enumerate(articles[:10]):  # Limiter à 10 articles maximum
            # Prendre les 300 premiers caractères de chaque article
            excerpt = article[:300] + "..." if len(article) > 300 else article
            processed_articles.append(f"Article {i+1}: {excerpt}")
        
        articles_text = "\n\n".join(processed_articles)
        prompt = prompt_template.format(articles=articles_text)
        
        # Utiliser la méthode ask qui gère déjà tous les providers
        response = self.ask(prompt, max_tokens=max_tokens, temperature=temperature)
        
        # Parser la réponse pour extraire le titre et le résumé
        title = ""
        summary = ""
        
        try:
            lines = response.strip().split('\n')
            for line in lines:
                if line.startswith("TITRE:") or line.startswith("TITLE:"):
                    title = line.split(':', 1)[1].strip()
                elif line.startswith("RÉSUMÉ:") or line.startswith("RESUME:") or line.startswith("SUMMARY:"):
                    summary = line.split(':', 1)[1].strip()
        except Exception as e:
            logger.error(f"Erreur lors du parsing de la réponse LLM: {e}")
            # Si le parsing échoue, retourner la réponse brute comme titre
            if not title:
                title = response[:50] + "..." if len(response) > 50 else response
        
        return (title, summary)

    def get_topic_names(self, top_words_per_topic, **kwargs):
        """
        Génère une liste de noms de topics pour plusieurs groupes de mots.
        """
        return [self.get_topic_name(words, **kwargs) for words in top_words_per_topic]
        
    def ask(self, prompt, max_tokens=100, temperature=0.3):
        """
        Méthode générique pour envoyer une requête au LLM et obtenir une réponse.
        
        Args:
            prompt: Le texte de la requête à envoyer au LLM
            max_tokens: Nombre maximum de tokens dans la réponse
            temperature: Température pour contrôler la créativité (0.0-1.0)
            
        Returns:
            La réponse du LLM sous forme de texte
        """
        logger.info(f"Envoi d'une requête au LLM ({self.provider}/{self.model})")
        logger.debug(f"Prompt: {prompt[:100]}..." if len(prompt) > 100 else f"Prompt: {prompt}")
        
        if self.provider == "mistral":
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            data = {
                "model": self.model,
                "messages": [
                    {"role": "user", "content": prompt}
                ],
                "max_tokens": max_tokens,
                "temperature": temperature
            }
            try:
                response = requests.post(self.endpoint, headers=headers, json=data)
                if response.status_code == 200:
                    result = response.json()["choices"][0]["message"]["content"].strip()
                    logger.info(f"Réponse LLM reçue: {result[:100]}..." if len(result) > 100 else f"Réponse LLM reçue: {result}")
                    return result
                else:
                    error_msg = f"Erreur API {self.provider}: {response.status_code} - {response.text}"
                    logger.error(error_msg)
                    return f"(Erreur API: {response.status_code})"
            except Exception as e:
                error_msg = f"Exception lors de l'appel au LLM: {str(e)}"
                logger.error(error_msg)
                return f"(Erreur: {str(e)})"
        elif self.provider == "openai":
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            data = {
                "model": self.model,
                "messages": [
                    {"role": "user", "content": prompt}
                ],
                "max_tokens": max_tokens,
                "temperature": temperature
            }
            try:
                # Ajouter un timeout pour éviter les blocages indéfinis
                response = requests.post(self.endpoint, headers=headers, json=data, timeout=30)
                
                if response.status_code == 200:
                    result = response.json()["choices"][0]["message"]["content"].strip()
                    logger.info(f"Réponse OpenAI reçue: {result[:100]}..." if len(result) > 100 else f"Réponse OpenAI reçue: {result}")
                    return result
                elif response.status_code == 429:
                    # Rate limit atteint
                    error_msg = f"Limite de requêtes OpenAI atteinte: {response.status_code} - {response.text}"
                    logger.error(error_msg)
                    raise Exception(f"Limite de requêtes OpenAI atteinte: {response.status_code}")
                elif response.status_code in [500, 502, 503, 504]:
                    # Erreur serveur
                    error_msg = f"Erreur serveur OpenAI: {response.status_code} - {response.text}"
                    logger.error(error_msg)
                    raise Exception(f"Erreur serveur OpenAI: {response.status_code}")
                else:
                    error_msg = f"Erreur API {self.provider}: {response.status_code} - {response.text}"
                    logger.error(error_msg)
                    raise Exception(f"Erreur API OpenAI: {response.status_code}")
            except requests.exceptions.Timeout:
                error_msg = "Timeout lors de la connexion à l'API OpenAI"
                logger.error(error_msg)
                raise Exception(error_msg)
            except requests.exceptions.ConnectionError:
                error_msg = "Erreur de connexion à l'API OpenAI"
                logger.error(error_msg)
                raise Exception(error_msg)
            except Exception as e:
                error_msg = f"Exception lors de l'appel à OpenAI: {str(e)}"
                logger.error(error_msg)
                raise Exception(error_msg)
        else:
            error_msg = f"Provider non supporté: {self.provider}"
            logger.error(error_msg)
            raise NotImplementedError(error_msg)
