import os
import requests

class LLMClient:
    """
    Client générique pour interagir avec des LLMs (Mistral, OpenAI, etc.)
    La configuration (clé API, endpoint, modèle, etc.) est passée au constructeur.
    """
    def __init__(self, config: dict):
        self.provider = config.get("provider", "mistral")
        self.api_key = config.get("api_key") or os.environ.get("MISTRAL_API_KEY")
        self.model = config.get("model", "mistral-small")
        self.endpoint = config.get("endpoint", "https://api.mistral.ai/v1/chat/completions")
        self.language = config.get("language", "fr")
        if not self.api_key:
            raise ValueError("Clé API LLM manquante (api_key ou variable d'environnement)")

    def get_topic_name(self, words, prompt_template=None, max_tokens=20, temperature=0.3):
        if not prompt_template:
            prompt_template = (
                "Tu es un assistant expert en analyse de texte et en fouille de données. "
                "On t’a extrait les mots les plus représentatifs d’un thème (topic) identifié automatiquement par un algorithme de topic modeling (LDA, NMF, etc.) sur un grand corpus d’articles de presse. "
                "Donne un titre court, explicite et pertinent pour ce groupe de mots-clés, qui résume le thème principal représenté par ces mots. "
                "Les mots-clés du topic sont : {words} "
                "Réponds uniquement par le titre proposé, sans phrase introductive."
            )
        prompt = prompt_template.format(words=", ".join(words))
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
            response = requests.post(self.endpoint, headers=headers, json=data)
            if response.status_code == 200:
                return response.json()["choices"][0]["message"]["content"].strip()
            else:
                return f"(Erreur API: {response.status_code})"
        else:
            raise NotImplementedError(f"Provider non supporté: {self.provider}")

    def get_topic_names(self, top_words_per_topic, **kwargs):
        """
        Génère une liste de noms de topics pour plusieurs groupes de mots.
        """
        return [self.get_topic_name(words, **kwargs) for words in top_words_per_topic]
