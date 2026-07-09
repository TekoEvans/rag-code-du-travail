"""
Client LLM (Groq).

Enveloppe minimale autour de l'API « chat completions » de Groq. L'API n'a
aucune mémoire entre deux appels : chaque requête est autonome. La clé est
lue depuis l'environnement, jamais codée en dur.
"""


class GroqClient:
    def __init__(self, api_key):
        if not api_key:
            raise RuntimeError(
                "Clé Groq absente. Renseignez GROQ_API_KEY dans un fichier .env "
                "(voir .env.example)."
            )
        from groq import Groq
        self.client = Groq(api_key=api_key)

    def completer(self, modele, messages, temperature=0.1, max_tokens=1024,
                  json_mode=False):
        """Envoie une conversation au modèle et renvoie le texte de la réponse."""
        params = {
            "model": modele,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if json_mode:
            # Force une sortie JSON valide (utilisé par le modérateur).
            params["response_format"] = {"type": "json_object"}
        reponse = self.client.chat.completions.create(**params)
        return reponse.choices[0].message.content
