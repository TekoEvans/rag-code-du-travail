"""
Agent modérateur (Jalon 6, obligatoire).

Un premier appel LLM classe la question AVANT tout le pipeline, en sortie JSON :
  - laisse passer les questions de droit du travail ;
  - refuse poliment les questions hors sujet ;
  - bloque les tentatives d'injection de prompt.

En cas de doute (JSON illisible, panne réseau), on choisit la prudence côté
disponibilité : on laisse passer, car le prompt système de réponse reste, lui,
strictement borné au corpus.
"""

import json


class Moderateur:
    def __init__(self, client_groq, modele, gabarit_prompt):
        self.client = client_groq
        self.modele = modele
        self.gabarit = gabarit_prompt  # contenu de prompts/moderator.txt

    def classer(self, question):
        """Renvoie (categorie, raison). categorie ∈ {code_du_travail, hors_sujet, injection}."""
        prompt = self.gabarit.format(question=question)
        try:
            brut = self.client.completer(
                modele=self.modele,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=200,
                json_mode=True,
            )
            donnees = json.loads(brut)
            categorie = donnees.get("categorie", "code_du_travail")
            raison = donnees.get("raison", "")
        except Exception:
            # Prudence : en cas d'échec du modérateur, on ne bloque pas l'utilisateur.
            categorie, raison = "code_du_travail", "modérateur indisponible"

        if categorie not in ("code_du_travail", "hors_sujet", "injection"):
            categorie = "code_du_travail"
        return categorie, raison
