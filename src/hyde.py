"""
HyDE — Hypothetical Document Embeddings (levier côté question).

Une question ne ressemble pas à une réponse ; or les chunks indexés SONT des
réponses (des articles de loi). HyDE demande d'abord au LLM de rédiger un court
passage hypothétique « à la manière d'un article », puis c'est ce passage —
plus proche des vrais articles — que l'on encode pour la recherche.

Coût : un appel LLM de plus par question. On combine le passage hypothétique
avec la question d'origine pour ne pas perdre les termes exacts (numéros, sigles).
"""


class HyDE:
    def __init__(self, client_groq, modele, gabarit_prompt):
        self.client = client_groq
        self.modele = modele
        self.gabarit = gabarit_prompt  # contenu de prompts/hyde.txt

    def reformuler(self, question):
        """Renvoie un texte de recherche enrichi (question + passage hypothétique)."""
        prompt = self.gabarit.format(question=question)
        try:
            hypothese = self.client.completer(
                modele=self.modele,
                messages=[{"role": "user", "content": prompt}],
                temperature=0,      # un peu de liberté pour couvrir le vocabulaire
                max_tokens=256,
            )
        except Exception:
            # Si HyDE échoue, on se rabat sur la question brute.
            return question
        # On garde la question d'origine devant : conserve les termes exacts.
        return f"{question}\n{hypothese.strip()}"
