"""
Pipeline d'interrogation (Jalons 4, 5 et 6).

Enchaîne : modération -> (HyDE) -> recherche vectorielle -> seuil de confiance
-> génération avec citations. L'avertissement juridique est garanti par le code,
pas seulement par le prompt : c'est l'endroit le plus fiable pour l'imposer.
"""

REFUS_HORS_BASE = "Je ne trouve pas cette information dans ma base."
REFUS_HORS_SUJET = (
    "Je suis un assistant dédié au Code du travail français et je ne traite "
    "que les questions qui s'y rapportent."
)
REFUS_INJECTION = (
    "Cette demande ne peut pas être traitée : je réponds uniquement à des "
    "questions sur le Code du travail, à partir de ma base d'articles."
)


class Pipeline:
    def __init__(self, moderateur, embedder, store, client_groq,
                 gabarit_reponse, config, hyde=None):
        self.moderateur = moderateur
        self.embedder = embedder
        self.store = store
        self.client = client_groq
        self.gabarit_reponse = gabarit_reponse   # contenu de prompts/system_answer.txt
        self.cfg = config
        self.hyde = hyde

    # --- Assemblage du contexte numéroté ---------------------------------
    def _construire_contexte(self, chunks):
        lignes = []
        for i, c in enumerate(chunks, 1):
            m = c["metadata"]
            lignes.append(
                f"[{i}] Article {m['article']} — {m['section']} "
                f"(similarité {c['similarite']:.2f})\n{c['texte']}"
            )
        return "\n\n".join(lignes)

    def _sources(self, chunks):
        """Liste dédupliquée des articles cités, du plus au moins pertinent."""
        vus, sources = set(), []
        for c in chunks:
            art = c["metadata"]["article"]
            if art in vus:
                continue
            vus.add(art)
            sources.append({
                "article": art,
                "section": c["metadata"]["section"],
                "similarite": round(c["similarite"], 3),
            })
        return sources

    def _garantir_avertissement(self, texte):
        """Ajoute l'avertissement s'il manque : présence garantie à 100 %."""
        avert = self.cfg.AVERTISSEMENT_JURIDIQUE
        if avert.split(".")[0] not in texte:
            texte = texte.rstrip() + "\n\n" + avert
        return texte

    def _resultat(self, reponse, refuse, categorie, sources=None, sim_max=None):
        return {
            "reponse": self._garantir_avertissement(reponse),
            "refuse": refuse,
            "categorie": categorie,
            "sources": sources or [],
            "similarite_max": sim_max,
        }

    # --- Point d'entrée --------------------------------------------------
    def repondre(self, question, utiliser_hyde=True, top_k=None):
        top_k = top_k or self.cfg.TOP_K

        # 1) Modération : injection / hors-sujet filtrés en amont.
        categorie, _ = self.moderateur.classer(question)
        if categorie == "injection":
            return self._resultat(REFUS_INJECTION, True, categorie)
        if categorie == "hors_sujet":
            return self._resultat(REFUS_HORS_SUJET, True, categorie)

        # 2) HyDE : reformulation orientée « réponse » pour mieux chercher.
        texte_recherche = question
        if utiliser_hyde and self.hyde is not None:
            texte_recherche = self.hyde.reformuler(question)

        # 3) Recherche vectorielle.
        vecteur = self.embedder.encoder_question(texte_recherche)
        chunks = self.store.rechercher(vecteur, top_k=top_k)

        # 4) Seuil de confiance : si le meilleur score est trop bas, on refuse
        #    plutôt que d'inventer (score de confiance calibré sur le corpus).
        sim_max = chunks[0]["similarite"] if chunks else 0.0
        if not chunks or sim_max < self.cfg.SEUIL_CONFIANCE:
            return self._resultat(REFUS_HORS_BASE, True, categorie, sim_max=sim_max)

        # 5) Génération avec citations, à température basse.
        prompt = self.gabarit_reponse.format(
            date_corpus=self.store.date_corpus(),
            contexte=self._construire_contexte(chunks),
            question=question,
            avertissement=self.cfg.AVERTISSEMENT_JURIDIQUE,
        )
        reponse = self.client.completer(
            modele=self.cfg.MODELE_REPONSE,
            messages=[{"role": "user", "content": prompt}],
            temperature=self.cfg.TEMPERATURE,
            max_tokens=self.cfg.MAX_TOKENS_REPONSE,
        )

        return self._resultat(
            reponse, refuse=False, categorie=categorie,
            sources=self._sources(chunks), sim_max=sim_max,
        )
