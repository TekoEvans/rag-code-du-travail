class Embedder:
    def __init__(self, nom_modele):
        # Import différé : sentence-transformers est lourd à charger.
        from sentence_transformers import SentenceTransformer

        self.nom_modele = nom_modele
        self._modele = SentenceTransformer(nom_modele)
        # Détection heuristique des modèles e5 (préfixes requis).
        self._e5 = "e5" in nom_modele.lower()

    @property
    def dimension(self):
        return self._modele.get_sentence_embedding_dimension()

    def _prefixer(self, textes, prefixe):
        if self._e5:
            return [f"{prefixe}{t}" for t in textes]
        return textes

    def encoder_documents(self, textes, batch_size=64, progression=False):
        """Encode des passages à indexer -> matrice numpy normalisée."""
        textes = self._prefixer(textes, "passage: ")
        return self._modele.encode(
            textes,
            batch_size=batch_size,
            normalize_embeddings=True,   # indispensable pour la similarité cosinus
            show_progress_bar=progression,
            convert_to_numpy=True,
        )

    def encoder_question(self, question):
        """Encode une question -> vecteur normalisé (liste de floats)."""
        vecteurs = self._modele.encode(
            self._prefixer([question], "query: "),
            normalize_embeddings=True,
            convert_to_numpy=True,
        )
        return vecteurs[0].tolist()
