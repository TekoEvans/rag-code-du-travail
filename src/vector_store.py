import chromadb


class VectorStore:
    def __init__(self, dossier_persistance, nom_collection):
        self.client = chromadb.PersistentClient(path=str(dossier_persistance))
        self.nom_collection = nom_collection
        self.collection = None

    # --- Création / réinitialisation (phase d'indexation) ----------------
    def creer(self, nom_modele_embedding, date_corpus, reset=False):
        if reset:
            try:
                self.client.delete_collection(self.nom_collection)
            except Exception:
                pass  # la collection n'existait pas encore
        self.collection = self.client.get_or_create_collection(
            name=self.nom_collection,
            # cosine : cohérent avec des embeddings normalisés
            metadata={
                "hnsw:space": "cosine",
                "embedding_model": nom_modele_embedding,   # tracé avec la base
                "date_corpus": date_corpus,
            },
        )
        return self.collection

 # --- Chargement (phase d'interrogation) ------------------------------
    def charger(self, nom_modele_embedding_attendu=None):
        """Charge une collection existante et vérifie le modèle d'embedding."""
        self.collection = self.client.get_collection(self.nom_collection)
        meta = self.collection.metadata or {}
        modele_base = meta.get("embedding_model")
        if (
            nom_modele_embedding_attendu
            and modele_base
            and modele_base != nom_modele_embedding_attendu
        ):
            raise ValueError(
                "Incohérence de modèle d'embedding : la base a été construite avec "
                f"« {modele_base} » mais la configuration demande "
                f"« {nom_modele_embedding_attendu} ». Réindexez ou corrigez la config."
            )
        return self.collection

    def date_corpus(self):
        meta = self.collection.metadata or {}
        return meta.get("date_corpus", "inconnue")

    def nombre(self):
        return self.collection.count()

    def identifiants_existants(self):
        """Renvoie l'ensemble des ids déjà indexés (pour la mise à jour)."""
        return set(self.collection.get(include=[])["ids"])

