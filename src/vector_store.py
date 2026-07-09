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

   