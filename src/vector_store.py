import chromadb


class VectorStore:
    def __init__(self, dossier_persistance, nom_collection):
        self.client = chromadb.PersistentClient(path=str(dossier_persistance))
        self.nom_collection = nom_collection
        self.collection = None

