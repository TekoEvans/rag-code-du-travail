import argparse
import time

import config  # noqa: F401  (garde le package importable)
from src import data_preparation as dp
from src.embeddings import Embedder
from src.vector_store import VectorStore


def main():
    parseur = argparse.ArgumentParser(description="Indexation du Code du travail.")
    parseur.add_argument("--reset", action="store_true",
                         help="reconstruit entièrement la base")
    parseur.add_argument("--update", action="store_true",
                         help="ajoute uniquement les articles absents de la base")
    parseur.add_argument("--limit", type=int, default=None,
                         help="limite le nombre de chunks (test rapide)")
    args = parseur.parse_args()

    # 1) Préparation des données -----------------------------------------
    print("→ Préparation du corpus…")
    chunks, date = dp.preparer(
        config.CHEMIN_CORPUS,
        config.CHUNK_MAX_CARACTERES,
        config.CHUNK_OVERLAP,
    )
    if args.limit:
        chunks = chunks[:args.limit]
    print(f"  {len(chunks)} chunks · corpus daté du {date}")

    # Contrôle qualité : relire quelques chunks (recommandation du cours).
    print("\n→ Contrôle qualité (chunks au hasard) :")
    dp.apercu_aleatoire(chunks, n=5)

    # 2) Base vectorielle -------------------------------------------------
    store = VectorStore(config.DOSSIER_BASE, config.NOM_COLLECTION)

    if args.update:
        store.charger()  # base existante
        deja = store.identifiants_existants()
        chunks = [c for c in chunks if c["id"] not in deja]
        print(f"\n→ Mise à jour : {len(chunks)} nouveaux chunks à ajouter.")
        if not chunks:
            print("  Rien à ajouter, la base est à jour.")
            return
    else:
        # Refus de réindexer par mégarde : une base existante non vide bloque.
        try:
            store.charger()
            if store.nombre() > 0 and not args.reset:
                print(
                    f"\n⚠ Une base de {store.nombre()} chunks existe déjà.\n"
                    "  Utilisez --reset pour la reconstruire, ou --update pour "
                    "ajouter de nouveaux articles."
                )
                return
        except Exception:
            pass  # pas de base existante : cas normal du premier lancement
        store.creer(config.MODELE_EMBEDDING, date, reset=args.reset)

    # 3) Embedding + insertion -------------------------------------------
    print(f"\n→ Chargement du modèle d'embedding « {config.MODELE_EMBEDDING} »…")
    embedder = Embedder(config.MODELE_EMBEDDING)

    print("→ Encodage des chunks (peut prendre quelques minutes)…")
    t = time.time()
    vecteurs = embedder.encoder_documents(
        [c["embed_text"] for c in chunks], progression=True
    )
    print(f"  encodage terminé en {time.time() - t:.1f}s")

    print("→ Insertion dans la base…")
    n = store.ajouter(chunks, vecteurs, ignorer_existants=True)
    print(f"\n✓ Terminé : {n} chunks indexés · base totale = {store.nombre()} "
          f"· modèle « {config.MODELE_EMBEDDING} » tracé avec la base.")
    print(f"  Base persistée dans : {config.DOSSIER_BASE}")


if __name__ == "__main__":
    main()
