"""
Script d'INTERROGATION (phase 2 — le script principal).

Recharge la base existante (sans réindexer) et répond à une question.

Deux usages :
    python ask.py "Quelle est la durée légale du préavis pour un CDI ?"
    python ask.py                 # boucle interactive (tapez /quit pour sortir)

Options :
    --no-hyde        désactive la reformulation HyDE
    --k N            nombre de chunks envoyés au LLM (défaut : config.TOP_K)
    --show-scores    affiche la similarité du meilleur chunk
"""

import argparse
import sys

import config
from src import lire_prompt
from src.embeddings import Embedder
from src.vector_store import VectorStore
from src.llm_client import GroqClient
from src.moderator import Moderateur
from src.hyde import HyDE
from src.rag_pipeline import Pipeline


def construire_pipeline():
    """Charge la base et branche tous les composants du pipeline."""
    # Base : chargée telle quelle, avec vérification du modèle d'embedding.
    store = VectorStore(config.DOSSIER_BASE, config.NOM_COLLECTION)
    try:
        store.charger(config.MODELE_EMBEDDING)
    except Exception as e:
        sys.exit(
            f"Impossible de charger la base : {e}\n"
            "Avez-vous lancé l'indexation ?  ->  python index.py --reset"
        )

    embedder = Embedder(config.MODELE_EMBEDDING)
    client = GroqClient(config.GROQ_API_KEY)

    moderateur = Moderateur(
        client, config.MODELE_MODERATEUR, lire_prompt(config.DOSSIER_PROMPTS / "moderator.txt")
    )
    hyde = HyDE(
        client, config.MODELE_HYDE, lire_prompt(config.DOSSIER_PROMPTS / "hyde.txt")
    )
    gabarit_reponse = lire_prompt(config.DOSSIER_PROMPTS / "system_answer.txt")

    return Pipeline(moderateur, embedder, store, client, gabarit_reponse, config, hyde=hyde)


def afficher(resultat, show_scores):
    """Affiche la réponse, les articles sources et l'avertissement."""
    print("\n" + resultat["reponse"].strip())

    if resultat["sources"]:
        print("\nArticles sources :")
        for s in resultat["sources"]:
            score = f"  (similarité {s['similarite']})" if show_scores else ""
            print(f"  · {s['article']} — {s['section']}{score}")

    if show_scores and resultat["similarite_max"] is not None:
        print(f"\n[confiance : meilleur score = {resultat['similarite_max']:.3f} "
              f"· seuil = {config.SEUIL_CONFIANCE}]")


def main():
    parseur = argparse.ArgumentParser(description="Assistant Code du travail (RAG).")
    parseur.add_argument("question", nargs="*", help="la question (entre guillemets)")
    parseur.add_argument("--no-hyde", action="store_true", help="désactive HyDE")
    parseur.add_argument("--k", type=int, default=None, help="nb de chunks (top-k)")
    parseur.add_argument("--show-scores", action="store_true",
                         help="affiche les scores de similarité")
    args = parseur.parse_args()

    pipeline = construire_pipeline()
    utiliser_hyde = not args.no_hyde

    # Mode « une question en argument » -----------------------------------
    if args.question:
        question = " ".join(args.question)
        resultat = pipeline.repondre(question, utiliser_hyde=utiliser_hyde, top_k=args.k)
        afficher(resultat, args.show_scores)
        return

    # Mode interactif (Jalon 5) ------------------------------------------
    print("Assistant Code du travail — tapez votre question, /quit pour sortir.")
    while True:
        try:
            question = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nAu revoir.")
            break
        if not question:
            continue
        if question.lower() in ("/quit", "/exit", "/q"):
            print("Au revoir.")
            break
        resultat = pipeline.repondre(question, utiliser_hyde=utiliser_hyde, top_k=args.k)
        afficher(resultat, args.show_scores)


if __name__ == "__main__":
    main()
