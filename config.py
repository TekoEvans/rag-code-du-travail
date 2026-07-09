"""
Configuration centrale de l'assistant.

Tout est regroupé ici pour qu'un seul fichier décrive les choix techniques :
modèles, chemins, paramètres de recherche et de découpage. Les valeurs
sensibles ou dépendantes de la machine sont lues depuis l'environnement
(fichier .env), jamais codées en dur.
"""

import os
from pathlib import Path

try:
    # python-dotenv charge automatiquement le fichier .env s'il existe.
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:  # dotenv est optionnel : on peut aussi exporter les variables à la main.
    pass

# --- Chemins du projet ---------------------------------------------------
RACINE = Path(__file__).resolve().parent
DOSSIER_DATA = RACINE / "data"
DOSSIER_PROMPTS = RACINE / "prompts"
DOSSIER_BASE = RACINE / "vector_store"          # base vectorielle persistée sur disque

CHEMIN_CORPUS = DOSSIER_DATA / "corpus_legifrance.json"
NOM_COLLECTION = "code_du_travail"

# --- Modèle d'embedding (sentence-transformers) --------------------------
# Le MÊME modèle sert à indexer les articles et à encoder les questions.
# Son nom est stocké avec la base : au chargement on vérifie qu'il correspond.
# multilingual-e5-small : léger, rapide sur CPU, bon en français.
# Passer à "intfloat/multilingual-e5-base" ou "-large" pour plus de qualité.
MODELE_EMBEDDING = os.getenv("MODELE_EMBEDDING", "intfloat/multilingual-e5-small")

# --- Modèles Groq (LLM) --------------------------------------------------
# Les anciens noms Llama sont dépréciés côté Groq : on cible la famille gpt-oss.
MODELE_REPONSE = os.getenv("MODELE_REPONSE", "openai/gpt-oss-120b")   # qualité pour la réponse finale
MODELE_MODERATEUR = os.getenv("MODELE_MODERATEUR", "openai/gpt-oss-20b")  # rapide pour le filtrage
MODELE_HYDE = os.getenv("MODELE_HYDE", "openai/gpt-oss-20b")          # rapide pour la reformulation

GROQ_API_KEY = os.getenv("GROQ_API_KEY")  # jamais écrit en dur — voir .env.example

# --- Paramètres de recherche --------------------------------------------
TOP_K = int(os.getenv("TOP_K", "5"))                 # nb de chunks envoyés au LLM
NB_CANDIDATS = int(os.getenv("NB_CANDIDATS", "20"))  # pool récupéré avant sélection du top-k
# Seuil de confiance : en dessous, on considère que le corpus ne répond pas.
# Similarité cosinus, à recalibrer sur son corpus (les valeurs absolues
# dépendent du modèle d'embedding).
SEUIL_CONFIANCE = float(os.getenv("SEUIL_CONFIANCE", "0.78"))

# --- Paramètres de découpage (chunking) ---------------------------------
# Stratégie « par structure » : un article = une unité. Les articles trop
# longs sont découpés sur des frontières de phrase, jamais en plein milieu.
CHUNK_MAX_CARACTERES = int(os.getenv("CHUNK_MAX_CARACTERES", "1200"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "150"))

# --- Génération ----------------------------------------------------------
TEMPERATURE = float(os.getenv("TEMPERATURE", "0.1"))  # fidélité avant créativité
MAX_TOKENS_REPONSE = int(os.getenv("MAX_TOKENS_REPONSE", "1024"))

# --- Avertissement juridique (garanti par le code, pas seulement le prompt) ---
AVERTISSEMENT_JURIDIQUE = (
    "Cet assistant ne fournit pas de conseil juridique. "
    "Consultez un avocat ou l'inspection du travail pour votre situation personnelle."
)
