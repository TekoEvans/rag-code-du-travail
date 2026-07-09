import json
import random
import re
from datetime import datetime, timezone

SOURCE = "Code du travail — Légifrance"


def charger_corpus(chemin):
    """Charge le fichier JSON brut de l'arborescence Légifrance."""
    with open(chemin, encoding="utf-8") as f:
        return json.load(f)


def date_corpus(corpus):
    """Date de la version du corpus (pour informer sur la fraîcheur)."""
    d = corpus.get("data", {})
    return d.get("dateDebutVersion") or d.get("dateModif") or "inconnue"


def _ms_vers_date(ms):
    """Convertit un timestamp Légifrance (millisecondes) en date ISO lisible."""
    if not ms:
        return ""
    try:
        # Certaines dates de fin valent l'an 2999 : on les ignore silencieusement.
        annee = datetime.fromtimestamp(ms / 1000, tz=timezone.utc).year
        if annee > 2100:
            return ""
        return datetime.fromtimestamp(ms / 1000, tz=timezone.utc).strftime("%Y-%m-%d")
    except (ValueError, OverflowError, OSError):
        return ""


def nettoyer_texte(txt):
    """Nettoyage léger : le champ `texte` de Légifrance est déjà du texte brut."""
    if not txt:
        return ""
    txt = txt.replace("\xa0", " ").replace("\r", " ")
    txt = re.sub(r"[ \t]+", " ", txt)          # espaces multiples
    txt = re.sub(r"\n{3,}", "\n\n", txt)        # sauts de ligne multiples
    return txt.strip()


def iterer_articles(corpus, etats_gardes=("VIGUEUR",)):
    """
    Parcourt l'arbre en profondeur et renvoie un dictionnaire par article.

    On conserve le chemin des sections traversées pour reconstituer la
    « section thématique » de chaque article.
    """
    articles = []

    def descendre(noeud, chemin_sections):
        type_noeud = noeud.get("type")
        data = noeud.get("data", {})

        if type_noeud == "section":
            titre = nettoyer_texte(data.get("title") or "")
            nouveau_chemin = chemin_sections + [titre] if titre else chemin_sections
        elif type_noeud == "article":
            etat = data.get("etat")
            if etat in etats_gardes:
                articles.append(_construire_article(data, chemin_sections))
            return  # un article n'a pas d'enfants pertinents
        else:
            nouveau_chemin = chemin_sections

        for enfant in noeud.get("children", []) or []:
            descendre(enfant, nouveau_chemin)

    descendre(corpus, [])
    return articles


def _construire_article(data, chemin_sections):
    """Construit le dictionnaire normalisé d'un article à partir du noeud brut."""
    section_proche = chemin_sections[-1] if chemin_sections else ""
    return {
        "article": (data.get("num") or "").strip(),
        "texte": nettoyer_texte(data.get("texte") or ""),
        "section": section_proche,
        "hierarchie": " > ".join(chemin_sections[-4:]),  # 4 derniers niveaux, lisibles
        "source": SOURCE,
        "etat": data.get("etat", ""),
        "date_debut": _ms_vers_date(data.get("dateDebut")),
        "id_legifrance": data.get("id", ""),
    }
