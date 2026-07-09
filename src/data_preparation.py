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

# --- Découpage (chunking) ------------------------------------------------

def _decouper_en_phrases(texte):
    """Découpe grossièrement en phrases sur la ponctuation forte."""
    # On coupe après . ; : ! ? suivis d'un espace. Simple et suffisant ici.
    morceaux = re.split(r"(?<=[.;:!?])\s+", texte)
    return [m.strip() for m in morceaux if m.strip()]


def _chunks_par_phrases(texte, taille_max, overlap):
    """
    Regroupe les phrases en blocs de taille <= taille_max, sans couper une
    phrase, avec un chevauchement (overlap) entre blocs consécutifs pour ne
    pas perdre le contexte aux jointures.
    """
    phrases = _decouper_en_phrases(texte)
    blocs, courant, taille = [], [], 0
    for ph in phrases:
        if taille + len(ph) > taille_max and courant:
            blocs.append(" ".join(courant))
            # On garde les dernières phrases pour l'overlap.
            reprise, taille = [], 0
            for p in reversed(courant):
                if taille + len(p) > overlap:
                    break
                reprise.insert(0, p)
                taille += len(p)
            courant = reprise
        courant.append(ph)
        taille += len(ph)
    if courant:
        blocs.append(" ".join(courant))
    return blocs


def _texte_a_embarquer(article, section, texte):
    """
    Texte réellement encodé par le modèle d'embedding.

    On préfixe le numéro d'article et la section : cela renforce la recherche
    (signal lexical sur le numéro) et rappelle au modèle la référence.
    """
    entete = f"Article {article}"
    if section:
        entete += f" — {section}"
    return f"{entete}\n{texte}"


def construire_chunks(articles, taille_max, overlap):
    """
    Transforme la liste d'articles en liste de chunks indexables.

    Un article court = un seul chunk. Un article long est découpé en
    plusieurs chunks (#p0, #p1...) qui gardent tous le même numéro d'article
    dans leurs métadonnées.
    """
    chunks = []
    for art in articles:
        if not art["texte"] or not art["article"]:
            continue

        if len(art["texte"]) <= taille_max:
            morceaux = [art["texte"]]
        else:
            morceaux = _chunks_par_phrases(art["texte"], taille_max, overlap)

        multi = len(morceaux) > 1
        for i, morceau in enumerate(morceaux):
            chunk_id = f"{art['article']}#p{i}" if multi else art["article"]
            chunks.append({
                "id": chunk_id,
                "embed_text": _texte_a_embarquer(art["article"], art["section"], morceau),
                "texte": morceau,
                "metadata": {
                    "article": art["article"],
                    "section": art["section"],
                    "hierarchie": art["hierarchie"],
                    "source": art["source"],
                    "etat": art["etat"],
                    "date_debut": art["date_debut"],
                    "partie": i,          # indice du morceau dans l'article
                },
            })
    return chunks


def apercu_aleatoire(chunks, n=10, graine=42):
    """Contrôle qualité : affiche n chunks au hasard pour relecture humaine."""
    random.seed(graine)
    echantillon = random.sample(chunks, min(n, len(chunks)))
    for c in echantillon:
        m = c["metadata"]
        print(f"\n[{c['id']}] section: {m['section'][:70]}")
        apercu = c["texte"][:280].replace("\n", " ")
        print(f"   {apercu}{'…' if len(c['texte']) > 280 else ''}")
