import os
import re
import yaml
import pandas as pd
import unidecode
import fitz
import numpy as np
import json
from sentence_transformers import SentenceTransformer


def norm(txt):
    txt = str(txt or "").strip()
    txt = unidecode.unidecode(txt)
    txt = txt.lower()
    txt = re.sub(r"[^a-z0-9\s\-]", " ", txt)
    txt = re.sub(r"\s+", " ", txt).strip()
    return txt


def lire_pdf(path):
    doc = fitz.open(path)

    texte = ""

    for page in doc:
        texte += page.get_text("text") + "\n"

    return texte


def chunk_text(text, size=500, overlap=80):

    chunks = []

    start = 0

    while start < len(text):

        chunks.append(
            text[start:start + size]
        )

        start += size - overlap

    return chunks


ALIASES = {
    # AOD
    "eliquis": ["AOD"],
    "apixaban": ["AOD"],
    "xarelto": ["AOD"],
    "rivaroxaban": ["AOD"],
    "pradaxa": ["AOD"],
    "dabigatran": ["AOD"],
    "lixiana": ["AOD"],
    "edoxaban": ["AOD"],
    "aod": ["AOD"],
    "anticoagulant oral direct": ["AOD"],

    # AVK
    "previscan": ["AVK", "Anti-vitamine K"],
    "fluindione": ["AVK", "Anti-vitamine K"],
    "sintrom": ["AVK", "Anti-vitamine K"],
    "acenocoumarol": ["AVK", "Anti-vitamine K"],
    "coumadine": ["AVK", "Anti-vitamine K"],
    "warfarine": ["AVK", "Anti-vitamine K"],
    "avk": ["AVK", "Anti-vitamine K"],

    # AAP
    "kardegic": ["AAP (Anti-agrégants plaquettaires)"],
    "aspegic": ["AAP (Anti-agrégants plaquettaires)"],
    "aspirine": ["AAP (Anti-agrégants plaquettaires)"],
    "plavix": ["AAP (Anti-agrégants plaquettaires)"],
    "clopidogrel": ["AAP (Anti-agrégants plaquettaires)"],
    "brilique": ["AAP (Anti-agrégants plaquettaires)"],
    "ticagrelor": ["AAP (Anti-agrégants plaquettaires)"],
    "efient": ["AAP (Anti-agrégants plaquettaires)"],
    "prasugrel": ["AAP (Anti-agrégants plaquettaires)"],
    "duoplavin": ["AAP (Anti-agrégants plaquettaires)"],
    "aap": ["AAP (Anti-agrégants plaquettaires)"],

    # Diabète
    "metformine": ["Metformine"],
    "glucophage": ["Metformine"],

    "forxiga": ["SGLT2 - FORXIGA", "SGLT2 (Gliflozines)"],
    "dapagliflozine": ["SGLT2 - FORXIGA", "SGLT2 (Gliflozines)"],
    "jardiance": ["SGLT2 (Gliflozines)"],
    "empagliflozine": ["SGLT2 (Gliflozines)"],
    "sglt2": ["SGLT2 (Gliflozines)"],
    "gliflozine": ["SGLT2 (Gliflozines)"],

    "ozempic": ["GLP-1 (Analogues : Ozempic, Saxenda...)"],
    "semaglutide": ["GLP-1 (Analogues : Ozempic, Saxenda...)"],
    "wegovy": ["GLP-1 (Analogues : Ozempic, Saxenda...)"],
    "trulicity": ["GLP-1 (Analogues : Ozempic, Saxenda...)"],
    "dulaglutide": ["GLP-1 (Analogues : Ozempic, Saxenda...)"],
    "saxenda": ["GLP-1 (Analogues : Ozempic, Saxenda...)"],
    "liraglutide": ["GLP-1 (Analogues : Ozempic, Saxenda...)"],
    "victoza": ["GLP-1 (Analogues : Ozempic, Saxenda...)"],
    "glp1": ["GLP-1 (Analogues : Ozempic, Saxenda...)"],

    # Héparines
    "lovenox": ["Enoxaparine – lovenox"],
    "enoxaparine": ["Enoxaparine – lovenox"],
    "inohep": ["Tinzaparine - Inohep"],
    "tinzaparine": ["Tinzaparine - Inohep"],
    "arixtra": ["Fundaparinux – arixtra"],
    "fondaparinux": ["Fundaparinux – arixtra"],
    "hnf": ["HNF"],
    "heparine": ["HNF", "Enoxaparine – lovenox"],

    # Corticoïdes
    "hydrocortisone": ["HYDROCORTISONE SYSTEMIQUE", "CORTICOIDES TOPIQUES"],
    "hydrocortisone orale": ["HYDROCORTISONE SYSTEMIQUE"],
    "hydrocortisone per os": ["HYDROCORTISONE SYSTEMIQUE"],
    "hydrocortisone systemique": ["HYDROCORTISONE SYSTEMIQUE"],
    "hydrocortisone creme": ["CORTICOIDES TOPIQUES"],
    "hydrocortisone crème": ["CORTICOIDES TOPIQUES"],
    "hydrocortisone pommade": ["CORTICOIDES TOPIQUES"],
    "corticoide topique": ["CORTICOIDES TOPIQUES"],
    "corticoides": ["CORTICOIDES", "HYDROCORTISONE SYSTEMIQUE"],

    # Cardio
    "ramipril": ["SRAA"],
    "perindopril": ["SRAA"],
    "candesartan": ["SRAA"],
    "losartan": ["SRAA"],
    "valsartan": ["SRAA"],
    "entresto": ["Entresto"],
    "bisoprolol": ["Bêta-bloquants"],
    "atenolol": ["Bêta-bloquants"],
    "metoprolol": ["Bêta-bloquants"],
    "nebivolol": ["Bêta-bloquants"],
    "atorvastatine": ["Statines"],
    "rosuvastatine": ["Statines"],
    "pravastatine": ["Statines"],
    "simvastatine": ["Statines"],
    "statine": ["Statines"],


    "ado": ["ADO"],
    "sulfamide": ["ADO"],
    "sulfamides": ["ADO"],
    "glinide": ["ADO"],
    "glinides": ["ADO"],
    "dpp4": ["ADO"],
    "antidiabetique oral": ["ADO"],
    "antidiabetiques oraux": ["ADO"],


}


def construire_index(base_dir):
    yaml_path = os.path.join(base_dir, "regles_sfar.yaml")

    with open(yaml_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}

    
    questions_path = os.path.join(base_dir, "questions.json")

    questions_dataset = []

    if os.path.exists(questions_path):
        with open(questions_path, "r", encoding="utf-8") as f:
            questions_dataset = json.load(f)


    atc_map = {}

    csv_path = os.path.join(base_dir, "dci_atc.fichier.csv")
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path, sep=";")
        df.columns = [norm(c).upper() for c in df.columns]

        cols_nom = [
            "MEDICAMENT_SOURCE",
            "NOM_COMMERCIAL",
            "DCI"
        ]

        for _, row in df.iterrows():
            atc = str(row.get("CODE_ATC", "")).upper().strip()
            if not atc or atc == "NAN":
                continue

            for col in cols_nom:
                if col in df.columns:
                    nom = row.get(col)
                    if pd.notna(nom):
                        atc_map[norm(nom)] = atc

    passages_docs = []

    docs_dir = os.path.join(base_dir, "docs_sfar")

    if os.path.exists(docs_dir):
        for file in os.listdir(docs_dir):
            path = os.path.join(docs_dir, file)

            if file.lower().endswith(".pdf"):
                texte = lire_pdf(path)

                for chunk in chunk_text(texte):
                    passages_docs.append({
                        "source": file,
                        "texte": chunk
                    })

    model = SentenceTransformer(
        "FremyCompany/BioLORD-2023-M"
    )

    if passages_docs:
        embeddings_docs = model.encode(
            [p["texte"] for p in passages_docs],
            normalize_embeddings=True,
            show_progress_bar=False
        )

        embeddings_docs = np.array(
            embeddings_docs,
            dtype="float32"
        )
    else:
        embeddings_docs = np.empty((0, 768), dtype="float32")

    return {
        "yaml": data,
        "atc_map": atc_map,
        "passages_docs": passages_docs,
        "embeddings_docs": embeddings_docs,
        "questions_dataset": questions_dataset
    }, passages_docs, model



    return {
        "yaml": data,
        "atc_map": atc_map
    }, [], None


def sources_regle(data, regle):
    sources = []

    if regle.get("source_url"):
        if isinstance(regle["source_url"], list):
            sources.extend(regle["source_url"])
        else:
            sources.append(regle["source_url"])

    ref = regle.get("source_ref")
    if ref:
        bloc = data.get("sources_regles", {}).get(ref, {})
        src = bloc.get("sources", [])
        if isinstance(src, list):
            sources.extend(src)
        elif src:
            sources.append(str(src))

    return list(dict.fromkeys([str(s) for s in sources if str(s).strip()]))


def trouver_atc(question, atc_map):
    q = norm(question)

    meilleurs = []

    for nom, atc in atc_map.items():
        if len(nom) < 3:
            continue

        if nom in q:
            meilleurs.append((len(nom), nom, atc))

    if not meilleurs:
        return None, None

    meilleurs.sort(reverse=True)
    _, nom, atc = meilleurs[0]
    return atc, nom


def regle_match_atc(regle, atc):
    atc = str(atc or "").upper().strip()

    if not atc:
        return False

    if regle.get("atc_codes"):
        codes = [str(c).upper().strip() for c in regle["atc_codes"]]
        if atc in codes:
            return True

    if regle.get("atc_prefix"):
        prefixes = [str(p).upper().strip() for p in regle["atc_prefix"]]
        if any(atc.startswith(p) for p in prefixes):
            return True

    return False


def trouver_regles(question, data_store):
    data = data_store["yaml"]
    atc_map = data_store["atc_map"]

    q = norm(question)
    resultats = []

    for mot, categories in ALIASES.items():
        if norm(mot) in q:
            for regle in data.get("regles_medicaments", []):
                cat = norm(regle.get("categorie", ""))
                for cible in categories:
                    if norm(cible) in cat:
                        resultats.append(regle)

    if resultats:
        return resultats

    atc, nom = trouver_atc(question, atc_map)

    if atc:
        for regle in data.get("regles_medicaments", []):
            if regle_match_atc(regle, atc):
                resultats.append(regle)

    return resultats

def choisir_condition(question, regle):
    q = norm(question)
    conditions = regle.get("conditions", [])

    if not conditions:
        return None

    # hydrocortisone topique
    if any(x in q for x in ["creme", "crème", "pommade", "lotion", "topique"]):
        for c in conditions:
            if "topique" in norm(c):
                return c

    # hydrocortisone systémique / orale
    if any(x in q for x in ["orale", "per os", "systemique", "systémique"]):
        for c in conditions:
            if "systemique" in norm(c):
                return c

    # =================================================
    # DFG
    # ====================================================

    q_simple = (
        q.replace(" ", "")
         .replace("≥", ">=")
         .replace("≤", "<=")
    )

    # DFG > 30
    if (
        "dfg>30" in q_simple
        or "dfgsuperieur30" in q_simple
        or "dfgsup30" in q_simple
        or "dfg>=30" in q_simple
    ):

        for c in conditions:

            txt = norm(str(c)).replace(" ", "")

            if (
                "dfg>30" in txt
                or "dfg_ge_30" in txt
                or "dfgsuperieur30" in txt
                or "dfg>=30" in txt
            ):
                return c

    # DFG < 30
    if (
        "dfg<30" in q_simple
        or "dfginferieur30" in q_simple
        or "dfginf30" in q_simple
        or "dfg<=30" in q_simple
    ):

        for c in conditions:

            txt = norm(str(c)).replace(" ", "")

            if (
                "dfg<30" in txt
                or "dfg_inf_30" in txt
                or "dfginferieur30" in txt
                or "dfg<=30" in txt
            ):
                return c

    # DFG 30-49
    if (
        "30-49" in q_simple
        or "3049" in q_simple
    ):

        for c in conditions:

            txt = norm(str(c)).replace(" ", "")

            if (
                "30-49" in txt
                or "3049" in txt
                or "dfg_30_49" in txt
            ):
                return c

    # DFG > 50
    if (
        "dfg>50" in q_simple
        or "dfgsuperieur50" in q_simple
        or "dfg>=50" in q_simple
    ):

        for c in conditions:

            txt = norm(str(c)).replace(" ", "")

            if (
                "dfg>50" in txt
                or "dfg_ge_50" in txt
                or "dfgsuperieur50" in txt
                or "dfg>=50" in txt
            ):
                return c


    # contexte chirurgie
    if "urgence" in q:
        for c in conditions:
            if "urgence" in norm(c):
                return c

    if "ambulatoire" in q:
        for c in conditions:
            if "ambulatoire" in norm(c):
                return c

    if "neuro" in q or "intracran" in q or "intracr" in q:
        for c in conditions:
            if "neurochir" in norm(c) or "intracran" in norm(c):
                return c

    if "alr" in q or "neurax" in q or "rachianesthesie" in q:
        for c in conditions:
            if "alr" in norm(c) or "neurax" in norm(c):
                return c

    # risque hémorragique
    if "faible" in q:
        for c in conditions:
            if "faible" in norm(c):
                return c

    if "majeur" in q or "eleve" in q or "élevé" in q or "important" in q:
        for c in conditions:
            if "majeur" in norm(c) or "eleve" in norm(c) or "important" in norm(c):
                return c

    # héparines
    if "curative" in q:
        for c in conditions:
            if "curative" in norm(c):
                return c

    if "preventive" in q or "préventive" in q:
        for c in conditions:
            if "preventive" in norm(c):
                return c

    # défaut
    for c in conditions:
        if c.get("default") is True:
            return c

    return conditions[0]


def format_reponse(data, regle, cond):
    categorie = regle.get("categorie", "-")

    if cond:
        action = cond.get("action", regle.get("action", "-"))
        jour = cond.get("jour", regle.get("jour", "-"))
        note = (
            cond.get("note")
            or cond.get("precision")
            or regle.get("note")
            or regle.get("precision")
            or "-"
        )
    else:
        action = regle.get("action", "-")
        jour = regle.get("jour", "-")
        note = regle.get("note") or regle.get("precision") or "-"

    sources = sources_regle(data, regle)

    texte = f"Catégorie : {categorie}\n"
    texte += f"Action : {action}\n"
    texte += f"Jour : {jour}\n"
    texte += f"Note : {note}\n"

    if sources:
        texte += "\nSources :\n"
        for s in sources:
            texte += f"- {s}\n"

    return texte


def chercher_documents(question, data_store, model, k=3):

    passages_docs = data_store["passages_docs"]
    embeddings_docs = data_store["embeddings_docs"]

    if len(passages_docs) == 0:
        return ""

    q_emb = model.encode(
        [question],
        normalize_embeddings=True
    )

    q_emb = np.array(q_emb, dtype="float32")

    scores = np.dot(
        embeddings_docs,
        q_emb[0]
    )

    meilleurs = np.argsort(scores)[::-1][:k]

    textes = []

    for idx in meilleurs:

        p = passages_docs[idx]

        textes.append(
            f" {p['source']}\n\n"
            f"{p['texte'][:1200]}"
        )

    return "\n\n------------------\n\n".join(textes)


def repondre_rag(question, index, passages, model):
    data_store = index
    data = data_store["yaml"]

    regles = trouver_regles(question, data_store)

    if not regles:
        return "Médicament ou classe non retrouvé dans les règles SFAR."

    regle = regles[0]
    cond = choisir_condition(question, regle)

    reponse = format_reponse(data, regle, cond)

    q = norm(question)

    mots_doc = [
        "pourquoi",
        "justification",
        "documentation",
        "source",
        "explique",
        "reference",
        "preuve"
    ]

    if any(m in q for m in mots_doc):
        docs = chercher_documents(question, data_store, model)

        if docs:
            docs = docs.replace("\n", " ")
            docs = " ".join(docs.split())
            docs = docs[:1800]

            reponse += "\n\nDocumentation SFAR :\n\n" + docs

    return reponse
