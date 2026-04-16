import yaml
import streamlit as st
import os
import math
import pandas as pd
import whisper
import numpy as np
import easyocr
from PIL import Image, ImageOps, ImageFilter
from rapidfuzz import process, fuzz
import unidecode
import tempfile
import re
import io
import fitz
import copy
import shutil
import difflib

from datetime import date, timedelta
from collections import defaultdict

def format_jour_avec_date(jour, date_intervention):
    if not jour or not date_intervention:
        return jour

    if str(jour).startswith("J-"):
        nb = int(str(jour).replace("J-", ""))
        date_calc = date_intervention - timedelta(days=nb)
        return f"{jour} ({date_calc.strftime('%d/%m/%Y')})"

    if jour == "J0":
        return f"{jour} ({date_intervention.strftime('%d/%m/%Y')})"

    return jour


def enrichir_note_avec_dates(note, date_intervention):
    note = str(note or "")

    if not date_intervention:
        return note

    def repl(match):
        jour = match.group(0)
        return format_jour_avec_date(jour, date_intervention)

    return re.sub(r"\bJ-\d+\b|\bJ0\b", repl, note)


# =========================================================
# CONFIGURATION
# =========================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

st.set_page_config(page_title="IA CARE - Expert SFAR", layout="wide")
st.markdown("""
<style>
.chat-floating-box {
    position: fixed;
    bottom: 20px;
    right: 20px;
    width: 420px;
    max-height: 70vh;
    overflow-y: auto;
    background: white;
    border: 1px solid #ddd;
    border-radius: 16px;
    padding: 16px;
    box-shadow: 0 8px 30px rgba(0,0,0,0.15);
    z-index: 9999;
}

.chat-title {
    font-weight: 700;
    font-size: 18px;
    margin-bottom: 12px;
}
</style>
""", unsafe_allow_html=True)

REGLES = {}
yaml_path = os.path.join(BASE_DIR, "regles_sfar.yaml")
if os.path.exists(yaml_path):
    try:
        with open(yaml_path, "r", encoding="utf-8") as f:
            REGLES = yaml.safe_load(f) or {}
    except Exception as e:
        st.warning(f"Impossible de charger regles_sfar.yaml : {e}")



@st.cache_resource
def get_whisper_model_cached(model_name="base"):
    return whisper.load_model(model_name)


@st.cache_resource
def get_easyocr_reader_cached():
    return easyocr.Reader(["fr"], gpu=False)


def preprocess_image_for_ocr(image):
    img = image.convert("L")
    img = ImageOps.autocontrast(img)
    img = img.filter(ImageFilter.MedianFilter(size=3))
    img = img.point(lambda p: 255 if p > 170 else 0)
    return img


def extraire_texte_tesseract_image(image):
    return ""

def extraire_lignes_ocr_image(image):
    lignes = []

    try:
        reader = get_easyocr_reader_cached()
        results = reader.readtext(np.array(image.convert("RGB")), detail=1, paragraph=False)
        lignes.extend(regrouper_ocr_en_lignes(results, tol_y=18))
    except Exception:
        pass

    lignes_finales = []
    vus = set()
    for ligne in lignes:
        l = str(ligne).strip()
        if not l:
            continue
        key = normalize_text(l)
        if key not in vus:
            vus.add(key)
            lignes_finales.append(l)

    return lignes_finales



def afficher_pdf(uploaded_pdf):
    contenu = uploaded_pdf.getvalue()
    doc = fitz.open(stream=contenu, filetype="pdf")

    for i, page in enumerate(doc):
        pix = page.get_pixmap(matrix=fitz.Matrix(2, 2), alpha=False)
        image = Image.open(io.BytesIO(pix.tobytes("png"))).convert("RGB")
        st.image(
            image,
            caption=f"Page {i+1}",
            use_container_width=True
        )



def extraire_texte_pdf(uploaded_pdf):
    contenu = uploaded_pdf.getvalue()
    doc = fitz.open(stream=contenu, filetype="pdf")
    lignes = []

    for page in doc:
        texte_natif = page.get_text("text") or ""
        if texte_natif.strip():
            lignes.extend(decouper_texte_en_entrees_medicaments(texte_natif))

        pix = page.get_pixmap(matrix=fitz.Matrix(2, 2), alpha=False)
        image = Image.open(io.BytesIO(pix.tobytes("png"))).convert("RGB")
        lignes.extend(extraire_lignes_ocr_image(image))

    lignes_finales = []
    vus = set()
    for ligne in lignes:
        l = str(ligne).strip()
        if not l:
            continue
        key = normalize_text(l)
        if key not in vus:
            vus.add(key)
            lignes_finales.append(l)

    return lignes_finales


def corriger_texte_vocal_medicamenteux(texte, ref):
    if not texte:
        return ""

    txt = normalize_text(texte)

    # petites normalisations phonétiques simples
    txt = txt.replace(" PH ", " F ")
    txt = txt.replace(" Y ", " I ")
    txt = txt.replace("-", " ")
    txt = re.sub(r"\s+", " ", txt).strip()

    # on corrige mot par mot avec tolérance orthographique
    mots = txt.split()
    mots_corriges = []

    # petits mots à ne pas toucher
    mots_a_ignorer = {
        "LE", "LA", "LES", "DE", "DU", "DES", "ET", "OU", "UN", "UNE",
        "MATIN", "MIDI", "SOIR", "JOUR", "JOURS", "SI", "BESOIN"
    }

    for mot in mots:
        mot_clean = normalize_text(mot)

        if len(mot_clean) < 4 or mot_clean in mots_a_ignorer:
            mots_corriges.append(mot_clean)
            continue

        # variantes phonétiques très simples
        variantes = {
            mot_clean,
            mot_clean.replace("Z", "S"),
            mot_clean.replace("PH", "F"),
            mot_clean.replace("Y", "I"),
            mot_clean.replace("C", "K"),
        }

        meilleur_nom = None
        meilleur_score = 0

        for variante in variantes:
            match = process.extractOne(variante, ref, scorer=fuzz.ratio)
            if match:
                nom_match, score_match, _ = match
                if score_match > meilleur_score:
                    meilleur_nom = nom_match
                    meilleur_score = score_match

        # seuil voix un peu plus souple
        if meilleur_nom and meilleur_score >= 82:
            mots_corriges.append(meilleur_nom)
        else:
            mots_corriges.append(mot_clean)

    texte_corrige = " ".join(mots_corriges)
    texte_corrige = re.sub(r"\s+", " ", texte_corrige).strip()
    return texte_corrige


def transcrire_audio_robuste(uploaded_audio):
    audio_path = None
    try:
        model = get_whisper_model_cached("small")  

        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            tmp.write(uploaded_audio.getvalue())
            audio_path = tmp.name

        result = model.transcribe(
            audio_path,
            language="fr",
            fp16=False,
            initial_prompt=(
                "Liste de médicaments en français. "
                "Pradaxa, Bisoprolol, Ramipril, Kardegic, Lasilix, "
                "Amlodipine, Atorvastatine, Metformine, Levothyrox, "
                "Eliquis, Xarelto, Previscan, Sintrom."
            )
        )

        texte = (result or {}).get("text", "")
        return extraire_medicaments_depuis_transcription_vocale(texte, ref)

    finally:
        try:
            if audio_path:
                os.unlink(audio_path)
        except Exception:
            pass

def extraire_medicaments_depuis_transcription_vocale(texte, ref):
    if not texte:
        return []

    txt = normalize_text(texte)

    # =========================
    # SUPPRESSION DES DOSAGES
    # =========================
    txt = re.sub(r"\b\d+[.,]?\d*\s*(MG|G|MCG|UG|ML|UI|MUI)\b", " ", txt)
    txt = re.sub(r"\b\d+[.,]?\d*\b", " ", txt)

    # mots de posologie à ignorer
    txt = re.sub(
        r"\b(MG|G|MCG|UG|ML|UI|MUI|COMPRIME|COMPRIMES|GELULE|GELULES|AMP|AMPOULE|SACHET|SACHETS|MATIN|MIDI|SOIR|JOUR|JOURS|PAR|FOIS|BESOIN)\b",
        " ",
        txt
    )

    txt = re.sub(r"[^A-Z0-9\s\-]", " ", txt)
    txt = re.sub(r"\s+", " ", txt).strip()

    mots = txt.split()
    candidats = []
    vus = set()

    # 1 mot
    for i in range(len(mots)):
        candidats.append(mots[i])

    # 2 mots
    for i in range(len(mots) - 1):
        candidats.append(mots[i] + " " + mots[i + 1])

    # 3 mots
    for i in range(len(mots) - 2):
        candidats.append(mots[i] + " " + mots[i + 1] + " " + mots[i + 2])

    meds_trouves = []

    mots_a_ignorer = {
        "LE", "LA", "LES", "DE", "DU", "DES", "ET", "OU",
        "UN", "UNE", "AVEC", "SANS", "PENDANT"
    }

    for cand in candidats:
        cand = normalize_text(cand)

        if len(cand) < 4:
            continue
        if cand in mots_a_ignorer:
            continue

        variantes = {
            cand,
            cand.replace("Z", "S"),
            cand.replace("PH", "F"),
            cand.replace("Y", "I"),
        }

        meilleur_nom = None
        meilleur_score = 0

        for variante in variantes:
            match = process.extractOne(variante, ref, scorer=fuzz.WRatio)
            if match:
                nom_match, score_match, _ = match
                if score_match > meilleur_score:
                    meilleur_nom = nom_match
                    meilleur_score = score_match

        if meilleur_nom and meilleur_score >= 86:
            nom_norm = normalize_text(meilleur_nom)
            if nom_norm not in vus:
                vus.add(nom_norm)
                meds_trouves.append(meilleur_nom)

    return meds_trouves

st.markdown("""
    <style>
    .sticky-header {
        position: -webkit-sticky;
        position: sticky;
        top: 0;
        background-color: white;
        z-index: 1000;
        padding: 10px;
        border-bottom: 2px solid #f0f2f6;
        margin-bottom: 20px;
    }
    .box-profil {
        border: 1px solid #e6e6e6;
        border-radius: 12px;
        padding: 16px;
        margin-bottom: 12px;
        background: #fafafa;
    }
    </style>
""", unsafe_allow_html=True)


# ===================================================
# OUTILS GENERAUX
# =========================================================
def normalize_colname(col):
    col = str(col).strip()
    col = unidecode.unidecode(col)
    col = col.upper()
    col = re.sub(r"\s+", " ", col)
    return col

def normalize_text(txt):
    txt = str(txt).strip()
    txt = unidecode.unidecode(txt)
    txt = txt.upper()
    txt = re.sub(r"\s+", " ", txt)
    return txt

def val_upper(x, default=""):
    if pd.isna(x):
        return default
    s = str(x).strip()
    s = unidecode.unidecode(s)
    return s.upper()

def clean_display_value(x, default="Non renseigné"):
    if pd.isna(x):
        return default
    s = str(x).strip()
    return s if s else default

def corriger_nom_profil(nom):
    if not nom:
        return nom

    n = normalize_text(nom)

    mapping = {
        "DT2": "Diabète Type 2",
        "DT 2": "Diabète Type 2",
        "DIABETE T2": "Diabète Type 2",
        "DIABETE TYPE 2": "Diabète Type 2",
        "HTA": "Hypertension Artérielle",
        "FA": "Fibrillation Auriculaire",
        "BPCO": "Bronchopneumopathie Chronique Obstructive",
        "IRC": "Insuffisance Rénale Chronique",
        "IC": "Insuffisance Cardiaque",
        "SCA": "Syndrome Coronarien Aigu",
        "AVC": "Accident Vasculaire Cérébral",
        "MTEV": "Maladie Thromboembolique Veineuse",
        "CORONARIEN": "Coronarien",
    }

    return mapping.get(n, nom)

def asa_acte_to_int(asa_val):
    s = str(asa_val).strip()
    if not s or s.lower() == "nan":
        return None
    m = re.search(r"\d+", s)
    if m:
        return int(m.group(0))
    return None

def calculer_asa(age, nb_medocs, risque_chir):
    # 1. On définit le score de base : Minimum ASA 2 (ta règle)
    score = 2
    
    # 2. Règle de l'âge : Plus de 60 ans -> ASA 3 d'office
    if age >= 60:
        score = 3
        
    # 3. Règle des médicaments : Plus de 3 médicaments -> ASA 3 d'office
    elif nb_medocs > 3:
        score = 3

    # 4. Sécurité pour les cas très lourds 
    # Si chirurgie majeure ET (très âgé ou énormément de médicaments)
    risque_clean = str(risque_chir).upper().strip()
    if (age > 70 or nb_medocs > 5) and risque_clean in ["MAJEUR", "MAJEURE", "IMPORTANT"]:
        score = 4

    return score
def extraire_nb_jours(texte):
    if not texte:
        return None
    m = re.search(r"J-(\d+)", str(texte).upper())
    if m:
        return int(m.group(1))
    return None
#--------------------------------------------------------------
#----------chargement yaml--------------------------------------------------------

def charger_yaml_regles():
    file_path = os.path.join(BASE_DIR, "regles_sfar.yaml")
    with open(file_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {
            "metadata": {},
            "sources_regles": {},
            "regles_medicaments": []
        }




ALIASES = {
    "sraa": "sraa",
    "iec": "sraa",
    "ara2": "sraa",
    "ara ii": "sraa",

    "metfo": "metformine",
    "metformin": "metformine",

    "aap": "aap",
    "plaquettes": "aap",
    "antiagregants": "aap",
    "anti agregants": "aap",

    "aod": "aod",
    "avk": "avk",
    "ains": "ains",
    "insuline": "insuline",
}



def trouver_regle_par_categorie(data, categorie):
    if not data or "regles_medicaments" not in data:
        return None, None

    cat_user = nettoyer_texte(categorie)
    cat_user = ALIASES.get(cat_user, cat_user)

    meilleurs = []

    for i, regle in enumerate(data.get("regles_medicaments", [])):
        cat_yaml = regle.get("categorie", "")
        cat_yaml_clean = nettoyer_texte(cat_yaml)

        score = difflib.SequenceMatcher(None, cat_user, cat_yaml_clean).ratio()

        if cat_user in cat_yaml_clean:
            score += 0.35
        if cat_yaml_clean in cat_user:
            score += 0.15

        alias_yaml = ALIASES.get(cat_yaml_clean, cat_yaml_clean)
        if cat_user == alias_yaml:
            score += 0.4

        meilleurs.append((score, i, regle))

    if not meilleurs:
        return None, None

    meilleurs.sort(reverse=True, key=lambda x: x[0])
    meilleur_score, idx, regle = meilleurs[0]

    if meilleur_score < 0.45:
        return None, None

    return idx, regle




def valider_bloc_regle(bloc):
    if not isinstance(bloc, dict):
        return False, "Le bloc proposé n'est pas un dictionnaire."

    if not bloc.get("categorie"):
        return False, "Le champ 'categorie' est obligatoire."

    if not any(k in bloc for k in ["atc_codes", "atc_prefix", "conditions"]):
        return False, "La règle doit contenir au moins 'atc_codes', 'atc_prefix' ou 'conditions'."

    if "conditions" in bloc:
        if not isinstance(bloc["conditions"], list):
            return False, "'conditions' doit être une liste."
        for cond in bloc["conditions"]:
            if not isinstance(cond, dict):
                return False, "Chaque condition doit être un dictionnaire."
            if "if" not in cond and "default" not in cond:
                return False, "Chaque condition doit contenir 'if' ou 'default'."

    return True, None





def clean_medicament_name(name):
    if not name:
        return name

    pattern = r"\b(BOUFFEES?|INHALATIONS?|CP|COMPRIMES?|GELULES?|SPRAY|AEROSOL)\b"
    return re.sub(pattern, "", name, flags=re.IGNORECASE).strip()


def nettoyer_nom_affichage_medicament(name):
    if not name:
        return ""

    s = normalize_text(name)
    s = DOSE_PATTERN.sub("", s)
    s = re.sub(r"\b\d+[.,]?\d*\b", " ", s)
    s = re.sub(r"\b(MG|G|MCG|UG|ML|UI|MUI|CP|COMPRIME|COMPRIMES|GELULE|GELULES|AMP|AMPOULE|SACHET|SACHETS)\b", " ", s)
    s = re.sub(r"\s+", " ", s).strip(" -,:;")
    return clean_medicament_name(s)

def parser_commande_chat(message):
    msg = " ".join(str(message).strip().lower().split())

    # modifie + remplace
    m = re.match(
        r"^modifie(?:r)?\s+(?:la\s+)?(?:regle|règle)\s+(?P<categorie>.+?)\s*,?\s*remplace\s+(?P<old>.+?)\s+par\s+(?P<new>.+)$",
        msg,
        flags=re.IGNORECASE
    )
    if m:
        categorie = m.group("categorie").strip()
        categorie = re.sub(r"^(de|du|des|la|le|les)\s+", "", categorie, flags=re.IGNORECASE)
        return {
            "intent": "replace_value",
            "categorie": categorie,
            "old": m.group("old").strip(),
            "new": m.group("new").strip()
        }

    # modifie sans détail
    m = re.match(
        r"^modifie(?:r)?\s+(?:la\s+)?(?:regle|règle)\s+(?P<categorie>.+)$",
        msg,
        flags=re.IGNORECASE
    )
    if m:
        categorie = m.group("categorie").strip()
        categorie = re.sub(r"^(de|du|des|la|le|les)\s+", "", categorie, flags=re.IGNORECASE)
        return {
            "intent": "ask_update",
            "categorie": categorie
        }

    # affiche
    m = re.match(
        r"^affiche(?:r)?\s+(?:la\s+)?(?:regle|règle)\s+(?P<categorie>.+)$",
        msg,
        flags=re.IGNORECASE
    )
    if m:
        categorie = m.group("categorie").strip()
        categorie = re.sub(r"^(de|du|des|la|le|les)\s+", "", categorie, flags=re.IGNORECASE)
        return {
            "intent": "show_rule",
            "categorie": categorie
        }

    # supprime
    m = re.match(
        r"^supprime(?:r)?\s+(?:la\s+)?(?:regle|règle)\s+(?P<categorie>.+)$",
        msg,
        flags=re.IGNORECASE
    )
    if m:
        categorie = m.group("categorie").strip()
        categorie = re.sub(r"^(de|du|des|la|le|les)\s+", "", categorie, flags=re.IGNORECASE)
        return {
            "intent": "delete_rule",
            "categorie": categorie
        }

    return {"intent": "unknown"}

def remplacer_valeur_dans_objet(obj, old_value, new_value):
    modifie = False

    if isinstance(obj, dict):
        for k, v in obj.items():
            if isinstance(v, str) and v.strip().lower() == old_value.strip().lower():
                obj[k] = new_value
                modifie = True
            else:
                if remplacer_valeur_dans_objet(v, old_value, new_value):
                    modifie = True

    elif isinstance(obj, list):
        for i, item in enumerate(obj):
            if isinstance(item, str) and item.strip().lower() == old_value.strip().lower():
                obj[i] = new_value
                modifie = True
            else:
                if remplacer_valeur_dans_objet(item, old_value, new_value):
                    modifie = True

    return modifie


def preparer_modification_depuis_commande(cmd):
    data = charger_yaml_regles()

    if cmd["intent"] == "ask_update":
        idx, regle = trouver_regle_par_categorie(data, cmd["categorie"])

        if idx is None:
            return False, f"Règle introuvable : {cmd['categorie']}", None, None, None

        vrai_nom = regle.get("categorie", cmd["categorie"])
        return False, (
            f"Que veux-tu modifier dans la règle {vrai_nom} ?\n"
            f"Exemple : modifie la règle {vrai_nom}, remplace J0 matin par J-1"
        ), None, None, None

    if cmd["intent"] == "show_rule":
        idx, regle = trouver_regle_par_categorie(data, cmd["categorie"])
        if idx is None:
            return False, f"Règle introuvable : {cmd['categorie']}", None, None, None
        return True, "Règle trouvée.", "show_rule", regle.get("categorie", cmd["categorie"]), regle

    if cmd["intent"] == "delete_rule":
        idx, regle = trouver_regle_par_categorie(data, cmd["categorie"])
        if idx is None:
            return False, f"Règle introuvable : {cmd['categorie']}", None, None, None
        return True, "Suppression prête.", "delete_rule", regle.get("categorie", cmd["categorie"]), regle

    if cmd["intent"] == "replace_value":
        idx, regle = trouver_regle_par_categorie(data, cmd["categorie"])
        if idx is None:
            return False, f"Règle introuvable : {cmd['categorie']}", None, None, None

        nouveau_bloc = copy.deepcopy(regle)
        modifie = remplacer_valeur_dans_objet(nouveau_bloc, cmd["old"], cmd["new"])

        if not modifie:
            return False, f"Valeur '{cmd['old']}' introuvable dans la règle {regle.get('categorie', cmd['categorie'])}.", None, None, None

        ok, err = valider_bloc_regle(nouveau_bloc)
        if not ok:
            return False, f"Règle modifiée invalide : {err}", None, None, None

        return True, "Modification prête.", "update_rule", regle.get("categorie", cmd["categorie"]), nouveau_bloc

    return False, "Commande non reconnue.", None, None, None

# =========================================================
# OCR / DETECTION MEDICAMENTS
# =========================================================
DOSE_PATTERN = re.compile(r"\b\d+[.,]?\d*\s*(MG|G|MCG|UG|ML|UI|MUI)\b", re.IGNORECASE)

def contient_dose(ligne):
    return bool(DOSE_PATTERN.search(str(ligne)))

def est_ligne_posologie(ligne):
    l = normalize_text(ligne)
    mots_posologie = [
        "COMPRIME", "COMPRIMES", "GELULE", "GELULES", "SACHET", "SACHETS",
        "AMP", "AMPOULE", "LE MATIN", "LE SOIR", "MIDI", "PAR JOUR",
        "PAR SEMAINE", "PRISE", "INJECTION", "AU BESOIN", "SI BESOIN",
        "MATIN", "SOIR", "JOUR", "JOURS"
    ]
    return any(m in l for m in mots_posologie)

def est_ligne_non_medicamenteuse(ligne):
    l = normalize_text(ligne)

    if not l or len(l) < 2:
        return True

    stopwords = [
        "DOCTEUR", "DR", "CARDIOLOGUE", "NICE"
        "TEL", "TELEPHONE", "FAX", "EMAIL", "MAIL",
        "PLACE", "RUE", "AVENUE", "BOULEVARD",
        "PARIS", "LYON", "MARSEILLE", "TOULOUSE", "LILLE",
        "SIGNATURE", "CABINET", "MADAME", "MONSIEUR", "MME", "MR",
        "ORDONNANCE", "DATE", "MEDECIN", "RENOUVELABLE", 
        "MATIN", "MIDI", "SOIR", "JOUR", "JOURS", "SEMAINE", "SEMAINES",
        "CLINIQUE", "HOPITAL", "SERVICE", "TIMONE",
        "PRISE", "POSOLOGIE", "ADMINISTRATION",
        "SI BESOIN", "AU BESOIN"
    ]

    return any(f" {sw} " in f" {l} " for sw in stopwords)

def nettoyer_ligne_medicament(ligne):
    l = normalize_text(ligne)
    l = re.sub(r"^[\-\•\.\*\s]+", "", l)
    l = DOSE_PATTERN.sub("", l)
    l = re.sub(r"[^A-Z0-9\s\-]", " ", l)
    l = re.sub(r"\s+", " ", l).strip()
    return l

def nettoyer_ligne_medicament_manuscrit(ligne):
    l = normalize_text(ligne)
    l = re.sub(r"^[\-\•\.\*\s]+", "", l)
    l = DOSE_PATTERN.sub("", l)
    l = re.sub(r"\b\d+\b", " ", l)
    l = re.sub(r"\b(MG|ML|UI|MUI|G)\b", " ", l)
    l = re.sub(r"[^A-Z0-9\s\-]", " ", l)
    l = re.sub(r"\s+", " ", l).strip()
    return l




def nettoyer_texte(txt):
    txt = str(txt).lower()

    txt = re.sub(r"\(.*?\)", "", txt)
    txt = re.sub(r"(anti|systeme|chronique|classe|traitement)", "", txt)
    txt = re.sub(r"\b(de|du|des|la|le|les)\b", "", txt)
    txt = re.sub(r"\s+", " ", txt).strip()

    return txt

def regrouper_ocr_en_lignes(results, tol_y=18):
    items = []

    for r in results:
        box, text, conf = r
        if not text or not str(text).strip():
            continue

        xs = [p[0] for p in box]
        ys = [p[1] for p in box]

        items.append({
            "text": str(text).strip(),
            "x": min(xs),
            "y": sum(ys) / len(ys),
            "conf": conf
        })

    items = sorted(items, key=lambda z: (z["y"], z["x"]))

    lignes = []
    for item in items:
        placed = False
        for ligne in lignes:
            if abs(ligne["y_mean"] - item["y"]) <= tol_y:
                ligne["items"].append(item)
                ys = [it["y"] for it in ligne["items"]]
                ligne["y_mean"] = sum(ys) / len(ys)
                placed = True
                break

        if not placed:
            lignes.append({
                "y_mean": item["y"],
                "items": [item]
            })

    lignes_finales = []
    for ligne in lignes:
        ligne["items"] = sorted(ligne["items"], key=lambda z: z["x"])
        txt = " ".join([it["text"] for it in ligne["items"]]).strip()
        if txt:
            lignes_finales.append(txt)

    return lignes_finales

def decouper_texte_en_entrees_medicaments(txt):
    txt = str(txt).replace(",", "\n").replace(";", "\n")
    lignes = []

    for bloc in txt.split("\n"):
        bloc = bloc.strip()
        if not bloc:
            continue

        sous_blocs = re.split(r"\s+\+\s+|\s+\-\s+", bloc)

        for s in sous_blocs:
            s = s.strip()
            if s:
                lignes.append(s)

    return lignes

def extraire_lignes_candidates_imprime(txt):
    lignes = decouper_texte_en_entrees_medicaments(txt)
    candidates = []

    formes_galeniques = [
        "GEL", "CREME", "CRÈME", "POMMADE", "LOTION", "SOLUTION",
        "INHALATEUR", "SPRAY", "COLLYRE", "PATCH"
    ]

    for ligne in lignes:
        if est_ligne_non_medicamenteuse(ligne):
            continue
        if est_ligne_posologie(ligne):
            continue

        l_norm = normalize_text(ligne)
        a_dose = contient_dose(ligne)
        a_forme = any(f in l_norm for f in formes_galeniques)

        if not a_dose and not a_forme:
            pass

        ligne_nettoyee = nettoyer_ligne_medicament(ligne)
        if len(ligne_nettoyee) >= 4:
            candidates.append((ligne, ligne_nettoyee))

    return candidates

def extraire_lignes_candidates_manuscrit(txt):
    lignes = decouper_texte_en_entrees_medicaments(txt)
    candidates = []

    for ligne in lignes:
        if est_ligne_non_medicamenteuse(ligne):
            continue
        if est_ligne_posologie(ligne):
            continue

        ligne_nettoyee = nettoyer_ligne_medicament_manuscrit(ligne)
        nb_alpha = len(re.findall(r"[A-Z]", ligne_nettoyee))
        if len(ligne_nettoyee) >= 4 and nb_alpha >= 4:
            candidates.append((ligne, ligne_nettoyee))

    return candidates


def meilleur_match_medicament(candidate, ref):
    cand = normalize_text(candidate)

    mots_interdits = ["INHALATION", "BOUFFEES", "SPRAY", "AEROSOL"]
    if cand in mots_interdits:
        return None, 0

    if len(cand) < 3:
        return None, 0

    # 1. match exact prioritaire
    if cand in ref:
        return cand, 100

    # 2. pour les noms courts / un seul mot : très strict
    if len(cand.split()) == 1:
        match = process.extractOne(cand, ref, scorer=fuzz.ratio)
        if match:
            nom_match, score_match, _ = match

            # n'accepte que si quasi identique
            if score_match >= 96:
                return nom_match, score_match

        return None, 0

    # 3. pour les noms multi-mots : un peu plus souple
    best_name = None
    best_score = 0

    for r in ref:
        r_norm = normalize_text(r)

        if cand in r_norm or r_norm in cand:
            score = fuzz.WRatio(cand, r_norm)
            if score > best_score:
                best_name = r
                best_score = score

    if best_name is not None and best_score >= 94:
        return best_name, best_score

    match = process.extractOne(cand, ref, scorer=fuzz.WRatio)
    if match:
        nom_match, score_match, _ = match
        nom_match_norm = normalize_text(nom_match)

        cand_words = {w for w in cand.split() if len(w) >= 5}
        match_words = {w for w in nom_match_norm.split() if len(w) >= 5}
        mots_communs = cand_words & match_words

        if score_match >= 93 and mots_communs:
            return nom_match, score_match

    return None, 0

# =========================================================
# MOTEUR YAML
def conditions_match(ctx, regle, atc=None):
    conditions = regle.get("conditions", [])
    if not conditions:
        return None

    
    atc_clean = str(atc or "").upper().strip()
    

    def norm(t):
        s = str(t or "").strip()
        s = unidecode.unidecode(s)
        s = s.upper().replace("≤", "<=").replace("_", " ")
        s = re.sub(r"\s+", " ", s).strip()
        return s

    meilleure_cond = None

    for cond in conditions:
        if "if" not in cond:
            if cond.get("default") is True and meilleure_cond is None:
                meilleure_cond = cond
            continue

        bloc_if = cond.get("if", {}) or {}
        match_ok = True

        for cle, val_yaml in bloc_if.items():
            val_ctx = ctx.get(cle)

            if cle == "atc_codes":
                liste_codes = [str(c).upper().strip() for c in (val_yaml if isinstance(val_yaml, list) else [val_yaml])]
                if atc_clean not in liste_codes:
                    match_ok = False
                    break
                continue

            if cle == "atc_prefix":
                prefixes = [str(p).upper().strip() for p in (val_yaml if isinstance(val_yaml, list) else [val_yaml])]
                if not any(atc_clean.startswith(prefix) for prefix in prefixes):
                    match_ok = False
                    break
                continue
            if cle in ["type_chir", "type_chir_neuro"]:
                v_ctx_n = norm(ctx.get(cle))
                v_yaml_n = norm(val_yaml)

                if v_ctx_n != v_yaml_n:
                    match_ok = False
                    break
                continue

            if isinstance(val_yaml, bool):
                if bool(val_ctx) != val_yaml:
                    match_ok = False
                    break
                continue

            if isinstance(val_yaml, list):
                liste_vals = [norm(v) for v in val_yaml]
                if norm(val_ctx) not in liste_vals:
                    match_ok = False
                    break
                continue

            if norm(val_ctx) != norm(val_yaml):
                match_ok = False
                break

        if match_ok:
            return cond

    return meilleure_cond


        
def moteur_yaml(atc, ctx):
    atc = str(atc).upper().strip()
    liste_regles = REGLES.get("regles_medicaments") or []

    for famille in liste_regles:
        match_atc = False

        if famille.get("atc_prefix"):
            match_atc = any(atc.startswith(str(p).upper().strip()) for p in famille["atc_prefix"])

        if famille.get("atc_codes"):
            match_atc = atc in [str(c).upper().strip() for c in famille["atc_codes"]]

        if famille.get("conditions") and not famille.get("atc_prefix") and not famille.get("atc_codes"):
            match_atc = True

        if not match_atc:
            continue

        lien_sfar = ""
        source_url = famille.get("source_url")
        if source_url:
            if isinstance(source_url, list):
                lien_sfar = " | ".join([str(s).strip() for s in source_url if str(s).strip()])
            else:
                lien_sfar = str(source_url).strip()
        elif famille.get("sources"):
            sources = famille.get("sources", [])
            if isinstance(sources, list):
                lien_sfar = " | ".join([str(s).strip() for s in sources if str(s).strip()])
        elif famille.get("source_ref"):
            ref = famille.get("source_ref")
            source_table = REGLES.get("sources_regles", {})
            if ref in source_table:
                sources = source_table[ref].get("sources", [])
                if isinstance(sources, list):
                    lien_sfar = " | ".join([str(s).strip() for s in sources if str(s).strip()])

        res = {
            "action": famille.get("action", "POURSUITE"),
            "jour": famille.get("jour", "J0"),
            "note": famille.get("precision") or famille.get("note") or "-",
            "source": lien_sfar,
        }

        res_cond = conditions_match(ctx, famille, atc=atc)
        if res_cond:
            return {
                "action": res_cond.get("action", res["action"]),
                "jour": res_cond.get("jour", res["jour"]),
                "note": res_cond.get("precision") or res_cond.get("note") or res["note"],
                "source": lien_sfar,
            }

        if not famille.get("conditions"):
            return res

    return None

def moteur_global(atc, ctx):
    atc_clean = str(atc or "").upper().strip()
    ctx["corticoides"] = atc_clean.startswith("H02") or ctx.get("corticoides", False)

    ans_yaml = moteur_yaml(atc_clean, ctx)
    if ans_yaml:
        return ans_yaml

    return {
        "action": "POURSUITE",
        "jour": "J0",
        "note": "Aucune règle spécifique retrouvée dans le référentiel pour ce médicament.",
        "source": ""
    }
# =========================================================
# regles SFAR
# =========================================================
def moteur_expert_sfar(atc, ctx):
    """
    Transcription Python des règles YAML SFAR visibles dans le référentiel courant.
    Priorité clinique :
    1) règles très spécifiques
    2) règles générales
    3) défaut
    """
    atc = str(atc).upper().strip()

    # ----------------------------
    # Helpers contexte
    # ----------------------------
    def u(v):
        return str(v or "").upper().strip()

    type_chir = u(ctx.get("type_chir"))
    is_neurochir = ctx.get("type_chir_neuro") == "NEUROCHIR_INTRACRANIENNE"
    r_hem = u(ctx.get("r_hem"))
    alr = u(ctx.get("alr"))
    stress_chir = u(ctx.get("stress_chir"))
    ind_sraa = u(ctx.get("ind_sraa"))

    is_ambu = "AMBULATOIRE" in type_chir
    is_urg = "URGENCE" in type_chir
    alr_majore = alr in ["NEURAXIAL", "PROFOND"]

    dfg_ctx = ctx.get("dfg")
    dfg_connu_ctx = ctx.get("dfg_connu")

    if dfg_connu_ctx == "Oui" and dfg_ctx is not None:
        ctx["dfg_ge_50"] = dfg_ctx >= 50
        ctx["dfg_ge_30"] = dfg_ctx >= 30
        ctx["dfg_30_49"] = 30 <= dfg_ctx <= 49
    else:
        ctx["dfg_ge_50"] = False
        ctx["dfg_ge_30"] = False
        ctx["dfg_30_49"] = False


    prev_primaire = bool(ctx.get("prev_primaire", False))
    prev_secondaire = bool(ctx.get("prev_secondaire", False))
    bitherapie_aap = bool(ctx.get("bitherapie_aap", False))

    stent_1m = bool(ctx.get("stent_1m", False))
    stent_6m_haut_risque = bool(ctx.get("stent_6m_haut_risque", False))
    idm_6m = bool(ctx.get("idm_6m", False))

    aspirine_sup_200 = bool(ctx.get("aspirine_sup_200", False))
    dose_aspirine_inf_300 = bool(ctx.get("dose_aspirine_inf_300", False))

    # ----------------------------
    # 1. SRAA (IEC / ARA II)
    # ----------------------------
    if atc.startswith(("C09AA", "C09CA")):
        if ind_sraa == "HTA":
            return {
                "action": "ARRET",
                "jour": ">=12h",
                "note": "Risque d'hypotension peropératoire."
            }
        return {
            "action": "POURSUITE",
            "jour": "J0",
            "note": "Indication Insuffisance Cardiaque : maintien recommandé."
        }

    # Entresto
    if atc in ["C09DX04", "C09BA03"]:
        return {
            "action": "POURSUITE",
            "jour": "J-1",
            "precision": "SRAA(ARNI - sacubitril/valsartan) Sacubitril/valsartan (Entresto) : dernière prise la veille de l’intervention (J-1). Proposition faible (absence de recommandations établies).",
            
        }
        
     

    # ----------------------------
    # 2. Diurétiques
    # ----------------------------
    if atc.startswith("C03"):
        return {
            "action": "Pas de prise le matin",
            "jour": "J0 matin",
            "note": "Sauf si décompensation cardiaque aiguë."
        }

    # ----------------------------
    # 3. Antiarythmiques
    # ----------------------------
    if atc in ["C01BD01", "C07AA07"]:
        return {"action": "POURSUITE", "jour": "J0"}

    if atc.startswith(("C01BA", "C01BB", "C01BC")):
        return {"action": "ARRET", "jour": "J-1", "note": "Dernière prise J-1."}

    # ----------------------------
    # 4. Diabète
    # ----------------------------
    # Metformine
    if atc == "A10BA02":
        if is_ambu:
            return {"action": "POURSUITE", "jour": "J0", "note": "Metformine: poursuite si ambulatoire ou chirurgie courte avec ≤ 1 repas jeûné"}
        if is_urg:
            return {"action": "ARRET", "jour": "Immédiat", "note": "Metformine : arrêt immédiat en urgence."}
        return {"action": "STOP MATIN", "jour": "J0 matin", "note": "Metformine : ne pas prendre le matin si chirurgie avec ≥ 2 repas jeûnés."}

    # ADO
    if atc.startswith(("A10BB", "A10BX", "A10BF", "A10BH")):
        if is_ambu:
            return {"action": "POURSUITE", "jour": "J0", "note": "ADO : poursuite si ambulatoire ou chirurgie courte avec ≤ 1 repas jeûné."}
        if is_urg:
            return {"action": "ARRET", "jour": "Immédiat", "note": "ADO : arrêt immédiat en urgence."}
        return {"action": "STOP MATIN", "jour": "J0 matin", "note": "ADO : ne pas prendre le matin si chirurgie mineure ou majeure ou avec ≥ 2 repas jeûnés."}

    # SGLT2
    if atc.startswith("A10BK"):
        return {
            "action": "ARRET",
            "jour": "J-3",
            "note": "SGLT2 : dernière prise à J-3 (risque d’acidocétose euglycémique). Si prise à moins de J-3 : cétonémie capillaire obligatoire."
        }

    # GLP-1
    if atc.startswith("A10BJ"):
        if is_urg:
            return {
                "action": "ARRET",
                "jour": "Immédiat",
                "precision": "GLP-1 : arrêt immédiat en urgence. Risque d’estomac plein. Considérer le patient comme estomac plein, réaliser une échographie gastrique si possible et discuter une induction en séquence rapide."
            }
        if ctx.get("ind_glp1_obesite"):
            return {
                "action": "POURSUITE",
                "jour": "J0",
                "precision": "GLP-1 : que ce soit pour DT2 ou pour obésité, poursuite du traitement quel que soit le risque de la chirurgie (ambulatoire / mineure / majeure). Risque estomac plein. Favoriser ALR, surtout si signe de gastroparésie, ancienneté du DT2, microangiopathie ou autres traitements ralentissant la vidange gastrique. Si AG nécessaire, réaliser une échographie gastrique et discuter une induction en séquence rapide."
            }
        if ctx.get("ind_glp1_dt2"):
            return {
                "action": "POURSUITE",
                "jour": "J0",
                "precision": "GLP-1 : que ce soit pour DT2 ou pour obésité, poursuite du traitement quel que soit le risque de la chirurgie (ambulatoire / mineure / majeure). Risque estomac plein. Favoriser ALR, surtout si signe de gastroparésie, ancienneté du DT2, microangiopathie ou autres traitements ralentissant la vidange gastrique. Si AG nécessaire, réaliser une échographie gastrique et discuter une induction en séquence rapide."
            }
        return {
            "action": "POURSUITE",
            "jour": "J0",
            "precision": "GLP-1 : que ce soit pour DT2 ou pour obésité, poursuite du traitement quel que soit le risque de la chirurgie (ambulatoire / mineure / majeure). Risque estomac plein. Favoriser ALR, surtout si signe de gastroparésie, ancienneté du DT2, microangiopathie ou autres traitements ralentissant la vidange gastrique. Si AG nécessaire, réaliser une échographie gastrique et discuter une induction en séquence rapide."
        }

    # Insuline rapide ou mixte
    if atc.startswith(("A10AB", "A10AD")):
        if is_ambu:
            return {"action": "POURSUITE", "jour": "J0", "note": "Insuline rapide ou mixte : poursuite si ambulatoire ou chirurgie courte avec ≤ 1 repas jeûné."}
        if is_urg:
            return {"action": "ARRET", "jour": "Immédiat", "note": "Insuline rapide ou mixte : arrêt immédiat en urgence."}
        return {"action": "STOP MATIN", "jour": "J0 matin", "note": "Insuline rapide ou mixte : pas d’injection le matin si chirurgie mineure ou majeure ou avec ≥ 2 repas jeûnés."}

    # Insuline basale
    if atc.startswith("A10AE"):
        if is_ambu:
            return {"action": "POURSUITE", "jour": "J0", "note": "Insuline basale :poursuite si ambulatoire ou chirurgie courte avec ≤ 1 repas jeûné."}
        if is_urg:
            return {"action": "ARRET et relais insuline IVSE", "jour": "Immédiat", "note": "Insuline basale : arrêt et relais IVSE"}
        return {"action": "STOP MATIN", "jour": "J0 matin", "note": "Insuline basale : pas d'injection le matin, sauf chez le DT1 où l'injection doit être maintenue. Si jeûne et insuline lente injectée, perfusion de G10% 40 mL/h à partir du premier repas jeûné."}

    # Pompe à insuline
    if ctx.get("dispositif_insuline") == "pompe":
        if is_ambu:
            return {"action": "POURSUITE", "jour": "J0", "note": "Pompe à insuline : poursuite si ambulatoire ou chirurgie courte avec ≤ 1 repas jeûné. Maintien possible si intervention courte < 2 h, perturbations du contrôle glycémique non attendues, dispositif visible et à distance du champ opératoire, accord du patient et de l’équipe d’anesthésie, gestion précoce du matériel par le patient en postopératoire, et avis du diabétologue pour adaptation des débits."}
        if is_urg:
            return {"action": "ARRET et relais insuline IVSE", "jour": "Immédiat", "note": "Pompe à insuline : arrêt immédiat et relais IVSE."}
        return {"action": "ARRET DE LA POMPE AU BLOC", "jour": "J0", "note": "Pompe à insuline : perfusion de G10% 40 mL/h à partir du premier repas jeûné ; arrêt de la pompe au bloc ; relais IVSE."}




    # ----------------------------
    # 5. AAP
    # ----------------------------
    if atc in ["B01AC01", "B01AC06", "B01AC04", "B01AC24", "B01AC22"]:

        # Incohérence prévention primaire + P2Y12
        if prev_primaire and atc in ["B01AC04", "B01AC24", "B01AC22"]:
            return {
                "action": "INFO",
                "jour": "J0",
                "precision": "Incohérence : inhibiteur P2Y12 non indiqué en prévention primaire."
            }

        # Neurochirurgie
        if is_neurochir:
            if atc in ["B01AC01", "B01AC06"]:
                return {
                    "action": "ARRET",
                    "jour": "J-5",
                    "precision": "Neurochirurgie intracrânienne ou intrarachidienne : arrêt de l’aspirine à J-5."
                }
            if atc in ["B01AC04", "B01AC24"]:
                return {
                    "action": "ARRET",
                    "jour": "J-7",
                    "precision": "Neurochirurgie intracrânienne ou intrarachidienne : arrêt de l'anti P2Y12 à J-7."
                }
            if atc == "B01AC22":
                return {
                    "action": "ARRET",
                    "jour": "J-9",
                    "precision": "Neurochirurgie intracrânienne ou intrarachidienne : arrêt du prasugrel à J-9."
                }

        bitherapie_aap = ctx.get("bitherapie_aap", False)
        stent_1m = ctx.get("stent_1m", False)
        stent_6m_haut_risque = ctx.get("stent_6m_haut_risque", False)
        idm_6m = ctx.get("idm_6m", False)

        # Bithérapie haut risque thrombotique = priorité absolue
        if bitherapie_aap and (stent_1m or stent_6m_haut_risque or idm_6m):
            label = (
                "Stent ≤ 1 mois" if stent_1m else
                "Stent < 6 mois à haut risque thrombotique" if stent_6m_haut_risque else
                "IDM < 6 mois"
            )

            # FAIBLE
            if r_hem == "FAIBLE":
                return {
                    "action": "DIFFERER",
                    "jour": "J0",
                    "precision": f"{label} : différer la procédure. Si impossibilité, poursuivre les 2 AAP."
                }

            # INTERMEDIAIRE
            if r_hem == "INTERMEDIAIRE":
                if atc in ["B01AC01", "B01AC06"]:
                    return {
                        "action": "POURSUITE",
                        "jour": "J0",
                        "precision": f"{label} : si impossibilité de différer, poursuivre l’aspirine."
                    }
                if atc == "B01AC04":
                    return {
                        "action": "ARRET",
                        "jour": "J-5",
                        "precision": f"{label} : si impossibilité de différer, arrêt du clopidogrel à J-5."
                    }
                if atc == "B01AC24":
                    return {
                        "action": "ARRET",
                        "jour": "J-5",
                        "precision": f"{label} : si impossibilité de différer, arrêt du ticagrelor à J-5."
                    }
                if atc == "B01AC22":
                    return {
                        "action": "ARRET",
                        "jour": "J-7",
                        "precision": f"{label} : si impossibilité de différer, arrêt du prasugrel à J-7."
                    }

            # ELEVE / IMPORTANT / MAJEUR
            if r_hem in ["ELEVE", "IMPORTANT", "MAJEUR"]:
                if atc in ["B01AC01", "B01AC06"]:
                    txt = f"{label} : différer la procédure. Si impossibilité, arrêter les 2 AAP."
                    if stent_1m:
                        txt += " Envisager un relais par AAP injectable."
                    return {
                        "action": "ARRET",
                        "jour": "J-3",
                        "precision": txt
                    }
                if atc in ["B01AC04", "B01AC24"]:
                    txt = f"{label} : différer la procédure. Si impossibilité, arrêter les 2 AAP."
                    if stent_1m:
                        txt += " Envisager un relais par AAP injectable."
                    return {
                        "action": "ARRET",
                        "jour": "J-5",
                        "precision": txt
                    }
                if atc == "B01AC22":
                    txt = f"{label} : différer la procédure. Si impossibilité, arrêter les 2 AAP."
                    if stent_1m:
                        txt += " Envisager un relais par AAP injectable."
                    return {
                        "action": "ARRET",
                        "jour": "J-7",
                        "precision": txt
                    }

        # ALR
        if alr_majore:
            if atc in ["B01AC01", "B01AC06"]:
                if aspirine_sup_200:
                    return {
                        "action": "ARRET",
                        "jour": "J-3",
                        "precision": "Aspirine > 200 mg : arrêt à J-3. Préférer si possible la rachianesthésie en    ponction unique à la péridurale."
                    }
                return {
                    "action": "POURSUITE",
                    "jour": "J0",
                    "precision": "Aspirine ≤200 mg : autorisée pour ALR. Préférer rachianesthésie en ponction unique à la péridurale."
                }
            if atc in ["B01AC04", "B01AC24"]:
                return {"action": "ARRET", "jour": "J-5", "precision": "ALR : arrêt du P2Y12."}
            if atc == "B01AC22":
                return {"action": "ARRET", "jour": "J-7", "precision": "ALR : arrêt du prasugrel."}

        # Prévention primaire
        if prev_primaire:
            if atc in ["B01AC01", "B01AC06"] and r_hem == "FAIBLE":
                return {
                    "action": "DISCUTER",
                    "jour": "J0",
                    "precision": "Aspirine en prévention primaire : arrêt ou poursuite selon balance bénéfice-risque."
                }
            if atc in ["B01AC01", "B01AC06"]:
                return {
                    "action": "ARRET",
                    "jour": "J-3",
                    "precision": "Aspirine en prévention primaire : arrêt à J-3."
                }

        # Prévention secondaire
        if prev_secondaire:
            if atc in ["B01AC01", "B01AC06"]:
                if r_hem in ["FAIBLE", "INTERMEDIAIRE"]:
                    return {
                        "action": "POURSUITE",
                        "jour": "J0",
                        "precision": "Aspirine en prévention secondaire : poursuite."
                    }
                if r_hem in ["ELEVE", "IMPORTANT", "MAJEUR"]:
                    return {
                        "action": "ARRET",
                        "jour": "J-3",
                        "precision": "Aspirine en prévention secondaire : arrêt."
                    }

            if atc == "B01AC04":
                if r_hem == "FAIBLE":
                    return {
                        "action": "POURSUITE",
                        "jour": "J0",
                        "precision": "Clopidogrel : poursuite."
                    }
                if r_hem == "INTERMEDIAIRE":
                    return {
                        "action": "ARRET",
                        "jour": "J-5",
                        "precision": "Arrêt du clopidogrel entre 5 et 7 jours avant la chirurgie, avec un relais par aspirine 75 à 100mg le lendemain de l'arrêt. Proposition forte."
                    }
                if r_hem in ["ELEVE", "IMPORTANT", "MAJEUR"]:
                    return {
                        "action": "ARRET",
                        "jour": "J-5",
                        "precision": "Clopidogrel : arrêt."
                    }

        # Aspirine faible dose
        if atc in ["B01AC01", "B01AC06"] and dose_aspirine_inf_300:
            return {
                "action": "POURSUITE",
                "jour": "J0",
                "precision": "Aspirine ≤ 300 mg/j : ne pas réduire la posologie avant la chirurgie (proposition forte)."
            }

        # Aspirine 75–300 mg
        if atc in ["B01AC01", "B01AC06"] and prev_secondaire and r_hem in ["FAIBLE", "INTERMEDIAIRE"]:
            return {
                "action": "POURSUITE",
                "jour": "J0",
                "precision": "Aspirine (75–300 mg) : poursuite sans relais."
            }

        # Défaut AAP
        return {
            "action": "ARRET",
            "jour": "J-5",
            "precision": "Arrêt standard AAP."
        }



    # ----------------------------
    # 6. AOD
    # ----------------------------
    if atc.startswith(("B01AE", "B01AF")):
        if r_hem in ["FAIBLE", "NUL"]:
            return {
                "action": "ARRET",
                "jour": "J-1",
                "note": "Risque faible. Ne pas prendre la veille au soir et le matin de l'intervention."
            }

        if is_neurochir:
            if dfg_connu and dfg is not None and dfg >= 50:
                return {"action": "ARRET", "jour": "J-5", "note": "Neurochirurgie + Fonction rénale normale. Arrêt J-5."}
            return {
                "action": "DOSAGE",
                "note": "Si fonction rénale rénale altérée (= DFG < 30 pour les -xabans ou DFG < 50 pour dabigatran) alors dosage AOD veille intervention"
            }

        if atc.startswith("B01AF"):
            if dfg_connu and dfg is not None and dfg >= 30:
                return {"action": "ARRET", "jour": "J-3", "note": "Xaban + DFG >= 30. Dernière prise J-3."}

        if atc == "B01AE07":
            if dfg_connu and dfg is not None and dfg >= 50:
                return {"action": "ARRET", "jour": "J-4", "note": "Dabigatran + DFG >= 50. Dernière prise J-4."}
            if dfg_connu and dfg is not None and 30 <= dfg <= 49:
                return {"action": "ARRET", "jour": "J-5", "note": "Dabigatran + DFG entre 30 et 49. Dernière prise J-5."}

        if not dfg_connu:
            return {
                "action": "INFO",
                "note": "DFG INCONNU. Rappel : Riva/Apixa J-3 si DFG>30. Pradaxa J-4 si DFG>50 ou J-5 si 30-49. Neurochir J-5."
            }

        return {"action": "DOSAGE", "note": "Cas complexe : Vérifier le DFG et réaliser un dosage."}

    # ----------------------------
    # 7. AINS
    # ----------------------------
    if atc.startswith("M01A"):
        if r_hem in ["ELEVE", "IMPORTANT", "MAJEUR"]:
            return {"action": "ARRET", "note": "Arrêt selon 5 demi-vies."}
        return {"action": "POURSUITE"}

    # ----------------------------
    # 8. Psy / neuro
    # ----------------------------
    if atc == "M03BX01":
        if u(ctx.get("voie_baclofene")) == "PER_OS":
            return {"action": "POURSUITE"}
        return {"action": "DISCUTER", "note": "Si voie intrathécale : avis spécialisé."}

    if atc in ["N06AA04", "N06AA09"]:
        if ctx.get("atcd_cv"):
            return {"action": "ARRET", "jour": "J-5"}
        if ctx.get("ASA") in [1, 2]:
            return {"action": "POURSUITE"}
        return {"action": "INFO_MANQUANTE"}

    # ----------------------------
    # 9. Corticoïdes
    # ----------------------------
    if atc.startswith("H02AB"):
        if stress_chir == "MINEUR":
            return {"action": "POURSUITE"}
        if stress_chir == "MODERE":
            return {"action": "POURSUITE", "note": "Dose habituelle + Hydrocortisone 50mg."}
        if stress_chir == "MAJEUR":
            return {"action": "POURSUITE", "note": "Dose habituelle + Hydrocortisone 100mg."}
        return {"action": "POURSUITE"}

    # ----------------------------
    # 10. AVK
    # ----------------------------
    if atc.startswith("B01AA"):
        if ctx.get("valve_mecanique") or ctx.get("acfa_atcd") or ctx.get("mtev_haut_risque"):
            return {
                "action": "ARRET",
                "jour": "J-5",
                "note": "Relais curatif requis (HBPM 2/j ou HNF)."
            }

        if r_hem == "FAIBLE":
            return {
                "action": "POURSUITE",
                "jour": "J0",
                "note": "Si INR entre 2 et 3."
            }

        if is_neurochir:
            return {
                "action": "ARRET",
                "jour": "J-5",
                "note": "Objectif INR < 1.2."
            }

        return {
            "action": "ARRET",
            "jour": "J-5",
            "note": "Objectif INR < 1.5."
        }

    # ----------------------------
    # Défaut global
    # ----------------------------
    return {"action": "POURSUITE", "jour": "J0", "note": "Médicament reconnu dans le référentiel, sans règle spécifique identifiée : poursuite, sans impact anesthésique évident, à vérifier selon le contexte clinique."}




def get_classe(atc, classe_map):
    if not atc:
        return "Inconnue"

    atc = str(atc).upper().strip()

    # 1. match exact
    if atc in classe_map:
        return classe_map[atc]

    # 2. fallback par préfixe (A10BJ06 → A10BJ)
    for i in range(len(atc), 2, -1):
        prefix = atc[:i]
        if prefix in classe_map:
            return classe_map[prefix]

    # 3. fallback logique
    if atc.startswith("A10"):
        return "Antidiabétique"

    return "Inconnue"



# =========================
# DETECTION CONTEXTE CORTICOIDES
# =========================
def contexte_corticoide_detecte(txt, ref, atc_map):
    if not txt or not str(txt).strip():
        return False

    candidats = []

    for brute, nettoyee in extraire_lignes_candidates_imprime(txt):
        candidats.append(nettoyee)

    for brute, nettoyee in extraire_lignes_candidates_manuscrit(txt):
        candidats.append(nettoyee)

    for cand in candidats:
        meilleur_nom, meilleur_score = meilleur_match_medicament(cand, ref)

        if meilleur_nom and meilleur_score >= 75:
            atc = str(atc_map.get(meilleur_nom, "")).upper().strip()
            if atc.startswith("H02"):
                return True

    txt_norm = normalize_text(txt)
    mots_cortico = [
        "PREDNISONE", "PREDNISOLONE", "HYDROCORTISONE",
        "DEXAMETHASONE", "METHYLPREDNISOLONE",
        "BETAMETHASONE", "CORTANCYL", "SOLUPRED",
        "MEDROL", "CELESTENE", "SOLUMEDROL"
    ]

    return any(mot in txt_norm for mot in mots_cortico)


def contexte_famille_detecte(txt, ref, atc_map, atc_prefixes=None, atc_codes=None, mots_secours=None):
    if not txt or not str(txt).strip():
        return False

    atc_prefixes = atc_prefixes or []
    atc_codes = atc_codes or []
    mots_secours = mots_secours or []

    candidats = []

    for brute, nettoyee in extraire_lignes_candidates_imprime(txt):
        candidats.append(nettoyee)

    for brute, nettoyee in extraire_lignes_candidates_manuscrit(txt):
        candidats.append(nettoyee)

    for cand in candidats:
        meilleur_nom, meilleur_score = meilleur_match_medicament(cand, ref)

        if meilleur_nom and meilleur_score >= 75:
            atc = str(atc_map.get(meilleur_nom, "")).upper().strip()

            if atc in [str(c).upper().strip() for c in atc_codes]:
                return True

            if any(atc.startswith(str(p).upper().strip()) for p in atc_prefixes):
                return True

    txt_norm = normalize_text(txt)
    return any(normalize_text(mot) in txt_norm for mot in mots_secours)


def ressemble_a_un_medicament(txt):
    if not txt:
        return False

    t = str(txt).strip()
    tn = normalize_text(t)

    if len(tn) < 4:
        return False

    # trop de mots = souvent pas un médicament
    mots = t.split()
    if len(mots) > 2:
        return False

    # lignes de posologie / texte médical non médicament
    mots_interdits = [
        "MEDECIN", "DOCTEUR", "DR", "GENERALISTE", "DERMATOLOGUE",
        "CABINET", "CENTRE", "RUE", "AVENUE", "BOULEVARD",
        "TEL", "TELEPHONE", "MAIL","NICE",
        "APPLICATION", "APPLIQUER", "BOUFFEE", "BOUFFEES",
        "GELULE", "GELULES", "COMPRIME", "COMPRIMES",
        "MATIN", "MIDI", "SOIR", "JOUR", "JOURS",
        "SEMAINE", "SEMAINES", "PENDANT", "FOIS",
        "TRAITER", "ZONE", "SI", "SUR", "AVEC"
    ]

    if any(mot in tn for mot in mots_interdits):
        return False


    # mots trop courts ou suite bizarre type "C MARTIN"
    if len(mots) == 2:
        if len(mots[0]) <= 2 or len(mots[1]) <= 2:
            return False

    # lettres minimum
    nb_lettres = sum(ch.isalpha() for ch in t)
    if nb_lettres < 4:
        return False

    return True




def extraire_nom_medicament_debut_ligne(txt):
    if not txt:
        return ""

    t = str(txt).strip()

    separateurs = [
        " mg", " g", " ml", " µg", " mcg", " ui", "%",
        " comprimé", " comprimés", " cp", " gélule", " gélules",
        " sachet", " sachets", " ampoule", " ampoules",
        " gel", " crème", " creme", " pommade", " spray",
        " matin", " midi", " soir",
        " 1/jour", " 2/jour", " 3/jour",
        " bouffée", " bouffées", " si besoin",
        " application", " appliquer"
    ]

    t_lower = " " + t.lower()

    positions = []
    for sep in separateurs:
        pos = t_lower.find(sep)
        if pos != -1:
            positions.append(pos)

    if positions:
        cut = min(positions)
        nom = t[:cut].strip(" -,:;")
    else:
        nom = t.strip(" -,:;")

    return nom








def detecter_medicaments_depuis_texte(txt, ref, atc_map, classe_map, ctx):
    resultats = []
    vus = set()
    vus_resultats = set()

    candidats_imprime = extraire_lignes_candidates_imprime(txt)
    candidats_manuscrit = extraire_lignes_candidates_manuscrit(txt)

    tous_candidats = []
    deja = set()

    for brute, nettoyee in candidats_imprime:
        key = (normalize_text(brute), normalize_text(nettoyee))
        if key not in deja:
            tous_candidats.append((brute, nettoyee, "imprime"))
            deja.add(key)

    for brute, nettoyee in candidats_manuscrit:
        key = (normalize_text(brute), normalize_text(nettoyee))
        if key not in deja:
            tous_candidats.append((brute, nettoyee, "manuscrit"))
            deja.add(key)

    for brute, nettoyee, mode in tous_candidats:
        meilleur_nom, meilleur_score = meilleur_match_medicament(nettoyee, ref)
        seuil = 85 if mode == "imprime" else 75

        # =========================
        # CAS 1 — PAS DE MATCH BASE FIABLE
        # =========================
        if not meilleur_nom or meilleur_score < seuil:
            nom_affiche = nettoyer_nom_affichage_medicament(brute if brute else nettoyee)

            if not ressemble_a_un_medicament(nom_affiche):
                continue

            cle_resultat = ("INCONNU", normalize_text(nom_affiche))
            if cle_resultat in vus_resultats:
                continue
            vus_resultats.add(cle_resultat)

            resultats.append({
                "Médicament": nom_affiche,
                "Code ATC": "",
                "Classe": "",
                "Action": "AVIS SPECIALISE",
                "Date": "-",
                "Note": " ATC non reconnu,médicament absent du référentiel ATC/libellé : laisser la décision à l’anesthésiste.",
                "Lien": ""
            })
            continue

        ligne_upper = normalize_text(brute)
        nom_upper = normalize_text(meilleur_nom)
        mots_nom = nom_upper.split()
        mots_significatifs = [m for m in mots_nom if len(m) >= 5]

        if mots_significatifs:
            if not any(m in ligne_upper for m in mots_significatifs):
                if meilleur_score < 92:
                    nom_affiche = nettoyer_nom_affichage_medicament(
                        extraire_nom_medicament_debut_ligne(brute if brute else nettoyee)
                    )

                    if not ressemble_a_un_medicament(nom_affiche):
                        continue

                    cle_resultat = ("INCONNU", normalize_text(nom_affiche))
                    if cle_resultat in vus_resultats:
                        continue
                    vus_resultats.add(cle_resultat)

                    resultats.append({
                        "Médicament": nom_affiche,
                        "Code ATC": "",
                        "Classe": "",
                        "Action": "AVIS SPECIALISE",
                        "Date": "-",
                        "Note": "ATC non reconnu,médicament absent du référentiel ATC/libellé : laisser la décision à l’anesthésiste.",
                        "Lien": ""
                    })
                    continue

        atc = atc_map.get(meilleur_nom)
        ans = None

        # =========================
        # CAS 2 — MÉDICAMENT RECONNU MAIS PAS D’ATC
        # =========================
        if not atc or str(atc).upper() == "NAN":
            nom_affiche = nettoyer_nom_affichage_medicament(meilleur_nom)

            cle_resultat = ("INCONNU", normalize_text(nom_affiche))
            if cle_resultat in vus_resultats:
                continue
            vus_resultats.add(cle_resultat)

            resultats.append({
                "Médicament": nom_affiche,
                "Code ATC": "",
                "Classe": "",
                "Action": "AVIS SPECIALISE",
                "Date": "-",
                "Note": "ATC non reconnu,médicament absent du référentiel ATC/libellé : laisser la décision à l’anesthésiste.",
                "Lien": ""
            })
            continue

        # =========================
        # CAS 3 — ATC TROUVÉ
        # =========================
        atc = str(atc).upper().strip()

        if atc in vus:
            continue

        vus.add(atc)
        ans = moteur_global(atc, ctx)

        if not ans:
            ans = {
                "action": "POURSUITE",
                "jour": "J0",
                "note": "Médicament reconnu dans le référentiel, sans règle spécifique identifiée : poursuite, sans impact anesthésique évident, à vérifier selon le contexte clinique.",
                "source": ""
            }

        atc_affiche = atc
        classe_affiche = get_classe(atc_affiche, classe_map)
        nom_resultat = nettoyer_nom_affichage_medicament(brute or meilleur_nom)

        cle_resultat = atc_affiche if atc_affiche else normalize_text(nom_resultat)
        if cle_resultat in vus_resultats:
            continue
        vus_resultats.add(cle_resultat)

        resultats.append({
            "Médicament": nom_resultat,
            "Code ATC": atc_affiche,
            "Classe": ans.get("classe", classe_affiche),
            "Action": ans.get("action", "POURSUITE"),
            "Date": ans.get("jour", "J0"),
            "Note": ans.get("note") or ans.get("precision") or "-",
            "Lien": str(ans.get("source", "")).strip()
        })

    debug_candidates = [(b, n, m) for b, n, m in tous_candidats]
    return resultats, vus, debug_candidates

# =========================================================
# CHARGEMENT DES DONNEES
# =========================================================

def get_stress_cortico_from_id(id_acte, df_inter_cortico):
    if not id_acte or df_inter_cortico.empty:
        return False, "", ""

    id_acte = str(id_acte).strip().upper()

    df_tmp = df_inter_cortico.copy()
    df_tmp.columns = [str(c).strip().upper() for c in df_tmp.columns]

    if "ID" not in df_tmp.columns:
        return False, "", ""

    df_tmp["ID"] = df_tmp["ID"].astype(str).str.strip().str.upper()

    row = df_tmp[df_tmp["ID"] == id_acte]

    if row.empty:
        return False, "", ""

    stress_raw = str(row.iloc[0].get("STRESS CHIR CORTICO", "")).strip()
    stress_norm = stress_raw.upper()

    stress_faible = stress_norm == "FAIBLE"

    return stress_faible, stress_raw, stress_norm



@st.cache_data
def load_data():
    try:
        atc = pd.read_csv(os.path.join(BASE_DIR, "dci_atc_fichier.csv"), sep=";")
        inter = pd.read_csv(os.path.join(BASE_DIR, "Feuille1-Tableau 1.csv"), sep=";")
        inter_cortico = pd.read_csv(os.path.join(BASE_DIR, "Feuil1-Tableau 1cortico.csv"), sep=";")
        taxo = pd.read_csv(os.path.join(BASE_DIR, "TAXONOMIE-Tableau 1.csv"), sep=";")
        libelles = pd.read_csv(os.path.join(BASE_DIR, "LISTE_FINALE_AVEC_LIBELLES.csv"), sep=";")
        sentinelles = pd.read_csv(os.path.join(BASE_DIR, "Medicaments Sentinelles-Tableau.csv"), sep=";")
        profils = pd.read_csv(os.path.join(BASE_DIR, "Profils Pathologiques-Tableau.csv"), sep=";")

        atc.columns = [normalize_colname(c) for c in atc.columns]
        inter.columns = [normalize_colname(c) for c in inter.columns]
        inter_cortico.columns = [normalize_colname(c) for c in inter_cortico.columns]
        taxo.columns = [normalize_colname(c) for c in taxo.columns]
        libelles.columns = [normalize_colname(c) for c in libelles.columns]
        sentinelles.columns = [normalize_colname(c) for c in sentinelles.columns]
        profils.columns = [normalize_colname(c) for c in profils.columns]

        atc_map = {}

        # pour éviter les bugs accents / espaces
        def norm(x):
            return normalize_text(x)

        # 1. MEDICAMENT_SOURCE
        for k, v in zip(atc["MEDICAMENT_SOURCE"], atc["CODE_ATC"]):
            if pd.notna(k) and pd.notna(v):
                atc_map[norm(k)] = str(v).upper().strip()

        # 2. NOM_COMMERCIAL (IMPORTANT)
        if "NOM_COMMERCIAL" in atc.columns:
            for k, v in zip(atc["NOM_COMMERCIAL"], atc["CODE_ATC"]):
                if pd.notna(k) and pd.notna(v):
                    atc_map[norm(k)] = str(v).upper().strip()

        # 3. DCI (IMPORTANT)
        if "DCI" in atc.columns:
            for k, v in zip(atc["DCI"], atc["CODE_ATC"]):
                if pd.notna(k) and pd.notna(v):
                    atc_map[norm(k)] = str(v).upper().strip()
  
        # nettoyage doublons / erreurs critiques
        corrections = {
            "FLUINDIONE": "B01AA12",
            "PREVISCAN": "B01AA12",
            "SINTROM": "B01AA07",
        }

        for k, v in corrections.items():
            atc_map[norm(k)] = v

        classe_map = {
            str(k).upper().strip(): str(v).strip()
            for k, v in zip(libelles["CODE_ATC"], libelles["NOM_DEUXIEME_CLASSE"])
        }

        ref = list(atc_map.keys())

        return atc_map, ref, classe_map, inter, inter_cortico, taxo, sentinelles, profils

    except Exception as e:
        st.error(f"Erreur fichiers CSV : {e}")
        return {}, [], {}, pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()


atc_map, ref, classe_map, df_inter, df_inter_cortico, df_taxo, df_sentinelles, df_profils = load_data()

# =========================================================
# PROFILS PATHOLOGIQUES
# =========================================================
@st.cache_data
def prepare_profile_tables(df_sentinelles, df_profils):
    if df_sentinelles.empty or df_profils.empty:
        return pd.DataFrame(), pd.DataFrame()

    df_s = df_sentinelles.copy()
    df_p = df_profils.copy()

    if "CODE ATC" in df_s.columns:
        df_s["CODE ATC"] = df_s["CODE ATC"].astype(str).str.upper().str.strip()

    if "PROFIL IDENTIFIE" in df_s.columns:
        df_s["PROFIL IDENTIFIE_NORM"] = df_s["PROFIL IDENTIFIE"].apply(normalize_text)

    if "PROFIL PATHOLOGIQUE" in df_p.columns:
        df_p["PROFIL PATHOLOGIQUE_NORM"] = df_p["PROFIL PATHOLOGIQUE"].apply(normalize_text)

    return df_s, df_p

df_sentinelles_ready, df_profils_ready = prepare_profile_tables(df_sentinelles, df_profils)

def inferer_profils_structures(codes_atc_detectes, df_sentinelles_ready, df_profils_ready):
    if not codes_atc_detectes or df_sentinelles_ready.empty or df_profils_ready.empty:
        return pd.DataFrame()

    codes_atc_detectes = [str(c).upper().strip() for c in codes_atc_detectes]
    hits = df_sentinelles_ready[df_sentinelles_ready["CODE ATC"].isin(codes_atc_detectes)].copy()

    if hits.empty:
        return pd.DataFrame()

    profile_scores = defaultdict(float)
    profile_atc = defaultdict(list)
    profile_certitudes = defaultdict(list)
    profile_sentinelles = defaultdict(list)

    poids_certitude = {
        "TRES ELEVEE": 3.0,
        "ELEVEE": 2.5,
        "MOYENNE": 1.5,
        "FAIBLE": 1.0
    }

    for _, row in hits.iterrows():
        profil = clean_display_value(row.get("PROFIL IDENTIFIE", ""))
        profil_norm = normalize_text(profil)
        atc = clean_display_value(row.get("CODE ATC", ""))
        medic = clean_display_value(row.get("MEDICAMENT SENTINELLE", ""))
        certitude = normalize_text(clean_display_value(row.get("CERTITUDE", "MOYENNE")))
        seul_suffit = normalize_text(clean_display_value(row.get("SEUL SUFFIT", "NON")))

        score = poids_certitude.get(certitude, 1.5)
        if seul_suffit == "OUI":
            score += 1.0

        profile_scores[profil_norm] += score
        profile_atc[profil_norm].append(atc)
        profile_certitudes[profil_norm].append(certitude)
        profile_sentinelles[profil_norm].append(medic)

    rows = []
    for profil_norm, score in profile_scores.items():
        sub = df_profils_ready[df_profils_ready["PROFIL PATHOLOGIQUE_NORM"] == profil_norm]

        if not sub.empty:
            rowp = sub.iloc[0]
            libelle = corriger_nom_profil(clean_display_value(rowp.get("PROFIL PATHOLOGIQUE", "")))
            asa_min = clean_display_value(rowp.get("ASA MIN", ""))
        else:
            libelle = corriger_nom_profil(profil_norm.title())
            asa_min = ""

        certs = sorted(set(profile_certitudes[profil_norm]))
        if "TRES ELEVEE" in certs:
            niveau = "Très forte"
        elif "ELEVEE" in certs:
            niveau = "Forte"
        else:
            niveau = "Modérée"

        rows.append({
            "Profil": libelle,
            "Score": score,
            "Niveau": niveau,
            "ASA min": asa_min,
            "ATC": ", ".join(sorted(set(profile_atc[profil_norm]))),
            "Sentinelles": ", ".join(sorted(set(profile_sentinelles[profil_norm])))
        })

    return pd.DataFrame(rows).sort_values("Score", ascending=False).head(3).reset_index(drop=True)

#=========================================================
# ASSISTANT INTELLIGENT & MODIFICATION YAML
# =========================================================
def appliquer_modification_yaml(intent, categorie_cible, nouveau_bloc_dict=None):
    file_path = os.path.join(BASE_DIR, "regles_sfar.yaml")

    try:
        data = charger_yaml_regles()
        regles = data.get("regles_medicaments", [])

        idx, ancienne_regle = trouver_regle_par_categorie(data, categorie_cible)

        if intent in ["add_rule", "update_rule"]:
            ok, err = valider_bloc_regle(nouveau_bloc_dict)
            if not ok:
                return False, err

        if intent == "update_rule":
            if idx is None:
                return False, f"Règle introuvable : {categorie_cible}"
            regles[idx] = nouveau_bloc_dict

        elif intent == "add_rule":
            regles.append(nouveau_bloc_dict)

        elif intent == "delete_rule":
            if idx is None:
                return False, f"Règle introuvable : {categorie_cible}"
            regles.pop(idx)

        else:
            return False, f"Intent inconnu : {intent}"

        data["regles_medicaments"] = regles

        with open(file_path, "w", encoding="utf-8") as f:
            yaml.dump(data, f, allow_unicode=True, sort_keys=False, indent=2)

        with open(file_path, "r", encoding="utf-8") as f:
            yaml.safe_load(f)

        return True, "Modification enregistrée."

    except Exception as e:
        return False, str(e)

# =========================================================
# CHAT DANS LA SIDEBAR
# =========================================================

if "messages" not in st.session_state:
    st.session_state.messages = []

if "proposition_ia" not in st.session_state:
    st.session_state.proposition_ia = None

# =========================================================
# SIDEBAR
# =========================================================
with st.sidebar:
    st.header("Dossier Patient")
    iep = st.text_input("Numéro IEP")
    age = st.number_input("Âge", 0, 115, 65)
    date_op = st.date_input("Date intervention", date.today() + timedelta(days=7))




    type_chir = "MINEURE"


    st.divider()
    st.header("Chirurgie")

    if not df_taxo.empty and "SPECIALITE" in df_taxo.columns:
        spe = st.selectbox("Spécialité", sorted(df_taxo["SPECIALITE"].dropna().unique()))
        df_grp = df_taxo[df_taxo["SPECIALITE"] == spe]

        grp = st.selectbox("Groupe", sorted(df_grp["GROUPE"].dropna().unique()))
        code_grp_series = df_taxo[
            (df_taxo["SPECIALITE"] == spe) & (df_taxo["GROUPE"] == grp)
        ]["CODE GRP"]

        code_grp = code_grp_series.iloc[0] if not code_grp_series.empty else None

        if code_grp is not None and not df_inter.empty and "ID" in df_inter.columns:
            code_grp = str(code_grp).strip().upper()

            row_grp = df_taxo[
                (df_taxo["SPECIALITE"] == spe) & (df_taxo["GROUPE"] == grp)
            ]

            code_spe = str(row_grp["CODE SPE"].iloc[0]).strip().upper()
            ids_inter = df_inter["ID"].astype(str).str.strip().str.upper()

            mask_acte = ids_inter.str.startswith(code_spe + "-" + code_grp + "-")
            df_actes_filtre = df_inter[mask_acte].copy()

            liste_actes = sorted(
                df_actes_filtre["INTERVENTION CHIRURGICALE"].dropna().unique()
            )
        else:
            df_actes_filtre = pd.DataFrame()
            liste_actes = []

        acte_nom = st.selectbox(
            "Intervention",
            liste_actes if liste_actes else ["Aucune intervention trouvée"],
            key="intervention_chirurgie"
        )

        # Détection automatique neurochirurgie / rachis
        is_neuro = False
        if spe is not None and str(spe).strip().upper() in ["NEUROCHIRURGIE", "RACHIS"]:
            is_neuro = True

        if not df_actes_filtre.empty and acte_nom != "Aucune intervention trouvée":
            data_acte_filtre = df_actes_filtre[
                df_actes_filtre["INTERVENTION CHIRURGICALE"] == acte_nom
            ].copy()

           
        if not data_acte_filtre.empty:
            data_acte = data_acte_filtre.iloc[0]

         

            risque_acte = val_upper(data_acte.get("RISQUE HEMORRAGIQUE", "NON RENSEIGNE"))
            controle_hemorragique = val_upper(data_acte.get("CONTROLE HEMORRAGIQUE", "STANDARD"))
            asa_acte = clean_display_value(data_acte.get("ASA", ""))
            antibio = data_acte.get("ANTIBIOPROPHYLAXIE", "Non renseignée")
            dose_antibio = data_acte.get("DOSE", "Non renseignée")

            id_acte = clean_display_value(data_acte.get("ID", ""))

            id_acte = clean_display_value(data_acte.get("ID", "")).strip().upper()

        # LISTE DES CHIRURGIES A STRESS FAIBLE
            ids_faibles = [
                "OPH-ANT-01",   # cataracte
                "THO-END-05",   # endoscopie
      
            ]


            if id_acte in ids_faibles:
                stress_chir = "faible"
            else:
                stress_chir = "modéré/élevé"







            stress_cortico_faible, stress_cortico_raw, stress_cortico_norm = get_stress_cortico_from_id(
                id_acte,
                df_inter_cortico
            )
            
            stress_chir = "faible" if stress_cortico_faible else "modéré/élevé"

        else:
            data_acte = pd.Series(dtype=object)
            risque_acte = "NON RENSEIGNE"
            controle_hemorragique = "STANDARD"
            asa_acte = ""
            antibio = "Non renseignée"
            dose_antibio = "Non renseignée"
            stress_cortico_raw = ""
            stress_cortico_norm = ""
            stress_cortico_faible = False
            stress_chir = "modéré/élevé"





    else:
        spe = None
        grp = None
        acte_nom = None
        is_neuro = False
        data_acte = pd.Series(dtype=object)
        risque_acte = "NON RENSEIGNE"
        controle_hemorragique = "STANDARD"
        asa_acte = ""
        antibio = "Non renseignée"
        dose_antibio = "Non renseignée"
        stress_cortico_raw = ""
        stress_cortico_norm = ""
        stress_cortico_faible = False
        st.warning("Taxonomie chirurgie indisponible ou colonnes non reconnues.")

   


   
    type_alr = st.selectbox("ALR prévue", ["AUCUNE", "SUPERFICIEL", "NEURAXIAL", "PROFOND"])



#----------------------------
#---- RAPPEL ALR PROFONDES
#-----------------------------
    if type_alr == "PROFOND":
        with st.expander(" Rappel – ALR profondes"):
            st.markdown("""
- Ganglion stellaire  
- Plexus cervical profond  
- Paravertébral cervical  
- Infraclaviculaire  
- Épidural  
- Paravertébral thoracique  
- Plexus lombaire  
- Compartiment psoas  
- Sympathectomie lombaire  
- Paravertébral lombaire  
- Quadratus lumborum  
- Fascia transversalis  
- Plexus sacré  
- PENG (Pericapsular Nerve Group)  
- Sciatique (approches proximales)  
- Spinal  
- Épidural  
            """)
# ----------------------------
# RAPPEL ALR SUPERFICIELLES
# ----------------------------
    if type_alr == "SUPERFICIEL":
        with st.expander(" Rappel – ALR superficielles"):
            st.markdown("""
- Occipital  
- Péribulbaire  
- Sub-Ténon  
- Plexus cervical superficiel  
- Interscalénique  
- Supraclaviculaire  
- Axillaire  
- Suprascapulaire  
- Ulnaire, radial, médian (avant-bras ou poignet)  
- Parasternal (intercostal, profond ou superficiel)  
- Serratus anterior (profond ou superficiel)  
- Erector spinae plane  
- Intercostal  
- Interpectoral plane  
- Pecto-serratus plane  
- Ilio-inguinal  
- Ilio-hypogastrique  
- TAP block  
- Gaine des droits  
- Branche génitale du nerf génito-fémoral  
- Nerf pudendal  
- Fémoral  
- Triangle fémoral  
- Canal des adducteurs  
- Sciatique (sous-glutéal, poplité)  
- Fascia iliaca  
- Nerf cutané latéral de la cuisse  
- Branche fémorale du nerf génito-fémoral  
- Sural, saphène, tibial, fibulaire (profond ou superficiel)  
            """)



    








    st.divider()
    st.header("Contexte patient / chirurgie")

    st.subheader("Fonction rénale")
    st.caption("DFG utile notamment pour l’adaptation des anticoagulants oraux d'action direct (AOD) en contexte périopératoire.")

    dfg_connu = st.radio(
        "DFG connu ?",
        ["Oui", "Non"],
        index=0
    )

    dfg = None
    if dfg_connu == "Oui":
        dfg = st.number_input(
            "DFG (mL/min)",
            min_value=0,
            max_value=200,
            value=80,
            step=1
        )






  
    


with st.sidebar:
    st.divider()
    st.subheader("💬 Assistant Expert SFAR")

    # affichage messages style bulles simples
    chat_container = st.container()

    with chat_container:
        for message in st.session_state.messages:
            if message["role"] == "user":
                st.markdown(
                    f"""
                    <div style="
                        background:#DCF8C6;
                        padding:10px 14px;
                        border-radius:14px;
                        margin:8px 0;
                        text-align:left;
                    ">
                        <b>Toi :</b><br>{message["content"]}
                    </div>
                    """,
                    unsafe_allow_html=True
                )
            else:
                st.markdown(
                    f"""
                    <div style="
                        background:#F1F0F0;
                        padding:10px 14px;
                        border-radius:14px;
                        margin:8px 0;
                        text-align:left;
                    ">
                        <b>IA :</b><br>{message["content"]}
                    </div>
                    """,
                    unsafe_allow_html=True
                )

    prompt = st.text_input(
        "Écris ta demande",
        key="sidebar_chat_input",
        placeholder="Ex: modifie la règle Metformine"
    )

    col1, col2 = st.columns(2)

    with col1:
        envoyer = st.button("Envoyer", use_container_width=True)

    with col2:
        vider = st.button("Vider", use_container_width=True)

   
    if vider:
        st.session_state.messages = []
        st.session_state.proposition_ia = None
        st.rerun()
    
    if envoyer and prompt.strip():
        st.session_state.messages.append({"role": "user", "content": prompt})

        cmd = parser_commande_chat(prompt)
        ok, msg, intent, categorie, bloc = preparer_modification_depuis_commande(cmd)
  
        if not ok:
            st.session_state.messages.append({"role": "assistant", "content": msg})
        else:
            if intent == "show_rule":
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": f"Règle trouvée : {categorie}"
                })
                st.session_state.proposition_ia = {
                    "intent": "show_rule",
                    "target_category": categorie,
                    "proposed_rule": bloc
                }



# Proposition en attente
if st.session_state.proposition_ia:
    prop = st.session_state.proposition_ia
    st.divider()
    st.markdown("### Proposition")

    if prop["intent"] == "show_rule":
        st.code(
            yaml.dump(prop["proposed_rule"], allow_unicode=True, sort_keys=False),
            language="yaml"
        )

    elif prop["intent"] == "update_rule":
        data = charger_yaml_regles()
        _, ancienne = trouver_regle_par_categorie(data, prop["target_category"])

        st.markdown("**Ancienne règle**")
        st.code(
            yaml.dump(ancienne, allow_unicode=True, sort_keys=False),
            language="yaml"
        )

        st.markdown("**Nouvelle règle**")
        st.code(
            yaml.dump(prop["proposed_rule"], allow_unicode=True, sort_keys=False),
            language="yaml"
        )

        if st.button("Confirmer la modification"):
            ok, msg = appliquer_modification_yaml(
                prop["intent"],
                prop["target_category"],
                prop["proposed_rule"]
            )
            st.session_state.messages.append({"role": "assistant", "content": msg})
            st.session_state.proposition_ia = None
            st.rerun()

    elif prop["intent"] == "delete_rule":
        st.markdown(f"**Suppression de :** {prop['target_category']}")

        if st.button("Confirmer la suppression"):
            ok, msg = appliquer_modification_yaml(
                prop["intent"],
                prop["target_category"],
                None
            )
            st.session_state.messages.append({"role": "assistant", "content": msg})
            st.session_state.proposition_ia = None
            st.rerun()


       
         
     


st.markdown("</div>", unsafe_allow_html=True)


# =========================================================
# INTERFACE PRINCIPALE
# =========================================================
st.title("IA CARE - système d'aide à la décision")

col_input, col_scan = st.columns(2)

with col_input:
    audio = st.audio_input(" Dictée vocale, veuillez parler clairement et lentement, en énonçant les médicaments un par un.")

with col_scan:
    photo = st.file_uploader("Scan Ordonnance ou PDF", type=["jpg", "png", "jpeg", "pdf"])

if photo is not None:
    try:
        if str(getattr(photo, "type", "")).lower() == "application/pdf":
            st.subheader("Ordonnance scannée")
            afficher_pdf(photo)
            photo.seek(0)
        else:
            img_preview = Image.open(photo).convert("RGB")
            st.subheader("Ordonnance scannée")
            st.image(img_preview, caption="Aperçu de l'ordonnance", use_container_width=True)
            photo.seek(0)
    except Exception as e:
        st.warning(f"Impossible d'afficher le document : {e}")



if "txt" not in st.session_state:
    st.session_state.txt = ""

if "ocr_lines" not in st.session_state:
    st.session_state.ocr_lines = []

if "manual_meds_buffer" not in st.session_state:
    st.session_state.manual_meds_buffer = ""

if "manual_meds_validated" not in st.session_state:
    st.session_state.manual_meds_validated = ""

manual_meds = st.text_area(
    "Saisie manuelle des médicaments",
    value=st.session_state.manual_meds_buffer,
    height=120,
    placeholder="Exemple :\nAtorvastatine\nBisoprolol\nRamipril\nKardegic"
)

col_manual_1, col_manual_2 = st.columns([1, 1])

with col_manual_1:
    if st.button("Valider saisie manuelle"):
        st.session_state.manual_meds_buffer = manual_meds
        st.session_state.manual_meds_validated = manual_meds
        st.success("Saisie manuelle enregistrée.")

with col_manual_2:
    if st.button("Effacer saisie manuelle"):
        st.session_state.manual_meds_buffer = ""
        st.session_state.manual_meds_validated = ""
        st.rerun()


if audio and st.button("Transcrire Voix"):
    try:
        lignes = transcrire_audio_robuste(audio)
        st.session_state.txt = "\n".join(lignes)
        st.session_state.ocr_lines = lignes
    except Exception as e:
        st.error(f"Erreur transcription audio : {e}")



if st.session_state.get("txt"):

    col1, col2 = st.columns(2)

    with col1:
        if st.button("Effacer la transcription"):
            st.session_state.txt = ""
            st.session_state.ocr_lines = []
            st.rerun()

   


if photo and st.button("Lancer Scan Document"):
    try:
        if str(getattr(photo, "type", "")).lower() == "application/pdf":
            lignes = extraire_texte_pdf(photo)
        else:
            img = Image.open(photo).convert("RGB")
            lignes = extraire_lignes_ocr_image(img)

        st.session_state.ocr_lines = lignes
        st.session_state.txt = "\n".join(lignes)
        photo.seek(0)
    except Exception as e:
        st.error(f"Erreur OCR document : {e}")

st.write("---")

txt_source = st.session_state.txt
manual_text = st.session_state.manual_meds_validated.strip()

if manual_text:
    if txt_source.strip():
        txt_source = txt_source + "\n" + manual_text
    else:
        txt_source = manual_text

txt_final = st.text_area("Données détectées :", txt_source, height=180)

sraa_detecte = contexte_famille_detecte(
    txt_final,
    ref,
    atc_map,
    atc_prefixes=["C09"],
    mots_secours=[
        "RAMIPRIL", "PERINDOPRIL", "ENALAPRIL",
        "LISINOPRIL", "CAPTOPRIL",
        "LOSARTAN", "VALSARTAN", "IRBESARTAN",
        "CANDESARTAN", "TELMISARTAN"
    ]
)








# =========================
# DETECTION CONTEXTES
# =========================

# Corticoïdes
corticoide_detecte = contexte_corticoide_detecte(txt_final, ref, atc_map)

# SRAA
sraa_detecte = contexte_famille_detecte(
    txt_final,
    ref,
    atc_map,
    atc_prefixes=["C09"],
    mots_secours=[
        "RAMIPRIL", "PERINDOPRIL", "ENALAPRIL", "LISINOPRIL", "CAPTOPRIL",
        "LOSARTAN", "VALSARTAN", "IRBESARTAN", "CANDESARTAN", "TELMISARTAN",
        "OLMESARTAN", "ENTRESTO"
    ]
)

# AAP
aap_detecte = contexte_famille_detecte(
    txt_final,
    ref,
    atc_map,
    atc_codes=["B01AC01", "B01AC06", "B01AC04", "B01AC24", "B01AC22"],
    mots_secours=[
        "ASPIRINE", "KARDEGIC", "CLOPIDOGREL", "PLAVIX",
        "TICAGRELOR", "BRILIQUE", "PRASUGREL", "EFIENT"
    ]
)

# AVK
avk_detecte = contexte_famille_detecte(
    txt_final,
    ref,
    atc_map,
    atc_prefixes=["B01AA"],
    mots_secours=[
        "WARFARINE", "COUMADINE", "PREVISCAN",
        "SINTROM", "FLUINDIONE", "ACENOCOUMAROL"
    ]
)

# Diabète
diabete_detecte = contexte_famille_detecte(
    txt_final,
    ref,
    atc_map,
    atc_prefixes=["A10"],
    mots_secours=[
        "METFORMINE", "GLUCOPHAGE",
        "INSULINE", "LANTUS", "LEVEMIR", "NOVORAPID",
        "TRULICITY", "OZEMPIC", "VICTOZA",
        "BYDUREON", "BYETTA",
        "JARDIANCE", "FORXIGA", "INVOKANA",
        "JANUVIA", "GALVUS", "ONGLYZA",
        "DIAMICRON", "AMAREL"
    ]
)

# =========================
# UI CONTEXTES
# =========================

ind_sraa = None

if sraa_detecte:
    st.divider()
    st.header("Système rénine-angiotensine (SRAA)")

    ind_sraa = st.radio(
        "Indication du traitement (IEC / ARA II)",
        ["HTA", "Insuffisance Cardiaque"],
        index=0
    )


type_traitement_aap = None
contexte_stent = "Aucun critère"
dose_aspirine = 75
indication_aap = None

if aap_detecte:
    st.divider()
    st.header("Antiagrégants plaquettaires (AAP)")

    type_traitement_aap = st.radio(
        "Type de traitement",
        ["Prévention primaire", "Prévention secondaire", "Bithérapie"],
        index=0
    )

    if type_traitement_aap == "Bithérapie":
        contexte_stent = st.selectbox(
            "Contexte thrombotique",
            [
                "Aucun critère",
                "Stent ≤ 1 mois",
                "Stent ≤ 6 mois à haut risque thrombotique",
                "IDM < 6 mois"
            ]
        )

        with st.expander("Définition haut risque thrombotique"):
            st.markdown("""
- thrombose de stent sous traitement
- diabète + coronaropathie diffuse
- DFG < 60
- ≥ 3 stents ou lésions
- bifurcation complexe
- longueur totale de stents > 60 mm
            """)

    dose_aspirine = st.number_input(
        "Dose aspirine (mg)",
        min_value=0,
        max_value=500,
        value=75,
        step=25
    )

    indication_aap = (
        "prev_prim" if type_traitement_aap == "Prévention primaire"
        else "prev_second" if type_traitement_aap == "Prévention secondaire"
        else "bitherapie"
    )


valves = False
acfa_atcd = False
mtev_hr = False
inr_disponible = "Non"
inr_valeur = None

if avk_detecte:
    st.divider()
    st.header("Anti-vitamine K (AVK)")

    st.info("""
### Objectif INR péri-opératoire (AVK)

Objectif standard :
- INR < 1,5
- INR < 1,2 si neurochirurgie
- Exception : chirurgie à faible risque hémorragique → INR en zone thérapeutique possible

---

### Facteurs pro-thrombotiques (à rechercher)

Si ≥ 1 facteur présent → **augmenter la cible INR de +0,5**

- Fibrillation atriale
- Dysfonction VG (FEVG < 35 %)
- État hypercoagulable
- Événement thrombotique récent (< 12 mois : AVC, TVP, EP)

---

### Valves mécaniques

- Valve mitrale / tricuspide / ancienne génération
  → **INR cible = 3 (2,5 – 3,5)**

- Valve aortique moderne (bileaflet)
  → **INR cible = 2,5 (2 – 3)**

---

### Autres indications (sans valve mécanique)

**INR cible 2 – 3 :**
- Fibrillation atriale non valvulaire
- Prévention et traitement TVP / EP
- Chirurgie de hanche (prévention TVP/EP)
- Syndrome des antiphospholipides (selon terrain)

**INR cible 3 – 4,5 :**
- Valvulopathie mitrale avec :
  - dilatation oreillette gauche
  - contraste spontané
  - thrombus intra-auriculaire gauche
""")

    valves = st.checkbox("Valve mécanique")
    acfa_atcd = st.checkbox("ACFA avec antécédent embolique")
    mtev_hr = st.checkbox("MTEV à haut risque")

    if mtev_hr:
        st.markdown("""
**Rappel MTEV à haut risque :**
- TVP proximale et/ou EP < 3 mois
- MTEV récidivante idiopathique (≥ 2 épisodes, dont ≥ 1 sans facteur déclenchant)

*Discuter la mise en place d’un filtre cave au cas par cas.*
""")

    st.subheader("INR")

    inr_disponible = st.radio(
        "INR récent disponible ?",
        ["Oui", "Non"],
        index=1
    )

    if inr_disponible == "Oui":
        inr_valeur = st.number_input(
            "Valeur INR",
            min_value=0.8,
            max_value=10.0,
            value=2.5,
            step=0.1
        )


    ctx = {}

    ctx["valve_mecanique"] = valves
    ctx["acfa_atcd"] = acfa_atcd
    ctx["mtev_haut_risque"] = mtev_hr
    ctx["inr_disponible"] = inr_disponible

    

    resultats, vus, candidats_retenus = detecter_medicaments_depuis_texte(
        txt=txt_final,
        ref=ref,
        atc_map=atc_map,
        classe_map=classe_map,
        ctx=ctx
    )

# =========================
# CONTEXTE PATIENT / CHIRURGIE
# =========================


ind_glp1 = None

if diabete_detecte:
    st.divider()
    st.header("Contexte diabète")

    ind_glp1 = st.radio(
        "Si prise de GLP-1, dans quel cas ?",
        ["Diabète", "Obésité", "Inconnue"],
        index=0
    )



    type_chir = st.selectbox(
        "Type de chirurgie",
        [
            "AMBULATOIRE ou chirurgie courte avec ≤ 1 repas jeûné",
            "chirurgie mineure ou majeure ou avec ≥ 2 repas jeûné",
            "URGENCE"
        ],
        key="type_chir_diabete"
    )

    pompe = st.checkbox(
        "Patient sous pompe à insuline",
        key="pompe_insuline"
    )

    if pompe:
        dispositif_insuline = "pompe"



corticoides_connus = False
duree_sup_4sem = False
dose_pred_sup_5 = False
dose_hc_inf_40 = False
dose_hc_sup_40 = False
chirurgie_courte = False
post_op_jeun_sup_24h = False
reprise_precoce = False
complications_postop = False
obstetrique = False

if "stress_cortico_faible" not in locals():
    stress_cortico_faible = False

if "stress_chir" not in locals():
    stress_chir = "modéré/élevé"



if corticoide_detecte:
    st.divider()

    st.subheader("Contexte corticoïdes")

    st.info(
        "Equivalence : Prednisone 5 mg = Méthylprednisolone 4 mg = "
        "Hydrocortisone 20 mg = Dexaméthasone 0.75 mg = Cortisone 25 mg"
    )

    
    duree_cortico = st.selectbox(
        "Durée du traitement corticoïde",
        ["< 4 semaines", "≥ 4 semaines"],
        key="ui_duree_cortico"
    )
    duree_sup_4sem = duree_cortico == "≥ 4 semaines"

    dose_pred = st.number_input(
        "Dose équivalente prednisone (mg/j)",
        min_value=0.0,
        step=0.5,
        key="ui_dose_pred"
    )
    dose_pred_sup_5 = dose_pred >= 5

 


    st.warning("""
    **Interprétation clinique :**

    - Corticothérapie ≥ 4 semaines et ≥ 5 mg prednisone :  
      → risque de suppression surrénalienne, adapter selon le stress chirurgical (voir plus bas).

    - Corticothérapie sans critère de risque :  
      → poursuite simple, sans supplémentation.
    """)




    st.subheader("Stress chirurgical (corticoïdes)")
    st.caption(
        "Déterminé automatiquement à partir de l’intervention sélectionnée. "
    )

    stress_chir = "faible" if stress_cortico_faible else "modéré/élevé"

    if stress_chir == "faible":
        st.success("Faible")
    else:
        st.warning("Modéré / Élevé")

    chirurgie_courte = False
    post_op_jeun_sup_24h = False
    reprise_precoce = False
    complications_postop = False
    obstetrique = False

    if stress_chir == "modéré/élevé":
        chirurgie_courte = st.checkbox(
            "Chirurgie modérée courte avec reprise rapide",
            key="ui_chirurgie_courte"
        )
        post_op_jeun_sup_24h = st.checkbox(
            "Jeûne postopératoire > 24h",
            key="ui_postop_jeun"
        )
        reprise_precoce = st.checkbox(
            "Reprise alimentaire < 24h",
            key="ui_reprise_precoce"
        )
        complications_postop = st.checkbox(
            "Complications postopératoires",
            key="ui_complications"
        )
        obstetrique = st.checkbox(
            "Accouchement / Césarienne",
            key="ui_obstetrique"
        )

# =========================
# CONTEXTE GLOBAL 
# =========================
ctx = {
    "type_chir_neuro": "NEUROCHIR_INTRACRANIENNE" if spe in ["Neurochirurgie", "Rachis"] else None,
    "type_chir": type_chir,
    "stress_chir": stress_chir,
    "is_neuro": is_neuro,
    "r_hem": risque_acte,
    "alr": type_alr,
    "ind_sraa": ind_sraa if ind_sraa else "",
    "indication_aap": indication_aap,
    "aspirine_dose": dose_aspirine,
    "aspirine_sup_100": dose_aspirine > 100,
    "aspirine_sup_200": dose_aspirine > 200,
    "dose_aspirine_inf_300": dose_aspirine <= 300,
    "controle_hem": controle_hemorragique,
    "categorie_geste": None,
    "demi_vie_heures": None,
    "voie_baclofene": None,

    "ASA": asa_acte_to_int(asa_acte) if 'asa_acte' in locals() else None,

    "atcd_cv": None,
    "dfg_connu": dfg_connu,
    "dfg": dfg,


    "ind_glp1_obesite": st.session_state.get("ind_glp1_diabete") == "Obésité",
    "ind_glp1_dt2": st.session_state.get("ind_glp1_diabete") == "Diabète",
    "dispositif_insuline": "pompe" if st.session_state.get("pompe_insuline", False) else None,

    "valve_mecanique": valves,
    "acfa_atcd": acfa_atcd,
    "mtev_haut_risque": mtev_hr,

    "type_traitement_aap": type_traitement_aap if type_traitement_aap else "",
    "bitherapie_aap": type_traitement_aap == "Bithérapie",
    "prev_secondaire": type_traitement_aap == "Prévention secondaire",
    "prev_primaire": type_traitement_aap == "Prévention primaire",
    "stent_1m": contexte_stent == "Stent ≤ 1 mois",
    "stent_6m_haut_risque": contexte_stent == "Stent ≤ 6 mois à haut risque thrombotique",
    "idm_6m": contexte_stent == "IDM < 6 mois",
    "aucun_critere_stent": contexte_stent == "Aucun critère",
    "inr_disponible": inr_disponible,
    "inr_valeur": inr_valeur,

    "corticoides": corticoides_connus,
    "duree_sup_4sem": duree_sup_4sem,
    "dose_pred_sup_5": dose_pred_sup_5,
    "dose_hc_inf_40": dose_hc_inf_40,
    "dose_hc_sup_40": dose_hc_sup_40,
    "stress_cortico_faible": stress_cortico_faible,
    "chirurgie_courte": chirurgie_courte,
    "post_op_jeun_sup_24h": post_op_jeun_sup_24h,
    "reprise_precoce": reprise_precoce,
    "complications_postop": complications_postop,
    "obstetrique": obstetrique,

    "inr_therapeutique_2_3": inr_disponible == "Oui" and inr_valeur is not None and 2 <= inr_valeur <= 3,
    "inr_hors_zone_2_3": inr_disponible == "Oui" and inr_valeur is not None and not (2 <= inr_valeur <= 3),
    "inr_non_connu": inr_disponible != "Oui",
    
    }
# =========================
# DFG
# =========================
dfg_ctx = ctx.get("dfg")
dfg_connu_ctx = ctx.get("dfg_connu")

if dfg_connu_ctx == "Oui" and dfg_ctx is not None:
    ctx["dfg_ge_50"] = dfg_ctx >= 50
    ctx["dfg_ge_30"] = dfg_ctx >= 30
    ctx["dfg_30_49"] = 30 <= dfg_ctx <= 49
else:
    ctx["dfg_ge_50"] = False
    ctx["dfg_ge_30"] = False
    ctx["dfg_30_49"] = False



# =======================
# ANALYSE 
# =========================
resultats, vus, candidats_retenus = detecter_medicaments_depuis_texte(
    txt=txt_final,
    ref=ref,
    atc_map=atc_map,
    classe_map=classe_map,
    ctx=ctx
)


# =========================
# DETECTION IMIPRAMINIQUES
# =========================
imipraminiques_detectes = any(
    str(r.get("Code ATC", "")).upper().strip() in ["N06AA04", "N06AA09"]
    for r in resultats
)




if imipraminiques_detectes:
    st.info("Antidépresseur imipraminique détecté")

    atcd_cv_ui = st.radio(
        "Patient avec antécédent cardiovasculaire ?",
        ["Non", "Oui"],
        help="Exemples : infarctus, angor, stent, insuffisance cardiaque, AVC, trouble du rythme."
    )

    ctx["atcd_cv"] = (atcd_cv_ui == "Oui")

    # recalcul avec la réponse du médecin
    resultats, vus, candidats_retenus = detecter_medicaments_depuis_texte(
        txt=txt_final,
        ref=ref,
        atc_map=atc_map,
        classe_map=classe_map,
        ctx=ctx
    )












with st.expander("Voir les lignes retenues pour la détection"):
    if candidats_retenus:
        for brute, nettoyee, mode in candidats_retenus:
            st.write(f"**Mode :** {mode}")
            st.write(f"**Brut :** {brute}")
            st.write(f"**Nettoyé :** {nettoyee}")
            st.write("---")
    else:
        st.write("Aucune ligne candidate retenue.")


# =========================
# RESULTATS
# =========================
codes_atc_detectes = []

if resultats:
    codes_atc_detectes = list(vus)
    asa_calcule = calculer_asa(age, len(codes_atc_detectes), None)

    asa_affiche = ctx.get("ASA")

    if asa_affiche is not None:
        st.subheader(f"ASA retenu pour l'intervention : {asa_affiche}")
    else:
        st.subheader(f"Score ASA Prédit : {asa_calcule}")

    df_final = pd.DataFrame(resultats)

    def format_lien_unique(liens):
        if not liens:
            return ""
        return str(liens).split(" | ")[0].strip()

    df_final["Lien"] = df_final["Lien"].apply(format_lien_unique)

    # =========================
    # CALENDRIER
    # =========================
    st.subheader("Calendrier Patient")

    au_moins_un_arret = False

    for r in resultats:
        if r["Action"] == "ARRET":
            date_txt = str(r["Date"]).upper().strip()

            jours = extraire_nb_jours(date_txt)
            if jours is not None:
                d_stop = date_op - timedelta(days=jours)
                st.write(
                    f"- **{r['Médicament']}** : dernière prise le **{d_stop.strftime('%d/%m/%Y')}**"
                )
                au_moins_un_arret = True
                continue

            match_h = re.search(r"(\d+)\s*H", date_txt)
            if match_h:
                heures = int(match_h.group(1))
                st.write(
                    f"- **{r['Médicament']}** : dernière prise à **H-{heures}** avant l’intervention"
                )
                au_moins_un_arret = True
                continue

    if au_moins_un_arret:
        st.info("Poursuivre le reste du traitement jusqu'au jour de l'intervention avec un peu d'eau.")
    else:
        st.info("Aucun arrêt médicamenteux daté à planifier selon les règles actuelles.")

    st.divider()

    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        st.markdown("**Patient**")
        st.markdown(f"### {age} ans")

    with col2:
        st.markdown("**Date chirurgie**")
        st.markdown(f"### {date_op.strftime('%d/%m/%Y')}")

    with col3:
        st.markdown("**Risque chirurgical**")
        st.markdown(f"### {risque_acte}")

    with col4:
        st.markdown("**Contrôle Hém.**")
        st.markdown(f"### {controle_hemorragique}")

    with col5:
        st.markdown("**ALR**")
        st.markdown(f"### {type_alr}")

    st.subheader("Tableau des recommandations")

    col1, col2, col3, col4, col5, col6, col7, col8, col9 = st.columns(
        [2.2, 1.1, 3.0, 2.0, 1.0, 3.4, 2.3, 1.8, 2.8]
    )

    col1.markdown("**Médicament**")
    col2.markdown("**ATC**")
    col3.markdown("**Classe**")
    col4.markdown("**Consigne**")
    col5.markdown("**Date**")
    col6.markdown("**Précisions**")
    col7.markdown("**Sources**")
    col8.markdown("**Validation médecin**")
    col9.markdown("**Commentaire médecin**")

    st.divider()

    for i, r in enumerate(resultats):
        c1, c2, c3, c4, c5, c6, c7, c8, c9 = st.columns(
            [2.2, 1.1, 3.0, 2.0, 1.0, 3.4, 2.3, 1.8, 2.8]
        )

        c1.write(r.get("Médicament", ""))
        c2.write(r.get("Code ATC", ""))
        c3.write(r.get("Classe", ""))
        c4.write(r.get("Action", ""))
        c5.write(format_jour_avec_date(r.get("Date", ""), date_op))
        c6.write(enrichir_note_avec_dates(r.get("Note", ""), date_op))

        liens_bruts = str(r.get("Lien", "")).strip()
        liens_list = [l.strip() for l in liens_bruts.split(" | ") if l.strip()] if liens_bruts else []

        if liens_list:
            with c7:
                with st.popover(f"{len(liens_list)} source(s)"):
                    for j, lien in enumerate(liens_list):
                        st.link_button(f"Source {j+1}", lien)
        else:
            c7.write("")

        with c8:
            st.selectbox(
                "",
                ["Oui", "Non"],
                key=f"validation_medecin_{i}",
                label_visibility="collapsed"
            )

        with c9:
            st.text_input(
                "",
                key=f"commentaire_medecin_{i}",
                placeholder="Commentaire médecin",
                label_visibility="collapsed"
            )

    st.divider()

    feedback_path = os.path.join(BASE_DIR, "retours_medecin.csv")

    if st.button("Enregistrer validation médecin"):
        rows = []

        for i, r in enumerate(resultats):
            rows.append({
                "Médicament": r.get("Médicament", ""),
                "ATC": r.get("Code ATC", ""),
                "Classe": r.get("Classe", ""),
                "Consigne": r.get("Action", ""),
                "Date": format_jour_avec_date(r.get("Date", ""), date_op),
                "Précisions": enrichir_note_avec_dates(r.get("Note", ""), date_op),
                "Sources": r.get("Lien", ""),
                "Validation médecin": st.session_state.get(f"validation_medecin_{i}", "À valider"),
                "Commentaire médecin": st.session_state.get(f"commentaire_medecin_{i}", ""),
                "Date analyse": date.today().isoformat()
            })

        df_feedback = pd.DataFrame(rows)

        try:
            if os.path.exists(feedback_path):
                ancien = pd.read_csv(feedback_path)
                df_feedback = pd.concat([ancien, df_feedback], ignore_index=True)

            df_feedback.to_csv(feedback_path, index=False, encoding="utf-8-sig")
            st.success("Validation médecin enregistrée")
        except Exception as e:
            st.error(f"Erreur : {e}")




#------- Profils pathologiques probables --------------------------
st.subheader("Profils pathologiques probables")

df_profils_patient = inferer_profils_structures(
    codes_atc_detectes,
    df_sentinelles_ready,
    df_profils_ready
)

# =========================
# PROFILS PATIENT
# =========================
if df_profils_patient is not None and not df_profils_patient.empty:
    profil_principal = df_profils_patient.iloc[0]

    st.success(
        f"**Profil principal probable : {profil_principal['Profil']}** — certitude {profil_principal['Niveau']}"
    )

    for _, row in df_profils_patient.iterrows():
        st.markdown(f"""
        <div class="box-profil">
            <b>{row['Profil']}</b><br>
            <b>Certitude :</b> {row['Niveau']}<br>
            <b>ASA minimum suggéré :</b> {row['ASA min']}<br>
            <b>Médicaments sentinelles :</b> {row['Sentinelles']}<br>
            <b>ATC retrouvés :</b> {row['ATC']}
        </div>
        """, unsafe_allow_html=True)

    with st.expander("Voir le détail clinique des profils retenus"):
        st.dataframe(
            df_profils_patient[[
                "Profil",
                "Niveau",
                "ASA min",
                "Sentinelles",
                "ATC"
            ]],
            use_container_width=True
        )
else:
    st.info("Aucun profil pathologique fort identifié à partir des médicaments détectés.")




# =========================
# QUESTIONNAIRE DE SATISFACTION
# =========================
st.divider()

afficher_satisfaction = st.checkbox("Remplir le questionnaire de satisfaction")

if afficher_satisfaction:

    st.subheader("Questionnaire de satisfaction AI CARE")

    st.info("""
Merci d’évaluer votre expérience :

1 = Pas du tout d’accord  
2 = Plutôt pas d’accord  
3 = Neutre  
4 = Plutôt d’accord  
5 = Tout à fait d’accord  
""")

    st.markdown("### Profil utilisateur")

    profil = st.radio(
        "Vous êtes :",
        ["Interne", "Médecin sénior"],
        horizontal=True
    )

    questions = [
        "1. Je pense que j’aimerais utiliser cette application fréquemment.",
        "2. J’ai trouvé cette application inutilement complexe.",
        "3. J’ai trouvé cette application facile à utiliser.",
        "4. J’aurais besoin d’une aide technique pour l’utiliser.",
        "5. Les fonctions sont bien intégrées.",
        "6. Il y a trop d’incohérences dans cette application.",
        "7. Les utilisateurs apprendront rapidement à l’utiliser.",
        "8. Application encombrante à utiliser.",
        "9. Je me suis senti confiant en l’utilisant.",
        "10. J’ai dû apprendre beaucoup de choses avant de l’utiliser."
    ]

    reponses = []

    for i, question in enumerate(questions, start=1):
        rep = st.radio(
            question,
            [1, 2, 3, 4, 5],
            horizontal=True,
            key=f"q_{i}"
        )
        reponses.append(rep)

    commentaire = st.text_area("Commentaire libre", key="commentaire")



    if st.button("Enregistrer le questionnaire"):

        # chemin du fichier dans le même dossier que app.py
        fichier_satisfaction = os.path.join(BASE_DIR, "satisfaction_utilisateurs.csv")

        # créer une ligne de données
        data = {
            "date": date.today().isoformat(),
            "profil": profil,
            "commentaire": commentaire
        }

        # ajouter les 10 questions
        for i, rep in enumerate(reponses, start=1):
            data[f"Q{i}"] = rep

        df_new = pd.DataFrame([data])

        try:
            # si fichier existe → on ajoute
            if os.path.exists(fichier_satisfaction):
                df_old = pd.read_csv(fichier_satisfaction)
                df_final = pd.concat([df_old, df_new], ignore_index=True)
            else:
                df_final = df_new

            df_final.to_csv(fichier_satisfaction, index=False, encoding="utf-8-sig")

            st.success("Questionnaire enregistré dans le fichier satisfaction_utilisateurs.csv")

        except Exception as e:
            st.error(f"Erreur sauvegarde : {e}")
