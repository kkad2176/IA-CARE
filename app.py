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
import html
import streamlit.components.v1 as components
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import cm
from reportlab.pdfgen import canvas
from reportlab.lib import colors
from datetime import date, timedelta
from collections import defaultdict
from reportlab.lib.utils import ImageReader
from reportlab.platypus import Paragraph, Table, TableStyle
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.enums import TA_LEFT

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

def generer_pdf_patient(
    ville,
    date_doc,
    civilite,
    nom_prenom,
    lignes,
    phrase,
    bilan_texte="",
    scanner_texte="",
    allergies_texte="",
    medecin="",
    tableau_avk=None
):

    import tempfile, os
    import fitz

    fd, path = tempfile.mkstemp(suffix=".pdf")
    os.close(fd)

    c = canvas.Canvas(path, pagesize=A4)
    w, h = A4

    fond_path = os.path.join(
        os.path.dirname(__file__),
        "SKM_451i26051814360.pdf"
    )

    try:
        doc_fond = fitz.open(fond_path)
        page = doc_fond[0]

        pix = page.get_pixmap(
            matrix=fitz.Matrix(2, 2),
            alpha=False
        )

        bg_path = os.path.join(
            os.path.dirname(__file__),
            "fond_temp.png"
        )

        pix.save(bg_path)

        c.drawImage(
            bg_path,
            0,
            0,
            width=w,
            height=h
        )

    except Exception as e:
        print(f"Erreur fond PDF : {e}")

    x = 6.2 * cm
    y = h - 8.0 * cm

    c.setFillColorRGB(0, 0, 0)

    c.setFont("Helvetica", 10)
    c.drawRightString(
        w - 2.2 * cm,
        h - 4.0 * cm,
        f"{ville}, le {date_doc}"
    )

    c.setFont("Helvetica", 11)
    c.drawString(
        x,
        y,
        f"{civilite} {nom_prenom}"
    )
    y -= 1.1 * cm

    y -= 0.4 * cm

    c.setFont("Helvetica", 11)

    for l in lignes:
        c.drawString(
            x,
            y,
            f"- {l}"
        )
        y -= 0.75 * cm

    y -= 0.4 * cm

    if phrase:

        c.setFillColorRGB(0.91, 0.95, 1.0)
        c.roundRect(
            x,
            y - 0.35 * cm,
            13.5 * cm,
            1.0 * cm,
            8,
            fill=1,
            stroke=0
        )

        c.setFillColorRGB(0.12, 0.35, 0.66)
        c.setFont("Helvetica", 9.5)
        c.drawString(
            x + 0.3 * cm,
            y,
            phrase
        )

        c.setFillColorRGB(0, 0, 0)
        y -= 1.4 * cm




# =====================================================
# SCHÉMA AVK PRÉOPE
# =====================================================

    if tableau_avk:

        y -= 0.35 * cm

        c.setFillColorRGB(0, 0, 0)
        c.setFont("Helvetica-Bold", 11)
        c.drawString(
            x,
            y,
            "Schéma thérapeutique préopératoire"
        )

        y -= 0.55 * cm

        data_tableau = [
            ["Moment", "Instruction"]
        ] + tableau_avk

        tableau = Table(
            data_tableau,
            colWidths=[3.2 * cm, 10.3 * cm]
        )

        tableau.setStyle(
            TableStyle([
       
                ("BACKGROUND", (0, 0), (-1, -1), colors.white),

      
                ("TEXTCOLOR", (0, 0), (-1, -1), colors.black),

      
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),

        
                ("FONTNAME", (0, 1), (-1, -1), "Helvetica"),

        
                ("FONTSIZE", (0, 0), (-1, -1), 8.5),
                ("LEADING", (0, 0), (-1, -1), 10),

     
                ("BOX", (0, 0), (-1, -1), 1, colors.black),

       
                ("INNERGRID", (0, 0), (-1, -1), 0.5, colors.black),

        
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),

  
                ("LEFTPADDING", (0, 0), (-1, -1), 5),
                ("RIGHTPADDING", (0, 0), (-1, -1), 5),
                ("TOPPADDING", (0, 0), (-1, -1), 4),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ])
        )




        largeur_tableau, hauteur_tableau = tableau.wrap(
            13.5 * cm,
            h
        )

        tableau.drawOn(
            c,
            x,
            y - hauteur_tableau
        )

        y -= hauteur_tableau + 0.35 * cm

        c.setFont("Helvetica-Bold", 8.5)
        c.drawString(
            x,
            y,
            "J0 = jour de l’intervention"
        )

        y -= 0.6 * cm







    if bilan_texte or scanner_texte or allergies_texte:

        y -= 0.2 * cm

        c.setFillColorRGB(0.96, 0.98, 1.0)
        c.roundRect(
            x,
            y - 5.6 * cm,
            13.5 * cm,
            5.9 * cm,
            10,
            fill=1,
            stroke=0
        )

        c.setFillColorRGB(0.12, 0.35, 0.66)
        c.setFont("Helvetica-Bold", 13)
        c.drawString(
            x + 0.4 * cm,
            y - 0.3 * cm,
            "Préparation pré-opératoire"
        )

        y -= 1.1 * cm

        def bloc_preop(titre, texte, y):
            if not texte:
                return y

            c.setFont("Helvetica-Bold", 10.5)
            c.setFillColorRGB(0.10, 0.25, 0.45)
            c.drawString(
                x + 0.6 * cm,
                y,
                titre
            )

            y -= 0.5 * cm

            c.setFont("Helvetica", 9.5)
            c.setFillColorRGB(0, 0, 0)

            for ligne in texte.split("\n"):
                if ligne.strip():
                    c.drawString(
                        x + 1.0 * cm,
                        y,
                        f"- {ligne.strip()}"
                    )
                    y -= 0.42 * cm

            return y - 0.2 * cm

        y = bloc_preop("Bilans à prévoir", bilan_texte, y)
        y = bloc_preop("Examens complémentaires", scanner_texte, y)
        y = bloc_preop("Allergies / précautions", allergies_texte, y)

    if medecin:

        c.setFillColorRGB(0, 0, 0)

        c.setFont("Helvetica", 10)

        c.drawRightString(
            w - 3*cm,
            3.2*cm,
            f"Dr {medecin}"
        )

        c.drawRightString(
            w - 3*cm,
            2.6*cm,
            "Signature :"
        )

    c.save()
    return path


## ORDO POUR PHARMACIE CAS HEPARINE


def generer_pdf_ordonnance_pharmacie(
    ville,
    date_doc,
    civilite,
    nom_prenom,
    ordonnance_pharmacie,
    medecin=""
):
    import tempfile
    import os
    import fitz
    from reportlab.pdfgen import canvas
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.units import cm

    if not ordonnance_pharmacie:
        return None

    fd, path = tempfile.mkstemp(suffix=".pdf")
    os.close(fd)

    c = canvas.Canvas(path, pagesize=A4)
    w, h = A4



    fond_path = os.path.join(
        os.path.dirname(__file__),
        "SKM_451i26051814360.pdf"
    )

    try:
        doc_fond = fitz.open(fond_path)
        page = doc_fond[0]

        pix = page.get_pixmap(
            matrix=fitz.Matrix(2, 2),
            alpha=False
        )

        bg_path = os.path.join(
            os.path.dirname(__file__),
            "fond_temp_ordonnance.png"
        )

        pix.save(bg_path)

        c.drawImage(
            bg_path,
            0,
            0,
            width=w,
            height=h
        )

    except Exception as e:
        print(f"Erreur fond PDF ordonnance pharmacie : {e}")


    x = 6.2 * cm
    y = h - 8.0 * cm

    c.setFillColorRGB(0, 0, 0)

    c.setFont("Helvetica", 10)
    c.drawRightString(
        w - 2.2 * cm,
        h - 4.0 * cm,
        f"{ville}, le {date_doc}"
    )

    c.setFont("Helvetica", 11)
    c.drawString(
        x,
        y,
        f"{civilite} {nom_prenom}"
    )

    y -= 1.2 * cm



    style_pharmacie = ParagraphStyle(
        "style_pharmacie",
        fontName="Helvetica",
        fontSize=10.5,
        leading=14,
        alignment=TA_LEFT
    )

    largeur_texte = 13.2 * cm

    for ligne in ordonnance_pharmacie.split("\n"):

        ligne = ligne.strip()

        if not ligne:
            y -= 0.25 * cm
            continue

        if ligne == "ORDONNANCE":
            continue



        if ligne.startswith("ORDONNANCE – RELAIS PRÉOPÉRATOIRE"):
            ligne = f"<b>{ligne}</b>"

        elif ligne.startswith("À DESTINATION DE LA PHARMACIE"):
            ligne = f"<b>{ligne}</b>"

        elif ligne.startswith("Indication :"):
            ligne = ligne.replace(
                "Indication :",
                "<b>Indication :</b>",
                1
            )


        elif ligne.startswith("Si énoxaparine avec adaptation pondérale :"):
            ligne = ligne.replace(
                "Si énoxaparine avec adaptation pondérale :",
                "<b>Si énoxaparine avec adaptation pondérale :</b>",
                1
            )

        p = Paragraph(
            ligne,
            style_pharmacie
        )
 
        largeur_p, hauteur_p = p.wrap(
            largeur_texte,
            h
        )

        p.drawOn(
            c,
            x,
            y - hauteur_p
        )

        y -= hauteur_p + 0.18 * cm


 

    if medecin:

        c.setFont("Helvetica", 10)

        c.drawRightString(
            w - 3 * cm,
            3.2 * cm,
            f"Dr {medecin}"
        )

        c.drawRightString(
            w - 3 * cm,
            2.6 * cm,
            "Signature :"
        )

    c.save()

    return path

def generer_pdf_prescription_ide(
    ville,
    date_doc,
    civilite,
    nom_prenom,
    prescription_ide,
    medecin=""
):
    import tempfile
    import os
    import fitz

    from reportlab.pdfgen import canvas
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.units import cm
    from reportlab.platypus import Paragraph
    from reportlab.lib.styles import ParagraphStyle
    from reportlab.lib.enums import TA_CENTER, TA_LEFT

    if not prescription_ide:
        return None

    fd, path = tempfile.mkstemp(suffix=".pdf")
    os.close(fd)

    c = canvas.Canvas(path, pagesize=A4)
    w, h = A4


    fond_path = os.path.join(
        os.path.dirname(__file__),
        "SKM_451i26051814360.pdf"
    )

    try:
        doc_fond = fitz.open(fond_path)
        page = doc_fond[0]

        pix = page.get_pixmap(
            matrix=fitz.Matrix(2, 2),
            alpha=False
        )

        bg_path = os.path.join(
            os.path.dirname(__file__),
            "fond_temp_ide.png"
        )

        pix.save(bg_path)

        c.drawImage(
            bg_path,
            0,
            0,
            width=w,
            height=h
        )

    except Exception as e:
        print(f"Erreur fond PDF IDE : {e}")



    c.setFont("Helvetica", 10)

    c.drawRightString(
        w - 2.2 * cm,
        h - 4.0 * cm,
        f"{ville}, le {date_doc}"
    )


    x = 6.2 * cm
    largeur = 13.2 * cm
    y = h - 7.3 * cm

    if nom_prenom:
        c.setFont("Helvetica", 11)
        c.drawString(
            x,
            y,
            f"{civilite} {nom_prenom}".strip()
        )
        y -= 1.0 * cm



    style_titre = ParagraphStyle(
        "titre_ide",
        fontName="Helvetica-Bold",
        fontSize=12,
        leading=15,
        alignment=TA_CENTER,
        spaceAfter=14
    )

    style_normal = ParagraphStyle(
        "normal_ide",
        fontName="Helvetica",
        fontSize=10.2,
        leading=14,
        alignment=TA_LEFT,
        spaceAfter=6
    )

    style_fort = ParagraphStyle(
        "fort_ide",
        fontName="Helvetica-Bold",
        fontSize=10.2,
        leading=14,
        alignment=TA_LEFT,
        spaceAfter=8
    )



    def ajouter_paragraphe(texte, style, y_actuel):
        p = Paragraph(texte, style)
        largeur_p, hauteur_p = p.wrap(largeur, h)

        p.drawOn(
            c,
            x,
            y_actuel - hauteur_p
        )

        return y_actuel - hauteur_p - 0.15 * cm



    lignes = [
        ligne.strip()
        for ligne in prescription_ide.split("\n")
    ]

    for ligne in lignes:

        if not ligne:
            y -= 0.15 * cm
            continue

  
        if ligne.startswith(
            "ORDONNANCE DE SOINS INFIRMIERS"
        ):
            y = ajouter_paragraphe(
                "ORDONNANCE DE SOINS INFIRMIERS – RELAIS PRÉOPÉRATOIRE DES AVK PAR HÉPARINE",
                style_titre,
                y
            )
            continue

  
        if ligne.startswith(
            "Faire pratiquer à domicile"
        ):
            y = ajouter_paragraphe(
                f"<b>{ligne}</b>",
                style_normal,
                y
            )
            continue

        if ligne.startswith(
            "Ne réaliser aucune injection supplémentaire"
        ):
            y = ajouter_paragraphe(
                f"<b>{ligne}</b>",
                style_normal,
                y
            )
            continue


        if ligne.startswith("Modalités :"):
            texte = ligne.replace(
                "Modalités :",
                "<b>Modalités :</b>",
                1
            )

            y = ajouter_paragraphe(
                texte,
                style_normal,
                y
            )
            continue

     
        if ligne.startswith(
            "Pour l’énoxaparine en seringue graduée :"
        ):
            texte = ligne.replace(
                "Pour l’énoxaparine en seringue graduée :",
                "<b>Pour l’énoxaparine en seringue graduée :</b>",
                1
            )

            y = ajouter_paragraphe(
                texte,
                style_normal,
                y
            )
            continue

   
        if ligne.startswith("Contexte opératoire :"):
            texte = ligne.replace(
                "Contexte opératoire :",
                "<b>Contexte opératoire :</b>",
                1
            )

            y = ajouter_paragraphe(
                texte,
                style_normal,
                y
            )
            continue


        prefixes_soulignes = [
            "Dose :",
            "Rythme :",
            "Première injection :",
            "Injection(s) suivante(s) :",
            "Dernière injection préopératoire :"
        ]

        prefixe_trouve = None

        for prefixe in prefixes_soulignes:
            if ligne.startswith(prefixe):
                prefixe_trouve = prefixe
                break

        if prefixe_trouve:

            reste = ligne[len(prefixe_trouve):]

            texte = (
                f"<u>{prefixe_trouve}</u>"
                f"{reste}"
            )

            y = ajouter_paragraphe(
                texte,
                style_normal,
                y
            )
            continue


        if ligne.startswith("- "):
            y = ajouter_paragraphe(
                ligne,
                style_normal,
                y
            )
            continue

      
        y = ajouter_paragraphe(
            ligne,
            style_normal,
            y
        )


    if medecin:

        c.setFont("Helvetica", 10)

        c.drawRightString(
            w - 3 * cm,
            3.2 * cm,
            f"Dr {medecin}"
        )

        c.drawRightString(
            w - 3 * cm,
            2.6 * cm,
            "Signature :"
        )

    c.save()

    return path






# =========================================================
# CONFIGURATION
# =========================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))


st.set_page_config(page_title="IA CARE - Expert SFAR", layout="wide")

REGLES = {}
yaml_path = os.path.join(BASE_DIR, "regles_sfar.yaml")
if os.path.exists(yaml_path):
    try:
        with open(yaml_path, "r", encoding="utf-8") as f:
            REGLES = yaml.safe_load(f) or {}
    except Exception as e:
        st.warning(f"Impossible de charger regles_sfar.yaml : {e}")

MED_TO_ATC = {}


st.set_page_config(page_title="IA CARE - Expert SFAR", layout="wide")


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
   
    txt = txt.replace(" PH ", " F ")
    txt = txt.replace(" Y ", " I ")
    txt = txt.replace("-", " ")
    txt = re.sub(r"\s+", " ", txt).strip()

    mots = txt.split()
    mots_corriges = []

    mots_a_ignorer = {
        "LE", "LA", "LES", "DE", "DU", "DES", "ET", "OU", "UN", "UNE",
        "MATIN", "MIDI", "SOIR", "JOUR", "JOURS", "SI", "BESOIN"
    }

    for mot in mots:
        mot_clean = normalize_text(mot)

        if len(mot_clean) < 4 or mot_clean in mots_a_ignorer:
            mots_corriges.append(mot_clean)
            continue

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

    # SUP DES DOSAGES

    txt = re.sub(r"\b\d+[.,]?\d*\s*(MG|G|MCG|UG|ML|UI|MUI)\b", " ", txt)
    txt = re.sub(r"\b\d+[.,]?\d*\b", " ", txt)

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

    for i in range(len(mots)):
        candidats.append(mots[i])

    for i in range(len(mots) - 1):
        candidats.append(mots[i] + " " + mots[i + 1])

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

        if meilleur_nom and meilleur_score >= 80:
            nom_norm = normalize_text(meilleur_nom)
            if nom_norm not in vus:
                vus.add(nom_norm)
                meds_trouves.append(meilleur_nom)

    return meds_trouves

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
    # Minimum ASA 2 
    score = 2
    
    # 2 Règle de l'âge : Plus de 60 ans -> ASA 3 d'office
    if age >= 60:
        score = 3
        
    # 3 Plus de 3 médicaments -> ASA 3 d'office
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

# =========================================================
# OCR / DETECTION MEDICAMENTS
# =========================================================
DOSE_PATTERN = re.compile(
    r"\b\d+[.,]?\d*\s*(MG|G|MCG|UG|µG|ML|UI|MUI)\b|\b\d+[.,]?\d*\s*%",
    re.IGNORECASE
)

def contient_dose(ligne):
    return bool(DOSE_PATTERN.search(str(ligne)))


FORMES_SANS_DOSAGE = [
    "INHALATEUR",
    "INHALATION",
    "SPRAY",
    "AEROSOL",
    "AÉROSOL",
    "COLLYRE",
    "POMMADE",
    "CREME",
    "CRÈME",
    "GEL",
    "PATCH",
    "STYLO",
    "SOLUTION",
    "BOUFFEE",
    "BOUFFÉE",
    "BOUFFEES",
    "BOUFFÉES",
]

def est_forme_sans_dosage(ligne):
    l = normalize_text(ligne)
    return any(mot in l for mot in FORMES_SANS_DOSAGE)


def filtrer_lignes_scan_avec_dosage(lignes):
    lignes_filtrees = []

    for ligne in lignes:
        l = str(ligne).strip()
        if not l:
            continue

        if contient_dose(l) or est_forme_sans_dosage(l):
            lignes_filtrees.append(l)

    return lignes_filtrees


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
        "DOCTEUR", "DR", "CARDIOLOGUE", "NICE",
        "TEL", "TELEPHONE", "FAX", "EMAIL", "MAIL",
        "PLACE", "RUE", "AVENUE", "BOULEVARD",
        "PARIS", "LYON", "MARSEILLE", "TOULOUSE", "LILLE",
        "SIGNATURE", "CABINET", "MADAME", "MONSIEUR", "MME", "MR",
        "ORDONNANCE", "DATE", "MEDECIN", "RENOUVELABLE",
        "CLINIQUE", "HOPITAL", "SERVICE", "TIMONE"
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

        # if est_ligne_posologie(ligne):
        # continue


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


    if cand in ref:
        return cand, 100


    if len(cand.split()) == 1:
        match = process.extractOne(cand, ref, scorer=fuzz.ratio)
        if match:
            nom_match, score_match, _ = match

            # n'accepte que si quasi identique
            if score_match >= 85:
                return nom_match, score_match

        return None, 0

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




def construire_schema_relais(ctx):
    schema = {
        "indique": False,
        "type": None,
        "molecule": None,
        "dose": None,
        "frequence": None,
        "voie": None,
        "debut": None,
        "fin": None,
        "surveillance": None,
    }

    if not ctx.get("relais_avk"):
        return schema

    schema["indique"] = True

    dfg = ctx.get("dfg_relais_avk")
    mode = ctx.get("mode_prise_en_charge_relais")
    type_relais = ctx.get("type_heparine_relais")
    schema_hbpm = ctx.get("schema_hbpm")

    # =====================================================
    # DFG > 30
    # =====================================================

    if dfg == "DFG > 30":

        if schema_hbpm == "2 injections par jour (Enoxaparine 100 UI/kg toutes les 12h)":
            schema.update({
                "type": "HBPM",
                "molecule": "Enoxaparine",
                "dose": "100 UI/kg",
                "frequence": "toutes les 12 h",
                "voie": "SC",
                "debut": "J-3 soir",
                "fin": "J-1 matin",
                "surveillance": None,
            })

        elif schema_hbpm == "1 injection par jour (Tinzaparine 175 UI/kg x 1/j)":
            schema.update({
                "type": "HBPM",
                "molecule": "Tinzaparine",
                "dose": "175 UI/kg",
                "frequence": "1 injection par jour",
                "voie": "SC",
                "debut": "J-3 soir",
                "fin": "J-2 soir",
                "surveillance": None,
            })


    # =====================================================
    # DFG < 30 + HOSPITALISATION
    # =====================================================

    elif dfg in ["15 ≤ DFG < 30", "DFG < 15"] and mode == "Hospitalisation prévue":

        schema.update({
            "type": "HNF IVSE",
            "molecule": "HNF",
            "dose": "Selon protocole local ou 12 à 15 UI/kg/h max",
            "frequence": "IVSE continue",
            "voie": "IV",
            "debut": "J-3 soir",
            "fin": "6 h avant la procédure",
            "surveillance": "1ère activité anti-Xa à H6 puis adaptation posologique selon le protocole d’ajustement local",
        })


    # =====================================================
    # DFG < 30 + EXTRAHOSPITALIER + HNF CALCIQUE
    # =====================================================

    elif (
        dfg in ["15 ≤ DFG < 30", "DFG < 15"]
        and mode == "Prise en charge extrahospitalière"
        and type_relais == "HNF calcique SC"
    ):

        schema.update({
            "type": "HNF CALCIQUE SC",
            "molecule": "HNF calcique",
            "dose": "333 UI/kg puis 250 UI/kg",
            "frequence": "toutes les 12 h après la première injection",
            "voie": "SC",
            "debut": "J-3 soir",
            "fin": "J-1 matin",
            "surveillance": None,
        })


    # =====================================================
    # DFG ENTREE 15-29 EXTRAHOSPITALIER + HBPM
    # =====================================================

    elif (
        dfg == "15 ≤ DFG < 30"
        and mode == "Prise en charge extrahospitalière"
        and type_relais == "HBPM"
    ):

        schema.update({
            "type": "HBPM",
            "molecule": "Tinzaparine ou Enoxaparine",
            "dose": "Tinzaparine 175 UI/kg x1/j OU Enoxaparine 100 UI/kg x1/j",
            "frequence": "1 injection par jour",
            "voie": "SC",
            "debut": "J-3 soir",
            "fin": "J-2 soir",
            "surveillance": "Aucune HBPM à J-1",
        })


    # =====================================================
    # DFG < 15 + EXTRAHOSPITALIER + HBPM
    # =====================================================

    elif (
        dfg == "DFG < 15"
        and mode == "Prise en charge extrahospitalière"
        and type_relais == "HBPM"
    ):

        schema.update({
            "type": "HBPM",
            "molecule": "Tinzaparine",
            "dose": "175 UI/kg",
            "frequence": "1 injection par jour",
            "voie": "SC",
            "debut": "J-3 soir",
            "fin": "J-2 soir",
            "surveillance": "Pas d’HBPM à J-1. Les autres HBPM ne sont pas recommandées.",
        })

    return schema



# =====================================================
# CALCUL PR CALENDRIER DES INJECTIONS 
# =====================================================

def calculer_injections_relais(schema_relais, date_op):

    if not schema_relais or not date_op:
        return []

    molecule = schema_relais.get("molecule")

    injections = []

    # ====================================
    # ENOXAPARINE 2 injections / jour
    # ======================================

    if molecule == "Enoxaparine":

        injections = [
            {
                "moment": "J-3 soir",
                "date": date_op - timedelta(days=3)
            },
            {
                "moment": "J-2 matin",
                "date": date_op - timedelta(days=2)
            },
            {
                "moment": "J-2 soir",
                "date": date_op - timedelta(days=2)
            },
            {
                "moment": "J-1 matin",
                "date": date_op - timedelta(days=1)
            },
        ]

    # =======================================
    # TINZAPARINE 1 injection / jour
    # ================================================

    elif molecule == "Tinzaparine":

        injections = [
            {
                "moment": "J-3 soir",
                "date": date_op - timedelta(days=3)
            },
            {
                "moment": "J-2 soir",
                "date": date_op - timedelta(days=2)
            },
        ]

    # =====================================================
    # HNF CALCIQUE
    # J-3 soir puis toutes les 12 h -> J-1 matin
    #==================================

    elif molecule == "HNF calcique":

        injections = [
            {
                "moment": "J-3 soir",
                "date": date_op - timedelta(days=3)
            },
            {
                "moment": "J-2 matin",
                "date": date_op - timedelta(days=2)
            },
            {
                "moment": "J-2 soir",
                "date": date_op - timedelta(days=2)
            },
            {
                "moment": "J-1 matin",
                "date": date_op - timedelta(days=1)
            },
        ]

    return injections




def generer_ordonnance_pharmacie(schema_relais, poids_kg=None, date_op=None):

    if not schema_relais or not schema_relais.get("indique"):
        return None

    if not poids_kg or poids_kg <= 0:
        return "Poids du patient nécessaire pour calculer la prescription d’héparine."

    molecule = schema_relais.get("molecule")

    injections = calculer_injections_relais(
        schema_relais,
        date_op
    )

    nb_seringues = len(injections)

    if injections:
        premiere = injections[0]
        derniere = injections[-1]

        date_debut = (
            f"{premiere['moment']} "
            f"({premiere['date'].strftime('%d/%m/%Y')})"
        )

        date_derniere = (
            f"{derniere['moment']} "
            f"({derniere['date'].strftime('%d/%m/%Y')})"
        )
    else:
        date_debut = ""
        date_derniere = ""

    # =====================================================
    # CALCUL DOSES ET FREQUENCE
    # =====================================================

    dose = ""
    frequence = ""

    if molecule == "Enoxaparine":

        poids_calcul = min(poids_kg, 100)
        dose_ui = poids_calcul * 100

        dose = f"{dose_ui:.0f}"
        frequence = "1 injection toutes les 12 h"

    elif molecule == "Tinzaparine":

        poids_calcul = min(poids_kg, 100)
        dose_ui = poids_calcul * 175

        dose = f"{dose_ui:.0f}"
        frequence = "1 injection par jour"

    elif molecule == "HNF calcique":

        dose_initiale = poids_kg * 333
        dose_suivante = poids_kg * 250

        dose = (
            f"{dose_initiale:.0f} UI pour la première injection, "
            f"puis {dose_suivante:.0f}"
        )

        frequence = "selon protocole sélectionné"

    elif molecule == "HNF":

        # HNF IVSE = pas d'ordonnance pharmacie ambulatoire standard
        return None

    elif molecule == "Tinzaparine ou Enoxaparine":

        return (
            "Choisir la molécule HBPM avant de générer "
            "l’ordonnance pharmacie : Tinzaparine ou Enoxaparine."
        )

    else:
        return None


    lignes = []

    lignes.append(
        "ORDONNANCE – RELAIS PRÉOPÉRATOIRE DES AVK PAR HÉPARINE"
    )
    lignes.append(
        "À DESTINATION DE LA PHARMACIE"
    )
    lignes.append("")

    lignes.append(
        "Indication : relais anticoagulant préopératoire d’un traitement par AVK."
    )

    lignes.append("")

    lignes.append(
        f"Poids du patient : {poids_kg} kgs"
    )

    lignes.append("")

    lignes.append(
        f"Molécule : {molecule.upper()}"
    )

    lignes.append(
        "Présentation : seringue préremplie graduée"
    )

    lignes.append(
        f"Dose à administrer : {dose} UI anti-Xa ou UI par injection, par voie sous-cutanée."
    )

    lignes.append(
        f"Fréquence : {frequence}"
    )

    lignes.append(
        f"Début : {date_debut}"
    )

    lignes.append(
        f"Dernière injection préopératoire : {date_derniere}"
    )

    lignes.append(
        f"Quantité totale à délivrer : {nb_seringues} seringues préremplies."
    )

    lignes.append(
        "Non renouvelable."
    )

    lignes.append("")

    lignes.append(
        "Si énoxaparine avec adaptation pondérale : délivrer une présentation en seringue préremplie graduée permettant l’administration de la dose prescrite."
    )

    return "\n".join(lignes)





def generer_prescription_ide(schema_relais, poids_kg=None, date_op=None):

    if not schema_relais or not schema_relais.get("indique"):
        return None

    if not poids_kg or poids_kg <= 0:
        return None

    molecule = schema_relais.get("molecule")

    if molecule == "HNF":
        return None

    injections = calculer_injections_relais(
        schema_relais,
        date_op
    )

    if not injections:
        return None


    if molecule == "Enoxaparine":

        poids_calcul = min(poids_kg, 100)
        dose_ui = poids_calcul * 100

        dose = f"{dose_ui:.0f}"
        rythme = "toutes les 12 heures"

    elif molecule == "Tinzaparine":

        poids_calcul = min(poids_kg, 100)
        dose_ui = poids_calcul * 175

        dose = f"{dose_ui:.0f}"
        rythme = "1 fois par jour"

    elif molecule == "HNF calcique":

        dose_initiale = poids_kg * 333
        dose_suivante = poids_kg * 250

        dose = (
            f"{dose_initiale:.0f} UI pour la première injection, "
            f"puis {dose_suivante:.0f}"
        )

        rythme = "selon protocole sélectionné"

    else:
        return None

 
    premiere = injections[0]
    derniere = injections[-1]

    date_premiere = (
        f"{premiere['moment']} "
        f"({premiere['date'].strftime('%d/%m/%Y')})"
    )

    date_derniere = (
        f"{derniere['moment']} "
        f"({derniere['date'].strftime('%d/%m/%Y')})"
    )

    injections_suivantes = injections[1:-1]

    if injections_suivantes:
        dates_suivantes = "\n".join(
            f"- {inj['moment']} ({inj['date'].strftime('%d/%m/%Y')})"
            for inj in injections_suivantes
        )
    else:
        dates_suivantes = "Aucune injection intermédiaire."



    lignes = []

    lignes.append(
        "ORDONNANCE DE SOINS INFIRMIERS – RELAIS PRÉOPÉRATOIRE DES AVK PAR HÉPARINE"
    )

    lignes.append("")

    lignes.append(
        f"Poids du patient : {poids_kg} kgs"
    )

    lignes.append("")

    lignes.append(
        "Faire pratiquer à domicile par un(e) infirmier(ère) diplômé(e) d’État :"
    )

    lignes.append("")

    lignes.append(
        f"Injection sous-cutanée de {molecule.upper()}."
    )

    lignes.append(
        f"Dose : {dose} UI anti-Xa ou UI par injection."
    )

    lignes.append(
        f"Rythme : {rythme}."
    )

    lignes.append(
        f"Première injection : {date_premiere}"
    )

    lignes.append(
        "Injection(s) suivante(s) :"
    )

    lignes.append(
        dates_suivantes
    )

    lignes.append(
        f"Dernière injection préopératoire : {date_derniere}"
    )

    lignes.append("")

    lignes.append(
        "Ne réaliser aucune injection supplémentaire sans nouvelle prescription médicale."
    )

    lignes.append("")

    lignes.append(
        "Modalités : soins à domicile. Prescription qualitative et quantitative. Si les dates concernées comprennent un dimanche ou un jour férié, soins à réaliser de façon rigoureusement quotidienne selon le calendrier prescrit."
    )

    lignes.append("")

    lignes.append(
        "Pour l’énoxaparine en seringue graduée : administrer la dose exacte figurant sur l’ordonnance. Lorsque la présentation contient une quantité supérieure, éliminer l’excédent avant l’injection conformément au RCP."
    )

    lignes.append("")

    if date_op:
        lignes.append(
            f"Contexte opératoire : intervention prévue le {date_op.strftime('%d/%m/%Y')}."
        )

    return "\n".join(lignes)


def moteur_yaml(atc, ctx):
    atc = str(atc).upper().strip()
    liste_regles = REGLES.get("regles_medicaments") or []


    # ================= AOD =================
 

    if atc.startswith(("B01AE", "B01AF")):

        matchs_aod = []
        sources_aod = []

        for famille_aod in liste_regles:

            match_atc_aod = False

            if famille_aod.get("atc_prefix"):
                match_atc_aod = any(
                    atc.startswith(str(p).upper().strip())
                    for p in famille_aod["atc_prefix"]
                )

            if famille_aod.get("atc_codes"):
                match_atc_aod = atc in [
                    str(c).upper().strip()
                    for c in famille_aod["atc_codes"]
                ]


       

            if (
                famille_aod.get("conditions")
                and not famille_aod.get("atc_prefix")
                and not famille_aod.get("atc_codes")
            ):

                categorie_aod = str(
                    famille_aod.get("categorie", "")
                ).upper()

                conditions_aod = famille_aod.get("conditions", [])

                cles_aod = {
                    "reprise_aod_differee",
                    "aod_repris",
                    "thromboprophylaxie_indiquee_aod",
                    "heparine_curative_indiquee",
                    "dfg_ge_30",
                    "dfg_ge_50",
                    "dfg_30_49",
                    "dfg_15_29",
                    "dfg_inf_15",
                    "poids_ge_100",
                    "poids_inf_50",
                    "indication_aod",
                }

                contient_condition_aod = any(
                    any(
                        cle in cles_aod
                        for cle in (cond.get("if", {}) or {}).keys()
                    )
                    for cond in conditions_aod
                    if isinstance(cond, dict)
                )

                if "AOD" in categorie_aod or contient_condition_aod:
                    match_atc_aod = True


            if not match_atc_aod:
                continue


            for cond in famille_aod.get("conditions", []):

                if "if" not in cond:
                    continue

                test_regle = {
                    "conditions": [cond]
                }

                match = conditions_match(
                    ctx,
                    test_regle,
                    atc=atc
                )

                if match:
                    matchs_aod.append(match)


            # ================================
            # Sources AOD
            # ================================

            source_url = famille_aod.get("source_url")

            if source_url:

                if isinstance(source_url, list):

                    sources_aod.extend([
                        str(s).strip()
                        for s in source_url
                        if str(s).strip()
                    ])

                else:

                    sources_aod.append(
                        str(source_url).strip()
                    )


            elif famille_aod.get("sources"):

                sources = famille_aod.get("sources", [])

                if isinstance(sources, list):

                    sources_aod.extend([
                        str(s).strip()
                        for s in sources
                        if str(s).strip()
                    ])


            elif famille_aod.get("source_ref"):

                ref = famille_aod.get("source_ref")

                source_table = REGLES.get(
                    "sources_regles",
                    {}
                )

                if ref in source_table:

                    sources = source_table[ref].get(
                        "sources",
                        []
                    )

                    if isinstance(sources, list):

                        sources_aod.extend([
                            str(s).strip()
                            for s in sources
                            if str(s).strip()
                        ])


        if matchs_aod:

            principale = None


            # ================================
            # ARRET préop prioritaire
            # ================================

            for m in matchs_aod:

                if (
                    str(
                        m.get("action", "")
                    ).upper().strip()
                    == "ARRET"
                ):

                    principale = m
                    break


            if principale is None:
                principale = matchs_aod[0]



            notes = []

            for m in matchs_aod:

                texte = (
                    m.get("precision")
                    or m.get("note")
                    or ""
                )

                if texte and texte not in notes:
                    notes.append(texte)


            sources_aod = list(
                dict.fromkeys(
                    s
                    for s in sources_aod
                    if s
                )
            )


            return {
                "action": principale.get(
                    "action",
                    "INFO"
                ),
                "jour": principale.get(
                    "jour",
                    ""
                ),
                "note": "\n\n".join(notes),
                "source": " | ".join(sources_aod),
            }




    for famille in liste_regles:

        match_atc = False

        if famille.get("atc_prefix"):

            match_atc = any(
                atc.startswith(
                    str(p).upper().strip()
                )
                for p in famille["atc_prefix"]
            )

        if famille.get("atc_codes"):

            match_atc = atc in [
                str(c).upper().strip()
                for c in famille["atc_codes"]
            ]

        if (
            famille.get("conditions")
            and not famille.get("atc_prefix")
            and not famille.get("atc_codes")
        ):
            match_atc = True

        if not match_atc:
            continue


        lien_sfar = ""

        source_url = famille.get("source_url")

        if source_url:

            if isinstance(source_url, list):

                lien_sfar = " | ".join([
                    str(s).strip()
                    for s in source_url
                    if str(s).strip()
                ])

            else:

                lien_sfar = str(
                    source_url
                ).strip()


        elif famille.get("sources"):

            sources = famille.get(
                "sources",
                []
            )

            if isinstance(sources, list):

                lien_sfar = " | ".join([
                    str(s).strip()
                    for s in sources
                    if str(s).strip()
                ])


        elif famille.get("source_ref"):

            ref = famille.get("source_ref")

            source_table = REGLES.get(
                "sources_regles",
                {}
            )

            if ref in source_table:

                sources = source_table[
                    ref
                ].get(
                    "sources",
                    []
                )

                if isinstance(sources, list):

                    lien_sfar = " | ".join([
                        str(s).strip()
                        for s in sources
                        if str(s).strip()
                    ])


        res = {
            "action": famille.get(
                "action",
                "POURSUITE"
            ),
            "jour": famille.get(
                "jour",
                "J0"
            ),
            "note": (
                famille.get("precision")
                or famille.get("note")
                or "-"
            ),
            "source": lien_sfar,
        }


        res_cond = conditions_match(
            ctx,
            famille,
            atc=atc
        )



        # ================= AVK =================

        if atc.startswith("B01AA"):

            matchs = []

            for cond in famille.get(
                "conditions",
                []
            ):

                if "if" not in cond:
                    continue

                test_regle = {
                    "conditions": [cond]
                }

                match = conditions_match(
                    ctx,
                    test_regle,
                    atc=atc
                )

                if match:
                    matchs.append(match)


            if matchs:

                # == PRIORITE REGLES SPE =====

                indication = ctx.get(
                    "indication_avk"
                )

                actions_specifiques = {
                    m.get("action")
                    for m in matchs
                    if (
                        (m.get("if", {}) or {})
                        .get("indication_avk")
                        == indication
                    )
                }


                matchs = [
                    m
                    for m in matchs
                    if not (
                        (m.get("if", {}) or {})
                        .get("indication_avk") is None

                        and m.get("action")
                        in actions_specifiques

                        and m.get("action")
                        in [
                            "REPRISE AVK",
                            "RELAIS POSTOPERATOIRE"
                        ]
                    )
                ]



                dfg_relais = ctx.get(
                    "dfg_relais_avk"
                )


                if dfg_relais in [
                    "15 ≤ DFG < 30",
                    "DFG < 15"
                ]:

                    matchs = [
                        m
                        for m in matchs
                        if not (
                            m.get("action")
                            == "RELAIS PREOPERATOIRE"

                            and

                            "Réaliser un relais pré-procédural par HBPM"
                            in (
                                m.get("precision")
                                or m.get("note")
                                or ""
                            )
                        )
                    ]



                if dfg_relais in [
                    "15 ≤ DFG < 30",
                    "DFG < 15"
                ]:

                    relais_postop_renal_present = any(
                        m.get("action")
                        == "ADAPTATION RELAIS POSTOPERATOIRE"
                        for m in matchs
                    )


                    if relais_postop_renal_present:

                        matchs = [
                            m
                            for m in matchs
                            if m.get("action")
                            != "RELAIS POSTOPERATOIRE"
                        ]



                arret_heparine_deja_dans_specifique = any(

                    (m.get("if", {}) or {})
                    .get("indication_avk")
                    == indication

                    and

                    "Arrêter l’héparine dès le premier INR ≥ 2"
                    in (
                        m.get("precision")
                        or m.get("note")
                        or ""
                    )

                    for m in matchs
                )


                if arret_heparine_deja_dans_specifique:

                    matchs = [
                        m
                        for m in matchs
                        if not (
                            (m.get("if", {}) or {})
                            .get("indication_avk") is None

                            and

                            m.get("action")
                            == "ARRET HEPARINE"
                        )
                    ]


                principale = None



                # LVAD prioritaire

                if ctx.get(
                    "indication_avk"
                ) == "LVAD":

                    for m in matchs:

                        bloc_if = (
                            m.get(
                                "if",
                                {}
                            )
                            or {}
                        )

                        if (
                            bloc_if.get(
                                "indication_avk"
                            )
                            == "LVAD"
                        ):

                            principale = m
                            break



                # Sinon : arrêt/poursuite AVK

                if principale is None:

                    for m in matchs:

                        bloc_if = (
                            m.get(
                                "if",
                                {}
                            )
                            or {}
                        )

                        if (
                            "atc_codes"
                            in bloc_if

                            or

                            (
                                "r_hem"
                                in bloc_if
                                and len(
                                    bloc_if
                                ) == 1
                            )
                        ):

                            principale = m
                            break


                if principale is None:
                    principale = matchs[0]



                # ================================
                # priorité chrono avk
                # ==========================================

                priorite_action_avk = {

                    "ARRET": 10,
                    "ARRET + RELAI": 10,

                    "RELAIS PREOPERATOIRE": 20,
                    "RELAIS HBPM": 21,
                    "HNF CALCIQUE SC": 21,
                    "ADAPTATION RELAIS": 22,
                    "ADAPTATION HNF IVSE": 22,
                    "VALIDATION MEDICALE": 23,

                    "CONTROLE INR": 30,
                    "VITAMINE K": 31,

                    "INFO": 40,
                    "DIFFERER PROCEDURE": 40,
                    "DISCUSSION MEDICALE": 40,
                    "AVIS_SPECIALISE": 40,

                    "REPRISE AVK": 50,
                    "PAS DE RELAIS POSTOPERATOIRE": 60,

                    "RELAIS POSTOPERATOIRE": 70,
                    "ADAPTATION RELAIS POSTOPERATOIRE": 71,

                    "ARRET HEPARINE": 80,
                    "DECISION MEDICALE": 90,
                    "THROMBOPROPHYLAXIE": 90,
                }


                matchs = sorted(
                    matchs,
                    key=lambda m:
                    priorite_action_avk.get(
                        str(
                            m.get(
                                "action",
                                ""
                            )
                        ).upper().strip(),
                        50
                    )
                )



                notes = []

                for m in matchs:

                    texte = (
                        m.get("precision")
                        or m.get("note")
                        or ""
                    )

                    if texte and texte not in notes:
                        notes.append(texte)





                if (
                    ctx.get("reprise_avk_24h", False)
                    and not ctx.get("relais_postop_indique", False)
                ):

                    texte_pas_relais_postop = (
                        "Ne pas réaliser de relais héparinique curatif postopératoire."
                    )
  
                    if texte_pas_relais_postop not in notes:
                        notes.append(texte_pas_relais_postop)




                if ctx.get("thromboprophylaxie_indiquee", False):

                    texte_thromboprophylaxie = (
                        "Réaliser une thromboprophylaxie veineuse postopératoire "
                        "selon les indications et modalités habituelles, en attendant "
                        "l’anticoagulation curative. L’interrompre dès que le patient "
                        "est anticoagulé à dose curative."
                    )

                    if texte_thromboprophylaxie not in notes:
                        notes.append(texte_thromboprophylaxie)


                return {
                    "action": principale.get(
                        "action",
                        res["action"]
                    ),
                    "jour": principale.get(
                        "jour",
                        res["jour"]
                    ),
                    "note": "\n\n".join(
                        notes
                    ),
                    "source": lien_sfar,
                }





        # ================= AUTRES MEDICAMENTS =================

        res_cond = conditions_match(
            ctx,
            famille,
            atc=atc
        )


        if res_cond:

            return {
                "action": res_cond.get(
                    "action",
                    res["action"]
                ),
                "jour": res_cond.get(
                    "jour",
                    res["jour"]
                ),
                "note": (
                    res_cond.get("precision")
                    or res_cond.get("note")
                    or res["note"]
                ),
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

    if ctx.get("corticoides"):
        return {
            "action": "POURSUITE",
            "jour": "J0",
            "note": "Poursuite simple, sans supplémentation.",
            "source": ""
        }

    return {
        "action": "NON SPECIFIE",
        "jour": "",
        "note": "Aucune recommandation spécifique retrouvée dans le référentiel.",
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
    def u(v):
        return str(v or "").upper().strip()

    type_chir = u(ctx.get("type_chir"))
    is_neurochir = ctx.get("type_chir_neuro") == "NEUROCHIR_INTRACRANIENNE"
    r_hem = u(ctx.get("r_hem"))
    alr = u(ctx.get("alr"))
    stress_cortico_faible = ctx.get("stress_cortico_faible", False)
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

            choix_sraa_hta = ctx.get("choix_sraa_hta", "")

            if choix_sraa_hta == "Arrêter":
                return {
                    "action": "ARRET",
                    "jour": ">=12h",
                    "note": (
                        "Recommandation SFAR actuelle : arrêt préopératoire lorsque le "
                        "traitement est prescrit pour une hypertension artérielle ; "
                        "données récentes de la littérature n’ont pas montré de bénéfice "
                        "clinique clair à l’arrêt systématique de ces médicaments avant "
                        "une chirurgie non cardiaque chez les patients traités pour "
                        "hypertension artérielle : la décision doit être adaptée au patient "
                        "et au contexte opératoire."
                    )
                }

            elif choix_sraa_hta == "Poursuivre":
                return {
                    "action": "POURSUITE",
                    "jour": "J0",
                    "note": (
                        "Poursuite choisie par l’anesthésiste. "
                        "La décision doit être adaptée au patient et au contexte opératoire."
                    )
                }

        return {
            "action": "POURSUITE",
            "jour": "J0",
            "note": "Indication Insuffisance Cardiaque : maintien recommandé."
        }



        # Bipreterax / association IEC + diurétique
        if atc == "C09BA04":

            if ind_sraa == "HTA":

                choix_sraa_hta = ctx.get("choix_sraa_hta", "")

                if choix_sraa_hta == "Arrêter":
                    return {
                        "action": "ARRET",
                        "jour": ">=12h",
                        "note": "Arrêt préopératoire choisi par l’anesthésiste."
                    }

                elif choix_sraa_hta == "Poursuivre":
                    return {
                        "action": "POURSUITE",
                        "jour": "J0",
                        "note": "Poursuite préopératoire choisie par l’anesthésiste."
                    }

            return {
                "action": "POURSUITE",
                "jour": "J0",
                "note": "Indication Insuffisance Cardiaque : maintien recommandé."
            }



    # Entresto
    if atc == "C09DX04":
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
    if atc.startswith("M01A"):
        if r_hem in ["ELEVE", "IMPORTANT", "MAJEUR"]:
            return {"action": "ARRET", "note": "Arrêt selon 5 demi-vies."}
        return {"action": "POURSUITE"}


    #Psy / neuro
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




    #  AVK
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

    if atc in classe_map:
        return classe_map[atc]

    for i in range(len(atc), 2, -1):
        prefix = atc[:i]
        if prefix in classe_map:
            return classe_map[prefix]

    if atc.startswith("A10"):
        return "Antidiabétique"

    return "Inconnue"


# =========================
#Bithérapie 2 aap détectés
# =========================
def compter_aap_dans_texte(txt, ref, atc_map):
  
    codes_trouves = set()

    candidats = []
    for brute, nettoyee in extraire_lignes_candidates_imprime(txt):
        candidats.append(nettoyee)

    for brute, nettoyee in extraire_lignes_candidates_manuscrit(txt):
        candidats.append(nettoyee)

    for cand in candidats:
        meilleur_nom, meilleur_score = meilleur_match_medicament(cand, ref)
        if meilleur_nom and meilleur_score >= 75:
            code = str(atc_map.get(meilleur_nom, "")).upper().strip()

            if code == "B01AC30":
                return 2

            if code.startswith("B01AC"):
                codes_trouves.add(code)

    return len(codes_trouves)


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

    mots = t.split()
    if len(mots) > 2:
        return False

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


    if len(mots) == 2:
        if len(mots[0]) <= 2 or len(mots[1]) <= 2:
            return False

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



def extraire_nom_propre(ligne):
    l = normalize_text(ligne)
    l = l.replace("µ", "U")

   
    l = re.sub(r"\(.*?\)", " ", l)

   
    l = re.sub(r"^[\-\•\.\*\s]+", "", l)

   
    l = re.split(
    r":| MG| G| ML| UI| MUI| MCG| UG| µG| SOL| BUV| AMP| AMPOULE| MATIN| SOIR| MIDI| JOUR| COMPRIME| COMPRIMES| GEL| GELULE| GELULES| SACHET| SACHETS| CP| SI BESOIN",
    l
    )[0]
   
    l = DOSE_PATTERN.sub("", l)
    l = re.sub(r"\b\d+[.,]?\d*\b", "", l)

  
    l = re.sub(r"[^A-Z\s\-]", " ", l)
    l = re.sub(r"\s+", " ", l).strip()

    return l

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
    # extrait seulement le nom du médicament au début de la ligne
    # Exemple : "Metformine 850 mg : 1 comprimé matin et soir" -> "Metformine"
        nom_court = extraire_nom_propre(brute)

        meilleur_nom, meilleur_score = meilleur_match_medicament(nom_court, ref)
        seuil = 90 if mode == "imprime" else 80

    
        
        # CAS 1 medicament non reconnu 
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

        ctx_med = ctx.copy()

        if atc.startswith(("B01AA", "B01AE", "B01AF")):
            ctx_med["r_hem"] = ctx.get("r_hem_aod_avk")

        elif atc.startswith("B01AC"):
            ctx_med["r_hem"] = ctx.get("r_hem_aap")

        else:
            ctx_med["r_hem"] = ""

        ans = moteur_global(atc, ctx_med)

   

        if not ans:
            ans = {
                "action": "POURSUITE",
                "jour": "J0",
                "note": "Médicament reconnu dans le référentiel, sans règle spécifique identifiée : poursuite, sans impact anesthésique évident, à vérifier selon le contexte clinique.",
                "source": ""
            }

        atc_affiche = atc
        classe_affiche = get_classe(atc_affiche, classe_map)
        nom_resultat = meilleur_nom.title()
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


@st.cache_data
def load_data():
    try:
        atc = pd.read_csv(os.path.join(BASE_DIR, "dci_atc.fichier.csv"), sep=";")
        inter = pd.read_csv(os.path.join(BASE_DIR, "risque.hemorragique2.csv"), sep=";")
        taxo = pd.read_csv(os.path.join(BASE_DIR, "TAXONOMIE-Tableau 1.csv"), sep=";")
        libelles = pd.read_csv(os.path.join(BASE_DIR, "LISTE_FINALE_AVEC_LIBELLES.csv"), sep=";")
        sentinelles = pd.read_csv(os.path.join(BASE_DIR, "Medicaments Sentinelles-Tableau.csv"), sep=";")
        profils = pd.read_csv(os.path.join(BASE_DIR, "Profils Pathologiques-Tableau.csv"), sep=";")

        atc.columns = [normalize_colname(c) for c in atc.columns]
        inter.columns = [normalize_colname(c) for c in inter.columns]
      
        taxo.columns = [normalize_colname(c) for c in taxo.columns]
        libelles.columns = [normalize_colname(c) for c in libelles.columns]
        sentinelles.columns = [normalize_colname(c) for c in sentinelles.columns]
        profils.columns = [normalize_colname(c) for c in profils.columns]

        atc_map = {}

        # pour éviter les bugs accents / espaces
        def norm(x):
            return normalize_text(x)

        for k, v in zip(atc["MEDICAMENT_SOURCE"], atc["CODE_ATC"]):
            if pd.notna(k) and pd.notna(v):
                atc_map[norm(k)] = str(v).upper().strip()

        if "NOM_COMMERCIAL" in atc.columns:
            for k, v in zip(atc["NOM_COMMERCIAL"], atc["CODE_ATC"]):
                if pd.notna(k) and pd.notna(v):
                    atc_map[norm(k)] = str(v).upper().strip()

        if "DCI" in atc.columns:
            for k, v in zip(atc["DCI"], atc["CODE_ATC"]):
                if pd.notna(k) and pd.notna(v):
                    atc_map[norm(k)] = str(v).upper().strip()
  
        corrections = {
            "PREVISCAN": "B01AA12",
            "FLUINDIONE": "B01AA12",
            "SINTROM": "B01AA07",
            "DUOPLAVIN": "B01AC30",
        }


        corrections.update({
            "HEPARINE": "B01AB01",
            "HEPARINE IV": "B01AB01",
            "HEPARINE IVSE": "B01AB01",
            "HEPARINE SC": "B01AB01",
            "HNF": "B01AB01",
            "HNF IV": "B01AB01",
            "HNF IVSE": "B01AB01",
            "HNF SC": "B01AB01"
        })


        for k, v in corrections.items():
            atc_map[norm(k)] = v

        classe_map = {
            str(k).upper().strip(): str(v).strip()
            for k, v in zip(libelles["CODE_ATC"], libelles["NOM_DEUXIEME_CLASSE"])
            if pd.notna(k) and pd.notna(v)
        }

        # fichier libelles dans atc_map
        if "MEDICAMENT" in libelles.columns and "CODE_ATC" in libelles.columns:
            for nom, code in zip(libelles["MEDICAMENT"], libelles["CODE_ATC"]):
                if pd.notna(nom) and pd.notna(code):
                    atc_map[norm(nom)] = str(code).upper().strip()

        # alias utiles formes galéniques
        if "EURAX" in atc_map and "EURAX CREME" not in atc_map:
            atc_map["EURAX CREME"] = atc_map["EURAX"]

        if "CUTACNYL" in atc_map and "CUTACNYL 5 GEL" not in atc_map:
            atc_map["CUTACNYL 5 GEL"] = atc_map["CUTACNYL"]

        if "CUTACNYL" in atc_map and "CUTACNYL 5% GEL" not in atc_map:
            atc_map["CUTACNYL 5% GEL"] = atc_map["CUTACNYL"]

        ref = list(atc_map.keys())

        return atc_map, ref, classe_map, inter, taxo, sentinelles, profils


    except Exception as e:
        st.error(f"Erreur fichiers CSV : {e}")
        return {}, [], {}, pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()


atc_map, ref, classe_map, df_inter, df_taxo, df_sentinelles, df_profils = load_data()
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

    hits_list = []

    df_sent = df_sentinelles_ready.copy()
    df_sent["CODE ATC CLEAN"] = df_sent["CODE ATC"].astype(str).str.upper().str.strip()

    df_prof = df_profils_ready.copy()
    df_prof["CODES ATC CLES CLEAN"] = df_prof["CODES ATC CLES"].astype(str).str.upper()

    for code_patient in codes_atc_detectes:
        code_patient = str(code_patient).upper().strip()

        exact = df_sent[df_sent["CODE ATC CLEAN"] == code_patient]
 
        if not exact.empty:
            hits_list.append(exact.iloc[0].to_dict())
            continue

        prefix6 = df_sent[
            df_sent["CODE ATC CLEAN"].apply(
                lambda code_ref: code_patient.startswith(str(code_ref)[:6])
            )
        ]

        if not prefix6.empty:
            hits_list.append(prefix6.iloc[0].to_dict())
            continue

        for _, row in df_prof.iterrows():
            texte_codes = str(row.get("CODES ATC CLES CLEAN", ""))
            codes_refs = re.findall(r"[A-Z]\d{2}[A-Z0-9]{0,4}", texte_codes)

            if any(code_patient.startswith(code_ref) for code_ref in codes_refs):
                hits_list.append({
                    "MEDICAMENT SENTINELLE": "",
                    "CODE ATC": code_patient,
                    "PROFIL IDENTIFIE": row.get("PROFIL PATHOLOGIQUE", ""),
                    "SEUL SUFFIT": "OUI",
                    "CERTITUDE": "ELEVEE"
                })
                break

    hits = pd.DataFrame(hits_list)

    hits.columns = [str(c).upper().strip() for c in hits.columns]

    if "PROFIL IDENTIFIE" in hits.columns:
        hits["_PROFIL_NORM"] = hits["PROFIL IDENTIFIE"].apply(normalize_text)
        hits = hits.drop_duplicates(subset=["_PROFIL_NORM"], keep="first")
        hits = hits.drop(columns=["_PROFIL_NORM"])

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

    rows_dict = {}

    for profil_norm, score in profile_scores.items():
        sub = df_profils_ready[df_profils_ready["PROFIL PATHOLOGIQUE_NORM"] == profil_norm]

        if not sub.empty:
            rowp = sub.iloc[0]
            libelle = corriger_nom_profil(clean_display_value(rowp.get("PROFIL PATHOLOGIQUE", "")))
            asa_min = clean_display_value(rowp.get("ASA MIN", ""))
        else:
            libelle = corriger_nom_profil(profil_norm.title())
            asa_min = ""

        cle = normalize_text(libelle)

        certs = sorted(set(profile_certitudes[profil_norm]))
        if "TRES ELEVEE" in certs:
            niveau = "Très forte"
        elif "ELEVEE" in certs:
            niveau = "Forte"
        else:
            niveau = "Modérée"

        if cle not in rows_dict or score > rows_dict[cle]["Score"]:
            rows_dict[cle] = {
                "Profil": libelle,
                "Score": score,
                "Niveau": niveau,
                "ASA min": asa_min,
                "ATC": ", ".join(sorted(set(profile_atc[profil_norm]))),
                "Sentinelles": ", ".join(sorted(set(profile_sentinelles[profil_norm])))
            }

    rows = list(rows_dict.values())

    return pd.DataFrame(rows).sort_values("Score", ascending=False).head(3).reset_index(drop=True)

# =========================================================
# SIDEBAR
# =========================================================
with st.sidebar:
    st.header("Dossier Patient")
    iep = st.text_input("Numéro IEP")
    age = st.number_input("Âge", 0, 115, 65)
    date_op = st.date_input("Date intervention", date.today() + timedelta(days=7))


    st.divider()
    st.header("Chirurgie")

    if not df_inter.empty and "SPECIALITE" in df_inter.columns:

        liste_specialites = sorted(
            s for s in df_inter["SPECIALITE"].dropna().unique()
            if str(s).strip().upper() != "ALR"
        )

        liste_specialites_affichage = [
            s.replace("CHIR ", "").title()
            for s in liste_specialites
        ]

        spe_affichage = st.selectbox(
            "Spécialité",
            liste_specialites_affichage,
            key="specialite_chirurgie"
        )

        mapping_specialites = dict(zip(liste_specialites_affichage, liste_specialites))
        spe = mapping_specialites[spe_affichage]

        

        df_grp = df_inter[df_inter["SPECIALITE"] == spe].copy()

        grp = st.selectbox(
            "Groupe",
            sorted(df_grp["SOUS-GROUPE"].dropna().unique()),
            key="groupe_chirurgie"
        )

        df_actes_filtre = df_grp[df_grp["SOUS-GROUPE"] == grp].copy()

        liste_actes = sorted(
            df_actes_filtre["INTERVENTION CHIRURGICALE"].dropna().unique()
        )

        acte_nom = st.selectbox(
            "Intervention",
            liste_actes if liste_actes else ["Aucune intervention trouvée"],
            key="intervention_chirurgie"
        )


        type_alr_affichage = st.selectbox(
            "ALR prévue",
           [
                "Aucune ALR",
                "Anesthésie neuraxiale",
                "Blocs périphériques profonds",
                "Blocs périphériques superficiels"
            ],
            key="type_alr_chirurgie"
        )

        mapping_alr = {
            "Aucune ALR": "",
            "Anesthésie neuraxiale": "NEURAXIAL",
            "Blocs périphériques profonds": "PROFOND",
            "Blocs périphériques superficiels": "SUPERFICIEL"
        }

        type_alr = mapping_alr[type_alr_affichage]


        technique_neuraxiale = ""

        if type_alr_affichage == "Anesthésie neuraxiale":
            technique_neuraxiale = st.selectbox(
                "Technique neuraxiale",
                [
                    "Rachianesthésie",
                    "Péridurale",
                    "Péri-rachi combinée"
                ],
                key="technique_neuraxiale"
             )



        type_chir = spe


        
        is_neuro = False
        if spe is not None and str(spe).strip().upper() in ["NEUROCHIRURGIE", "RACHIS"]:
            is_neuro = True
        data_acte_filtre = pd.DataFrame()
        if not df_actes_filtre.empty and acte_nom != "Aucune intervention trouvée":
            data_acte_filtre = df_actes_filtre[
                df_actes_filtre["INTERVENTION CHIRURGICALE"] == acte_nom
            ].copy()

           
        if not data_acte_filtre.empty:
            data_acte = data_acte_filtre.iloc[0]

         

            risque_aod_avk = val_upper(data_acte.get("RISQUE H AOD - AVK", "NON RENSEIGNE"))
            risque_aap = val_upper(data_acte.get("RISQUE H AAP", "NON RENSEIGNE"))
            asa_acte = clean_display_value(data_acte.get("ASA", ""))
            antibio = data_acte.get("ANTIBIOPROPHYLAXIE", "Non renseignée")
            dose_antibio = data_acte.get("DOSE", "Non renseignée")

            id_acte = clean_display_value(data_acte.get("ID", ""))

            id_acte = clean_display_value(data_acte.get("ID", "")).strip().upper()

            stress_cortico_raw = clean_display_value(
                data_acte.get(
                    "STRESS CHIR CORTICO",
                    data_acte.get("stress chir cortico", "")
                )
            )

            stress_cortico_norm = val_upper(stress_cortico_raw)
            stress_cortico_faible = stress_cortico_norm == "FAIBLE"



            stress_cortico_norm = val_upper(stress_cortico_raw)

            stress_cortico_faible = (stress_cortico_norm == "FAIBLE")


        else:
            data_acte = pd.Series(dtype=object)
            risque_aod_avk = "NON RENSEIGNE"
            risque_aap = "NON RENSEIGNE"
            asa_acte = ""
            antibio = "Non renseignée"
            dose_antibio = "Non renseignée"
            stress_cortico_raw = ""
            stress_cortico_norm = ""
            stress_cortico_faible = False
 

    else:
        spe = None
        grp = None
        acte_nom = None
        is_neuro = False
        data_acte = pd.Series(dtype=object)
        risque_aod_avk = "NON RENSEIGNE"
        risque_aap = "NON RENSEIGNE"
        asa_acte = ""
        antibio = "Non renseignée"
        dose_antibio = "Non renseignée"
        stress_cortico_raw = ""
        stress_cortico_norm = ""
        stress_cortico_faible = False
        st.warning("Taxonomie chirurgie indisponible ou colonnes non reconnues.")


#RAPPEL ALR PROFONDES

    if type_alr == "PROFOND":
        with st.expander(" Rappel – ALR profondes"):
            st.markdown("""
- Ganglion stellaire  
- Plexus cervical profond  
- Paravertébral cervical  
- Infraclaviculaire  
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
# ----------------------------
# RAPPEL ALR NEURAXIAL
# ----------------------------

    if type_alr == "NEURAXIAL":
        with st.expander("Rappel – ALR neuraxial"):
          st.markdown("""
- Ponction lombaire  
- Rachi-anesthésie  
- Péridurale  
- Péri-rachi anesthésie combinée  
        """)



# =========================================================
# INTERFACE PRINCIPALE
# =========================================================
st.title("IA CARE - système d'aide à la décision")

st.markdown("""
<style>

/* Tous les boutons */
div[data-testid="stButton"] button {
    background: linear-gradient(180deg, #ffffff 0%, #f8fbff 100%) !important;
    border: 1.5px solid #cfe0ff !important;
    border-radius: 16px !important;
    height: 68px !important;
    font-size: 18px !important;
    font-weight: 600 !important;
    color: #2f3140 !important;
    box-shadow: 0 3px 10px rgba(0,0,0,0.04) !important;
}

/* Bouton sélectionné */
div[data-testid="stButton"] button[kind="primary"] {
    background: linear-gradient(180deg, #eef5ff 0%, #e4efff 100%) !important;
    border: 2px solid #8ab6ff !important;
    color: #0757c2 !important;
    font-weight: 700 !important;
    box-shadow: 0 6px 16px rgba(7,87,194,0.12) !important;
}

/* Survol */
div[data-testid="stButton"] button:hover {
    border: 2px solid #8ab6ff !important;
    background: #f1f7ff !important;
    color: #0757c2 !important;
}

</style>
""", unsafe_allow_html=True)


if "mode_entree" not in st.session_state:
    st.session_state.mode_entree = None

if "txt" not in st.session_state:
    st.session_state.txt = ""

if "ocr_lines" not in st.session_state:
    st.session_state.ocr_lines = []

if "manual_meds_buffer" not in st.session_state:
    st.session_state.manual_meds_buffer = ""

if "manual_meds_validated" not in st.session_state:
    st.session_state.manual_meds_validated = ""

if "ajout_manuel" not in st.session_state:
    st.session_state.ajout_manuel = ""

st.markdown("### Choisir une méthode d'entrée")

col_voix, col_scan, col_manuel = st.columns(3)

with col_voix:
    voix_type = "primary" if st.session_state.mode_entree == "voix" else "secondary"

    if st.button("Dictée vocale", use_container_width=True, type=voix_type):
        st.session_state.mode_entree = "voix"
        st.rerun()

with col_scan:
    scan_type = "primary" if st.session_state.mode_entree == "scan" else "secondary"

    if st.button("Scan ordonnance", use_container_width=True, type=scan_type):
        st.session_state.mode_entree = "scan"
        st.rerun()

with col_manuel:
    manuel_type = "primary" if st.session_state.mode_entree == "manuel" else "secondary"

    if st.button("Saisie manuelle", use_container_width=True, type=manuel_type):
        st.session_state.mode_entree = "manuel"
        st.rerun()


if st.session_state.mode_entree == "voix":

    audio = st.audio_input(
        "Dictée vocale, veuillez parler clairement et lentement, en énonçant les médicaments un par un."
    )

    if audio and st.button("Transcrire voix"):
        try:
            lignes = transcrire_audio_robuste(audio)
            st.session_state.txt = "\n".join(lignes)
            st.session_state.ocr_lines = lignes
        except Exception as e:
            st.error(f"Erreur transcription audio : {e}")


elif st.session_state.mode_entree == "scan":

    photo = st.file_uploader(
        "Scan Ordonnance ou PDF",
        type=["jpg", "png", "jpeg", "pdf"]
    )

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

    if photo and st.button("Lancer Scan Document"):
        try:
            if str(getattr(photo, "type", "")).lower() == "application/pdf":
                lignes = extraire_texte_pdf(photo)
            else:
                img = Image.open(photo).convert("RGB")
                lignes = extraire_lignes_ocr_image(img)

            lignes_filtrees = filtrer_lignes_scan_avec_dosage(lignes)

            st.session_state.ocr_lines = lignes_filtrees
            st.session_state.txt = "\n".join(lignes_filtrees)
            photo.seek(0)

        except Exception as e:
            st.error(f"Erreur OCR document : {e}")


elif st.session_state.mode_entree == "manuel":

    manual_meds = st.text_area(
        "Saisie manuelle",
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


if "ajout_manuel" not in st.session_state:
    st.session_state.ajout_manuel = ""

question_ajout = st.checkbox("Compléter les traitements si besoin")

if question_ajout:

    ajout_manuel = st.text_area(
        "",
        value=st.session_state.ajout_manuel,
        height=80,
        placeholder="Ajouter un traitement..."
    )

    col_a, col_b = st.columns([1, 1])

    with col_a:
        if st.button("Ajouter"):
            st.session_state.ajout_manuel = ajout_manuel
            st.success("Traitement ajouté.")

    with col_b:
        if st.button("Effacer ajout"):
            st.session_state.ajout_manuel = ""
            st.rerun()


if (
    st.session_state.get("txt")
    or st.session_state.get("manual_meds_validated")
    or st.session_state.get("ajout_manuel")
):

    if st.button("Effacer les données détectées"):
        st.session_state.txt = ""
        st.session_state.ocr_lines = []
        st.session_state.manual_meds_buffer = ""
        st.session_state.manual_meds_validated = ""
        st.session_state.ajout_manuel = ""
        st.rerun()




txt_source = st.session_state.txt
manual_text = st.session_state.manual_meds_validated.strip()
ajout_text = st.session_state.ajout_manuel.strip()

if manual_text:
    if txt_source.strip():
        txt_source = txt_source + "\n" + manual_text
    else:
        txt_source = manual_text

if ajout_text:
    if txt_source.strip():
        txt_source = txt_source + "\n" + ajout_text
    else:
        txt_source = ajout_text

with st.expander("Voir les données détectées"):
    txt_final = st.text_area(
        "",
        txt_source,
        height=180
    )

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
    atc_codes=["B01AC01", "B01AC06", "B01AC04", "B01AC24", "B01AC22", "B01AC30"],
    mots_secours=[
        "ASPIRINE", "KARDEGIC", "CLOPIDOGREL", "PLAVIX",
        "TICAGRELOR", "BRILIQUE", "PRASUGREL", "EFIENT"
    ]
)


aspirine_seule_detectee = contexte_famille_detecte(
    txt_final,
    ref,
    atc_map,
    atc_codes=["B01AC06"],
    mots_secours=["ASPIRINE", "KARDEGIC"]
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
choix_sraa_hta = None

if sraa_detecte:
    st.divider()
    st.header("Système rénine–angiotensine (SRAA)")

    ind_sraa = st.radio(
        "Indication du traitement (IEC / ARA II)",
        ["HTA", "Insuffisance Cardiaque"],
        index=0
    )

    if ind_sraa == "HTA":

        choix_sraa_hta = st.radio(
            "Conduite préopératoire",
            ["Arrêter", "Poursuivre"],
            index=0,
            key="choix_sraa_hta"
        )

        st.caption(
            "Recommandation SFAR actuelle : arrêt préopératoire lorsque le traitement "
            "est prescrit pour une hypertension artérielle ; données récentes de la "
            "littérature n’ont pas montré de bénéfice clinique clair à l’arrêt "
            "systématique de ces médicaments avant une chirurgie non cardiaque chez "
            "les patients traités pour hypertension artérielle : la décision doit être "
            "adaptée au patient et au contexte opératoire."
        )



type_traitement_aap = None
contexte_stent = "Aucun critère"
dose_aspirine = 75
indication_aap = None
nb_aap_detectes = compter_aap_dans_texte(txt_final, ref, atc_map)
bitherapie_auto = nb_aap_detectes >= 2

if aap_detecte:
    st.divider()
    st.header("Antiagrégants plaquettaires (AAP)")

    ## Si duopavin ou 2 AAP détectés
    if bitherapie_auto:
        st.info("Deux AAP détectés : bithérapie présumée.")
        type_traitement_aap = "Bithérapie"

    ## kardegic / aspirine seule
    elif aspirine_seule_detectee:
        type_traitement_aap = st.radio(
            "Type de traitement",
            ["Prévention primaire", "Prévention secondaire"],
            index=0
        )

    ## plavix / brilique / efient / autre AAP seul
    else:
        type_traitement_aap = "Prévention secondaire"





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
- Antécédent de thrombose de stent sous bithérapie AAP
- Coronaropathie diffuse (surtout chez les diabétiques)
- Insuffisance rénale chronique (DFG < 60 mL/min)
- Occlusion coronaire chronique
- Stenting de la dernière artère coronaire perméable
- Au moins 3 stents implantés
- Au moins 3 lésions traitées
- Bifurcation avec 2 stents implantés
- Longueur totale des stents > 60 mm
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
relais_avk = False
dfg_relais_avk = ""
inr_disponible = "Non"
inr_valeur = None



poids_kg = 0
indication_avk = ""
lvad_reprise_24_48h = False

mtev_moins_1_mois = False
mtev_complexe = False
procedure_differable = False

fa_avc_moins_3_mois = False

reprise_avk_24h = False

valve_aortique_double_ailette = False
rythme_sinusal = False
atcd_thrombotique_valve = False
valve_faible_risque_postop = False

mtev_relais_postop = False
relais_postop_indique = False
inr_ge_2_postop = False
inr_disponible = "Non"
inr_valeur = None

poids_kg = 0

reprise_avk_24h = False
relais_postop_indique = False
inr_ge_2_postop = False
chevauchement_non_acceptable = False
anticoag_curative_non_reprise = False
thromboprophylaxie_indiquee = False
facteur_hemorragique_supplementaire = False

mtev_moins_1_mois = False
risque_recidive_mtev = None
deficit_proteine_c_s = False
lvad_sans_arret_avk = False
lvad_reprise_24_48h = False
hnf_calcique_sc_choisie = False
hnf_ivse_choisie = False
mode_prise_en_charge_relais = None
type_heparine_relais = None
schema_hbpm = None
reprise_avk_24h_bool = False




if avk_detecte:
    st.divider()
    st.header("Anti-vitamine K (AVK)")


    st.subheader("Poids")

    poids_kg = st.number_input(
        "Poids du patient (kg) si connu",
        min_value=0,
        value=0,
        step=1,
        key="poids_kg"
    )




    facteur_hemorragique_supplementaire = False

    if risque_aod_avk in ["FAIBLE", "NUL"]:
        facteur_hemorragique_supplementaire = st.checkbox(
            "Facteur hémorragique supplémentaire important signalé par le médecin",
            key="facteur_hemorragique_supplementaire"
        )


   # ===INR =============

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

if avk_detecte:
    show_inr = st.toggle("INR complément", key="toggle_inr_complement")

    if show_inr:
        st.info("""
### Objectif INR péri-opératoire (AVK)


Objectif standard :
- INR < 1,5
- INR < 1,2 si neurochirurgie
- Zone thérapeutique usuelle possible (INR compris entre 2 et 3)

---

### Rappels des objectifs d'INR, selon recommandations ESC 2025

### Valves mécaniques

- Valve mitrale / tricuspide / ancienne génération  
  → **INR cible = 3 (2,5 - 3,5)**

- Valve aortique moderne (bileaflet)  
  → **INR cible = 2,5 (2 - 3)**

### Autres indications

**INR cible 2 - 3 :**
- Fibrillation atriale non valvulaire
- Prévention et traitement TVP / EP
- Syndrome des antiphospholipides (selon terrain)


**Valvulopathie mitrale avec :**
- Dilatation de l’oreillette gauche  
- Contraste spontané en ETO  
- Thrombus intra-auriculaire gauche  
---

### Facteurs pro-thrombotiques à rechercher

Si ≥ 1 facteur présent → augmenter la cible INR de +0,5

- Fibrillation atriale  
- Dysfonction VG (FEVG < 35 %)  
- État hypercoagulable  
- Événement thrombotique récent (< 12 mois : AVC, TVP, EP)
""")







# ================= RELAI =========

if avk_detecte:

    if risque_aod_avk not in ["FAIBLE", "NUL"]:

        st.subheader("Indication de l'AVK")

        indication_avk = st.radio(
            "Indication",
            [
                "FA",
                "VALVE_MECANIQUE",
                "MTEV",
                "LVAD",
                "AUTRE"
            ],
            format_func=lambda x: "VALVE MECANIQUE" if x == "VALVE_MECANIQUE" else x,
            key="indication_avk"
        )

        
        valves = False
        acfa_atcd = False
        mtev_hr = False
        relais_avk = False
      

        # ====FA= 
        if indication_avk == "FA":

            st.markdown("**Complément**")
            acfa_atcd = st.checkbox(
                "Antécédent d’AVC, AIT ou embolie systémique"
            )

            if acfa_atcd:

                fa_avc_moins_3_mois = st.checkbox(
                    "AVC ischémique datant de moins de 3 mois"
                )

                if fa_avc_moins_3_mois:

                    col_vide, col_question = st.columns([0.06, 0.94])

                    with col_question:
                        procedure_differable = st.checkbox(
                            " La procédure peut être différée sans risque vital ou fonctionnel",
                            key="procedure_differable_fa"
                     )



            relais_avk = acfa_atcd


        # ==== VALVE =====

        elif indication_avk == "VALVE_MECANIQUE":

            valves = True
            relais_avk = True

            valve_aortique_double_ailette = st.checkbox(
                "Valve aortique mécanique à double ailette"
            )

            rythme_sinusal = st.checkbox(
                "Rythme sinusal"
            )

            atcd_thrombotique_valve = st.checkbox(
                "Antécédent thrombotique lié à la valve"
            )

            valve_faible_risque_postop = (
                valve_aortique_double_ailette
                and rythme_sinusal
                and not atcd_thrombotique_valve
            )

        # == MTEV ===========

        elif indication_avk == "MTEV":

            mtev_hr = st.checkbox(
                "EP ou TVP proximale datant de moins de 3 mois"
            )

            relais_avk = mtev_hr


            if mtev_hr:

                col_vide1, col_question1 = st.columns([0.06, 0.94])

                with col_question1:
                    mtev_moins_1_mois = st.checkbox(
                        " EP ou TVP proximale datant de moins de 1 mois",
                        key="mtev_moins_1_mois"
                    )
   
                col_vide2, col_question2 = st.columns([0.06, 0.94])

                with col_question2:
                    procedure_differable = st.checkbox(
                        " La procédure peut être différée sans risque vital ou fonctionnel",
                        key="procedure_differable_mtev"
                    )


            mtev_complexe = st.checkbox(
                "Cas complexe de MTEV"
            )
            if mtev_complexe:
                st.caption("""
Cas complexe si au moins un des éléments suivants est présent :

- Syndrome des antiphospholipides (SAPL)
- Hypertension pulmonaire thromboembolique chronique (HTP-TEC)
- Histoire clinique ou familiale inhabituelle faisant évoquer un risque thromboembolique élevé (ex. déficit en antithrombine, syndrome paranéoplasique thrombogène)
- Thrombopénie induite par l’héparine (TIH) en cours de traitement anticoagulant
- Récidive d’EP ou de TVP sous traitement anticoagulant ou précocement après son arrêt
                """)



            deficit_proteine_c_s = st.checkbox(
                "Déficit en protéine C ou S",
                key="deficit_proteine_c_s"
            )




            # ==CLASSE DU RISQUE DE RECIDIVE MTEV =====

            if mtev_complexe:
                risque_recidive_mtev = "A_EVALUER"

            elif mtev_moins_1_mois:
                risque_recidive_mtev = "TRES_ELEVE"

            elif mtev_hr:
                risque_recidive_mtev = "ELEVE"

            else:
                risque_recidive_mtev = "MODERE"




        # ==LVAD=======

        elif indication_avk == "LVAD":

            lvad_reprise_24_48h = st.checkbox(
                "Reprise d’une anticoagulation curative possible dans les 24 à 48 h"
            )

            relais_avk = not lvad_reprise_24_48h  

        lvad_sans_arret_avk = (
            indication_avk == "LVAD"
            and lvad_reprise_24_48h
        )



        if relais_avk:

            st.subheader("Relai AVK - fonction rénale")

            dfg_relais_avk = st.radio(
                "DFG du patient",
                [
                    "DFG > 30",
                    "15 ≤ DFG < 30",
                    "DFG < 15",
                    "DFG inconnu"
                ],
                key="dfg_relais_avk"
            )


            
            if dfg_relais_avk in ["15 ≤ DFG < 30", "DFG < 15"]:

                mode_prise_en_charge_relais = st.radio(
                    "Prise en charge du relais",
                    [

                        "Hospitalisation prévue",
                        "Prise en charge extrahospitalière"
                    ],
                    key="mode_prise_en_charge_relais"
                )

    
                if mode_prise_en_charge_relais == "Hospitalisation prévue":

                    st.info(
                        "HNF IVSE recommandée en cas d’hospitalisation prévue."
                    )

                    type_heparine_relais = st.radio(
                        "Choix du relais",
                        [
                            "HNF IVSE",
                            "HNF calcique SC",
                            "HBPM"
                        ],
                        key="type_heparine_relais_hospit"
                    )


   
                elif mode_prise_en_charge_relais == "Prise en charge extrahospitalière":

                    type_heparine_relais = st.radio(
                        "Choix du relais",
                        [
                            "HNF calcique SC",
                            "HBPM"
                        ],
                        key="type_heparine_relais_extra"
                    )


                if type_heparine_relais == "HBPM":

                    if dfg_relais_avk == "15 ≤ DFG < 30":

                        schema_hbpm = st.radio(
                            "HBPM SC à dose curative",
                            [
                                "Tinzaparine 175 UI/kg x 1/j",
                                "Enoxaparine 100 UI/kg x 1/j"
                            ],
                            key="schema_hbpm_dfg_15_30"
                        )

                    elif dfg_relais_avk == "DFG < 15":

                        schema_hbpm = st.radio(
                            "HBPM SC à dose curative",
                            [
                                "Tinzaparine 175 UI/kg x 1/j"
                            ],
                            key="schema_hbpm_dfg_inf_15"
                        )









            elif dfg_relais_avk == "DFG > 30":

                type_heparine_relais = "HBPM"

                schema_hbpm = st.radio(
                    "HBPM SC à dose curative",
                    [
                        "2 injections par jour (Enoxaparine 100 UI/kg toutes les 12h)",
                        "1 injection par jour (Tinzaparine 175 UI/kg x 1/j)"
                    ],
                    key="schema_hbpm"
                )




        if relais_avk and poids_kg >= 100:
            hnf_ivse_choisie = st.checkbox(
                "HNF IVSE curative choisie",
                key="hnf_ivse_choisie"
            )



        # ==== POST OP======

        st.subheader("Post-opératoire")

        relais_postop_indique = False
        inr_ge_2_postop = False
        chevauchement_non_acceptable = False
        thromboprophylaxie_indiquee = False

        reprise_avk_24h = st.radio(
            "Reprise de l'AVK possible dans les 24 premières heures ?",
            ["Oui", "Non"],
            key="reprise_avk_24h_radio"
        )

        reprise_avk_24h_bool = reprise_avk_24h == "Oui"



        raison_relais_postop = ""


        # ===== FA =====

        if indication_avk == "FA" and not reprise_avk_24h_bool:

            relais_postop_indique = True

            raison_relais_postop = (
                "FA avec impossibilité de reprendre les AVK dans les 24 premières heures."
            )


        # ===== VALVE MECANIQUE =====

        elif indication_avk == "VALVE_MECANIQUE":

            valve_exception_sans_relais = (
                valve_aortique_double_ailette
                and rythme_sinusal
                and not atcd_thrombotique_valve
                and reprise_avk_24h_bool
            )

            if not valve_exception_sans_relais:

                relais_postop_indique = True

                raison_relais_postop = (
                    "Valve mécanique : relais postopératoire curatif indiqué, "
                    "sauf valve aortique mécanique à double ailette + rythme sinusal "
                    "+ absence d'antécédent thrombotique lié à la valve "
                    "+ reprise AVK possible dans les 24 h."
                )


        # === MTEV < 3 MOIS == =

        elif indication_avk == "MTEV" and mtev_hr:

            relais_postop_indique = True

            raison_relais_postop = (
                "MTEV avec EP ou TVP proximale datant de moins de 3 mois."
            )


        #=DEFICIT PROTEINE C/S ===

        elif indication_avk == "MTEV" and deficit_proteine_c_s:

            relais_postop_indique = True

            raison_relais_postop = (
                "MTEV avec déficit en protéine C ou S."
            )

 
        elif (
            indication_avk == "MTEV"
            and not mtev_hr
            and not mtev_complexe
            and not deficit_proteine_c_s
            and not reprise_avk_24h_bool
        ):

            relais_postop_indique = True

            raison_relais_postop = (
                "MTEV sans indication de relais curatif postopératoire systématique, "
                "mais reprise de l’AVK impossible dans les 24 premières heures."
            )



        if relais_postop_indique:

            st.success(
                " Héparine curative postopératoire indiquée, "
                "à débuter de préférence 48 à 72 h après l’intervention."
            )

            if raison_relais_postop:
                st.caption(raison_relais_postop)


        else:

            st.info(
                "Pas d'indication automatique d'héparine curative postopératoire "
                "retrouvée dans les règles précédentes."
            )


            indication_postop_medicale = st.checkbox(
                "Indication médicale d'héparine curative postopératoire",
                key="indication_postop_medicale"
            )

            if indication_postop_medicale:

                relais_postop_indique = True

                st.caption("""
        - Voie entérale non disponible.
        - Gestion du risque hémorragique plus simple avec des anticoagulants de demi-vie courte (HBPM, voire HNF), par exemple en présence de drains ou d'un risque élevé de reprise chirurgicale.
        - Risque thrombo-embolique considéré comme trop élevé pour attendre que les AVK permettent d'obtenir une anticoagulation curative.
        """)



        if relais_postop_indique:

            st.markdown(
                "**En attendant la reprise de l’anticoagulation curative**"
            )

            col_fleche, col_question = st.columns([0.06, 0.94])

            with col_fleche:
                st.markdown(
                    """
                    <div style="
                        width: 28px;
                        height: 28px;
                        border: 2px solid #16883a;
                        border-radius: 50%;
                        color: #16883a;
                        display: flex;
                        align-items: center;
                        justify-content: center;
                        font-size: 20px;
                        font-weight: bold;
                         margin-top: 6px;
                    ">
                        →
                    </div>
                    """,
                    unsafe_allow_html=True
                )

            with col_question:
                thromboprophylaxie_indiquee = st.checkbox(
                    "Thromboprophylaxie veineuse indiquée en attendant la reprise de l’anticoagulation curative",
                    key="thromboprophylaxie_indiquee"
                )


            if reprise_avk_24h_bool:

                st.markdown(
                    "**Compléments si reprise de l’AVK possible dans les 24 premières heures**"
                )

                # ---- AVK repris et INR ≥ 2 --
                col_fleche, col_question = st.columns([0.06, 0.94])

                with col_fleche:
                    st.markdown(
                        """
                        <div style="
                            width: 28px;
                            height: 28px;
                            border: 2px solid #16883a;
                            border-radius: 50%;
                            color: #16883a;
                            display: flex;
                            align-items: center;
                            justify-content: center;
                            font-size: 20px;
                            font-weight: bold;
                            margin-top: 6px;
                        ">
                            →
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

                with col_question:
                    inr_ge_2_postop = st.checkbox(
                        "AVK repris et premier INR ≥ 2",
                        key="inr_ge_2_postop"
                    )


                # ---- Chevauchement héparine + AVK ----
                col_fleche, col_question = st.columns([0.06, 0.94])

                with col_fleche:
                    st.markdown(
                        """
                        <div style="
                            width: 28px;
                            height: 28px;
                            border: 2px solid #16883a;
                            border-radius: 50%;
                            color: #16883a;
                            display: flex;
                            align-items: center;
                            justify-content: center;
                            font-size: 20px;
                            font-weight: bold;
                            margin-top: 6px;
                        ">
                            →
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

                with col_question:
                    chevauchement_non_acceptable = st.checkbox(
                        "Chevauchement héparine curative + AVK jugé non acceptable pendant la première semaine",
                        key="chevauchement_non_acceptable"
                    )
  
                    if chevauchement_non_acceptable:
                        st.caption(
                            "Situations possibles : PTH, chirurgie du rachis, "
                            "cure d’éventration complexe..."
                        )




# =========================
# CONTEXTE PATIENT / CHIRURGIE
# =========================


ind_glp1 = None
indication_sglt2 = None

texte_detecte = str(txt_final).lower()

# =========================
# SGLT2 (FORXIGA)
# =========================
if "forxiga" in texte_detecte or "dapagliflozine" in texte_detecte:
    st.divider()
    st.subheader("Contexte SGLT2")

    indication_sglt2 = st.radio(
        "Indication du Forxiga",
        [
            "Diabète",
            "Insuffisance cardiaque",
            "Néphroprotection"
        ],
        key="indication_sglt2"
    )


# =========================
# CONTEXTE DIABÈTE
# =========================

afficher_contexte_diabete = diabete_detecte

# si SGLT2 non diabète → on cache
if indication_sglt2 in ["Insuffisance cardiaque", "Néphroprotection"]:
    afficher_contexte_diabete = False

# si SGLT2 diabète → on force affichage
if indication_sglt2 == "Diabète":
    afficher_contexte_diabete = True


if afficher_contexte_diabete:
    st.divider()
    st.header("Contexte diabète")

    type_chir = st.radio(
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


corticoides_connus = corticoide_detecte
duree_sup_4sem = False
dose_pred_sup_5 = False
dose_pred_sup_10 = False
dose_hc_inf_40 = False
dose_hc_sup_40 = False
chirurgie_courte = False
post_op_jeun_sup_24h = False
reprise_precoce = False
complications_postop = False
obstetrique = False
hydrocortisone_topique = False
hydrocortisone_systemique = False



if "stress_cortico_faible" not in locals():
    stress_cortico_faible = False

stress_cortico_affichage = (
    "FAIBLE"
    if stress_cortico_faible
    else "MODÉRÉ-ÉLEVÉ"
)



hydrocortisone_detectee = "hydrocortisone" in str(st.session_state).lower()

if hydrocortisone_detectee:
    type_hydrocortisone = st.radio(
        "Hydrocortisone : préciser la forme",
        ["Topique (crème, pommade, lotion)", "Autre forme / systémique"],
        key="ui_hydrocortisone_type"
    )

    if type_hydrocortisone == "Topique (crème, pommade, lotion)":
        hydrocortisone_topique = True
        hydrocortisone_systemique = False
        corticoides_connus = False
    else:
        hydrocortisone_topique = False
        hydrocortisone_systemique = True
        corticoides_connus = True


if corticoides_connus and not hydrocortisone_topique:
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
    dose_pred_sup_10 = dose_pred >= 10

    dose_hc = dose_pred * 4
    dose_hc_inf_40 = dose_hc < 40
    dose_hc_sup_40 = dose_hc >= 40

    st.warning("""
    **Interprétation clinique :**

    - Corticothérapie ≥ 4 semaines et ≥ 5 mg prednisone :
      → risque de suppression surrénalienne, adapter selon le stress chirurgical (voir plus bas).

    - Corticothérapie sans critère de risque :
      → poursuite simple, sans supplémentation.
    """)

    st.subheader("Stress chirurgical (corticoïdes)")
    st.caption(
        "Déterminé automatiquement à partir de l’intervention sélectionnée."
    )

    if stress_cortico_faible:
        st.success("FAIBLE")
    else:
        st.warning("MODÉRÉ-ÉLEVÉ")


    chirurgie_courte = False
    post_op_jeun_sup_24h = False
    reprise_precoce = False
    complications_postop = False

    obstetrique = (spe == "Obstétrique")



#----
def normaliser_risque_yaml(risque):
    r = str(risque or "").upper().strip()

    mapping = {
        "NUL": "NUL",
        "FAIBLE": "FAIBLE",
        "INTERMEDIAIRE": "INTERMEDIAIRE",
        "ELEVE": "ELEVE",
        "TRES ELEVE": "MAJEUR",
    }

    return mapping.get(r, r)


# =========================
# HEPARINES 
# =========================
voie_heparine = None
dose_heparine = None

# =========================
# CONTEXTE GLOBAL 
# =========================
ctx = {
    "type_chir_neuro": "NEUROCHIR_INTRACRANIENNE" if val_upper(spe) in ["NEUROCHIRURGIE", "RACHIS"] else None,
    "type_chir": type_chir,
    "is_neuro": is_neuro,
    "alr": type_alr,
    "ind_sraa": ind_sraa if ind_sraa else "",
    "choix_sraa_hta": choix_sraa_hta if choix_sraa_hta else "",
    "indication_aap": indication_aap,
    "aspirine_dose": dose_aspirine,
    "aspirine_sup_100": dose_aspirine > 100,
    "aspirine_sup_200": dose_aspirine > 200,
    "dose_aspirine_inf_300": dose_aspirine <= 300,
    "controle_hem": "",
    "r_hem_aod_avk": normaliser_risque_yaml(risque_aod_avk),
    "r_hem_aap": normaliser_risque_yaml(risque_aap),
    "r_hem": normaliser_risque_yaml(risque_aod_avk),
    "categorie_geste": None,
    "demi_vie_heures": None,
    "voie_baclofene": None,
    "indication_sglt2": indication_sglt2 if indication_sglt2 else "",
    "technique_neuraxiale": technique_neuraxiale,

    "ASA": asa_acte_to_int(asa_acte) if 'asa_acte' in locals() else None,

    "atcd_cv": None,
    "dfg": None,

    "dispositif_insuline": "pompe" if st.session_state.get("pompe_insuline", False) else None,

    "valve_mecanique": valves,
    "acfa_atcd": acfa_atcd,
    "mtev_haut_risque": mtev_hr,
    "relais_avk": relais_avk,
    "poids_kg": poids_kg if avk_detecte and poids_kg > 0 else None,
    "dfg_relais_avk": dfg_relais_avk,
    "indication_avk": indication_avk,
    "lvad_reprise_24_48h": lvad_reprise_24_48h,
    "facteur_hemorragique_supplementaire": facteur_hemorragique_supplementaire,

    "fa_avc_moins_3_mois": fa_avc_moins_3_mois,

    "mtev_moins_1_mois": mtev_moins_1_mois,
    "mtev_complexe": mtev_complexe,
    "procedure_differable": procedure_differable,
    "lvad_sans_arret_avk": lvad_sans_arret_avk,
    "valve_aortique_double_ailette": valve_aortique_double_ailette,
    "rythme_sinusal": rythme_sinusal,
    "atcd_thrombotique_valve": atcd_thrombotique_valve,
    "valve_faible_risque_postop": valve_faible_risque_postop,
    "poids_ge_100": avk_detecte and poids_kg >= 100,
    "poids_inf_50": avk_detecte and 0 < poids_kg < 50,
    "hnf_calcique_sc_choisie": hnf_calcique_sc_choisie,
    "poids_extreme_et_dfg_inf_30": (
        avk_detecte
        and (poids_kg >= 100 or 0 < poids_kg < 50)
        and dfg_relais_avk in ["15 ≤ DFG < 30", "DFG < 15"]
    ),

    "neuro_ou_neuraxial": (
        val_upper(spe) in ["NEUROCHIRURGIE", "RACHIS"]
        or bool(technique_neuraxiale)
    ),

    "inr_sup_seuil_avk": False,

    "reprise_avk_24h": reprise_avk_24h_bool,
    "mtev_relais_postop": mtev_relais_postop,
    "relais_postop_indique": relais_postop_indique,
    "inr_ge_2_postop": inr_ge_2_postop,
    "risque_recidive_mtev": risque_recidive_mtev,
    "chevauchement_non_acceptable": chevauchement_non_acceptable,
    "anticoag_curative_non_reprise": anticoag_curative_non_reprise,
    "thromboprophylaxie_indiquee": thromboprophylaxie_indiquee,
    "deficit_proteine_c_s": deficit_proteine_c_s,
    "hnf_ivse_choisie": hnf_ivse_choisie,
    "mode_prise_en_charge_relais": mode_prise_en_charge_relais,
    "type_heparine_relais": type_heparine_relais,
    "schema_hbpm": schema_hbpm,
    
    "type_traitement_aap": type_traitement_aap if type_traitement_aap else "",
    "bitherapie_aap": type_traitement_aap == "Bithérapie",
    "prev_secondaire": type_traitement_aap == "Prévention secondaire",
    "prev_primaire": type_traitement_aap == "Prévention primaire",
    "aspirine_inf_75": dose_aspirine < 75,
    "stent_1m": contexte_stent == "Stent ≤ 1 mois",
    "stent_6m_haut_risque": contexte_stent == "Stent ≤ 6 mois à haut risque thrombotique",
    "idm_6m": contexte_stent == "IDM < 6 mois",
    "aucun_critere_stent": contexte_stent == "Aucun critère",
    "inr_disponible": inr_disponible,
    "inr_valeur": inr_valeur,

    "corticoides": corticoides_connus,
    "duree_sup_4sem": duree_sup_4sem,
    "dose_pred_sup_5": dose_pred_sup_5,
    "dose_pred_sup_10": dose_pred_sup_10,
    "dose_hc_inf_40": dose_hc_inf_40,
    "dose_hc_sup_40": dose_hc_sup_40,
    "stress_cortico_faible": stress_cortico_faible,
    "chirurgie_courte": chirurgie_courte,
    "post_op_jeun_sup_24h": post_op_jeun_sup_24h,
    "reprise_precoce": reprise_precoce,
    "complications_postop": complications_postop,
    "obstetrique": obstetrique,
    "hydrocortisone_topique": hydrocortisone_topique,
    "hydrocortisone_systemique": hydrocortisone_systemique,
    "voie_heparine": voie_heparine,
    "dose_heparine": dose_heparine,
    
    "inr_therapeutique_2_3": inr_disponible == "Oui" and inr_valeur is not None and 2 <= inr_valeur <= 3,
    "inr_hors_zone_2_3": inr_disponible == "Oui" and inr_valeur is not None and not (2 <= inr_valeur <= 3),
    "inr_non_connu": inr_disponible != "Oui",
    
    }

schema_relais = construire_schema_relais(ctx)

ordonnance_pharmacie = generer_ordonnance_pharmacie(
    schema_relais,
    poids_kg=ctx.get("poids_kg"),
    date_op=date_op
)



prescription_ide = generer_prescription_ide(
    schema_relais,
    poids_kg=ctx.get("poids_kg"),
    date_op=date_op
)
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

codes_atc_detectes = [r.get("Code ATC") for r in resultats if r.get("Code ATC")]
codes_atc_detectes_upper = [str(c).upper().strip() for c in codes_atc_detectes]

aod_detecte = any(
    code.startswith(("B01AE", "B01AF"))
    for code in codes_atc_detectes_upper
)


# =========================
# AOD - fonction rénale
# =========================

dfg_aod = ""
aod_repris = False

if aod_detecte:
    st.divider()
    st.header("AOD - fonction rénale")


    poids_aod_kg = st.number_input(
        "Poids du patient (kg) si connu",
        min_value=0,
        value=0,
        step=1,
        key="poids_aod_kg"
    )

    with st.expander("Rappel posologie AOD selon DFGe"):

        st.info(
            "Vérifier que la posologie de l’AOD est adaptée au DFGe."
        )

        st.markdown("### Posologies des AOD dans la FA adaptées au DFG estimé")

        tableau_aod_dfg = pd.DataFrame(
            {
                "": [
                    "Élimination rénale",
                    "DFGe > 50 ml/min/1,73 m²",
                    "IRC modérée — DFGe = 30–50 ml/min/1,73 m²",
                    "IRC sévère — DFGe = 15–29 ml/min/1,73 m²",
                    "IRC terminale — DFGe < 15 ml/min/1,73 m²",
                ],

                "Apixaban": [
                    "25 %",
                    "5 mg x 2/j ¹",
                    "5 mg x 2/j ¹",
                    "2,5 mg x 2/j ³",
                    "Hors AMM — 2,5 mg x 2/j ⁴",
                ],

                "Rivaroxaban": [
                    "33 %",
                    "20 mg x 1/j",
                    "15 mg x 1/j",
                    "15 mg x 1/j",
                    "Hors AMM — 15 mg x 1/j ⁴",
                ],

                "Edoxaban": [
                    "50 %",
                    "60 mg x 1/j",
                    "30 mg x 1/j",
                    "30 mg x 1/j",
                    "NR",
                ],

                "Dabigatran": [
                    "> 85 %",
                    "150 mg x 2/j ²",
                    "110 mg x 2/j",
                    "NR",
                    "NR",
                ],
            }
        )

        st.dataframe(
            tableau_aod_dfg,
            use_container_width=True,
            hide_index=True
        )

        st.caption(
            "¹ 2,5 mg x 2/j si 2 facteurs de risque parmi : âge > 80 ans ; "
            "poids < 60 kg ; créatininémie > 133 µmol/L."
        ) 

        st.caption(
            "² 110 mg x 2/j si âge > 80 ans ou inhibiteurs de la P-gp."
        )

        st.caption(
            "³ ANSM et recommandations ESC / KDIGO."
        )

        st.caption(
            "⁴ Utilisation hors AMM ; l’apixaban et le rivaroxaban sont autorisés "
            "aux USA (FDA)."
        )

        st.caption("NR : non recommandé.")







    dfg_aod = st.radio(
        "DFG du patient",
        [
            "DFG > 50",
            "30 ≤ DFG ≤ 50",
            "15 ≤ DFG < 30",
            "DFG < 15",
            "DFG inconnu"
        ],
        key="dfg_aod"
    )



    # =========================
    # AOD reprise postop 
    # =========================

    reprise_aod_differee = st.checkbox(
        "Reprise de l’AOD différée",
        key="reprise_aod_differee"
    )

    aod_repris = st.checkbox(
        "AOD repris",
        key="aod_repris"
    )

    if reprise_aod_differee:
        thromboprophylaxie_indiquee_aod = st.checkbox(
            "Thromboprophylaxie indiquée",
            key="thromboprophylaxie_indiquee_aod"
        )

        heparine_curative_indiquee = st.checkbox(
            "Héparine curative indiquée",
            key="heparine_curative_indiquee"
        )


ctx["dfg_connu"] = dfg_aod != "DFG inconnu" and dfg_aod != ""
ctx["dfg_ge_30"] = dfg_aod in ["DFG > 50", "30 ≤ DFG ≤ 50"]

ctx["dfg_inf_30"] = dfg_aod in [
    "15 ≤ DFG < 30",
    "DFG < 15"
]

ctx["dfg_ge_50"] = dfg_aod == "DFG > 50"
ctx["dfg_30_49"] = dfg_aod == "30 ≤ DFG ≤ 50"
ctx["dfg_15_29"] = dfg_aod == "15 ≤ DFG < 30"

ctx["dfg_inf_15"] = dfg_aod == "DFG < 15"

ctx["reprise_aod_differee"] = locals().get("reprise_aod_differee", False)

ctx["aod_repris"] = locals().get("aod_repris", False)
ctx["thromboprophylaxie_indiquee_aod"] = locals().get("thromboprophylaxie_indiquee_aod", False)
ctx["heparine_curative_indiquee"] = locals().get("heparine_curative_indiquee", False)


ctx["poids_ge_100"] = bool(
    ctx.get("poids_ge_100", False)
    or (aod_detecte and poids_aod_kg >= 100)
)

ctx["poids_inf_50"] = bool(
    ctx.get("poids_inf_50", False)
    or (aod_detecte and 0 < poids_aod_kg < 50)
)

# =========================
# AOD 
# ====================

indication_aod = ""
FA_ATCD_AVC_ischemique = False
FA_delai_depuis_AVC_ischemique_mois = None
procedure_differable_sans_risque_vital_fonctionnel = False
FA_procedure_risque_eleve_terminee = False
FA_coronaropathie = False
MTEV_cas_complexe = False
MTEV_type = ""
MTEV_delai_mois = None
MTEV_EP_TVP_proximale_moins_3_mois = False
MTEV_EP_TVP_proximale_moins_1_mois = False
MTEV_procedure_differable_sans_risque_vital_fonctionnel = False
MTEV_procedure_risque_eleve_terminee = False
MTEV_procedure_maintenue = False
MTEV_filtre_cave_mis_en_place = False
MTEV_anticoagulation_curative_reprise = False
MTEV_nouvel_arret_non_prevu_court_terme = False

MTEV_TVP_distale_symptomatique = False
MTEV_TVP_distale_procedure_differable = False
MTEV_risque_thromboembolique_veineux_tres_eleve = False


if aod_detecte:
    st.divider()
    st.header("AOD - indication")

    indication_aod = st.radio(
        "Indication du traitement par AOD",
        [
            "FA",
            "MTEV",
            "Autre"
        ],
        key="indication_aod"
    )

    if indication_aod == "FA":
        FA_ATCD_AVC_ischemique_ui = st.radio(
            "Antécédent d'AVC ischémique ?",
            ["Non", "Oui"],
            key="FA_ATCD_AVC_ischemique"
        )

        FA_ATCD_AVC_ischemique = (
            FA_ATCD_AVC_ischemique_ui == "Oui"
        )

        if FA_ATCD_AVC_ischemique:
            FA_delai_depuis_AVC_ischemique_mois = st.number_input(
                "Délai depuis l'AVC ischémique (mois)",
                min_value=0.0,
                step=0.5,
                key="FA_delai_depuis_AVC_ischemique_mois"
            )

            if FA_delai_depuis_AVC_ischemique_mois < 3:
                procedure_differable_ui = st.radio(
                    "La procédure peut-elle être différée sans risque vital ou fonctionnel ?",
                    ["Non", "Oui"],
                    key="procedure_differable_sans_risque_vital_fonctionnel"
                )

                procedure_differable_sans_risque_vital_fonctionnel = (
                    procedure_differable_ui == "Oui"
                )

        # =========================
        # FA 3
        # =========================
        if normaliser_risque_yaml(risque_aod_avk) in [
            "ELEVE",
            "IMPORTANT",
            "MAJEUR"
        ]:
            FA_procedure_risque_eleve_terminee = st.checkbox(
                "Procédure à risque hémorragique élevé terminée",
                key="FA_procedure_risque_eleve_terminee"
            )

        # =========================
        # FA 4 coronaropathie
        # =========================
        FA_coronaropathie_ui = st.radio(
            "Coronaropathie associée ?",
            ["Non", "Oui"],
            key="FA_coronaropathie"
        )

        FA_coronaropathie = (
            FA_coronaropathie_ui == "Oui"
        )


    if indication_aod == "MTEV":

           

        MTEV_type = st.radio(
            "Type de maladie thromboembolique veineuse",
            [
                "EP",
                "TVP proximale",
                "TVP distale",
                "Autre"
            ],
            key="MTEV_type"
        )

        MTEV_delai_mois = st.number_input(
            "Délai depuis l'EP ou la TVP (mois)",
            min_value=0.0,
            step=0.5,
            key="MTEV_delai_mois"
        )

        MTEV_EP_TVP_proximale_moins_3_mois = (
            MTEV_type in ["EP", "TVP proximale"]
            and MTEV_delai_mois < 3
        )

        MTEV_EP_TVP_proximale_moins_1_mois = (
            MTEV_type in ["EP", "TVP proximale"]
            and MTEV_delai_mois < 1
        )

        MTEV_risque_thromboembolique_veineux_tres_eleve = (
            MTEV_EP_TVP_proximale_moins_1_mois
        )

       
        # =========================
        # MTEV 6
        # ====================

        if (
            MTEV_EP_TVP_proximale_moins_1_mois
            and normaliser_risque_yaml(risque_aod_avk) in [
                "ELEVE",
                "IMPORTANT",
                "MAJEUR"
            ]
        ):
            MTEV_procedure_maintenue_ui = st.radio(
                "La procédure est-elle maintenue ?",
                ["Non", "Oui"],
                key="MTEV_procedure_maintenue"
            )

            MTEV_procedure_maintenue = (
                MTEV_procedure_maintenue_ui == "Oui"
            )



        # =========================
        # MTEV 2 procédure différable
        # =========================

        if MTEV_EP_TVP_proximale_moins_3_mois:
            MTEV_procedure_differable_ui = st.radio(
                "La procédure peut-elle être différée sans risque vital ou fonctionnel ?",
                ["Non", "Oui"],
                key="MTEV_procedure_differable_sans_risque_vital_fonctionnel"
            )

            MTEV_procedure_differable_sans_risque_vital_fonctionnel = (
                MTEV_procedure_differable_ui == "Oui"
            )




        # =========================
        # MTEV 8 TVP distale symptomatique
        # =========================

        if MTEV_type == "TVP distale":
            MTEV_TVP_distale_symptomatique_ui = st.radio(
                "TVP distale symptomatique ?",
                ["Non", "Oui"],
                key="MTEV_TVP_distale_symptomatique"
            )

            MTEV_TVP_distale_symptomatique = (
                MTEV_TVP_distale_symptomatique_ui == "Oui"
            )

            if (
                MTEV_TVP_distale_symptomatique
                and normaliser_risque_yaml(risque_aod_avk) in [
                    "ELEVE",
                    "IMPORTANT",
                    "MAJEUR"
                ]
            ):
                MTEV_TVP_distale_procedure_differable_ui = st.radio(
                    "La procédure peut-elle être différée sans risque vital ou fonctionnel ?",
                    ["Non", "Oui"],
                    key="MTEV_TVP_distale_procedure_differable"
                )

                MTEV_TVP_distale_procedure_differable = (
                    MTEV_TVP_distale_procedure_differable_ui == "Oui"
                )





            st.markdown("**Recherche d'une situation MTEV complexe**")

            MTEV_SAPL = st.checkbox(
                "Syndrome des anticorps antiphospholipides (SAPL)",
                key="MTEV_SAPL"
            )

            MTEV_HTP_TEC = st.checkbox(
                "Hypertension pulmonaire thromboembolique chronique (HTP-TEC)",
                key="MTEV_HTP_TEC"
            )

            MTEV_histoire_inhabituelle = st.checkbox(
                "Histoire clinique ou familiale inhabituelle évoquant un risque thromboembolique élevé",
                key="MTEV_histoire_inhabituelle"
            )

            MTEV_TIH = st.checkbox(
                "TIH en cours de traitement anticoagulant",
                key="MTEV_TIH"
            )

            MTEV_recidive = st.checkbox(
                "Récidive d'EP ou de TVP sous traitement anticoagulant ou précocement après son arrêt",
                key="MTEV_recidive"
            )

            MTEV_cas_complexe = (
                MTEV_SAPL
                or MTEV_HTP_TEC
                or MTEV_histoire_inhabituelle
                or MTEV_TIH
                or MTEV_recidive
            )

            # =========================
            # MTEV 5
            # =========================

            if (
                not MTEV_cas_complexe
                and normaliser_risque_yaml(risque_aod_avk) in [
                    "ELEVE",
                    "IMPORTANT",
                    "MAJEUR"
                ]
            ):
                MTEV_procedure_risque_eleve_terminee = st.checkbox(
                    "Procédure à risque hémorragique élevé terminée",
                    key="MTEV_procedure_risque_eleve_terminee"
                )



            # =================
            # MTEV 7 retrait filtre cave
            # =========================

            MTEV_filtre_cave_mis_en_place = st.checkbox(
                "Filtre cave optionnel mis en place",
                key="MTEV_filtre_cave_mis_en_place"
            )
 
            if MTEV_filtre_cave_mis_en_place:
                MTEV_anticoagulation_curative_reprise = st.checkbox(
                    "Anticoagulation curative reprise",
                    key="MTEV_anticoagulation_curative_reprise"
                )

                if MTEV_anticoagulation_curative_reprise:
                    MTEV_nouvel_arret_non_prevu_court_terme = st.checkbox(
                        "Nouvel arrêt de l’anticoagulation non prévu à court terme",
                        key="MTEV_nouvel_arret_non_prevu_court_terme"
                    )


ctx["indication_aod"] = indication_aod

ctx["FA_ATCD_AVC_ischemique"] = FA_ATCD_AVC_ischemique

ctx["FA_AVC_moins_3_mois"] = (
    FA_ATCD_AVC_ischemique
    and FA_delai_depuis_AVC_ischemique_mois is not None
    and FA_delai_depuis_AVC_ischemique_mois < 3
)

ctx["procedure_differable_sans_risque_vital_fonctionnel"] = (
    procedure_differable_sans_risque_vital_fonctionnel
)

ctx["FA_procedure_risque_eleve_terminee"] = FA_procedure_risque_eleve_terminee
ctx["FA_coronaropathie"] = FA_coronaropathie
ctx["aap_detecte"] = aap_detecte
ctx["MTEV_cas_complexe"] = MTEV_cas_complexe

ctx["MTEV_type"] = MTEV_type
ctx["MTEV_delai_mois"] = MTEV_delai_mois
ctx["MTEV_EP_TVP_proximale_moins_3_mois"] = MTEV_EP_TVP_proximale_moins_3_mois
ctx["MTEV_EP_TVP_proximale_moins_1_mois"] = MTEV_EP_TVP_proximale_moins_1_mois
ctx["MTEV_procedure_differable_sans_risque_vital_fonctionnel"] = (
    MTEV_procedure_differable_sans_risque_vital_fonctionnel
)
ctx["MTEV_procedure_risque_eleve_terminee"] = MTEV_procedure_risque_eleve_terminee

ctx["MTEV_procedure_maintenue"] = MTEV_procedure_maintenue
ctx["MTEV_filtre_cave_mis_en_place"] = MTEV_filtre_cave_mis_en_place

ctx["MTEV_anticoagulation_curative_reprise"] = MTEV_anticoagulation_curative_reprise

ctx["MTEV_nouvel_arret_non_prevu_court_terme"] = (
    MTEV_nouvel_arret_non_prevu_court_terme
)
ctx["MTEV_TVP_distale_symptomatique"] = MTEV_TVP_distale_symptomatique

ctx["MTEV_TVP_distale_procedure_differable"] = (
    MTEV_TVP_distale_procedure_differable
)
ctx["MTEV_risque_thromboembolique_veineux_tres_eleve"] = (
    MTEV_risque_thromboembolique_veineux_tres_eleve
)


ctx["reprise_aod_differee"] = st.session_state.get(
    "reprise_aod_differee",
    False
)

ctx["aod_repris"] = st.session_state.get(
    "aod_repris",
    False
)

ctx["thromboprophylaxie_indiquee_aod"] = st.session_state.get(
    "thromboprophylaxie_indiquee_aod",
    False
)

ctx["heparine_curative_indiquee"] = st.session_state.get(
    "heparine_curative_indiquee",
    False
)



resultats, vus, candidats_retenus = detecter_medicaments_depuis_texte(
    txt=txt_final,
    ref=ref,
    atc_map=atc_map,
    classe_map=classe_map,
    ctx=ctx
)

# =========================
# AVK - fonction rénale relais
# =========================


# =========================
# HEPARINES ui
# =========================

voie_heparine = None
dose_heparine = None

# HNF
if "B01AB01" in codes_atc_detectes_upper:
    st.divider()
    st.subheader("Contexte héparine")

    voie_heparine_ui = st.radio(
        "Voie d'administration de l'héparine non fractionnée (HNF)",
        ["IVSE", "SC"],
        horizontal=True,
        key="ui_voie_heparine"
    )

    voie_heparine = "IVSE" if voie_heparine_ui == "IVSE" else "SC"

    if voie_heparine == "SC":
        dose_heparine_ui = st.radio(
            "Type d'anticoagulation HNF SC",
            ["Dose préventive", "Dose curative"],
            horizontal=True,
            key="ui_dose_hnf_sc"
        )

        dose_heparine = "préventive" if dose_heparine_ui == "Dose préventive" else "curative"


# HBPM / Fondaparinux
if any(c in codes_atc_detectes_upper for c in ["B01AB05", "B01AB10", "B01AX05"]):
    st.divider()
    st.subheader("Contexte héparine")

    dose_heparine_ui = st.radio(
        "Type d'anticoagulation",
        ["Dose préventive", "Dose curative"],
        horizontal=True,
        key="ui_dose_heparine"
    )

    dose_heparine = "préventive" if dose_heparine_ui == "Dose préventive" else "curative"

    if "B01AB10" in codes_atc_detectes_upper:
        st.info("""
Rappel posologiques

Préventif :
- 4500 UI/j en SC
- SI IMC > 40 alors 75UI/kg poids réel x 1/j

Curatif :
- 175 ui/kg x 1/j
""")

    if "B01AX05" in codes_atc_detectes_upper:
        st.info("""
Rappels posologiques

Préventif :
- 2.5mg/j SC
- si IMC > 40 alors 5mg/j SC

Curatif :
- Poids < 50kgs = 5mg/j SC
- Poids 50-100kgs = 7.5mg/j SC
- Poids > 100kgs = 10mg/j SC
""")

ctx["voie_heparine"] = voie_heparine
ctx["dose_heparine"] = dose_heparine


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

    resultats, vus, candidats_retenus = detecter_medicaments_depuis_texte(
        txt=txt_final,
        ref=ref,
        atc_map=atc_map,
        classe_map=classe_map,
        ctx=ctx
    )

st.markdown("""
<style>
.card {
    border-radius: 16px;
    padding: 16px;
    margin-bottom: 16px;
    box-shadow: 0 2px 8px rgba(0,0,0,0.04);
}

.card-green {
    border: 1px solid #bfe5c8;
    background-color: #f4fbf6;
}

.card-orange {
    border: 1px solid #ffd1a8;
    background-color: #fff8f2;
}

.card-blue {
    border: 1px solid #bfd8ff;
    background-color: #f6faff;
}

.card-title-green {
    color: #16813a;
    font-size: 20px;
    font-weight: 700;
    margin-bottom: 12px;
}

.card-title-blue {
    color: #0757c2;
    font-size: 26px;
    font-weight: 700;
    margin-bottom: 12px;
}

.card-title-orange {
    color: #f26b00;
    font-size: 26px;
    font-weight: 700;
    margin-bottom: 12px;
}

.green-box {
    background-color: #eaf7ee;
    color: #16813a;
    border-radius: 12px;
    padding: 16px;
    font-size: 18px;
    margin-top: 10px;
    line-height: 1.6;
}

.orange-box {
    background-color: #fff3e0;
    color: #e65100;
    border-radius: 12px;
    padding: 16px;
    font-size: 18px;
    margin-top: 10px;
    line-height: 1.6;
}

.blue-box {
    background-color: #eef5ff;
    color: #0757c2;
    border-radius: 12px;
    padding: 16px;
    font-size: 18px;
    margin-top: 10px;
    line-height: 1.6;
}

.big-asa {
    color: #16813a;
    font-size: 25px;
    font-weight: 700;
}
</style>
""", unsafe_allow_html=True)

if texte_detecte.strip():

    st.markdown("""
    <h2 style="
    font-weight:800;
    font-size:42px;
    margin-bottom:25px;">
    Synthèse de l'évaluation préopératoire
    </h2>
    """, unsafe_allow_html=True)


# =========================
# CALENDRIER
# =========================

au_moins_un_arret = False
lignes_pdf = []

for r in resultats:
    action = str(r["Action"]).upper().strip()
    date_txt = str(r["Date"]).upper().strip()
    note = str(r.get("Note", "")).lower()

    if "ARRET" in action:
        jours = extraire_nb_jours(date_txt)

        if jours is not None:
            d_stop = date_op - timedelta(days=jours)

            lignes_pdf.append(
                f"{r['Médicament']} : dernière prise le {d_stop.strftime('%d/%m/%Y')}"
            )

            if "relais par aspirine" in note:
                d_relais = d_stop + timedelta(days=1)

                lignes_pdf.append(
                    f"Relais par aspirine 75 à 100 mg à débuter le {d_relais.strftime('%d/%m/%Y')}"
                )

            au_moins_un_arret = True
            continue

        match_h = re.search(r"(\d+)\s*H", date_txt)

        if match_h:
            heures = int(match_h.group(1))

            lignes_pdf.append(
                f"{r['Médicament']} : dernière prise : {heures} heures avant l’intervention"
            )

            if "relais par aspirine" in note:
                lignes_pdf.append(
                    "Relais par aspirine 75 à 100 mg à débuter le lendemain de l’arrêt"
                )

            au_moins_un_arret = True
            continue

    if action == "PAS DE PRISE LE MATIN":
        lignes_pdf.append(
            f"{r['Médicament']} : ne pas prendre le matin de l'intervention, le   {date_op.strftime('%d/%m/%Y')}"
        )
        au_moins_un_arret = True

    elif action == "STOP MATIN":
        lignes_pdf.append(
            f"{r['Médicament']} : ne pas prendre le matin de l'intervention, le {date_op.strftime('%d/%m/%Y')}"
        )
        au_moins_un_arret = True

    elif action == "ARRET" and str(r.get("Date", "")).upper() == "IMMÉDIAT":
        lignes_pdf.append(
            f"{r['Médicament']} : arrêter le traitement dès maintenant."
        )
        au_moins_un_arret = True


#date intervention que pr avk
if avk_detecte and date_op:
    lignes_pdf.append(
        f"Intervention prévue le {date_op.strftime('%d/%m/%Y')}"
    )



phrase_pdf = ""

if au_moins_un_arret:
    phrase_pdf = "Poursuivre le reste des médicaments jusqu'au jour de l'intervention avec un peu d'eau."
else:
    phrase_pdf = "Aucun arrêt médicamenteux daté à planifier selon les règles actuelles."

lignes_html = ""

for ligne in lignes_pdf:
    lignes_html += f"""
    <div style="
        background:#e8f0ff;
        color:#0b57d0;
        padding:8px 14px;
        border-radius:10px;
        font-weight:700;
        display:inline-block;
        margin-bottom:12px;
    ">
        {ligne}
    </div><br>
    """

if lignes_html:
    contenu_calendrier = lignes_html + f"<br>{phrase_pdf}"
else:
    contenu_calendrier = phrase_pdf


if resultats:

    with st.container(border=True):

        st.markdown("## :blue[Calendrier Patient]")

        st.markdown(f"""
        <div class="blue-box">
            {contenu_calendrier}
        </div>
        """, unsafe_allow_html=True)


        texte_a_copier = "\n".join(lignes_pdf)


        if phrase_pdf:
            texte_a_copier += "\n" + phrase_pdf

        texte_html = html.escape(texte_a_copier)

        components.html(
            f"""
            <textarea id="texteCopie" style="position:absolute; left:-9999px;">
        {texte_html}
            </textarea>

            <button id="boutonCopie"
                    onclick="copierTexte()"
                    style="
                        background:#0b57d0;
                        color:white;
                        border:none;
                        padding:9px 16px;
                        border-radius:8px;
                        cursor:pointer;
                        font-weight:600;
                   ">
                  Copier vers le presse-papiers
            </button>

            <script>
            function copierTexte() {{
                var texte = document.getElementById("texteCopie");
                texte.select();
                texte.setSelectionRange(0, 999999);

                var succes = document.execCommand("copy");

                if (succes) {{
                    document.getElementById("boutonCopie").innerHTML = " Copié !";
                }} else {{
                    document.getElementById("boutonCopie").innerHTML = " Copie impossible";
                }}
            }}
            </script>
            """,
            height=55
        )




        st.markdown("### :blue[DOCUMENTS PRÉOPÉRATOIRES]")

        creer_ordonnance_patient = st.checkbox(
            "Créer l'ordonnance patient",
            key="creer_ordonnance_patient"
        )

        creer_ordonnance_pharmacie = False
        creer_prescription_ide = False
 
        if avk_detecte and ordonnance_pharmacie:

            creer_ordonnance_pharmacie = st.checkbox(
                "Créer l'ordonnance pharmacie",
                key="creer_ordonnance_pharmacie"
            )

            creer_prescription_ide = st.checkbox(
                "Créer la prescription IDE",
                key="creer_prescription_ide"
            )

    path_ide = None

    if (
        creer_ordonnance_patient
        or creer_ordonnance_pharmacie
        or creer_prescription_ide
    ):

       


        # ORDONNANCE PHARMACIE

        if creer_ordonnance_pharmacie and ordonnance_pharmacie:

            st.markdown("### Ordonnance pharmacie")

            st.text_area(
                "Ordonnance du relais",
                ordonnance_pharmacie,
                height=220,
                disabled=True
            )


        # =========================
        # PRESCRIPTION PR IDE
        # =========================

        if creer_prescription_ide and prescription_ide:

            st.markdown("### Prescription IDE")

            st.text_area(
                "Prescription à réaliser",
                prescription_ide,
                height=220,
                disabled=True
            )


        ville = st.text_input("Ville", value="Marseille")

        civilite = ""

        if creer_ordonnance_patient or creer_ordonnance_pharmacie:
            civilite = st.selectbox(
                "Civilité",
                ["Madame", "Monsieur"]
            )



        nom = st.text_input("Nom prénom")

        medecin = st.text_input(
            "Nom du médecin"
            
        )

        date_doc = st.date_input(
            "Date",
            value=date.today()
        )

           
        # ORDONNANCE PATIENT
      

        bilan_texte = ""
        scanner_texte = ""
        allergies_texte = ""

        if creer_ordonnance_patient:

            st.markdown("### Ordonnance patient")

            st.subheader("Préparation pré-opératoire")

            ajouter_prepa = st.checkbox(
                "Ajouter des examens ou précautions"
            )

            if ajouter_prepa:

                bilan_sanguin = st.checkbox("Bilan sanguin")
                scanner = st.checkbox("Scanner / imagerie")
                allergies = st.checkbox("Allergies")

                if bilan_sanguin:
                    bilan_texte = st.text_area(
                        "Bilans demandés",
                        placeholder="Ex : NFS, créatinine, TP/TCA..."
                    )

                if scanner:
                    scanner_texte = st.text_area(
                        "Examens complémentaires",
                        placeholder="Ex : scanner injecté, ECG..."
                    )

                if allergies:
                    allergies_texte = st.text_area(
                        "Allergies / précautions",
                        placeholder="Ex : iode, latex, héparine..."
                    )



# ========================================
# TABLEAU PATIENT AVK PRE OP
# ======================================

tableau_avk = []

if avk_detecte and schema_relais and schema_relais.get("indique"):


    for r in resultats:

        code_atc = str(r.get("Code ATC", "")).upper().strip()

        if code_atc.startswith("B01AA"):

            jour_arret = str(r.get("Date", "")).strip()
            medicament_avk = str(r.get("Médicament", "AVK")).strip()

            if jour_arret.startswith("J-"):

                nb_jours = extraire_nb_jours(jour_arret)

                if nb_jours is not None:
                    tableau_avk.append([
                        f"J-{nb_jours}",
                        f"Dernière prise de {medicament_avk}"
                    ])

            break

    # ---------------------------------
    # Relais choisi
    # -------------------------------------------

    molecule_relais = schema_relais.get("molecule")

    injections = calculer_injections_relais(
        schema_relais,
        date_op
    )

    if injections:

        premiere = injections[0]
        derniere = injections[-1]

        if molecule_relais == "Enoxaparine":
            type_instruction = "HBPM curative – Enoxaparine"

        elif molecule_relais == "Tinzaparine":
            type_instruction = "HBPM curative – Tinzaparine"

        elif molecule_relais == "HNF calcique":
            type_instruction = "HNF calcique curative"

        else:
            type_instruction = "Héparine curative"

        tableau_avk.append([
            premiere["moment"],
            f"Première injection de {type_instruction}"
        ])

        tableau_avk.append([
            derniere["moment"],
            f"Dernière injection préopératoire de {type_instruction}"
        ])

    # HNF IVSE : pas d'injection ponctuelle
    elif molecule_relais == "HNF":

        tableau_avk.append([
            schema_relais.get("debut", "J-3"),
            "Début de l’HNF IVSE curative"
        ])

        tableau_avk.append([
            "J0",
            "Arrêt de l’HNF IVSE 6 heures avant la procédure"
        ])

    # -----------------------------------------
    # INR 


    tableau_avk.append([
        "J-1",
        "Contrôle INR"
    ])

    tableau_avk.append([
        "J-1 si INR > seuil",
        "Vitamine K PO 2 à 5 mg"
    ])

    tableau_avk.append([
        "J0",
        "Contrôle INR si vitamine K administrée ; procédure si seuil atteint"
    ])


if st.button("Générer PDF"):

    path = None
    path_pharmacie = None
    path_ide = None

    # =================
    # ORDONNANCE PATIENT
    # =================

    if creer_ordonnance_patient:
        path = generer_pdf_patient(
            ville,
            date_doc.strftime("%d/%m/%Y"),
            civilite,
            nom,
            lignes_pdf,
            phrase_pdf,
            bilan_texte,
            scanner_texte,
            allergies_texte,
            medecin,
            tableau_avk
        )

    # =================
    # ORDO PHARMACIE
    # =================

    if creer_ordonnance_pharmacie and ordonnance_pharmacie:
        path_pharmacie = generer_pdf_ordonnance_pharmacie(
            ville,
            date_doc.strftime("%d/%m/%Y"),
            civilite,
            nom,
            ordonnance_pharmacie,
            medecin
        )

    # ===================
    # PRESCRIPTION INFIRMIERE
    # ===================

    if creer_prescription_ide and prescription_ide:
        path_ide = generer_pdf_prescription_ide(
            ville,
            date_doc.strftime("%d/%m/%Y"),
            civilite,
            nom,
            prescription_ide,
            medecin
        )

    # ===================
    # TELECHARGEMENT PATIENT
    # ===================

    if creer_ordonnance_patient:
        if path and os.path.exists(path):
            with open(path, "rb") as f:
                st.download_button(
                    "Télécharger l'ordonnance patient",
                    f,
                    "ordonnance_patient.pdf"
                )
        else:
            st.error("L'ordonnance patient n’a pas pu être générée.")

    # ======================
    # TELECHARGEMENT PHARMACIE
    # ======================

    if creer_ordonnance_pharmacie:
        if path_pharmacie and os.path.exists(path_pharmacie):
            with open(path_pharmacie, "rb") as f:
                st.download_button(
                    "Télécharger l'ordonnance du relais",
                    f,
                    "ordonnance_relais.pdf"
                )
        else:
            st.error("L'ordonnance du relais n’a pas pu être générée.")

    # =========================
    # TELECHARGEMENT INFIRMIER
    # =========================

    if creer_prescription_ide:
        if path_ide and os.path.exists(path_ide):
            with open(path_ide, "rb") as f:
                st.download_button(
                    "Télécharger la prescription IDE",
                    f,
                    "prescription_ide.pdf"
                )
        else:
            st.error("La prescription IDE n’a pas pu être générée.")











# =========================
# RESULTATS
# ========================
codes_atc_detectes = []

if resultats:
    codes_atc_detectes = list(vus)
    asa_calcule = calculer_asa(age, len(codes_atc_detectes), None)
    asa_affiche = ctx.get("ASA")
    asa_a_afficher = asa_affiche if asa_affiche is not None else asa_calcule

    df_final = pd.DataFrame(resultats)

    def format_lien_unique(liens):
        if not liens:
            return ""
        return str(liens).split(" | ")[0].strip()

    df_final["Lien"] = df_final["Lien"].apply(format_lien_unique)



    df_profils_patient = inferer_profils_structures(
        codes_atc_detectes,
        df_sentinelles_ready,
        df_profils_ready
    )

    col1, col2 = st.columns(2)

    with col1:
        st.markdown(f"""
        <div class="card card-green">
            <div class="card-title-green">Score ASA Prédit</div>
            <div class="green-box">
                Score ASA : <span class="big-asa">{asa_a_afficher}</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

  
    with col2:
        if df_profils_patient is not None and not df_profils_patient.empty:

            profils_html = ""

            for i, row in df_profils_patient.iterrows():
                titre = "Profil principal probable" if i == 0 else "Profil associé probable"

                profils_html += (
                    f"<b>{titre}</b><br>"
                    f"{row['Profil']}<br>"
                    f"Certitude : {row['Niveau']}<br><br>"
                )

            bloc_html = f"""
            <div class="card card-green">
                <div class="card-title-green">Profils pathologiques probables</div>
                <div class="green-box">
                    {profils_html}
                </div>
            </div>
            """

            st.markdown(bloc_html, unsafe_allow_html=True)



st.divider()


if resultats:

    afficher_risque_aod_avk = aod_detecte or avk_detecte
    afficher_risque_aap = aap_detecte

    colonnes = ["Patient", "Date chirurgie"]

    if afficher_risque_aod_avk:
        colonnes.append("Risque AOD/AVK")

    if afficher_risque_aap:
        colonnes.append("Risque AAP")

    colonnes.append("ALR")

    cols = st.columns(len(colonnes))

    i = 0

    with cols[i]:
        st.markdown("**Patient**")
        st.markdown(f"### {age} ans")
    i += 1

    with cols[i]:
        st.markdown("**Date chirurgie**")
        st.markdown(f"### {date_op.strftime('%d/%m/%Y')}")
    i += 1

    if afficher_risque_aod_avk:
        with cols[i]:
            st.markdown("**Risque AOD/AVK**")
            st.markdown(f"### {risque_aod_avk}")
        i += 1

    if afficher_risque_aap:
        with cols[i]:
            st.markdown("**Risque AAP**")
            st.markdown(f"### {risque_aap}")
        i += 1

    with cols[i]:
        st.markdown("**ALR**")
        st.markdown(f"### {type_alr}")




    st.subheader("Tableau des recommandations")

    col1, col2, col3, col4, col5, col6, col7, col8, col9 = st.columns(
        [2.0, 1.4, 2.8, 1.4, 1.8, 4.0, 1.6, 1.6, 2.0]
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
            [2.0, 1.4, 2.8, 1.4, 1.8, 4.0, 1.6, 1.6, 2.0]
        )

        c1.write(r.get("Médicament", ""))
        c2.write(r.get("Code ATC", ""))
        c3.write(r.get("Classe", ""))
        c4.write(r.get("Action", ""))
        c5.write(format_jour_avec_date(r.get("Date", ""), date_op))
        note_affichee = enrichir_note_avec_dates(r.get("Note", ""), date_op)

        if (
            str(r.get("Code ATC", "")).upper().strip().startswith("B01AA")
            and ctx.get("r_hem") not in ["FAIBLE", "NUL"]
        ):
            note_affichee += (
                "\n\n<div style='background-color:#e7f3ff; padding:10px; border-radius:8px; "
                "border-left:4px solid #1f77b4; font-size:13px;'>"
                "<b>Note AVK :</b><br>"
                "Faire un dosage INR à J-1, la veille de l’intervention.<br>"
                "<b>Seul cet INR à J-1 doit être utilisé pour décider d’une correction par vitamine   K.</b><br>"
                "Si l’INR à J-1 est supérieur au seuil hémostatique : vitamine K 2 à 5 mg per os le soir,<br>"
                "puis contrôle de l’INR le lendemain matin."
                "</div>"
            )









        c6.markdown(note_affichee, unsafe_allow_html=True)


        liens_bruts = str(r.get("Lien", "")).strip()

        liens_list = [l.strip() for l in liens_bruts.split(" | ") if l.strip()] if liens_bruts else []

        sources_alr = [
            "https://sfar.org/wp-content/uploads/2026/04/RFE-20.4.2026-deifinitif-et-validei.pdf",
            "https://journals.lww.com/ejanaesthesiology/fulltext/2022/02000/regional_anaesthesia_in_patients_on_antithrombotic.4.aspx",
            "https://sfar.org/wp-content/uploads/2019/10/rfe-anesthesie-loco-regionale-perinerveuse.pdf",
]

        if str(type_alr).upper().strip() in ["SUPERFICIEL", "PROFOND", "NEURAXIAL"]:
            liens_list.extend(sources_alr)

        liens_list = list(dict.fromkeys(liens_list))

        if liens_list:
            with c7:
                with st.popover(f"{len(liens_list)} source(s)"):
                    for j, lien in enumerate(liens_list):
                        st.link_button(f"Ouvrir source {j+1}", str(lien).strip())
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
    csv_feedback = df_feedback.to_csv(index=False, encoding="utf-8-sig")

    st.download_button(
        label="Télécharger validation médecin",
        data=csv_feedback,
        file_name=f"validation_medecin_{date.today().isoformat()}.csv",
        mime="text/csv"
    )


# =========================
# QUESTIONNAIRE DE SATISFACTION
# =========================


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

    if st.button("Valider le questionnaire"):

        data = {
            "date": date.today().isoformat(),
            "profil": profil,
            "commentaire": commentaire
        }

        for i, rep in enumerate(reponses, start=1):
            data[f"Q{i}"] = rep

        df_satisfaction = pd.DataFrame([data])
        csv_satisfaction = df_satisfaction.to_csv(index=False, encoding="utf-8-sig")

        st.download_button(
            label="Télécharger le questionnaire",
            data=csv_satisfaction,
            file_name=f"questionnaire_satisfaction_{date.today().isoformat()}.csv",
            mime="text/csv"
        )
