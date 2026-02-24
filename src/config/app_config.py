"""
Configuration des LLMs et utilitaires de chargement.
"""

import os
from dotenv import load_dotenv
load_dotenv()
import fitz
import pymupdf4llm
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI
import litellm
litellm.set_verbose = False


def load_pdf(pdf_path: str) -> str:
    """Convertit un PDF en texte Markdown via pymupdf4llm (structure + formatage)."""
    return pymupdf4llm.to_markdown(pdf_path)


def load_pdf_first_page_text(pdf_path: str) -> str:
    """Extrait le texte brut de la première page en ordre de lecture (haut → bas, gauche → droite).

    Utilise fitz directement pour capturer les headers/sidebars que pymupdf4llm
    peut ignorer ou réordonner sur les CV à mise en page complexe (bannières colorées,
    colonnes, boîtes décoratives).
    """
    doc = fitz.open(pdf_path)
    if not doc:
        return ""

    page = doc[0]
    # Récupère les blocs texte avec leurs coordonnées
    blocks = page.get_text("blocks")  # (x0, y0, x1, y1, text, block_no, block_type)
    # Filtre les blocs texte (type 0) non vides
    text_blocks = [b for b in blocks if b[6] == 0 and b[4].strip()]
    # Trie par ligne (y arrondi à 10px pour gérer l'alignement imparfait), puis par colonne (x)
    text_blocks.sort(key=lambda b: (round(b[1] / 10) * 10, b[0]))

    doc.close()
    return "\n".join(b[4].strip() for b in text_blocks)


OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")


def get_big_llm():
    """GPT-4o pour les tâches complexes — max_tokens élevé pour éviter la troncature JSON."""
    return ChatOpenAI(
        model="gpt-4o",
        temperature=0.0,
        max_tokens=16384,
        api_key=OPENAI_API_KEY
    )


def get_small_llm():
    """GPT-4o-mini pour l'extraction rapide."""
    return ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0.0,
        max_tokens=1500,
        api_key=OPENAI_API_KEY
    )


def get_fast_llm():
    """Groq llama-3.1-8b - Le plus rapide."""
    return ChatGroq(
        model="groq/llama-3.1-8b-instant",
        temperature=0.0,
        max_tokens=1500,
        groq_api_key=GROQ_API_KEY
    )
