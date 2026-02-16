
import os
from dotenv import load_dotenv
load_dotenv()
import pymupdf4llm
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI
import litellm
litellm.set_verbose = False

def load_pdf(pdf_path):
    md_text = pymupdf4llm.to_markdown(pdf_path)
    return md_text    

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")


def get_big_llm():
    """GPT-4o pour les tâches complexes."""
    return ChatOpenAI(
        model="gpt-4o",
        temperature=0.0,
        api_key=OPENAI_API_KEY
    )

def get_small_llm():
    """GPT-4o-mini pour l'extraction."""
    return ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0.0,
        max_tokens=1500,
        api_key=OPENAI_API_KEY
    )

def get_fast_llm():
    """Groq llama-3.1-8b - Le plus RAPIDE."""
    return ChatGroq(
        model="groq/llama-3.1-8b-instant",
        temperature=0.0,
        max_tokens=1500,
        groq_api_key=GROQ_API_KEY
    )

def get_openai_small_llm():
    """GPT-4o-mini - Fallback."""
    return ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0.0,
        api_key=OPENAI_API_KEY
    )