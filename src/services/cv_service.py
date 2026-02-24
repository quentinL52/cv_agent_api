"""
Service de parsing et analyse de CV enrichi.
Pipeline en 3 phases :
  1. Découpage en sections (avec extraction brute pour le header)
  2. Extraction parallèle (compétences, expériences, projets, etc.)
  3. Analyse et recommandation (poste visé, matching métiers, qualité CV, projets)
"""

import logging
from typing import Dict, Any

from src.config.app_config import load_pdf, load_pdf_first_page_text
from src.parser_flow.CV_agent_flow import CVAgentOrchestrator

logger = logging.getLogger(__name__)


async def parse_cv_enriched(pdf_path: str, file_name: str = "") -> Dict[str, Any]:
    """
    Parse un CV avec extraction + analyse complète.

    Retourne un JSON en 2 parties :
    - candidat : données parsées du CV (identité, compétences, expériences, projets…)
    - recommandations : analyse critique, matching métiers, qualité CV, header_analysis (poste_vise)
    """
    orchestrator = CVAgentOrchestrator()

    # Double extraction :
    # - cv_text : Markdown (bon pour la structure des sections)
    # - cv_raw_start : texte brut ordonné par position (fiable pour le header/nom/titre)
    cv_text = load_pdf(pdf_path)
    cv_raw_start = load_pdf_first_page_text(pdf_path)

    logger.info("Phase 1 : Découpage du CV en sections...")
    sections = await orchestrator.split_cv_sections(cv_text, cv_raw_start=cv_raw_start)

    logger.info("Phase 2 : Extraction parallèle des données...")
    extraction = await orchestrator.extract_all_sections(
        sections, cv_raw_start=cv_raw_start, file_name=file_name
    )

    logger.info("Phase 3 : Analyse et recommandation...")
    recommendations = await orchestrator.analyze_and_recommend(
        cv_full_text=cv_text,
        sections=sections,
        extraction=extraction,
        cv_raw_start=cv_raw_start,
    )

    candidat_raw = extraction.get("candidat", {})

    # Assemblage ordonné : identité → langues → compétences → parcours
    candidat = {
        "first_name":  candidat_raw.get("first_name"),
        "langues":     candidat_raw.get("langues", []),
        "compétences": candidat_raw.get("compétences", {}),
        "expériences": candidat_raw.get("expériences", []),
        "projets":     candidat_raw.get("projets", {}),
        "formations":  candidat_raw.get("formations", []),
        "etudiant":    candidat_raw.get("etudiant", {}),
        "reconversion": candidat_raw.get("reconversion", {}),
    }

    result = {
        "candidat": candidat,
        "recommandations": recommendations,
    }

    logger.info("Parsing et analyse terminés.")
    return result
