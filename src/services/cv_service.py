"""
Service de parsing et analyse de CV enrichi.
Pipeline ultra-optimisé :
  Phase 1 : Découpage en sections (cv_splitter)
  Phase 2 : Extraction et Analyse en STRICT PARALLÈLE (11 agents)

Gain estimé : Temps de traitement grandement réduit.
"""

import logging
from typing import Dict, Any

from src.config.app_config import load_pdf, load_pdf_first_page_text, get_pdf_page_count
from src.parser_flow.CV_agent_flow import CVAgentOrchestrator

logger = logging.getLogger(__name__)


async def parse_cv(pdf_path: str, file_name: str = "") -> Dict[str, Any]:
    """
    Parse un CV avec extraction + analyse complète en 2 phases.

    Retourne un JSON en 2 parties :
    - candidat : données parsées du CV (identité, compétences, expériences, projets...)
    - recommandations : analyse critique (7 critères JSON), matching métiers, qualité CV
    """
    orchestrator = CVAgentOrchestrator()
    cv_text = load_pdf(pdf_path)
    cv_raw_start = load_pdf_first_page_text(pdf_path)
    page_count = get_pdf_page_count(pdf_path)

    logger.info("Phase 1 : Découpage du CV en sections...")
    sections = await orchestrator.split_cv_sections(cv_text, cv_raw_start=cv_raw_start)

    logger.info("Phase 2 : Extraction et Analyse en strict parallèle...")
    result = await orchestrator.run_all_agents(
        sections,
        cv_raw_start=cv_raw_start,
        cv_full_text=cv_text,
        file_name=file_name,
        page_count=page_count
    )

    logger.info("Parsing et analyse terminés.")
    return result
