"""
Service de parsing et analyse de CV enrichi.
Pipeline optimisé :
  Phase 1  : Découpage en sections
  Phase 2  : Extraction parallèle (8 agents) — en // avec Phase 3a
  Phase 3a : Analyse d'en-tête (header_analyzer) — en // avec Phase 2
  Phase 3b : Analyse & Recommandation (3 agents parallèles)

Flux : Phase 1 → asyncio.gather(Phase 2, Phase 3a) → Phase 3b
Gain estimé : ~5-8 secondes vs pipeline séquentiel précédent.
"""

import asyncio
import logging
from typing import Dict, Any

from src.config.app_config import load_pdf, load_pdf_first_page_text
from src.parser_flow.CV_agent_flow import CVAgentOrchestrator

logger = logging.getLogger(__name__)


async def parse_cv(pdf_path: str, file_name: str = "") -> Dict[str, Any]:
    """
    Parse un CV avec extraction + analyse complète.

    Retourne un JSON en 2 parties :
    - candidat : données parsées du CV (identité, compétences, expériences, projets…)
    - recommandations : analyse critique, matching métiers, qualité CV, header_analysis (poste_vise)
    """
    orchestrator = CVAgentOrchestrator()

    # Double extraction PDF :
    # - cv_text    : Markdown (bon pour la structure des sections)
    # - cv_raw_start : texte brut ordonné par position (fiable pour le header/nom/titre)
    cv_text = load_pdf(pdf_path)
    cv_raw_start = load_pdf_first_page_text(pdf_path)

    # ── Phase 1 : Découpage du CV en sections (séquentielle, nécessaire pour la suite) ──
    logger.info("Phase 1 : Découpage du CV en sections...")
    sections = await orchestrator.split_cv_sections(cv_text, cv_raw_start=cv_raw_start)

    # ── Phase 2 + Phase 3a en PARALLÈLE ──────────────────────────────────────────────────
    # Phase 2  : 8 agents d'extraction (skills, expériences, projets, etc.)
    # Phase 3a : header_analyzer (poste visé) — ne dépend que de sections + cv_raw_start
    logger.info("Phase 2 + Phase 3a : Extraction et analyse d'en-tête en parallèle...")
    extraction, header_data = await asyncio.gather(
        orchestrator.extract_all_sections(
            sections, cv_raw_start=cv_raw_start, file_name=file_name
        ),
        orchestrator.run_header_analysis(
            sections, cv_raw_start=cv_raw_start, cv_full_text=cv_text
        ),
    )

    # ── Phase 3b : 3 agents d'analyse en parallèle ───────────────────────────────────────
    logger.info("Phase 3b : Analyse et recommandation...")
    recommendations = await orchestrator.analyze_and_recommend(
        cv_full_text=cv_text,
        sections=sections,
        extraction=extraction,
        cv_raw_start=cv_raw_start,
        header_data=header_data,
    )

    candidat_raw = extraction.get("candidat", {})

    # Assemblage ordonné : identité → langues → compétences → parcours
    candidat = {
        "first_name":   candidat_raw.get("first_name"),
        "langues":      candidat_raw.get("langues", []),
        "compétences":  candidat_raw.get("compétences", {}),
        "expériences":  candidat_raw.get("expériences", []),
        "projets":      candidat_raw.get("projets", {}),
        "formations":   candidat_raw.get("formations", []),
        "etudiant":     candidat_raw.get("etudiant", {}),
        "reconversion": candidat_raw.get("reconversion", {}),
    }

    result = {
        "candidat": candidat,
        "recommandations": recommendations,
    }

    logger.info("Parsing et analyse terminés.")
    return result
