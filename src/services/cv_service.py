import logging
from typing import Dict, Any
from src.config.app_config import load_pdf
from src.parser_flow.CV_agent_flow import CVAgentOrchestrator

logger = logging.getLogger(__name__)

async def parse_cv(pdf_path: str, user_id: str = None) -> Dict[str, Any]:
    orchestrator = CVAgentOrchestrator()
    cv_text = load_pdf(pdf_path)
    sections = await orchestrator.split_cv_sections(cv_text)
    cv_data = await orchestrator.extract_all_sections(sections)
    return cv_data

