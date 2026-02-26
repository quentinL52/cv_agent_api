"""
Orchestrateur CV enrichi avec 3 phases :
  Phase 1  : Découpage du CV en sections (cv_splitter)
  Phase 2  : Extraction parallèle (8 agents)
  Phase 3a : Analyse d'en-tête (run_header_analysis) — tourne en // avec Phase 2
  Phase 3b : Analyse & Recommandation — 3 agents en parallèle après Phase 2 + 3a

Flux optimisé : Phase 1 → (Phase 2 // Phase 3a) → Phase 3b
Produit un JSON en 2 parties : candidat + recommandations.
"""

import json
import logging
import os
import yaml
import asyncio
from datetime import datetime
from typing import Dict, Any, List

from crewai import Agent, Task, Crew, Process
from src.config.app_config import get_small_llm, get_big_llm

logger = logging.getLogger(__name__)

#_____________________________________________________________________________________

class CVAgentOrchestrator:
    """Orchestrateur multi-agents pour le parsing et l'analyse de CV."""

    def __init__(self):
        self.llm = get_small_llm()
        self.big_llm = get_big_llm()
        self.agents_config = self._load_yaml("agents.yaml")
        self.tasks_config = self._load_yaml("tasks.yaml")
        self.metiers_data = self._load_metiers()
        self._create_agents()

    def _load_yaml(self, filename: str) -> Dict:
        base_path = os.path.dirname(os.path.dirname(__file__))
        config_path = os.path.join(base_path, "config", filename)
        with open(config_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)

    def _load_metiers(self) -> List[Dict]:
        """Charge le référentiel de métiers (avec embeddings)."""
        base_path = os.path.dirname(os.path.dirname(__file__))
        metiers_path = os.path.join(base_path, "data", "metiers.json")
        with open(metiers_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data.get("metiers", [])

    def _create_agents(self):
        def make_agent(name, llm_override=None):
            return Agent(
                config=self.agents_config[name],
                llm=llm_override or self.llm,
                allow_delegation=False,
                verbose=True,
                max_iter=1,
                respect_context_window=True,
            )

        self.cv_splitter = make_agent("cv_splitter")
        self.skills_extractor = make_agent("skills_extractor")
        self.experience_extractor = make_agent("experience_extractor")
        self.project_extractor = make_agent("project_extractor")
        self.education_extractor = make_agent("education_extractor")
        self.reconversion_detector = make_agent("reconversion_detector")
        self.language_extractor = make_agent("language_extractor")
        self.etudiant_detector = make_agent("etudiant_detector")
        self.identity_extractor = make_agent("identity_extractor")

        self.header_analyzer = make_agent("header_analyzer")
        self.metier_matcher = make_agent("metier_matcher")
        self.cv_quality_checker = make_agent("cv_quality_checker", llm_override=self.big_llm)
        self.project_analyzer = make_agent("project_analyzer", llm_override=self.big_llm)

    # ──────────────────────────────────────────────
    # PHASE 1 : Découpage du CV en sections
    # ──────────────────────────────────────────────

    async def split_cv_sections(self, cv_content: str, cv_raw_start: str = "") -> Dict[str, str]:
        """Découpe le CV en sections via l'agent cv_splitter."""
        task_config = self.tasks_config["split_cv_task"].copy()
        # Échapper les accolades dans le contenu CV pour éviter les erreurs de format
        safe_content = cv_content[:20000].replace("{", "{{").replace("}", "}}")
        safe_raw = cv_raw_start[:2000].replace("{", "{{").replace("}", "}}")
        task_config["description"] = task_config["description"].format(
            cv_content=safe_content,
            cv_raw_start=safe_raw,
        )

        task = Task(config=task_config, agent=self.cv_splitter)
        crew = Crew(
            agents=[self.cv_splitter],
            tasks=[task],
            process=Process.sequential,
            verbose=False,
        )
        result = await crew.kickoff_async()
        parsed = self._parse_json_output(result, default_structure={})
        return parsed

    # ──────────────────────────────────────────────
    # PHASE 2 : Extraction et Analyse Parallèles
    # ──────────────────────────────────────────────

    async def run_all_agents(
        self, sections: Dict[str, str], cv_raw_start: str = "", cv_full_text: str = "", file_name: str = "", page_count: int = 1
    ) -> Dict[str, Any]:
        """Exécute toutes les tâches d'extraction et d'analyse en parallèle."""
        
        raw_header = sections.get("header", "")
        raw_experiences = sections.get("experiences", "")
        raw_projects = sections.get("projects", "")
        raw_skills = sections.get("skills", "")
        raw_education = sections.get("education", "")
        raw_languages = sections.get("languages", "")
        safe_cv_raw = cv_raw_start[:2000].replace("{", "{{").replace("}", "}}")
        safe_header = raw_header.replace("{", "{{").replace("}", "}}")

        from src.services.metier_pre_filter import get_top_k_metiers
        top_metiers = get_top_k_metiers(
            metiers_data=self.metiers_data,
            experiences_summary=raw_experiences[:2000],
            projects_summary=raw_projects[:2000],
            hard_skills=raw_skills[:2000],
            soft_skills="",
            k=3
        )
        metiers_reference = self._prepare_metiers_for_prompt(top_metiers)

        def create_task_async(task_key, agent, **kwargs):
            t_config = self.tasks_config[task_key].copy()
            t_description = t_config["description"]
            try:
                t_config["description"] = t_description.format(**kwargs)
            except KeyError as e:
                logger.warning(f"KeyError formatting task '{task_key}': {e}. Falling back to manual replace.")
                desc = t_description
                for k, v in kwargs.items():
                    placeholder = "{" + k + "}"
                    if placeholder in desc:
                        desc = desc.replace(placeholder, str(v))
                t_config["description"] = desc
            except Exception as e:
                logger.error(f"Unexpected error formatting task '{task_key}': {e}")
            
            task = Task(config=t_config, agent=agent)
            c = Crew(agents=[agent], tasks=[task], verbose=False)
            return (task_key, c.kickoff_async())

        tasks_def = [
            ("skills_task", self.skills_extractor, {"experiences": raw_experiences, "projects": raw_projects, "skills": raw_skills, "education": raw_education}),
            ("experience_task", self.experience_extractor, {"experiences": raw_experiences}),
            ("project_task", self.project_extractor, {"projects": raw_projects}),
            ("education_task", self.education_extractor, {"education": raw_education}),
            ("reconversion_task", self.reconversion_detector, {"experiences": raw_experiences, "education": raw_education}),
            ("language_task", self.language_extractor, {"languages": raw_languages, "cv_raw_start": cv_raw_start[:500]}),
            ("etudiant_task", self.etudiant_detector, {"education": raw_education, "current_date": datetime.now().strftime("%Y-%m-%d")}),
            ("identity_task", self.identity_extractor, {"header": raw_header, "cv_raw_start": cv_raw_start[:1500], "file_name": file_name}),
            ("poste_visé_task", self.header_analyzer, {"header": safe_header, "cv_raw_start": safe_cv_raw}),
            ("cv_quality_task", self.cv_quality_checker, {
                "header": safe_header,
                "page_count": page_count,
                "cv_full_text": cv_full_text[:6000],
                "cv_raw_start": safe_cv_raw,
                "skills": raw_skills[:2000],
                "experiences": raw_experiences[:3000],
                "projects": raw_projects[:2000],
                "education": raw_education[:2000],
            }),
            ("metier_matching_task", self.metier_matcher, {
                "header": safe_header,
                "skills": raw_skills[:2000],
                "experiences": raw_experiences[:3000],
                "projects": raw_projects[:2000],
                "education": raw_education[:2000],
                "metiers_reference": metiers_reference,
            }),
            ("project_analysis_task", self.project_analyzer, {
                "header": safe_header,
                "projects": raw_projects[:3000],
            }),
        ]

        task_coroutines = [create_task_async(key, agent, **kwargs) for key, agent, kwargs in tasks_def]
        keys = [t[0] for t in task_coroutines]
        coroutines = [t[1] for t in task_coroutines]
        results_list = await asyncio.gather(*coroutines, return_exceptions=True)

        results_map = {}
        for key, result in zip(keys, results_list):
            if isinstance(result, Exception):
                logger.error(f"Task '{key}' failed: {result}")
            else:
                results_map[key] = result

        return self._build_final_json(results_map)

    def _build_final_json(self, results_map: Dict[str, Any]) -> Dict[str, Any]:
        """Agrège les résultats de toutes les tâches en un JSON final."""
        
        def get_parsed(key, default=None):
            if key not in results_map:
                return default
            return self._parse_json_output(results_map[key], default)

        # Extraction
        competences = get_parsed("skills_task", {"hard_skills": [], "soft_skills": []})
        experiences = get_parsed("experience_task", [])
        projets = get_parsed("project_task", {"professional": [], "personal": []})
        formations = get_parsed("education_task", [])
        reconversion = get_parsed("reconversion_task", {}).get("reconversion_analysis", {})
        etudiant_data = get_parsed("etudiant_task", {}).get("etudiant_analysis", {})
        
        latest_end_date = etudiant_data.get("latest_education_end_date")
        if latest_end_date:
            etudiant_data["is_etudiant"] = self._is_ongoing_date(latest_end_date)

        is_en_poste = False
        if isinstance(experiences, list):
            for exp in experiences:
                end_date = exp.get("end_date")
                if isinstance(exp, dict) and end_date:
                    if self._is_ongoing_date(end_date):
                        is_en_poste = True
                        break

        langues_raw = get_parsed("language_task", {})
        identity = get_parsed("identity_task", {})

        # Nettoyage des doublons dans hard_skills (case-insensitive)
        if isinstance(competences, dict):
            raw_skills = competences.get("hard_skills", [])
            seen = set()
            unique_skills = []
            for skill in raw_skills:
                key = str(skill).lower() if not isinstance(skill, str) else skill.lower()
                if key not in seen:
                    seen.add(key)
                    unique_skills.append(skill)
            competences["hard_skills"] = unique_skills

        candidat = {
            "first_name": identity.get("first_name") if isinstance(identity, dict) else None,
            "langues": langues_raw.get("langues", []) if isinstance(langues_raw, dict) else [],
            "compétences": competences,
            "expériences": experiences,
            "reconversion": reconversion,
            "projets": projets,
            "formations": formations,
            "etudiant": etudiant_data,
            "is_en_poste": is_en_poste,
        }

        # Analyse
        header_data = get_parsed("poste_visé_task", {"poste_vise": "Non identifié", "confiance": 0})
        metier_data = get_parsed("metier_matching_task", {"postes_recommandes": []})
        quality_data = get_parsed("cv_quality_task", {"score_global": 0, "red_flags": [], "conseils_prioritaires": []})
        project_data = get_parsed("project_analysis_task", {"analyse_projets": []})

        conseils = []
        if isinstance(quality_data, dict):
            conseils.extend(quality_data.get("conseils_prioritaires", []))

        # Filtre de sécurité : ne garder dans l'analyse de projets que ceux issus de l'extraction
        extracted_titles: set[str] = set()
        for p in (projets.get("professional", []) if isinstance(projets, dict) else []):
            if isinstance(p, dict) and p.get("title"):
                extracted_titles.add(p["title"].strip().lower())
        for p in (projets.get("personal", []) if isinstance(projets, dict) else []):
            if isinstance(p, dict) and p.get("title"):
                extracted_titles.add(p["title"].strip().lower())

        analyse_projets = project_data.get("analyse_projets", []) if isinstance(project_data, dict) else []
        if extracted_titles and isinstance(analyse_projets, list):
            def _is_extracted_project(titre: str) -> bool:
                t = titre.strip().lower()
                return t in extracted_titles or any(t in ref or ref in t for ref in extracted_titles)

            analyse_projets = [p for p in analyse_projets if isinstance(p, dict) and _is_extracted_project(p.get("titre", ""))]

        recommandations = {
            "header_analysis": header_data,
            "postes_recommandes": metier_data.get("postes_recommandes", []) if isinstance(metier_data, dict) else [],
            "analyse_poste_vise": metier_data.get("analyse_poste_vise", "") if isinstance(metier_data, dict) else "",
            "qualite_cv": quality_data,
            "analyse_projets": analyse_projets,
            "coherence_globale_projets": project_data.get("coherence_globale", {}) if isinstance(project_data, dict) else {},
            "conseils_amelioration": conseils,
        }

        return {
            "candidat": candidat,
            "recommandations": recommandations
        }

    def _prepare_metiers_for_prompt(self, metiers: List[Dict] = None) -> str:
        """Prépare le référentiel métiers restreint pour le prompt."""
        if metiers is None:
            metiers = self.metiers_data
            
        flat_list = []
        def _flatten(job_list):
            for job in job_list:
                if "metiers" in job:
                    _flatten(job["metiers"])
                elif "id" in job:
                    flat_list.append(job)
        _flatten(metiers)
        
        lines = []
        for m in flat_list:
            mid = m.get("id", "?")
            nom = m.get("nom", "?")
            cat = m.get("categorie", "?")
            comp = m.get("competences_techniques", [])
            outils = m.get("outils_technologies", [])
            soft = m.get("competences_soft", [])
            niveau = m.get("niveau_etude", "?")
            exp = m.get("experience_requise", "?")
            lines.append(
                f"[{mid}] {nom} ({cat})\n"
                f"  Compétences techniques: {', '.join(comp)}\n"
                f"  Outils: {', '.join(outils)}\n"
                f"  Soft skills: {', '.join(soft[:3])}\n"
                f"  Niveau: {niveau} | Expérience: {exp}"
            )
        return "\n\n".join(lines)



    # ──────────────────────────────────────────────
    # Utilitaires
    # ──────────────────────────────────────────────

    def _is_ongoing_date(self, date_str: str) -> bool:
        """Détermine si une date (fin d'étude ou fin d'expérience) est dans le futur ou en cours."""
        if not date_str:
            return False
        date_str = str(date_str).lower().strip()
        ongoing_keywords = [
            "present", "présent", "current", "cours", "aujourd'hui", "now"
        ]
        if any(keyword in date_str for keyword in ongoing_keywords):
            return True

        try:
            now = datetime.now()
            end_date = None
            if len(date_str) == 10 and date_str[4] == "-" and date_str[7] == "-":
                end_date = datetime.strptime(date_str, "%Y-%m-%d")
            elif len(date_str) == 7 and date_str[4] == "-":
                end_date = datetime.strptime(date_str, "%Y-%m")
            elif "/" in date_str:
                parts = date_str.split("/")
                if len(parts) == 2:
                    _, y = parts
                    if len(y) == 4:
                        end_date = datetime.strptime(date_str, "%m/%Y")
                    elif len(y) == 2:
                        end_date = datetime.strptime(date_str, "%m/%y")
            elif len(date_str) == 4 and date_str.isdigit():
                end_date = datetime.strptime(date_str, "%Y")
                end_date = end_date.replace(month=12, day=31)

            if end_date:
                return end_date >= now
            return False
        except (ValueError, IndexError):
            logger.warning(f"Date parsing failed for: {date_str}")
            return False

    def _parse_json_output(self, crew_output, default_structure=None) -> Any:
        """Parse la sortie JSON d'un agent CrewAI avec nettoyage robuste."""
        if crew_output is None:
            return default_structure if default_structure is not None else {}

        raw = crew_output.raw if hasattr(crew_output, "raw") else str(crew_output)

        # Extraire le bloc JSON si encapsulé dans des backticks
        if "```json" in raw:
            raw = raw.split("```json")[1].split("```")[0].strip()
        elif "```" in raw:
            parts = raw.split("```")
            if len(parts) >= 3:
                raw = parts[1].strip()

        raw = raw.strip().lstrip("\ufeff")

        def _try_parse(text: str):
            """Tente un parse direct puis un parse avec extraction du premier bloc JSON."""
            try:
                return json.loads(text)
            except json.JSONDecodeError:
                pass
            for start_char, end_char in [("{", "}"), ("[", "]")]:
                start_idx = text.find(start_char)
                end_idx = text.rfind(end_char)
                if start_idx != -1 and end_idx > start_idx:
                    try:
                        return json.loads(text[start_idx : end_idx + 1])
                    except json.JSONDecodeError:
                        pass
            return None

        result = _try_parse(raw)
        if result is not None:
            return result
        if "{{" in raw:
            cleaned = raw.replace("{{", "{").replace("}}", "}")
            result = _try_parse(cleaned)
            if result is not None:
                return result

        logger.error(f"JSON Parse Error (after cleanup): {raw[:200]}")
        return default_structure if default_structure is not None else {}
