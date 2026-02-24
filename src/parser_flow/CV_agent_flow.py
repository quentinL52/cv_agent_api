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

# Configuration du logger pour capturer la verbosité dans un fichier
verbose_logger = logging.getLogger("crewai_verbose")
verbose_logger.setLevel(logging.INFO)

# Création du fichier de log (écrase le précédent à chaque run avec 'w')
file_handler = logging.FileHandler("agents_trace.log", mode='w', encoding='utf-8')
formatter = logging.Formatter('%(asctime)s - %(message)s')
file_handler.setFormatter(formatter)
verbose_logger.addHandler(file_handler)

class CVAgentOrchestrator:
    """Orchestrateur multi-agents pour le parsing et l'analyse de CV."""

    def __init__(self):
        self.llm = get_small_llm()
        self.big_llm = get_big_llm()
        self.agents_config = self._load_yaml("agents.yaml")
        self.tasks_config = self._load_yaml("tasks.yaml")
        self.metiers_data = self._load_metiers()
        self.skill_domain_map = self._load_skill_domain_map()
        self._create_agents()

    # ──────────────────────────────────────────────
    # Chargement des configurations
    # ──────────────────────────────────────────────

    def _load_yaml(self, filename: str) -> Dict:
        base_path = os.path.dirname(os.path.dirname(__file__))
        config_path = os.path.join(base_path, "config", filename)
        with open(config_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)

    def _load_metiers(self) -> List[Dict]:
        """Charge le référentiel de métiers (sans les embeddings pour économiser la mémoire)."""
        base_path = os.path.dirname(os.path.dirname(__file__))
        metiers_path = os.path.join(base_path, "data", "metiers.json")
        with open(metiers_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        metiers = []
        for m in data.get("metiers", []):
            clean = {k: v for k, v in m.items() if k != "embedding"}
            metiers.append(clean)
        return metiers

    def _load_skill_domain_map(self) -> Dict[str, List[str]]:
        """Charge le mapping compétences -> domaines."""
        base_path = os.path.dirname(os.path.dirname(__file__))
        map_path = os.path.join(base_path, "config", "skill_domain_map.json")
        with open(map_path, "r", encoding="utf-8") as f:
            return json.load(f)

    # ──────────────────────────────────────────────
    # Création des agents
    # ──────────────────────────────────────────────



    def _create_agents(self):
        def make_agent(name, llm_override=None):
            return Agent(
                config=self.agents_config[name],
                llm=llm_override or self.llm,
                allow_delegation=False,
                verbose=True,
                max_iter=1,
                respect_context_window=True,
                # logs callbackagent
                step_callback=lambda step: verbose_logger.info(f"Agent {name} Step: {step}"),
            )

        # Phase 2 : Agents d'extraction (existants)
        self.cv_splitter = make_agent("cv_splitter", llm_override=self.big_llm)
        self.skills_extractor = make_agent("skills_extractor")
        self.experience_extractor = make_agent("experience_extractor")
        self.project_extractor = make_agent("project_extractor")
        self.education_extractor = make_agent("education_extractor")
        self.reconversion_detector = make_agent("reconversion_detector")
        self.language_extractor = make_agent("language_extractor")
        self.etudiant_detector = make_agent("etudiant_detector")
        self.identity_extractor = make_agent("identity_extractor")

        # Phase 3 : Agents d'analyse et recommandation (nouveaux)
        self.header_analyzer = make_agent("header_analyzer", llm_override=self.big_llm)
        self.metier_matcher = make_agent("metier_matcher", llm_override=self.big_llm)
        self.cv_quality_checker = make_agent("cv_quality_checker")
        self.project_analyzer = make_agent("project_analyzer")

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
    # PHASE 2 : Extraction parallèle (8 agents)
    # ──────────────────────────────────────────────

    async def extract_all_sections(
        self, sections: Dict[str, str], cv_raw_start: str = "", file_name: str = ""
    ) -> Dict[str, Any]:
        """Exécute les 8 tâches d'extraction en parallèle."""

        def create_task_async(task_key, agent, **kwargs):
            t_config = self.tasks_config[task_key].copy()
            t_description = t_config["description"]
            
            # Éviter les erreurs de formattage si des clés manquent ou sont mal échappées (ex: accolades dans le texte du CV)
            try:
                # Utiliser format_map pour plus de flexibilité si besoin, mais format() est standard
                t_config["description"] = t_description.format(**kwargs)
            except KeyError as e:
                logger.warning(f"KeyError formatting task '{task_key}': {e}. Falling back to manual replace.")
                # Fallback manuel sécurisé pour les clés présentes
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
            (
                "skills_task",
                self.skills_extractor,
                {
                    "experiences": sections.get("experiences", ""),
                    "projects": sections.get("projects", ""),
                    "skills": sections.get("skills", ""),
                    "education": sections.get("education", ""),
                },
            ),
            (
                "experience_task",
                self.experience_extractor,
                {"experiences": sections.get("experiences", "")},
            ),
            (
                "project_task",
                self.project_extractor,
                {"projects": sections.get("projects", "")},
            ),
            (
                "education_task",
                self.education_extractor,
                {"education": sections.get("education", "")},
            ),
            (
                "reconversion_task",
                self.reconversion_detector,
                {
                    "experiences": sections.get("experiences", ""),
                    "education": sections.get("education", ""),
                },
            ),
            (
                "language_task",
                self.language_extractor,
                {
                    "languages": sections.get("languages", ""),
                    "cv_raw_start": cv_raw_start[:500],
                },
            ),
            (
                "etudiant_task",
                self.etudiant_detector,
                {
                    "education": sections.get("education", ""),
                    "current_date": datetime.now().strftime("%Y-%m-%d"),
                },
            ),
            (
                "identity_task",
                self.identity_extractor,
                {
                    "header": sections.get("header", ""),
                    "cv_raw_start": cv_raw_start[:1500],
                    "file_name": file_name,
                },
            ),
        ]

        task_coroutines = [
            create_task_async(key, agent, **kwargs) for key, agent, kwargs in tasks_def
        ]
        keys = [t[0] for t in task_coroutines]
        coroutines = [t[1] for t in task_coroutines]
        results_list = await asyncio.gather(*coroutines, return_exceptions=True)

        results_map = {}
        for key, result in zip(keys, results_list):
            if isinstance(result, Exception):
                logger.error(f"Task '{key}' failed: {result}")
            else:
                results_map[key] = result

        return self._aggregate_extraction_results(results_map)

    # ──────────────────────────────────────────────
    # PHASE 3a : Analyse d'en-tête (indépendante, tourne en // avec Phase 2)
    # ──────────────────────────────────────────────

    async def run_header_analysis(
        self,
        sections: Dict[str, str],
        cv_raw_start: str = "",
        cv_full_text: str = "",
    ) -> Dict:
        """Extrait le poste visé depuis l'en-tête du CV.

        Ne dépend que de Phase 1 (sections) → peut tourner en PARALLÈLE avec Phase 2.
        """
        header_section = sections.get("header", "")
        raw_for_header = cv_raw_start[:2000] if cv_raw_start else cv_full_text[:2000]
        safe_cv_raw = raw_for_header.replace("{", "{{").replace("}", "}}")
        safe_header = header_section.replace("{", "{{").replace("}", "}}")

        header_data: Dict = {
            "poste_vise": "Non identifié",
            "niveau_seniorite": "non précisé",
            "confiance": 0,
        }

        try:
            t_config = self.tasks_config["poste_visé_task"].copy()
            t_config["description"] = t_config["description"].format(
                cv_raw_start=safe_cv_raw,
                header=safe_header,
            )
            task = Task(config=t_config, agent=self.header_analyzer)
            crew = Crew(agents=[self.header_analyzer], tasks=[task], verbose=False)
            header_result = await crew.kickoff_async()

            if header_result:
                header_data = self._parse_json_output(
                    header_result,
                    {"poste_vise": "Non identifié", "niveau_seniorite": "non précisé", "confiance": 0},
                )
                logger.info(
                    f"Header analyzer : poste_vise='{header_data.get('poste_vise')}', "
                    f"confiance={header_data.get('confiance')}"
                )
        except Exception as e:
            logger.error(f"Header analyzer failed: {e}", exc_info=True)

        # Fallback programmatique si le LLM n'a pas trouvé le poste
        if header_data.get("poste_vise", "Non identifié") == "Non identifié":
            logger.warning("Header analyzer 'Non identifié' → fallback programmatique...")
            fallback = self._fallback_extract_poste_vise(cv_full_text, header_section)
            if fallback:
                header_data["poste_vise"] = fallback
                header_data["source_detection"] = "fallback_programmatique"
                header_data["confiance"] = 70
                logger.info(f"Fallback found poste_vise: '{fallback}'")

        return header_data

    # ──────────────────────────────────────────────
    # PHASE 3b : Analyse & Recommandation (3 agents parallèles)
    # ──────────────────────────────────────────────

    async def analyze_and_recommend(
        self,
        cv_full_text: str,
        sections: Dict[str, str],
        extraction: Dict[str, Any],
        cv_raw_start: str = "",
        header_data: Dict = None,
    ) -> Dict[str, Any]:
        """Exécute les 3 tâches d'analyse en parallèle.

        header_data est pré-calculé par run_header_analysis (en // avec Phase 2).
        """
        if header_data is None:
            logger.warning("analyze_and_recommend sans header_data — valeurs par défaut utilisées.")
            header_data = {"poste_vise": "Non identifié", "niveau_seniorite": "non précisé", "confiance": 0}

        candidat = extraction.get("candidat", {})
        competences = candidat.get("compétences", {})
        hard_skills = competences.get("hard_skills", [])
        soft_skills = competences.get("soft_skills", [])
        skills_with_context = competences.get("skills_with_context", [])
        reconversion = candidat.get("reconversion", {})

        skill_domains = self._map_skills_to_domains(hard_skills)
        methodologies = self._extract_methodologies(hard_skills, skill_domains)

        experiences_summary = json.dumps(
            candidat.get("expériences", []), ensure_ascii=False
        )[:3000]
        projets = candidat.get("projets", {})
        professional_projects = json.dumps(
            projets.get("professional", []), ensure_ascii=False
        )[:2000]
        personal_projects = json.dumps(
            projets.get("personal", []), ensure_ascii=False
        )[:2000]
        projects_summary = f"Pro: {professional_projects}\nPerso: {personal_projects}"
        reconversion_data = json.dumps(reconversion, ensure_ascii=False) if reconversion else "{}"

        metiers_reference = self._prepare_metiers_for_prompt()

        poste_vise = header_data.get("poste_vise", "Non identifié")
        niveau_seniorite = header_data.get("niveau_seniorite", "non précisé")
        metier_reference_detail = self._get_metier_reference_for_poste(poste_vise)

        raw_for_header = cv_raw_start[:2000] if cv_raw_start else cv_full_text[:2000]
        safe_cv_raw = raw_for_header.replace("{", "{{").replace("}", "}}")

        def create_task_async(task_key, agent, **kwargs):
            t_config = self.tasks_config[task_key].copy()
            t_config["description"] = t_config["description"].format(**kwargs)
            task = Task(config=t_config, agent=agent)
            c = Crew(agents=[agent], tasks=[task], verbose=False)
            return (task_key, c.kickoff_async())

        # 3 agents en parallèle (quality + metier matching + project analysis)
        parallel_tasks = [
            (
                "cv_quality_task",
                self.cv_quality_checker,
                {
                    "cv_full_text": cv_full_text[:6000],
                    "cv_raw_start": safe_cv_raw,
                    "skills_with_context": json.dumps(skills_with_context, ensure_ascii=False)[:2000],
                    "experiences_summary": experiences_summary,
                    "projects_summary": projects_summary[:2000],
                    "niveau_seniorite": niveau_seniorite,
                    "reconversion_data": reconversion_data,
                },
            ),
            (
                "metier_matching_task",
                self.metier_matcher,
                {
                    "poste_vise": poste_vise,
                    "hard_skills": json.dumps(hard_skills, ensure_ascii=False),
                    "soft_skills": json.dumps(soft_skills, ensure_ascii=False),
                    "skill_domains": json.dumps(skill_domains, ensure_ascii=False),
                    "methodologies": json.dumps(methodologies, ensure_ascii=False),
                    "experiences_summary": experiences_summary,
                    "projects_summary": projects_summary[:2000],
                    "reconversion_data": reconversion_data,
                    "metiers_reference": metiers_reference,
                },
            ),
            (
                "project_analysis_task",
                self.project_analyzer,
                {
                    "poste_vise": poste_vise,
                    "metier_reference_detail": metier_reference_detail,
                    "professional_projects": professional_projects,
                    "personal_projects": personal_projects,
                    "reconversion_data": reconversion_data,
                },
            ),
        ]

        task_coroutines = [
            create_task_async(key, agent, **kwargs) for key, agent, kwargs in parallel_tasks
        ]
        keys = [t[0] for t in task_coroutines]
        coroutines = [t[1] for t in task_coroutines]
        results_list = await asyncio.gather(*coroutines, return_exceptions=True)

        analysis_results = {}
        for key, result in zip(keys, results_list):
            if isinstance(result, Exception):
                logger.error(f"Analysis task '{key}' failed: {result}")
            else:
                analysis_results[key] = result

        recommendations = self._aggregate_recommendations(analysis_results, header_data)

        # ── Filtre dur : ne garder que les projets issus de la section projets ──
        extracted_titles: set[str] = set()
        for p in projets.get("professional", []):
            if isinstance(p, dict) and p.get("title"):
                extracted_titles.add(p["title"].strip().lower())
        for p in projets.get("personal", []):
            if isinstance(p, dict) and p.get("title"):
                extracted_titles.add(p["title"].strip().lower())

        if extracted_titles:
            def _is_extracted_project(titre: str) -> bool:
                t = titre.strip().lower()
                if t in extracted_titles:
                    return True
                return any(t in ref or ref in t for ref in extracted_titles)

            recommendations["analyse_projets"] = [
                p for p in recommendations.get("analyse_projets", [])
                if isinstance(p, dict) and _is_extracted_project(p.get("titre", ""))
            ]
            logger.info(
                f"Filtre projets : {len(recommendations['analyse_projets'])} projets conservés "
                f"sur {len(extracted_titles)} extraits."
            )

        return recommendations

    # ──────────────────────────────────────────────
    # Mapping compétences -> domaines
    # ──────────────────────────────────────────────

    def _map_skills_to_domains(self, hard_skills: List[str]) -> Dict[str, List[str]]:
        """Mappe les compétences du candidat à leurs domaines métier."""
        result = {}
        for skill in hard_skills:
            skill_lower = skill.lower().strip()
            for domain, domain_skills in self.skill_domain_map.items():
                if skill_lower in domain_skills:
                    if domain not in result:
                        result[domain] = []
                    result[domain].append(skill)
                    break
        return result

    def _prepare_metiers_for_prompt(self) -> str:
        """Prépare le référentiel métiers COMPLET (30 métiers) pour le prompt."""
        lines = []
        for m in self.metiers_data:
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

    def _get_metier_reference_for_poste(self, poste_vise: str) -> str:
        """Trouve les métiers les plus proches du poste visé pour contextualiser l'analyse de projets."""
        if not poste_vise or poste_vise == "Non identifié":
            return "Aucun métier de référence spécifique. Analyser les projets selon leur qualité intrinsèque."

        poste_lower = poste_vise.lower()
        scored = []

        for m in self.metiers_data:
            nom_lower = m.get("nom", "").lower()
            id_lower = m.get("id", "").lower()
            desc_lower = m.get("description", "").lower()
            score = 0

            keywords = [w for w in poste_lower.replace("/", " ").replace("-", " ").split() if len(w) > 2]
            for kw in keywords:
                if kw in nom_lower:
                    score += 3
                if kw in id_lower:
                    score += 2
                if kw in desc_lower:
                    score += 1

            nom_keywords = [w for w in nom_lower.replace("/", " ").replace("-", " ").split() if len(w) > 2]
            for kw in nom_keywords:
                if kw in poste_lower:
                    score += 3

            if score > 0:
                scored.append((score, m))

        scored.sort(key=lambda x: -x[0])

        if not scored:
            return "Poste visé non trouvé dans le référentiel. Analyser les projets selon leur qualité intrinsèque."

        lines = ["Métier(s) de référence les plus proches du poste visé :"]
        for _, m in scored[:3]:
            mid = m.get("id")
            nom = m.get("nom")
            comp = m.get("competences_techniques", [])
            outils = m.get("outils_technologies", [])
            missions = m.get("missions_principales", [])
            lines.append(
                f"\n[{mid}] {nom}\n"
                f"  Compétences attendues: {', '.join(comp)}\n"
                f"  Outils attendus: {', '.join(outils)}\n"
                f"  Missions principales: {'; '.join(missions[:3])}"
            )
        return "\n".join(lines)

    def _extract_methodologies(self, hard_skills: List[str], skill_domains: Dict[str, List[str]]) -> List[str]:
        """Extrait les méthodologies de travail du candidat."""
        methodology_keywords = {
            "agile", "scrum", "kanban", "devops", "ci/cd", "cicd", "tdd", "bdd",
            "design thinking", "lean", "safe", "xp", "pair programming",
            "code review", "sprint", "product owner", "scrum master",
            "rgpd", "rgaa",
        }

        methodologies = []
        for skill in hard_skills:
            if skill.lower().strip() in methodology_keywords:
                methodologies.append(skill)

        if "gestion_projet" in skill_domains:
            for skill in skill_domains["gestion_projet"]:
                if skill not in methodologies:
                    methodologies.append(skill)

        if "devops" in skill_domains:
            for skill in skill_domains["devops"]:
                s = skill.lower()
                if any(kw in s for kw in ["ci", "cd", "github actions", "gitlab ci"]):
                    if skill not in methodologies:
                        methodologies.append(skill)

        return methodologies

    # ──────────────────────────────────────────────
    # Agrégation des résultats d'extraction (Phase 2)
    # ──────────────────────────────────────────────

    def _aggregate_extraction_results(self, results_map: Dict[str, Any]) -> Dict[str, Any]:
        """Agrège les résultats d'extraction (identique au module existant)."""

        def get_parsed(key, default=None):
            if key not in results_map:
                return default
            return self._parse_json_output(results_map[key], default)

        competences = get_parsed("skills_task", {"hard_skills": [], "soft_skills": []})
        experiences = get_parsed("experience_task", [])
        projets = get_parsed("project_task", {"professional": [], "personal": []})
        formations = get_parsed("education_task", [])
        reconversion = get_parsed("reconversion_task", {}).get(
            "reconversion_analysis", {}
        )
        etudiant_data = get_parsed("etudiant_task", {}).get("etudiant_analysis", {})
        latest_end_date = etudiant_data.get("latest_education_end_date")
        if latest_end_date:
            is_student_by_date = self._is_still_student(latest_end_date)
            etudiant_data["is_etudiant"] = is_student_by_date

        langues_raw = get_parsed("language_task", {})

        if isinstance(competences, dict):
            raw_skills = competences.get("hard_skills", [])
            seen = set()
            unique_skills = []
            for skill in raw_skills:
                key = (
                    str(skill).lower()
                    if not isinstance(skill, str)
                    else skill.lower()
                )
                if key not in seen:
                    seen.add(key)
                    unique_skills.append(skill)
            competences["hard_skills"] = unique_skills

        identity = get_parsed("identity_task", {})

        return {
            "candidat": {
                "first_name": (
                    identity.get("first_name")
                    if isinstance(identity, dict)
                    else None
                ),
                "compétences": competences,
                "expériences": experiences,
                "reconversion": reconversion,
                "projets": projets,
                "formations": formations,
                "etudiant": etudiant_data,
                "langues": (
                    langues_raw.get("langues", [])
                    if isinstance(langues_raw, dict)
                    else []
                ),
            }
        }

    # ──────────────────────────────────────────────
    # Agrégation des recommandations (Phase 3)
    # ──────────────────────────────────────────────

    def _aggregate_recommendations(
        self,
        analysis_results: Dict[str, Any],
        header_data: Dict,
    ) -> Dict[str, Any]:
        """Agrège les résultats d'analyse en un objet recommandations structuré."""

        def get_parsed(key, default=None):
            if key not in analysis_results:
                return default
            return self._parse_json_output(analysis_results[key], default)

        metier_data = get_parsed("metier_matching_task", {"postes_recommandes": []})
        quality_data = get_parsed(
            "cv_quality_task",
            {"score_global": 0, "red_flags": [], "conseils_prioritaires": []},
        )
        project_data = get_parsed("project_analysis_task", {"analyse_projets": []})

        # Conseils d'amélioration : uniquement les conseils qualité CV
        conseils = []
        if isinstance(quality_data, dict):
            conseils.extend(quality_data.get("conseils_prioritaires", []))

        return {
            "header_analysis": header_data,
            "postes_recommandes": (
                metier_data.get("postes_recommandes", [])
                if isinstance(metier_data, dict)
                else []
            ),
            "analyse_poste_vise": (
                metier_data.get("analyse_poste_vise", "")
                if isinstance(metier_data, dict)
                else ""
            ),
            "qualite_cv": quality_data,
            "analyse_projets": (
                project_data.get("analyse_projets", [])
                if isinstance(project_data, dict)
                else []
            ),
            "coherence_globale_projets": (
                project_data.get("coherence_globale", {})
                if isinstance(project_data, dict)
                else {}
            ),
            "conseils_amelioration": conseils,
        }

    # ──────────────────────────────────────────────
    # Utilitaires
    # ──────────────────────────────────────────────

    def _fallback_extract_poste_vise(
        self, cv_full_text: str, header_section: str
    ) -> str:
        """Extraction programmatique du poste visé en fallback.

        Cherche la ligne de titre dans l'en-tête du CV en filtrant les lignes
        qui ne sont clairement PAS un titre de poste (email, téléphone, liens,
        titres de section, compétences techniques).
        """
        import re

        # Patterns qui NE sont PAS un titre de poste
        skip_patterns = [
            r"^#{1,6}\s",                          # Titres markdown
            r"@",                                   # Email
            r"^\+?\d[\d\s\-\.]{7,}",              # Téléphone
            r"^http|^www\.|linkedin|github",        # URLs/liens
            r"^\*{1,3}[A-Z]",                      # Bold section headers
            r"^(CONTACT|LIENS|STACK|LANGUES|CENTRES|EXPERIENCE|FORMATION|PROJET|COMPÉTENCES|EDUCATION)",  # Section headings
            r"^(Python|SQL|JavaScript|React|FastAPI|Docker|AWS|Git|CI)",  # Skills
            r"^(Ile-de-France|Paris|Lyon|Marseille|France)",  # Locations
            r"^\d{2}\s?\d{2}\s?\d{2}",            # Phone numbers
            r"^(Français|Anglais|Portugais|Espagnol)",  # Languages
            r"^(Langages|Frameworks|Analytics|DevOps|Méthodologies|IA &|BI :)",  # Skill categories
            r"^(Blockchain|Jeux de rôle|Randonnée)",  # Interests
            r"^\s*$",                               # Empty lines
            r"^[\*\-\|]",                           # List items and table separators
        ]

        # Mots-clés qui INDIQUENT un titre de poste
        title_indicators = [
            "développeur", "developer", "ingénieur", "engineer", "chef de projet",
            "data analyst", "data scientist", "data engineer", "consultant",
            "architecte", "manager", "lead", "senior", "junior", "fullstack",
            "full-stack", "full stack", "backend", "frontend", "devops",
            "product", "project", "spécialiste", "expert", "analyste",
            "mlops", "ai", "ia", "machine learning", "nlp", "deep learning",
        ]

        def _has_title_indicator(text_lower: str) -> bool:
            for indicator in title_indicators:
                if len(indicator) <= 3:
                    if re.search(r"\b" + re.escape(indicator) + r"\b", text_lower):
                        return True
                else:
                    if indicator in text_lower:
                        return True
            return False

        def _is_likely_title(line: str) -> bool:
            stripped = line.strip().strip("#*_ ")
            if len(line.split()) > 10:
                return False
            for pattern in skip_patterns:
                if re.match(pattern, stripped, re.IGNORECASE):
                    return False
            return _has_title_indicator(stripped.lower())

        # Chercher dans toutes les sources, par ordre de priorité
        sources = [
            ("header", header_section),
            ("cv_text", cv_full_text[:3000]),
        ]

        for source_name, text in sources:
            if not text:
                continue
            lines = text.split("\n")
            for line in lines:
                if _is_likely_title(line):
                    clean = line.strip().strip("#*_ ")
                    logger.info(f"Fallback: found title in {source_name}: '{clean}'")
                    return clean

        return ""

    def _is_still_student(self, date_str: str) -> bool:
        """Détermine si le candidat est encore étudiant à partir de la date de fin d'études."""
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
