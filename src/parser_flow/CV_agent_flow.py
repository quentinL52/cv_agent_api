import json
import logging
import os
import yaml
import asyncio
from datetime import datetime
from typing import Dict, Any
from crewai import Agent, Task, Crew, Process
from src.config.app_config import get_small_llm, get_big_llm

logger = logging.getLogger(__name__)


class CVAgentOrchestrator:
    
    def __init__(self):
        self.llm = get_small_llm()
        self.big_llm = get_big_llm()
        self.agents_config = self._load_yaml("agents.yaml")
        self.tasks_config = self._load_yaml("tasks.yaml")
        self._create_agents()

    def _load_yaml(self, filename: str) -> Dict:
        base_path = os.path.dirname(os.path.dirname(__file__)) 
        config_path = os.path.join(base_path, "config", filename)
        with open(config_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)

    def _create_agents(self):
        def make_agent(name, llm_override=None):
            return Agent(
                config=self.agents_config[name],
                llm=llm_override or self.llm,
                allow_delegation=False,
                verbose=False,
                max_iter=1,
                respect_context_window=True
            )

        self.cv_splitter = make_agent('cv_splitter', llm_override=self.big_llm)
        self.skills_extractor = make_agent('skills_extractor')
        self.experience_extractor = make_agent('experience_extractor')
        self.project_extractor = make_agent('project_extractor')
        self.education_extractor = make_agent('education_extractor')
        self.reconversion_detector = make_agent('reconversion_detector')
        self.language_extractor = make_agent('language_extractor')
        self.etudiant_detector = make_agent('etudiant_detector')
        self.identity_extractor = make_agent('identity_extractor')

    async def split_cv_sections(self, cv_content: str) -> Dict[str, str]:
        """
        decoupage du cv en sections
        """
        task_config = self.tasks_config['split_cv_task'].copy()
        task_config['description'] = task_config['description'].format(cv_content=cv_content[:20000])
        
        task = Task(
            config=task_config,
            agent=self.cv_splitter
        )
        crew = Crew(
            agents=[self.cv_splitter],
            tasks=[task],
            process=Process.sequential,
            verbose=False
        )
        result = await crew.kickoff_async()
        parsed = self._parse_json_output(result, default_structure={})        
        return parsed

    async def extract_all_sections(self, sections: Dict[str, str]) -> Dict[str, Any]:
        """
        execution des taches en parraléle.
        """
        def create_task_async(task_key, agent, **kwargs):
            t_config = self.tasks_config[task_key].copy()
            t_config['description'] = t_config['description'].format(**kwargs)
            task = Task(config=t_config, agent=agent)
            c = Crew(agents=[agent], tasks=[task], verbose=False)
            return (task_key, c.kickoff_async())

        tasks_def = [
            ('skills_task', self.skills_extractor, {
                'experiences': sections.get('experiences', ''),
                'projects': sections.get('projects', ''),
                'skills': sections.get('skills', ''),
                'education': sections.get('education', '')
            }),
            ('experience_task', self.experience_extractor, {'experiences': sections.get('experiences', '')}),
            ('project_task', self.project_extractor, {'projects': sections.get('projects', '')}),
            ('education_task', self.education_extractor, {'education': sections.get('education', '')}),
            ('reconversion_task', self.reconversion_detector, {
                'experiences': sections.get('experiences', ''),
                'education': sections.get('education', '')
            }),
            ('language_task', self.language_extractor, {
                'languages': sections.get('languages', '')
            }),
            ('etudiant_task', self.etudiant_detector, {
                'education': sections.get('education', ''),
                'current_date': datetime.now().strftime("%Y-%m-%d")
            }),
            ('identity_task', self.identity_extractor, {
                'personal_info': sections.get('personal_info', '')
            })
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

        return self._aggregate_results(results_map)

    def _aggregate_results(self, results_map: Dict[str, Any]) -> Dict[str, Any]:

        def get_parsed(key, default=None):
            if key not in results_map:
                return default
            return self._parse_json_output(results_map[key], default)

        competences = get_parsed('skills_task', {"hard_skills": [], "soft_skills": []})
        experiences = get_parsed('experience_task', [])
        projets = get_parsed('project_task', {"professional": [], "personal": []})
        formations = get_parsed('education_task', [])
        reconversion = get_parsed('reconversion_task', {}).get("reconversion_analysis", {})
        etudiant_data = get_parsed('etudiant_task', {}).get("etudiant_analysis", {})
        latest_end_date = etudiant_data.get("latest_education_end_date")
        if latest_end_date:
            is_student_by_date = self._is_still_student(latest_end_date)
            etudiant_data["is_etudiant"] = is_student_by_date

        langues_raw = get_parsed('language_task', {})

        if isinstance(competences, dict):
            # Deduplicate hard_skills while preserving order
            raw_skills = competences.get("hard_skills", [])
            seen = set()
            unique_skills = []
            for skill in raw_skills:
                key = str(skill).lower() if not isinstance(skill, str) else skill.lower()
                if key not in seen:
                    seen.add(key)
                    unique_skills.append(skill)
            competences["hard_skills"] = unique_skills

        identity = get_parsed('identity_task', {})

        return {
            "candidat": {
                "first_name": identity.get("first_name") if isinstance(identity, dict) else None,
                "compétences": competences,
                "expériences": experiences,
                "reconversion": reconversion,
                "projets": projets,
                "formations": formations,
                "etudiant": etudiant_data,
                "langues": langues_raw.get("langues", []) if isinstance(langues_raw, dict) else [],
            }
        }

    def _is_still_student(self, date_str: str) -> bool:
        if not date_str:
            return False
        date_str = str(date_str).lower().strip()
        ongoing_keywords = ["present", "présent", "current", "cours", "aujourd'hui", "now"]
        if any(keyword in date_str for keyword in ongoing_keywords):
            return True
            
        try:
            now = datetime.now()
            end_date = None
            if len(date_str) == 10 and date_str[4] == '-' and date_str[7] == '-':
                 end_date = datetime.strptime(date_str, "%Y-%m-%d")
            elif len(date_str) == 7 and date_str[4] == '-':
                 end_date = datetime.strptime(date_str, "%Y-%m")
            elif '/' in date_str:
                parts = date_str.split('/')
                if len(parts) == 2:
                    m, y = parts
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
        raw = crew_output.raw if hasattr(crew_output, 'raw') else str(crew_output)

        if '```json' in raw:
            raw = raw.split('```json')[1].split('```')[0].strip()
        elif '```' in raw:
            parts = raw.split('```')
            if len(parts) >= 3:
                raw = parts[1].strip()

        # Clean common LLM artifacts
        raw = raw.strip().lstrip('\ufeff')  # BOM

        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            # Try to find the first JSON object or array in the output
            for start_char, end_char in [('{', '}'), ('[', ']')]:
                start_idx = raw.find(start_char)
                end_idx = raw.rfind(end_char)
                if start_idx != -1 and end_idx > start_idx:
                    try:
                        return json.loads(raw[start_idx:end_idx + 1])
                    except json.JSONDecodeError:
                        continue

            logger.error(f"JSON Parse Error (after cleanup): {raw[:200]}")
            return default_structure if default_structure is not None else {}