import math
import logging
from typing import Dict, List
from langchain_openai import OpenAIEmbeddings

logger = logging.getLogger(__name__)

def compute_cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
    """Computes cosine similarity between two vectors without relying on numpy."""
    numerator = sum(a * b for a, b in zip(vec1, vec2))
    norm1 = math.sqrt(sum(a * a for a in vec1))
    norm2 = math.sqrt(sum(b * b for b in vec2))
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return numerator / (norm1 * norm2)

def get_top_k_metiers(
    metiers_data: List[Dict],
    experiences_summary: str,
    projects_summary: str,
    hard_skills: str,
    soft_skills: str,
    k: int = 3
) -> List[Dict]:
    """
    Filters the job profiles (métiers) by cosine similarity to the candidate's profile.
    Returns the top K job profiles.
    """
    # flatten list to extract nested metiers from the JSON dataset
    flat_list = []
    def _flatten(job_list):
        for job in job_list:
            if "metiers" in job:
                _flatten(job["metiers"])
            elif "id" in job:
                flat_list.append(job)
    
    _flatten(metiers_data)

    try:
        if not flat_list:
            return []
            
        embeddings_model = OpenAIEmbeddings(model="text-embedding-3-small")
        
        # Stratégie Miroir
        text = f"EXPERIENCES_ET_ACTIONS: {experiences_summary}\n"
        text += f"PROJETS_ET_OUTCOMES: {projects_summary}\n"
        text += f"COMPETENCES_TECH_ET_SOFT: {hard_skills}, {soft_skills}"
        
        candidat_emb = embeddings_model.embed_query(text)
        
        scored_metiers = []
        for job in flat_list:
            # En cas de manque d'embedding, on met un score de 0
            if "embedding" in job and job["embedding"]:
                sim = compute_cosine_similarity(candidat_emb, job["embedding"])
            else:
                sim = 0.0
            scored_metiers.append((sim, job))
                    
        # Sort by similarity in descending order
        scored_metiers.sort(key=lambda x: x[0], reverse=True)
        top_k = [item[1] for item in scored_metiers[:k]]
        
        logger.info(f"Top {k} métiers sélectionnés par embedding: {[m.get('id') for m in top_k]}")
        return top_k
        
    except Exception as e:
        logger.error(f"Erreur lors du pré-filtrage des métiers: {e}", exc_info=True)
        # En cas d'erreur de clé d'API ou autre, on renvoie une liste par défaut (fallback)
        return flat_list[:k]
