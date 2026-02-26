import json
import os
import sys

# Add src's parent directory to path so we can run this directly if needed
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv

load_dotenv()

def get_job_text(job):
    """
    Formats job info using the "Miroir" strategy.
    """
    missions = ", ".join(job.get("missions_principales", []))
    tech = ", ".join(job.get("competences_techniques", []))
    outils = ", ".join(job.get("outils_technologies", []))
    soft = ", ".join(job.get("competences_soft", []))
    
    text = f"MISSIONS: {missions}\n"
    text += f"TECH_ET_OUTILS: {tech}, {outils}\n"
    text += f"SOFT_SKILLS: {soft}"
    return text

def embed_metiers_file():
    base_path = os.path.dirname(os.path.dirname(__file__))
    metiers_path = os.path.join(base_path, "data", "metiers.json")
    
    print(f"Loading {metiers_path}...")
    with open(metiers_path, "r", encoding="utf-8") as f:
        data = json.load(f)
        
    embeddings_model = OpenAIEmbeddings(model="text-embedding-3-small")
    
    count = 0
    def process_jobs(job_list):
        nonlocal count
        for job in job_list:
            if "metiers" in job:
                process_jobs(job["metiers"])
            elif "id" in job:
                print(f"Embedding {job.get('id')}...")
                text = get_job_text(job)
                emb = embeddings_model.embed_query(text)
                job["embedding"] = emb
                count += 1

    process_jobs(data.get("metiers", []))
    
    print(f"Writing {count} embeddings to {metiers_path}...")
    with open(metiers_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
        
    print("Done!")

if __name__ == "__main__":
    embed_metiers_file()
