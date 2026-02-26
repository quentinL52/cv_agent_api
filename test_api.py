import asyncio
import json
import traceback
from src.services.cv_service import parse_cv

async def main():
    pdf_path = "c:\\Users\\quent\\Documents\\Projets\\devellopement_AIRH - Copie\\CV - Quentin Loumeau - 2026.pdf"
    print(f"Testing CV Parser with file: {pdf_path}")
    
    try:
        # Define minimum metier data for the metier_matching_task
        metiers_reference = {
            "chef_projet_data_ia": {
                "nom": "Chef de Projet Data / IA",
                "competences": ["Python", "SQL", "Gestion de projet", "IA", "Machine Learning"]
            }
        }
        
        result = await parse_cv(pdf_path, "Chef de Projet Data / IA")
        print("Success! Output saved to test_result.json")
        with open("test_result.json", "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
            
    except Exception as e:
        print(f"Error occurred: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main())
