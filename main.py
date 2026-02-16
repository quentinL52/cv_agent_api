import os
import logging
import tempfile
import uuid
from langtrace_python_sdk import inject_additional_attributes

from fastapi import FastAPI, UploadFile, File, HTTPException

from fastapi.middleware.cors import CORSMiddleware

from src.services.cv_service import parse_cv
from langtrace_python_sdk import langtrace


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Optional: Langtrace initialization
LANGTRACE_API_KEY = os.getenv("LANGTRACE_API_KEY")
if LANGTRACE_API_KEY:
    langtrace.init(api_key=LANGTRACE_API_KEY)
else:
    logger.warning("LANGTRACE_API_KEY not found. Langtrace tracing is disabled.")


app = FastAPI(
    title="CV Parser API",
    description="parsing de CV agentique",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

ALLOWED_ORIGINS = os.getenv("CORS_ORIGINS", "http://localhost:8000").split(",")

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

from pydantic import BaseModel

class HealthCheck(BaseModel):
    status: str = "ok"

@app.get("/", response_model=HealthCheck, tags=["Status"])
async def health_check():
    return HealthCheck()

@app.post("/parse-cv/", tags=["CV Parsing"])
async def parse_cv_endpoint(
    file: UploadFile = File(...)
):
    """
    Parses a CV file (PDF) and returns the parsed data.
    """
    if file.content_type != "application/pdf":
        raise HTTPException(status_code=400, detail="PDF file required")

    contents = await file.read()


    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(contents)
        tmp_path = tmp.name

    try:
        session_id = str(uuid.uuid4())
        attributes = {
            "session.id": session_id,
            "user_id": session_id
        }

        async def _traced_parse():
            return await parse_cv(tmp_path)

        result = await inject_additional_attributes(
            _traced_parse,
            attributes
        )
    except Exception as e:
        logger.error(f"Error processing CV: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {str(e)}")
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

    if not result:
        raise HTTPException(status_code=500, detail="Failed to extract data from CV.")

    return result

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8001)) 
    uvicorn.run(app, host="0.0.0.0", port=port)