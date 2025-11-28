from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, Dict, List
import logging

from agent import agent, is_wash_query, WASH_DOMAINS  # from optimized agent.py

# ----------------------------------------------------
# Logging
# ----------------------------------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("WASH-API")

# ----------------------------------------------------
# FastAPI App Setup
# ----------------------------------------------------
app = FastAPI(
    title="WASH Bot API",
    description="AI assistant for Water, Sanitation, and Hygiene queries",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # later restrict to WhatsApp, web app, cloud domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ----------------------------------------------------
# Request and Response Models
# ----------------------------------------------------
class ChatRequest(BaseModel):
    query: str


class ChatResponse(BaseModel):
    answer: str
    sources: List[str]
    is_wash_related: bool
    search_performed: bool


class HealthResponse(BaseModel):
    status: str
    message: str


# ----------------------------------------------------
# Public Endpoints
# ----------------------------------------------------

@app.get("/", response_model=HealthResponse)
async def root():
    """Basic health endpoint for browser checks."""
    return HealthResponse(status="healthy", message="WASH Bot API is running ✔")


@app.get("/health", response_model=HealthResponse)
async def health():
    """Used by cloud platforms like Cloud Run for health checks."""
    return HealthResponse(status="healthy", message="Service operational")


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    Main chatbot endpoint.
    Accepts user query → returns structured AI response.
    """
    try:
        logger.info(f"Received query: {request.query}")
        response = agent(request.query)
        return ChatResponse(**response)

    except Exception as e:
        logger.error(f"Error during response: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@app.get("/check-domain")
async def check_domain(query: str):
    """Utility endpoint to test whether query is WASH related."""
    try:
        result = is_wash_query(query)
        return {"query": query, "is_wash_related": result}
    except Exception as e:
        logger.error(f"Error checking domain relevance: {e}")
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/domains")
async def domains():
    """Returns embedded knowledge base topics."""
    return {"count": len(WASH_DOMAINS), "domains": WASH_DOMAINS}


# ----------------------------------------------------
# WhatsApp Webhook Placeholder (Future Integration)
# ----------------------------------------------------

@app.post("/webhook")
async def whatsapp_webhook(data: Dict):
    """
    Future endpoint for WhatsApp API integration.

    WhatsApp cloud will send JSON → 
    we will extract message → send to agent() → Reply via WhatsApp API.
    """
    logger.info(f"Webhook request received: {data}")

    return {"status": "received", "note": "WhatsApp integration coming soon"}


# ----------------------------------------------------
# Local Development Runner (ignored in cloud)
# ----------------------------------------------------
#if __name__ == "__main__":
    #import uvicorn
    #uvicorn.run("main:app", host="0.0.0.0", port=8080, reload=True)

