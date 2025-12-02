import os
import logging
from functools import lru_cache
from dotenv import load_dotenv
from typing import List, Dict, Optional

from langchain_groq import ChatGroq
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS


from tavily import TavilyClient

# ----------------------------------------------------
# ENVIRONMENT & LOGGING
# ----------------------------------------------------
load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("WASH-Agent")

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
HF_TOKEN = os.getenv("HF_TOKEN")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

# ----------------------------------------------------
# DOMAIN KNOWLEDGE
# ----------------------------------------------------
WASH_DOMAINS = [
    "Water sanitation and hygiene practices are essential to prevent disease outbreaks",
    "Toilet and solid waste management should follow safety and hygiene rules",
    "Handwashing with soap reduces the chances of infection and disease transmission",
    "Safe drinking water and proper sanitation reduce public health risks",
    "Personal hygiene is necessary to maintain overall health and prevent infections",
]

WASH_KEYWORDS = {
    "toilet", "sanitation", " hygiene", "handwash", "sewage", "latrine",
    "menstrual", "waste", "septic", "cleanliness", "water safety"
}

# ----------------------------------------------------
# LAZY LOADING COMPONENTS
# ----------------------------------------------------

@lru_cache
def get_embeddings():
    logger.info("Loading embedding model...")
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"token": HF_TOKEN}
    )

@lru_cache
def get_vector_store():
    logger.info("Building FAISS vector store...")
    docs = [Document(page_content=domain, metadata={"label": "wash"}) for domain in WASH_DOMAINS]
    return FAISS.from_documents(docs, get_embeddings())

@lru_cache
def get_llm():
    logger.info("Initializing Groq model...")
    return ChatGroq(
        model="llama-3.3-70b-versatile",
        api_key=GROQ_API_KEY,
        temperature=0.2,
        max_tokens=500
    )

@lru_cache
def get_search_client():
    return TavilyClient(api_key=TAVILY_API_KEY)

# ----------------------------------------------------
# UTILITIES
# ----------------------------------------------------

def format_response(answer: str, sources: Optional[List[str]] = None, wash_related=True, searched=False) -> Dict:
    return {
        "answer": answer.strip(),
        "sources": sources or [],
        "is_wash_related": wash_related,
        "search_performed": searched
    }


def is_wash_query(query: str, threshold: float = 0.55) -> bool:
    """Check if query belongs to WASH domain via FAISS + keywords."""
    query = query.lower()

    # Keyword check
    if any(keyword in query for keyword in WASH_KEYWORDS):
        logger.info("Keyword matched → Treating as WASH domain.")
        return True

    # Vector similarity
    try:
        results = get_vector_store().similarity_search_with_score(query, k=1)
        _, score = results[0]

        similarity_score = 1 / (1 + score)  # normalize FAISS score
        logger.info(f"Similarity score: {similarity_score:.2f}")

        return similarity_score >= threshold

    except Exception as e:
        logger.warning(f"Similarity check failed: {e}")
        return False


# ----------------------------------------------------
# AGENT RESPONSE LOGIC
# ----------------------------------------------------

def generate_response(query: str) -> Dict:

    is_related = is_wash_query(query)

    if not is_related:
        logger.info("Query is NOT WASH related — using safe response.")
        return format_response(
            answer="I can only assist with topics related to sanitation, hygiene, wastewater, toilets, and cleanliness. Please ask in that domain 😊",
            wash_related=False
        )

    # First attempt: local domain knowledge
    logger.info("Generating local contextual response...")
    llm = get_llm()

    local_prompt = PromptTemplate(
        input_variables=["context", "question"],
        template=(
            "You are a WASH specialist assistant.\n"
            "Provide a clear, helpful, factual answer based ONLY on this information:\n\n"
            "{context}\n\n"
            "Question: {question}\n"
            "If the information is incomplete, respond with what is known but do NOT hallucinate.\n"
        )
    )

    context_docs = get_vector_store().similarity_search(query, k=3)
    context_text = "\n".join([doc.page_content for doc in context_docs])

    local_response = llm.invoke(local_prompt.format(context=context_text, question=query)).content

    # Decide if we need external search
    if len(local_response) >= 120:  
        return format_response(answer=local_response)

    # External Search (for factual or updated detail)
    logger.info("Performing external search...")
    search_client = get_search_client()
    search_results = search_client.search(query, max_results=3)

    search_summary = "\n".join(result["content"] for result in search_results.get("results", []))

    search_prompt = PromptTemplate(
        input_variables=["question", "search_results"],
        template=(
            "Based on the verified search results below, provide an accurate, concise answer.\n\n"
            "Search Results:\n{search_results}\n\n"
            "Question: {question}\n"
        )
    )

    final_answer = llm.invoke(
        search_prompt.format(question=query, search_results=search_summary)
    ).content

    return format_response(final_answer, sources=[r.get("url") for r in search_results.get("results", [])], searched=True)


# ----------------------------------------------------
# EXPORT FUNCTION
# ----------------------------------------------------

def agent(query: str) -> Dict:
    """Main public method the FastAPI app will call."""
    try:
        return generate_response(query)
    except Exception as e:
        logger.error(f"Agent error: {e}")
        return format_response("Something went wrong. Please try again.", wash_related=True)




