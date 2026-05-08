# ai_service/main.py
"""
MindfulMeals AI Microservice
━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Why this exists as Python (not just Node.js):
  ✅ sentence-transformers runs locally — FREE, no API calls for embeddings
  ✅ Vector similarity search with numpy (no paid vector DB needed)
  ✅ Future: custom mood classification model, fine-tuned Llama, LangChain chains
  ✅ Python's AI ecosystem is simply much richer than Node.js

Runs on: http://localhost:8000
Called by: Node.js API via internal HTTP
"""

from fastapi import FastAPI, Depends, HTTPException, Header
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import os
from dotenv import load_dotenv

from routers import mood, meals, restaurants, insights
from services.embeddings import EmbeddingService

load_dotenv()

# ── Startup: load embedding model once ───────────────────────
embedding_service = EmbeddingService()

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("🤖 Loading sentence-transformer model (free, local)...")
    await embedding_service.load()  
    print("✅ AI Service ready!")
    app.state.embeddings = embedding_service
    yield
    print("AI Service shutting down...")

app = FastAPI(
    title="MindfulMeals AI Service",
    version="3.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[os.getenv("BACKEND_URL", "http://localhost:3000"), "http://localhost:3000", "*"],
    allow_methods=["POST", "GET", "OPTIONS"],
    allow_headers=["*"],
)

# ── Internal Auth — only Node.js API can call this ─────────────
def verify_internal(x_service_secret: str = Header(...)):
    if x_service_secret != os.getenv("AI_SERVICE_SECRET"):
        raise HTTPException(status_code=401, detail="Unauthorized")
    return True

# ── Routes ────────────────────────────────────────────────────
app.include_router(mood.router,        prefix="/ai/mood",        tags=["mood"])
app.include_router(meals.router,       prefix="/ai/meals",       tags=["meals"])
app.include_router(restaurants.router, prefix="/ai/restaurants", tags=["restaurants"])
app.include_router(insights.router,    prefix="/ai/insights",    tags=["insights"])

@app.get("/health")
def health():
    return {
        "status": "ok",
        "service": "MindfulMeals AI",
        "model": "llama-3.3-70b-versatile (Groq FREE)",
        "embeddings": "all-MiniLM-L6-v2 (local FREE)",
    }
