# ai_service/models/schemas.py
from pydantic import BaseModel, Field
from typing import Optional, List
from enum import Enum


class Mood(str, Enum):
    calm = "Calm"
    energized = "Energized"
    comfort = "Comfort"
    focus = "Focus"
    happy = "Happy"


# ── Mood Intent ───────────────────────────────────────────────
class MoodIntentRequest(BaseModel):
    mood: Mood
    dietary_tags: List[str] = []
    allergy_tags: List[str] = []

class MoodIntentResponse(BaseModel):
    intent: str
    nutrients_focus: List[str] = []
    foods_to_emphasize: List[str] = []
    foods_to_avoid: List[str] = []
    cuisine_styles: List[str] = []
    ambience_needs: str = ""
    meal_tone: str = "calming"


# ── Meal Recommendations ──────────────────────────────────────
class MealCandidate(BaseModel):
    id: str
    title: str
    description: Optional[str] = ""
    calories: Optional[int] = 0
    mood_tags: List[str] = []
    dietary_tags: List[str] = []

class MealRecommendRequest(BaseModel):
    mood: Mood
    candidates: List[MealCandidate]
    user_preferences: dict = {}
    limit: int = Field(default=5, ge=1, le=10)

class MealRecommendation(BaseModel):
    id: str
    score: float
    mood_alignment: str
    quick_tip: str = ""

class MealRecommendResponse(BaseModel):
    recommendations: List[MealRecommendation]
    intent: MoodIntentResponse


# ── Restaurant Recommendations ────────────────────────────────
class RestaurantCandidate(BaseModel):
    id: str
    name: str
    cuisine_tags: List[str] = []
    mood_tags: List[str] = []
    description: Optional[str] = ""
    rating: Optional[float] = 0
    distance_km: Optional[float] = 0

class RestaurantRecommendRequest(BaseModel):
    mood: Mood
    candidates: List[RestaurantCandidate]
    limit: int = Field(default=6, ge=1, le=10)

class RestaurantRecommendation(BaseModel):
    id: str
    score: float
    mood_alignment: str
    matches_mood: bool = False

class RestaurantRecommendResponse(BaseModel):
    recommendations: List[RestaurantRecommendation]
    intent: MoodIntentResponse


# ── Embeddings ────────────────────────────────────────────────
class EmbedRequest(BaseModel):
    text: str

class EmbedResponse(BaseModel):
    embedding: List[float]
    model: str
    dimensions: int


# ── Mood Insights ─────────────────────────────────────────────
class MoodLog(BaseModel):
    mood: str
    mood_score: Optional[int] = None
    context: Optional[str] = None
    logged_at: str

class InsightsRequest(BaseModel):
    mood_logs: List[MoodLog]

class InsightsResponse(BaseModel):
    has_insights: bool
    summary: str = ""
    dominant_mood: str = ""
    mood_trend: str = ""  # improving | declining | stable
    insight: str = ""
    recommendation: str = ""
    affirmation: str = ""
    log_count: int = 0
    message: str = ""


# ── Smart Search ──────────────────────────────────────────────
class SearchRequest(BaseModel):
    query: str
    candidates: List[MealCandidate]
    top_k: int = Field(default=5, ge=1, le=10)

class SearchResponse(BaseModel):
    results: List[MealCandidate]
    extracted_filters: dict = {}
