# ai_service/routers/mood.py
from fastapi import APIRouter, Request
from models.schemas import MoodIntentRequest, MoodIntentResponse
from services.groq_client import chat_json, SMART_MODEL

router = APIRouter()

MOOD_INTENT_PROMPT = """You are MindfulMeals AI, an expert in nutritional psychology and mindful eating.
Given a user's emotional state and dietary preferences, generate structured nutrition guidance.

Return ONLY valid JSON matching this exact shape:
{
  "intent": "one sentence describing what they need nutritionally right now",
  "nutrients_focus": ["magnesium", "omega-3"],
  "foods_to_emphasize": ["dark leafy greens", "walnuts"],
  "foods_to_avoid": ["caffeine", "refined sugar"],
  "cuisine_styles": ["Japanese", "Mediterranean"],
  "ambience_needs": "quiet, calm, minimal distraction",
  "meal_tone": "calming"
}

meal_tone must be one of: calming, energizing, comforting, grounding, celebrating"""


@router.post("/intent", response_model=MoodIntentResponse)
async def mood_to_intent(req: MoodIntentRequest):
    messages = [
        {"role": "system", "content": MOOD_INTENT_PROMPT},
        {"role": "user", "content":
            f"Mood: {req.mood.value}\n"
            f"Dietary restrictions: {', '.join(req.dietary_tags) or 'none'}\n"
            f"Allergies: {', '.join(req.allergy_tags) or 'none'}"
        },
    ]
    data = await chat_json(messages)
    return MoodIntentResponse(**data)
