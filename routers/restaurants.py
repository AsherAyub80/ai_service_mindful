# ai_service/routers/restaurants.py
from fastapi import APIRouter, Request
from models.schemas import RestaurantRecommendRequest, RestaurantRecommendResponse, RestaurantRecommendation, MoodIntentRequest
from routers.mood import mood_to_intent
from services.groq_client import chat_json, SMART_MODEL

router = APIRouter()

RESTAURANT_PROMPT = """You are MindfulMeals AI, a mindful dining expert.
Rank these restaurants for a user based on their emotional state and needs.

Return a JSON array ONLY:
[
  {"id": "restaurant-id", "score": 0.95, "mood_alignment": "why this place suits the mood", "matches_mood": true},
  ...
]
score: 0.0–1.0. matches_mood: true if the restaurant's mood_tags include the user's mood."""


@router.post("/recommend", response_model=RestaurantRecommendResponse)
async def recommend_restaurants(req: RestaurantRecommendRequest, request: Request):
    intent = await mood_to_intent(MoodIntentRequest(mood=req.mood))

    # Semantic ranking first
    embeddings = request.app.state.embeddings
    query = f"{intent.intent} {intent.ambience_needs} {' '.join(intent.cuisine_styles)}"
    candidates_dicts = [c.model_dump() for c in req.candidates]
    semantically_ranked = embeddings.rank_by_similarity(
        query=query, candidates=candidates_dicts, text_field="description", top_k=10
    )

    messages = [
        {"role": "system", "content": RESTAURANT_PROMPT},
        {"role": "user", "content":
            f"User mood: {req.mood.value}\n"
            f"Needs: {intent.intent}\n"
            f"Ideal ambience: {intent.ambience_needs}\n"
            f"Preferred cuisines: {', '.join(intent.cuisine_styles)}\n\n"
            f"Restaurants:\n" +
            "\n".join(f"- id:{r['id']} | {r['name']} | cuisines:{','.join(r.get('cuisine_tags',[]))} | "
                      f"mood_tags:{','.join(r.get('mood_tags',[]))} | rating:{r.get('rating',0)} | "
                      f"{r.get('distance_km',0):.1f}km | {r.get('description','')[:60]}"
                      for r in semantically_ranked)
        },
    ]

    try:
        rankings = await chat_json(messages)
        if not isinstance(rankings, list):
            rankings = list(rankings.values())[0] if rankings else []
    except Exception:
        rankings = [
            {"id": r["id"], "score": round(r.get("_similarity_score", 0.75), 3),
             "mood_alignment": f"{r['name']} suits your {req.mood.value} mood",
             "matches_mood": req.mood.value in r.get("mood_tags", [])}
            for r in semantically_ranked
        ]

    recommendations = [RestaurantRecommendation(**r) for r in rankings[:req.limit]]
    return RestaurantRecommendResponse(recommendations=recommendations, intent=intent)
