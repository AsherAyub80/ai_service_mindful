# ai_service/routers/meals.py
from fastapi import APIRouter, Request
from models.schemas import (
    MealRecommendRequest, MealRecommendResponse, MealRecommendation,
    EmbedRequest, EmbedResponse, SearchRequest, SearchResponse, MoodIntentRequest,
)
from routers.mood import mood_to_intent
from services.groq_client import chat_json, SMART_MODEL, FAST_MODEL

router = APIRouter()

RANK_PROMPT = """You are MindfulMeals AI. Rank these meals for a user based on their mood and nutritional needs.

Return a JSON array ONLY — no markdown, no extra text:
[
  {"id": "meal-id", "score": 0.95, "mood_alignment": "why this meal helps this mood", "quick_tip": "brief serving suggestion"},
  ...
]

Rules:
- score range: 0.0 to 1.0 (higher = better match)
- mood_alignment: warm, personal, 1 sentence
- quick_tip: practical serving idea, 1 sentence
- Keep the order from best to worst match"""


@router.post("/recommend", response_model=MealRecommendResponse)
async def recommend_meals(req: MealRecommendRequest, request: Request):
    # Step 1: get mood intent
    intent = await mood_to_intent(MoodIntentRequest(
        mood=req.mood,
        dietary_tags=req.user_preferences.get("dietary_tags", []),
        allergy_tags=req.user_preferences.get("allergy_tags", []),
    ))

    # Step 2: semantic re-ranking using local embeddings
    embeddings = request.app.state.embeddings
    query_text = f"{intent.intent} {' '.join(intent.foods_to_emphasize)} {' '.join(intent.cuisine_styles)}"

    candidates_dicts = [c.model_dump() for c in req.candidates]
    semantically_ranked = embeddings.rank_by_similarity(
        query=query_text,
        candidates=candidates_dicts,
        text_field="description",
        top_k=min(15, len(candidates_dicts)),
    )

    # Step 3: Groq re-ranks the top semantic results with natural language reasoning
    messages = [
        {"role": "system", "content": RANK_PROMPT},
        {"role": "user", "content":
            f"User mood: {req.mood.value}\n"
            f"What they need: {intent.intent}\n"
            f"Foods to emphasize: {', '.join(intent.foods_to_emphasize)}\n"
            f"Foods to avoid: {', '.join(intent.foods_to_avoid)}\n"
            f"Meal tone needed: {intent.meal_tone}\n\n"
            f"Meals (pre-ranked by semantic similarity):\n"
            + "\n".join(f"- id:{m['id']} | {m['title']} | {m.get('calories',0)}cal | {m.get('description','')[:80]}"
                        for m in semantically_ranked)
        },
    ]

    try:
        rankings = await chat_json(messages)
        if not isinstance(rankings, list):
            rankings = rankings.get("recommendations", list(rankings.values())[0] if rankings else [])
    except Exception:
        # Fallback: use semantic scores
        rankings = [
            {"id": m["id"], "score": round(m.get("_similarity_score", 0.8), 3),
             "mood_alignment": f"A great choice for your {req.mood.value} state",
             "quick_tip": "Take a moment to eat mindfully and savour each bite."}
            for m in semantically_ranked
        ]

    recommendations = [MealRecommendation(**r) for r in rankings[:req.limit]]
    return MealRecommendResponse(recommendations=recommendations, intent=intent)


@router.post("/embed", response_model=EmbedResponse)
async def embed_text(req: EmbedRequest, request: Request):
    """Generate a free local embedding for a text string."""
    embeddings = request.app.state.embeddings
    vec = embeddings.embed(req.text)
    return EmbedResponse(embedding=vec, model="all-MiniLM-L6-v2", dimensions=len(vec))


@router.post("/search", response_model=SearchResponse)
async def semantic_search(req: SearchRequest, request: Request):
    """
    Natural language meal search using semantic embeddings.
    No Groq call needed for simple searches — pure local embeddings.
    """
    embeddings = request.app.state.embeddings
    candidates_dicts = [c.model_dump() for c in req.candidates]

    # For complex queries, extract filters with Groq first
    filters = {}
    if len(req.query.split()) > 4:
        try:
            messages = [
                {"role": "system", "content":
                    'Extract search filters from the query. Return JSON: {"mood": "Calm|null", "max_calories": 500, "dietary": ["vegan"], "keywords": ["light","japanese"]}'},
                {"role": "user", "content": f'Query: "{req.query}"'},
            ]
            filters = await chat_json(messages, model=FAST_MODEL, max_tokens=200)
        except Exception:
            pass

    results = embeddings.rank_by_similarity(
        query=req.query,
        candidates=candidates_dicts,
        text_field="description",
        top_k=req.top_k,
    )

    from models.schemas import MealCandidate
    return SearchResponse(
        results=[MealCandidate(**{k: v for k, v in r.items() if k != "_similarity_score"}) for r in results],
        extracted_filters=filters,
    )
