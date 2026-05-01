# ai_service/routers/insights.py
from fastapi import APIRouter
from models.schemas import InsightsRequest, InsightsResponse
from services.groq_client import chat_json

router = APIRouter()

INSIGHTS_PROMPT = """You are MindfulMeals AI wellness coach. Analyse these mood logs with warmth and compassion.
Give personalised, helpful, non-judgmental insights.

Return ONLY valid JSON:
{
  "summary": "2-3 warm sentences about the user's week",
  "dominant_mood": "the most frequent mood",
  "mood_trend": "improving|declining|stable",
  "insight": "one specific observation about their mood-food patterns",
  "recommendation": "one kind, actionable tip for next week",
  "affirmation": "a short, genuine encouraging sentence"
}"""


@router.post("/weekly", response_model=InsightsResponse)
async def weekly_insights(req: InsightsRequest):
    if len(req.mood_logs) < 3:
        return InsightsResponse(
            has_insights=False,
            message="Log at least 3 mood entries to unlock your weekly insights!",
        )

    logs_text = "\n".join(
        f"- {log.logged_at[:10]}: mood={log.mood}"
        + (f", score={log.mood_score}/10" if log.mood_score else "")
        + (f", context={log.context}" if log.context else "")
        for log in req.mood_logs
    )

    messages = [
        {"role": "system", "content": INSIGHTS_PROMPT},
        {"role": "user", "content": f"Mood logs for the past week:\n{logs_text}"},
    ]

    try:
        data = await chat_json(messages)
        return InsightsResponse(has_insights=True, log_count=len(req.mood_logs), **data)
    except Exception as e:
        return InsightsResponse(
            has_insights=False,
            message=f"Could not generate insights right now. Try again later.",
        )
