"""
AI integration routes.
Future-facing endpoints return structured placeholders that can be backed by
real models later without changing the public API.
"""

from fastapi import APIRouter, status

from app.api.deps import CurrentUser, DBSession
from app.schemas.ai import (
    AIConversationSummaryRequest,
    AIConversationSummaryResponse,
    AIEmailSummaryRequest,
    AIEmailSummaryResponse,
    AILeadScoreRequest,
    AILeadScoreResponse,
    AINextBestActionRequest,
    AINextBestActionResponse,
    AIRecommendationRequest,
    AIRecommendationResponse,
    DealInsightRequest,
    DealInsightResponse,
    SummaryRequest,
    SummaryResponse,
)
from app.services.ai_service import AIService

router = APIRouter()


@router.post("/lead-score", response_model=AILeadScoreResponse, status_code=status.HTTP_200_OK)
async def lead_score(payload: AILeadScoreRequest, current_user: CurrentUser, db: DBSession) -> AILeadScoreResponse:
    service = AIService(db)
    return await service.lead_score(current_user.organization_id, payload.lead_id, payload.context)


@router.post("/next-best-action", response_model=AINextBestActionResponse, status_code=status.HTTP_200_OK)
async def next_best_action(
    payload: AINextBestActionRequest,
    current_user: CurrentUser,
    db: DBSession,
) -> AINextBestActionResponse:
    service = AIService(db)
    return await service.next_best_action(current_user.organization_id, payload.context)


@router.post("/conversation-summary", response_model=AIConversationSummaryResponse, status_code=status.HTTP_200_OK)
async def conversation_summary(
    payload: AIConversationSummaryRequest,
    current_user: CurrentUser,
    db: DBSession,
) -> AIConversationSummaryResponse:
    service = AIService(db)
    return await service.conversation_summary(current_user.organization_id, payload.messages, payload.context)


@router.post("/email-summary", response_model=AIEmailSummaryResponse, status_code=status.HTTP_200_OK)
async def email_summary(payload: AIEmailSummaryRequest, current_user: CurrentUser, db: DBSession) -> AIEmailSummaryResponse:
    service = AIService(db)
    return await service.email_summary(current_user.organization_id, payload.email_ids, payload.context)


@router.post("/recommendations", response_model=AIRecommendationResponse, status_code=status.HTTP_200_OK)
async def recommendations(
    payload: AIRecommendationRequest,
    current_user: CurrentUser,
    db: DBSession,
) -> AIRecommendationResponse:
    service = AIService(db)
    return await service.recommendations(current_user.organization_id, payload.context)


@router.post("/deal-insight", response_model=DealInsightResponse, status_code=status.HTTP_200_OK)
async def deal_insight(payload: DealInsightRequest, current_user: CurrentUser, db: DBSession) -> DealInsightResponse:
    service = AIService(db)
    return await service.deal_insight(current_user.organization_id, payload.deal_id)


@router.post("/summary", response_model=SummaryResponse, status_code=status.HTTP_200_OK)
async def summary(payload: SummaryRequest, current_user: CurrentUser, db: DBSession) -> SummaryResponse:
    service = AIService(db)
    return await service.summarize(current_user.organization_id, payload.context)
