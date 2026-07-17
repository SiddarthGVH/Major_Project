"""
Gmail integration service.
"""
from __future__ import annotations

import secrets
from datetime import datetime, timedelta, timezone
from typing import Optional, Sequence, Tuple
from uuid import UUID, uuid4

from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.core.exceptions import ConflictException, NotFoundException
from app.models.email import Email, GmailConnection
from app.repositories.email_repository import EmailRepository, GmailConnectionRepository
from app.schemas.email import (
    EmailDetailResponse,
    EmailHistoryResponse,
    EmailResponse,
    EmailSyncMessageRequest,
    EmailSyncRequest,
    EmailSyncResultResponse,
    EmailThreadResponse,
    GmailConnectRequest,
    GmailOAuthCallbackRequest,
    GmailOAuthLoginResponse,
    GmailTokenRefreshRequest,
    GmailWebhookRequest,
)
from app.services.timeline_engine_service import TimelineEngineService
from app.utils.enums import EmailDirection, EmailSyncStatus, SortOrder


class EmailService:
    def __init__(self, db: AsyncSession) -> None:
        self.db = db
        self.connection_repo = GmailConnectionRepository(db)
        self.email_repo = EmailRepository(db)
        self.timeline = TimelineEngineService(db)

    async def connect_gmail(
        self,
        organization_id: UUID,
        created_by: UUID,
        user_id: UUID,
        email_address: str,
        access_token_encrypted: str,
        refresh_token_encrypted: Optional[str],
        token_expires_at: Optional[datetime],
        sync_cursor: Optional[str],
        scopes_json: Optional[list[str]],
    ) -> GmailConnection:
        existing = await self.connection_repo.get_by_user(organization_id, user_id)
        if existing:
            raise ConflictException("A Gmail connection already exists for this user.")

        connection = await self.connection_repo.create(
            organization_id=organization_id,
            created_by=created_by,
            user_id=user_id,
            email_address=email_address,
            access_token_encrypted=access_token_encrypted,
            refresh_token_encrypted=refresh_token_encrypted,
            token_expires_at=token_expires_at,
            sync_cursor=sync_cursor,
            scopes_json=scopes_json,
            sync_status=EmailSyncStatus.CONNECTED.value,
        )
        await self.timeline.record(
            organization_id=organization_id,
            created_by=created_by,
            entity_type="email",
            entity_id=connection.id,
            action="gmail_connected",
            title="Gmail connected",
            description=f"Connected Gmail account {email_address}.",
            payload={"gmail_connection_id": str(connection.id), "email_address": email_address},
            topic="gmail",
        )
        return connection

    async def start_oauth_login(self, organization_id: UUID, created_by: UUID, email_address: Optional[str] = None) -> GmailOAuthLoginResponse:
        state = secrets.token_urlsafe(24)
        scopes = settings.GOOGLE_OAUTH_SCOPES.replace(",", " ")
        redirect_uri = settings.GOOGLE_REDIRECT_URI or "http://localhost/oauth/google/callback"
        auth_url = (
            "https://accounts.google.com/o/oauth2/v2/auth"
            f"?client_id={settings.GOOGLE_CLIENT_ID or 'placeholder-client-id'}"
            f"&redirect_uri={redirect_uri}"
            f"&response_type=code"
            f"&scope={scopes}"
            f"&access_type=offline"
            f"&prompt=consent"
            f"&state={state}"
        )
        return GmailOAuthLoginResponse(authorization_url=auth_url, state=state)

    async def handle_oauth_callback(
        self,
        organization_id: UUID,
        created_by: UUID,
        payload: GmailOAuthCallbackRequest,
    ) -> GmailConnection:
        email_address = payload.email_address or f"user-{payload.state or 'gmail'}@example.com"
        access_token = f"access_{payload.code}"
        refresh_token = f"refresh_{payload.code}"
        token_expires_at = datetime.now(timezone.utc) + timedelta(hours=1)
        return await self.connect_gmail(
            organization_id=organization_id,
            created_by=created_by,
            user_id=created_by,
            email_address=str(email_address),
            access_token_encrypted=access_token,
            refresh_token_encrypted=refresh_token,
            token_expires_at=token_expires_at,
            sync_cursor=None,
            scopes_json=[scope.strip() for scope in settings.GOOGLE_OAUTH_SCOPES.split(",") if scope.strip()],
        )

    async def refresh_token(self, organization_id: UUID, created_by: UUID, payload: GmailTokenRefreshRequest) -> GmailConnection:
        connection = await self.connection_repo.get_by_id_in_org(organization_id, payload.gmail_connection_id)
        if not connection:
            raise NotFoundException("GmailConnection", payload.gmail_connection_id)
        connection.access_token_encrypted = f"access_{uuid4().hex}"
        connection.token_expires_at = datetime.now(timezone.utc) + timedelta(hours=1)
        connection.sync_status = EmailSyncStatus.ACTIVE.value
        await self.db.flush()
        return connection

    async def list_connections(self, organization_id: UUID) -> list[GmailConnection]:
        return await self.connection_repo.list_by_organization(organization_id)

    async def sync_messages(
        self,
        organization_id: UUID,
        created_by: Optional[UUID],
        payload: EmailSyncRequest,
    ) -> EmailSyncResultResponse:
        connection = await self.connection_repo.get_by_id_in_org(organization_id, payload.gmail_connection_id)
        if not connection:
            raise NotFoundException("GmailConnection", payload.gmail_connection_id)

        ingested: list[Email] = []
        skipped = 0
        messages: Sequence[EmailSyncMessageRequest] = payload.messages
        for message in messages:
            email, created = await self.ingest_email(
                organization_id=organization_id,
                created_by=created_by,
                gmail_connection_id=connection.id,
                gmail_message_id=message.gmail_message_id,
                thread_id=message.thread_id,
                direction=EmailDirection(message.direction),
                sender=str(message.sender),
                receiver=str(message.receiver) if message.receiver else None,
                subject=message.subject,
                body_preview=message.body_preview,
                sent_at=message.sent_at,
                attachment_metadata=[item.model_dump() for item in message.attachment_metadata],
                raw_payload=message.raw_payload,
                external_entity_type=message.external_entity_type,
                external_entity_id=message.external_entity_id,
                is_read=message.is_read,
            )
            if created:
                ingested.append(email)
            else:
                skipped += 1

        next_cursor = payload.sync_cursor or (messages[-1].gmail_message_id if messages else connection.sync_cursor)
        await self.connection_repo.update(
            connection,
            sync_cursor=next_cursor,
            sync_status=EmailSyncStatus.ACTIVE.value,
        )

        return EmailSyncResultResponse(
            gmail_connection_id=connection.id,
            synced_count=len(ingested),
            skipped_count=skipped,
            next_cursor=next_cursor,
            connection_status=EmailSyncStatus.ACTIVE.value,
            emails=[EmailResponse.model_validate(item) for item in ingested],
        )

    async def ingest_email(
        self,
        organization_id: UUID,
        created_by: Optional[UUID],
        gmail_connection_id: UUID,
        gmail_message_id: str,
        thread_id: Optional[str],
        direction: EmailDirection,
        sender: str,
        receiver: Optional[str],
        subject: str,
        body_preview: Optional[str],
        sent_at: datetime,
        attachment_metadata: Optional[list] = None,
        raw_payload: Optional[dict] = None,
        external_entity_type: Optional[str] = None,
        external_entity_id: Optional[UUID] = None,
        is_read: bool = False,
    ) -> tuple[Email, bool]:
        connection = await self.connection_repo.get_by_id_in_org(organization_id, gmail_connection_id)
        if not connection:
            raise NotFoundException("GmailConnection", gmail_connection_id)

        existing = await self.email_repo.get_by_message_id(organization_id, gmail_message_id)
        if existing:
            return existing, False

        email = await self.email_repo.create(
            organization_id=organization_id,
            created_by=created_by,
            gmail_message_id=gmail_message_id,
            thread_id=thread_id,
            direction=direction.value,
            sender=sender,
            receiver=receiver,
            subject=subject,
            body_preview=body_preview,
            sent_at=sent_at,
            attachment_metadata=attachment_metadata or [],
            raw_payload=raw_payload,
            gmail_connection_id=gmail_connection_id,
            external_entity_type=external_entity_type,
            external_entity_id=external_entity_id,
            is_read=is_read,
        )
        await self.timeline.record(
            organization_id=organization_id,
            created_by=created_by,
            entity_type="email",
            entity_id=email.id,
            action="email_received" if direction == EmailDirection.INBOUND else "email_sent",
            title=subject,
            description=body_preview,
            payload={
                "gmail_message_id": gmail_message_id,
                "thread_id": thread_id,
                "external_entity_type": external_entity_type,
                "external_entity_id": str(external_entity_id) if external_entity_id else None,
            },
            topic="gmail",
        )
        return email, True

    async def fetch_from_gmail(self, organization_id: UUID, connection_id: UUID, created_by: Optional[UUID]) -> EmailSyncResultResponse:
        connection = await self.connection_repo.get_by_id_in_org(organization_id, connection_id)
        if not connection:
            raise NotFoundException("GmailConnection", connection_id)
        payload = EmailSyncRequest(gmail_connection_id=connection.id, sync_cursor=connection.sync_cursor, messages=[])
        return await self.sync_messages(organization_id, created_by, payload)

    async def webhook_sync(self, organization_id: UUID, created_by: Optional[UUID], payload: GmailWebhookRequest) -> EmailSyncResultResponse:
        if payload.gmail_connection_id:
            return await self.fetch_from_gmail(organization_id, payload.gmail_connection_id, created_by)
        connections = await self.connection_repo.list_by_organization(organization_id)
        if not connections:
            raise NotFoundException("GmailConnection", UUID(int=0))
        return await self.fetch_from_gmail(organization_id, connections[0].id, created_by)

    async def sync_all_connections(self, organization_id: UUID, created_by: Optional[UUID]) -> list[EmailSyncResultResponse]:
        results: list[EmailSyncResultResponse] = []
        for connection in await self.connection_repo.list_by_organization(organization_id):
            results.append(await self.fetch_from_gmail(organization_id, connection.id, created_by))
        return results

    async def get_email(self, organization_id: UUID, email_id: UUID) -> Email:
        email = await self.email_repo.get_by_id_in_org(organization_id, email_id)
        if not email:
            raise NotFoundException("Email", email_id)
        return email

    async def list_emails(
        self,
        organization_id: UUID,
        search: Optional[str],
        direction: Optional[EmailDirection],
        thread_id: Optional[str],
        external_entity_type: Optional[str],
        external_entity_id: Optional[UUID],
        page: int,
        page_size: int,
        sort_order: SortOrder = SortOrder.DESC,
    ) -> Tuple[list[Email], int]:
        return await self.email_repo.list_by_organization(
            organization_id,
            search,
            direction.value if direction else None,
            thread_id,
            external_entity_type,
            external_entity_id,
            page,
            page_size,
            sort_order=sort_order,
        )

    async def get_contact_history(
        self,
        organization_id: UUID,
        contact_id: UUID,
        search: Optional[str],
        page: int,
        page_size: int,
        sort_order: SortOrder = SortOrder.DESC,
    ) -> Tuple[list[Email], int]:
        return await self.email_repo.list_entity_history(
            organization_id,
            "contact",
            contact_id,
            search,
            page,
            page_size,
            sort_order=sort_order,
        )

    async def get_deal_history(
        self,
        organization_id: UUID,
        deal_id: UUID,
        search: Optional[str],
        page: int,
        page_size: int,
        sort_order: SortOrder = SortOrder.DESC,
    ) -> Tuple[list[Email], int]:
        return await self.email_repo.list_entity_history(
            organization_id,
            "deal",
            deal_id,
            search,
            page,
            page_size,
            sort_order=sort_order,
        )

    async def get_thread_history(self, organization_id: UUID, thread_id: str) -> EmailThreadResponse:
        emails = await self.email_repo.list_thread_history(organization_id, thread_id)
        return EmailThreadResponse(
            thread_id=thread_id,
            emails=[EmailResponse.model_validate(email) for email in emails],
        )

    async def email_history_page(
        self,
        organization_id: UUID,
        entity_type: Optional[str],
        entity_id: Optional[UUID],
        search: Optional[str],
        page: int,
        page_size: int,
        direction: Optional[EmailDirection] = None,
        thread_id: Optional[str] = None,
        sort_order: SortOrder = SortOrder.DESC,
    ) -> EmailHistoryResponse:
        if entity_type and entity_id:
            records, total = await self.email_repo.list_entity_history(
                organization_id,
                entity_type,
                entity_id,
                search,
                page,
                page_size,
                sort_order=sort_order,
            )
        else:
            records, total = await self.list_emails(
                organization_id,
                search,
                direction,
                thread_id,
                entity_type,
                entity_id,
                page,
                page_size,
                sort_order=sort_order,
            )
        return EmailHistoryResponse(
            total=total,
            page=page,
            page_size=page_size,
            records=[EmailResponse.model_validate(record) for record in records],
        )

    async def get_by_id_response(self, organization_id: UUID, email_id: UUID) -> EmailDetailResponse:
        email = await self.get_email(organization_id, email_id)
        return EmailDetailResponse.model_validate(email)
