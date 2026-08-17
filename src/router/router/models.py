import asyncio
from datetime import datetime
from typing import Any

from pydantic import BaseModel


class PendingRequest(BaseModel):
    future: Any
    llm_id: str


class QueryReference(BaseModel):
    target: str
    dataset: str
    subset: str | None = None

    def has_subset(self):
        return self.subset is not None


class QueryRequest(BaseModel):
    query: str
    reference: QueryReference | None = None


class Query(BaseModel):
    query_id: str
    query: str
    origin_site: str | None = None
    reference: QueryReference | None = None

    def sent_from_router(self):
        return self.origin_site is not None

    def has_reference(self):
        return self.reference is not None


class QueryTimes(BaseModel):
    query_received: datetime | None
    query_published: datetime | None
    query_from_router_received: datetime | None
    model_query_received: datetime | None
    model_inference_started: datetime | None
    model_inference_finshed: datetime | None
    model_response_published: datetime | None
    response_received: datetime | None
    future_completed: datetime | None
