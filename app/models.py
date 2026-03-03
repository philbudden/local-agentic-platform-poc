"""Pydantic request/response schemas for the Ingress API."""

from typing import Literal

from pydantic import BaseModel


class ClassifierResponse(BaseModel):
    intent: Literal["execution", "decomposition", "novel_reasoning", "ambiguous"]
    confidence: float


class IngestRequest(BaseModel):
    input: str


class IngestResponse(BaseModel):
    intent: str
    confidence: float
    response: str
