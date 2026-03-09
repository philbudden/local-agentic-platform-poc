"""Pydantic schemas for the cortx distribution HTTP API."""

from __future__ import annotations

from pydantic import BaseModel, field_validator
from typing import Literal


class IngestRequest(BaseModel):
    input: str

    @field_validator("input")
    @classmethod
    def input_must_not_be_blank(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("input must not be empty or whitespace-only")
        return v


class IngestResponse(BaseModel):
    intent: str
    confidence: float
    response: str
