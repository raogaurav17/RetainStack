"""Pydantic response models for the RetainStack prediction API."""

from pydantic import BaseModel, Field


class PredictionResult(BaseModel):
    """Single prediction output returned by /api/v1/predict."""

    prediction: int = Field(
        ..., description="Binary prediction: 1 = purchase, 0 = no purchase"
    )
    purchase_probability: float = Field(
        ..., ge=0.0, le=1.0, description="Probability that the session ends in a purchase"
    )
    confidence: str = Field(
        ..., description="Human-readable confidence level: high, medium, or low"
    )


class HealthResponse(BaseModel):
    """Response from /health."""

    status: str = Field(..., description="Service status: 'ok' or 'degraded'")
    version: str = Field(..., description="Application version string")


class ReadyResponse(BaseModel):
    """Response from /ready."""

    ready: bool = Field(..., description="Whether the model is loaded and ready to serve")
    model_loaded: bool = Field(..., description="Whether model.pkl is loaded in memory")
    preprocessor_loaded: bool = Field(
        ..., description="Whether preprocessor.pkl is loaded in memory"
    )


class ErrorResponse(BaseModel):
    """Standard error envelope."""

    detail: str = Field(..., description="Human-readable error message")
