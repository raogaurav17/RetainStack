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


class BatchPredictionResult(BaseModel):
    """Batch prediction output returned by /api/v1/predict/batch.

    Contains one PredictionResult per input session, in the same order as the
    request, along with a convenience `total` count.
    """

    predictions: list[PredictionResult] = Field(
        ..., description="Ordered list of predictions — one per input session"
    )
    total: int = Field(
        ..., ge=1, description="Number of sessions scored in this request"
    )


class HealthResponse(BaseModel):
    """Response from /health."""

    status: str = Field(..., description="Service status: 'ok' or 'degraded'")
    version: str = Field(..., description="Application version string")


class ReadyResponse(BaseModel):
    """Response from /ready."""

    ready: bool = Field(..., description="Whether the model is loaded and ready to serve")
    model_loaded: bool = Field(..., description="Whether model.skops is loaded in memory")
    preprocessor_loaded: bool = Field(
        ..., description="Whether preprocessor.skops is loaded in memory"
    )


class ErrorResponse(BaseModel):
    """Standard error envelope."""

    detail: str = Field(..., description="Human-readable error message")
