import yaml
import pandas as pd
from fastapi import APIRouter, Depends, HTTPException, status

from src.api.dependencies import ModelStore, get_model_store
from src.api.schemas.request import SessionFeatures
from src.api.schemas.response import ErrorResponse, PredictionResult
from src.logger.logger import get_logger

logger = get_logger("api.predict")

router = APIRouter(prefix="/predict", tags=["Prediction"])


def _load_feature_order() -> list[str]:
    """Load feature order from params.yaml — single source of truth."""
    with open("params.yaml", "r") as f:
        params = yaml.safe_load(f)
    return params["data_preprocess"]["features"]


_RAW_FEATURE_ORDER = _load_feature_order()


def _confidence_label(probability: float) -> str:
    """Map a purchase probability to a human-readable confidence bucket."""
    if probability >= 0.75:
        return "high"
    if probability >= 0.40:
        return "medium"
    return "low"


@router.post(
    "",
    response_model=PredictionResult,
    responses={
        503: {"model": ErrorResponse, "description": "Model not loaded"},
    },
    summary="Predict purchase intent",
    description=(
        "Accept raw session features, apply the fitted preprocessor, and "
        "return the model's binary prediction with a purchase probability."
    ),
)
async def predict(
    session: SessionFeatures,
    store: ModelStore = Depends(get_model_store),
) -> PredictionResult:
    if not store.is_ready:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model or preprocessor not loaded. Check /ready.",
        )

    # Build a single-row DataFrame in the exact column order the
    # ColumnTransformer was fitted on.
    raw_dict = session.model_dump()
    row = {col: [raw_dict[col]] for col in _RAW_FEATURE_ORDER}
    df = pd.DataFrame(row)

    try:
        transformed = store.preprocessor.transform(df)
        prediction = int(store.model.predict(transformed)[0])
        probability = float(store.model.predict_proba(transformed)[0, 1])
    except Exception as exc:
        logger.error("Prediction failed: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Prediction failed due to an internal error.",
        ) from exc

    logger.info(
        "Prediction: %d | Probability: %.4f | Month: %s",
        prediction,
        probability,
        session.Month,
    )

    return PredictionResult(
        prediction=prediction,
        purchase_probability=round(probability, 4),
        confidence=_confidence_label(probability),
    )
