import yaml
import pandas as pd
from fastapi import APIRouter, Depends, HTTPException, status

from src.api.batcher import DynamicBatcher, _confidence_label, get_batcher
from src.api.dependencies import ModelStore, get_model_store
from src.api.schemas.request import BatchSessionRequest, SessionFeatures
from src.api.schemas.response import BatchPredictionResult, ErrorResponse, PredictionResult
from src.logger.logger import get_logger

logger = get_logger("api.predict")

router = APIRouter(prefix="/predict", tags=["Prediction"])


def _load_feature_order() -> list[str]:
    """Load feature order from params.yaml"""
    with open("params.yaml", "r") as f:
        params = yaml.safe_load(f)
    return params["data_preprocess"]["features"]


_RAW_FEATURE_ORDER = _load_feature_order()



@router.post(
    "",
    response_model=PredictionResult,
    responses={
        503: {"model": ErrorResponse, "description": "Model not loaded"},
    },
    summary="Predict purchase intent",
    description=(
        "Accept raw session features and return the model's binary prediction "
        "with a purchase probability. Requests are automatically coalesced by "
        "the server-side dynamic batcher for efficient vectorised inference."
    ),
)
async def predict(
    session: SessionFeatures,
    store: ModelStore = Depends(get_model_store),
    batcher: DynamicBatcher = Depends(get_batcher),
) -> PredictionResult:
    if not store.is_ready:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model or preprocessor not loaded. Check /ready.",
        )

    raw_dict = session.model_dump()
    row = {col: raw_dict[col] for col in _RAW_FEATURE_ORDER}

    try:
        result = await batcher.submit(row)
    except Exception as exc:
        logger.error("Prediction failed: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Prediction failed due to an internal error.",
        ) from exc

    logger.info(
        "Prediction: %d | Probability: %.4f | Month: %s",
        result.prediction,
        result.purchase_probability,
        session.Month,
    )

    return result


@router.post(
    "/batch",
    response_model=BatchPredictionResult,
    responses={
        503: {"model": ErrorResponse, "description": "Model not loaded"},
    },
    summary="Batch-predict purchase intent",
    description=(
        "Accept a list of 1–500 raw session-feature objects, apply the fitted "
        "preprocessor to all of them in a single pass, and return a prediction "
        "with purchase probability for each session in the same order."
    ),
)
async def predict_batch(
    body: BatchSessionRequest,
    store: ModelStore = Depends(get_model_store),
) -> BatchPredictionResult:
    # TODO(security): add rate limiting (e.g. slowapi) at the router level
    # to guard this and the single-predict endpoint against request flooding.
    if not store.is_ready:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model or preprocessor not loaded. Check /ready.",
        )

    # Extract and order session features for vectorized inference
    rows = [
        {col: raw_dict[col] for col in _RAW_FEATURE_ORDER}
        for raw_dict in (s.model_dump() for s in body.sessions)
    ]
    df = pd.DataFrame(rows)

    # Capture an immutable artifact snapshot for this batch pass.
    # Using get_artifacts() ensures we hold a consistent (model, preprocessor)
    # pair even if a hot-reload swaps the live pointer mid-request.
    try:
        container = store.get_artifacts()
    except RuntimeError as exc:
        logger.error("No artifacts available for batch prediction: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model or preprocessor not loaded. Check /ready.",
        ) from exc

    try:
        transformed = container.preprocessor.transform(df)
        predictions = container.model.predict(transformed).tolist()
        probabilities = container.model.predict_proba(transformed)[:, 1].tolist()
    except Exception as exc:
        logger.error("Batch prediction failed: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Batch prediction failed due to an internal error.",
        ) from exc

    results = [
        PredictionResult(
            prediction=int(pred),
            purchase_probability=round(float(prob), 4),
            confidence=_confidence_label(float(prob)),
        )
        for pred, prob in zip(predictions, probabilities)
    ]

    logger.info(
        "Batch prediction complete: %d sessions scored",
        len(results),
    )

    return BatchPredictionResult(predictions=results, total=len(results))
