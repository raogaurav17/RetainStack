"""Pydantic request models for the RetainStack prediction API.

Validates raw feature values before they reach the preprocessor pipeline.
Field constraints are based on the Online Shoppers Purchasing Intention Dataset.
"""

from pydantic import BaseModel, Field


# Valid months present in the UCI dataset
_VALID_MONTHS = frozenset(
    {"Jan", "Feb", "Mar", "Apr", "May", "June", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"}
)


class SessionFeatures(BaseModel):
    """Raw e-commerce session features — one visitor session.

    These are the *pre-preprocessor* features listed in params.yaml.
    The API applies the fitted ColumnTransformer (preprocessor.skops) before
    feeding them to the model, so callers send human-readable values.
    """

    Administrative: int = Field(
        ..., ge=0, description="Number of administrative pages visited"
    )
    Administrative_Duration: float = Field(
        ..., ge=0.0, description="Total seconds spent on administrative pages"
    )
    Informational_Duration: float = Field(
        ..., ge=0.0, description="Total seconds spent on informational pages"
    )
    ProductRelated: int = Field(
        ..., ge=0, description="Number of product-related pages visited"
    )
    ProductRelated_Duration: float = Field(
        ..., ge=0.0, description="Total seconds spent on product-related pages"
    )
    BounceRates: float = Field(
        ..., ge=0.0, le=1.0, description="Average bounce rate of visited pages"
    )
    ExitRates: float = Field(
        ..., ge=0.0, le=1.0, description="Average exit rate of visited pages"
    )
    PageValues: float = Field(
        ..., ge=0.0, description="Average page value of visited pages"
    )
    Month: str = Field(
        ..., description="Month of the visit (e.g. 'Feb', 'Mar', 'Nov')"
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "Administrative": 0,
                    "Administrative_Duration": 0.0,
                    "Informational_Duration": 0.0,
                    "ProductRelated": 53,
                    "ProductRelated_Duration": 1482.5,
                    "BounceRates": 0.02,
                    "ExitRates": 0.05,
                    "PageValues": 8.15,
                    "Month": "Nov",
                }
            ]
        }
    }


class BatchSessionRequest(BaseModel):
    """A batch of e-commerce session features for bulk prediction.

    Enforces a minimum of 1 and a maximum of 500 sessions per request to
    prevent empty payloads and guard against DoS via oversized requests.
    """

    sessions: list[SessionFeatures] = Field(
        ...,
        min_length=1,
        max_length=500,
        description="List of 1–500 session feature objects to score in one call",
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "sessions": [
                        {
                            "Administrative": 0,
                            "Administrative_Duration": 0.0,
                            "Informational_Duration": 0.0,
                            "ProductRelated": 53,
                            "ProductRelated_Duration": 1482.5,
                            "BounceRates": 0.02,
                            "ExitRates": 0.05,
                            "PageValues": 8.15,
                            "Month": "Nov",
                        },
                        {
                            "Administrative": 1,
                            "Administrative_Duration": 30.0,
                            "Informational_Duration": 5.0,
                            "ProductRelated": 10,
                            "ProductRelated_Duration": 300.0,
                            "BounceRates": 0.10,
                            "ExitRates": 0.15,
                            "PageValues": 0.0,
                            "Month": "Feb",
                        },
                    ]
                }
            ]
        }
    }
