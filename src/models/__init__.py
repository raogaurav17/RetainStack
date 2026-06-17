from . import xgboost_model
from . import random_forest_model

MODEL_REGISTRY = {
    "xgboost": xgboost_model.get_model,
    "random_forest": random_forest_model.get_model
}

def get_model(model_name: str, **kwargs):
    if model_name not in MODEL_REGISTRY:
        raise ValueError(f"Model {model_name} is not registered. Available models: {list(MODEL_REGISTRY.keys())}")
    return MODEL_REGISTRY[model_name](**kwargs)
