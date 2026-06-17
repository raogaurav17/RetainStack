import xgboost as xgb

def get_model(**kwargs):
    """
    Returns an instance of XGBClassifier with the given parameters.
    """
    return xgb.XGBClassifier(**kwargs)
