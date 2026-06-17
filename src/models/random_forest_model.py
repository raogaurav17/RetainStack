from sklearn.ensemble import RandomForestClassifier

def get_model(**kwargs):
    """
    Returns an instance of RandomForestClassifier with the given parameters.
    """
    return RandomForestClassifier(**kwargs)
