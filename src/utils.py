import yaml
import os

class PipelineError(Exception):
    """Exception raised for errors in the pipeline."""
    pass

def load_params(path: str) -> dict:
    """Load yaml parameters."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Params file not found at {path}")
    with open(path, 'r') as stream:
        params = yaml.safe_load(stream)
    return params
