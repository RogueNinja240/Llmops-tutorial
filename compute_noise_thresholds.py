import dagshub
import mlflow
import numpy as np
from pathlib import Path
import json
from utils.mlflow_utils import experiment_id,get_metrics_from_runs
import dagshub



ROOT_DIR = Path(__file__).parent 
JSON_FILE_PATH = ROOT_DIR / "historical_runs.json"
experiment_id = mlflow.get_experiment_by_name("rag-app").experiment_id
dagshub.init(repo_owner='RogueNinja240', repo_name='Llmops-tutorial', mlflow=True)
mlflow.set_tracking_uri("https://dagshub.com/RogueNinja240/Llmops-tutorial.mlflow")
experiment_id = mlflow.get_experiment_by_name("rag-app").experiment_id

def calculate_noise_threshold(metrics: list[dict[str, float]]):
    all_metrics = []
    
    for metric in metrics:
        metric_names = list(metric.keys())
        metric_values = list(metric.values())
        all_metrics.append(metric_values)
    
    
    matrix = np.array(all_metrics)
    stds = np.std(matrix, axis=0, ddof=1).tolist()
    
    return {key: val for key, val in zip(metric_names, stds)}
