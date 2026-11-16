from data.generate_eval_dataset import generate_evaluation_dataset
from evals.application_evals.evaluate_rag_app import evaluate_app
from dotenv import load_dotenv
from config.parameter_config import params_config
from pathlib import Path
import json 
from pprint import pprint
from langfuse import get_client

load_dotenv()

def flatten_params(params_dict: dict) -> dict:
    config_dict = {}
    
    for key, value in params_dict.items():
        if isinstance(value, dict):
            config_dict.update(flatten_params(value))
        else:
            if key in config_dict:
                raise ValueError(f"Duplicate key found: {key}")
            config_dict[key] = value
    
    return config_dict

def get_latest_results(path: Path | str):
    path_obj = Path(path)
    files = list(path_obj.glob("*.json"))
    
    if not files:
        print("No JSON files found.")
        return None

    recent_file = max(files, key=lambda f: f.name)
    return recent_file.stem

def get_metrics_from_results(results_json_path: Path | str) -> dict:
    metrics = {}
    
    with open(results_json_path, "r") as file:
        # Using the correct key from your first function definition
        metrics_results = json.load(file)["metricsScores"]
        
    for result in metrics_results:
        metric_name = result["metric"].removesuffix("[GEval]") if "[GEval]" in result["metric"] else result["metric"].strip()
        scores = result["scores"]
        avg_scores = round((sum(scores) / len(scores)), 2)
        
        metrics[metric_name] = avg_scores
    
    return metrics

def import_system_prompt(label: str="staging") -> str:
    langfuse = get_client()
    sys_prompt = langfuse.get_prompt(
        name="rag_app_system_prompt",
        type="text",
        label=label
    )
    
    return sys_prompt.prompt

# Usage:
ROOT_DIR = Path(__file__).parent
evaluation_result_dir = ROOT_DIR / "reports" / "evaluation_results"

filename = (evaluation_result_dir / get_latest_results(evaluation_result_dir)).as_posix() + ".json"
metrics = get_metrics_from_results(filename)
pprint(metrics)
