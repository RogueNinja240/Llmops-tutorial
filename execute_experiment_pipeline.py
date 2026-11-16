from data.generate_eval_dataset import generate_evaluation_dataset
from evals.application_evals.evaluate_rag_app import evaluate_app
from dotenv import load_dotenv
from config.parameter_config import params_config
from pathlib import Path
import json 
from langfuse import get_client
from typing import Literal 
import mlflow 
import logging 
from logging import INFO 

load_dotenv()

ROOT_DIR = Path(__file__).parent

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

# Fixed: Now actually uses the 'pattern' parameter
def get_latest_results(dir_path: Path | str, pattern: str = "*.json"):
    if isinstance(dir_path, str):
        dir_path = Path(dir_path)
    
    if not dir_path.is_dir():
        raise ValueError("Provided path is not a directory")

    filenames = []
    
    files = dir_path.glob(pattern)
    for file in files:
        filenames.append(file.stem)
        
    if not filenames:
        return None
        
    return max(filenames)

def get_metrics_from_results(results_json_path: Path | str) -> dict:
    metrics = {}
    
    with open(results_json_path, "r") as file:
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
    # Fixed: added return statement so mlflow gets the prompt
    return sys_prompt.get_langchain_prompt() 

def artifact_name(artifact_type: Literal["eval_dataset","golden_dataset"], artifact_dir: str) -> tuple[str, str]:
    DATA_PATH = ROOT_DIR / "data" / "evaluation"
    if artifact_type == "eval_dataset":
        artifact_path = (DATA_PATH / "eval_dataset" / params_config.evaluation_dataset.evaluation_dataset_filename).with_suffix(".json")
        return (artifact_path.as_posix(), artifact_dir)
    
    elif artifact_type == "golden_dataset":
        artifact_path = (DATA_PATH / "goldens" / params_config.golden_dataset.golden_dataset_filename).with_suffix(".json")
        return (artifact_path.as_posix(), artifact_dir)

def return_code_files():
    CODE_PATHS = ROOT_DIR / "src"
    
    app_code_path = CODE_PATHS / "app" /"rag_workflow.py"
    # clients_code_path = CODE_PATHS / "app" / "clients.py"
    # vector_store_code_path = CODE_PATHS / "app" / "vector_store.py"
    eval_data_path = CODE_PATHS / "data" / "generate_eval_dataset.py"
    evaluation_path = CODE_PATHS / "evals" / "application_evals" / "evaluate_rag_app.py"
    golden_dataset_path = CODE_PATHS / "data" / "generate_goldens.py"

    code_files = [app_code_path, eval_data_path, evaluation_path, golden_dataset_path]
    
    return [file.as_posix() for file in code_files]

# Fixed: double underscores added
if __name__ == "__main__": 
    mlflow.set_tracking_uri("http://127.0.0.1:5000/")
    mlflow.set_experiment("rag-app")
    
    logger = logging.getLogger(name="MLflow logger")
    handler = logging.StreamHandler()
    logger.addHandler(handler)
    logger.setLevel(INFO)
    
    # Fixed: Formatter syntax
    formatter = logging.Formatter(fmt="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    handler.setFormatter(fmt=formatter)

    all_params = params_config.model_dump()
    params_dict = flatten_params(all_params)

    mlflow.log_params(params_dict)
    logger.info("Parameters Logged")

    generate_evaluation_dataset()
    logger.info("Evaluation Dataset Created")

    evaluate_app()
    logger.info("Evaluation Completed")

    result_dir = ROOT_DIR / "reports" / "evaluation_results"
    report_dir = ROOT_DIR / "reports" / "evaluation_report"

    # Fixed: passing the pattern to the updated function
    results_file = (result_dir / get_latest_results(result_dir, "*.json")).as_posix() + ".json"
    report_file = (report_dir / get_latest_results(report_dir, "*.md")).as_posix() + ".md"

    mlflow.log_artifact(results_file, "results")
    mlflow.log_artifact(report_file, "reports")

    logger.info("Results and Reports logged")

    metrics = get_metrics_from_results(results_file)
    mlflow.log_metrics(metrics)
    logger.info("Metrics logged")

    system_prompt = import_system_prompt(params_config.rag_app.prompt_label)
    # Cast to string in case langfuse returns an object instead of pure text
    mlflow.log_text(str(system_prompt), artifact_file="system_prompt.txt")
    logger.info("System prompt logged")

    # Fixed: using artifact_dir instead of save_artifact_dir
    eval_dataset_artifact = artifact_name(artifact_type="eval_dataset", artifact_dir="eval_dataset")
    golden_dataset_artifact = artifact_name(artifact_type="golden_dataset", artifact_dir="golden_dataset")
    
    mlflow.log_artifact(eval_dataset_artifact[0], eval_dataset_artifact[1])
    mlflow.log_artifact(golden_dataset_artifact[0], golden_dataset_artifact[1])
    logger.info("Logged evaluation and golden datasets")

    code_files = return_code_files()
    for code_file in code_files:
        mlflow.log_artifact(code_file, "code")
    logger.info("Code files logged")