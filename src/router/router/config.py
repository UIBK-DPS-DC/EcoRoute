import yaml
from pydantic import BaseModel


class RouterConfig(BaseModel):
    model_ttl: int = 30
    router_heartbeat_interval_seconds: int = 15
    use_task_classification: bool = True
    router_log_file: str = "/data/db/app.log"
    router_logging_level: str = "info"
    query_timeout: float = 600.0


class RoutingConfig(BaseModel):
    mab_model_path: str | None = None
    mab_model_save_dir: str = "/data/mab"


class DatabaseConfig(BaseModel):
    file_path: str = "/data/db/"
    file_name: str = "metrics-{{ROUTER_ID}}.duckdb"


class TrainerConfig(BaseModel):
    training_interval_threshold_seconds: int = 10
    training_batch_size: int = 10
    normalizer: str = "mean"
    reward_output_quality_weight: float = 0.5
    reward_response_time_weight: float = 0.5
    reward_energy_consumption_weight: float = 0.5
    bounds_file_path: str = "/config/bounds.yml"


class TaskClassifierConfig(BaseModel):
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    dataset_path: str = "/data/hf_datasets/"
    temperature: float = 0.1


class AppConfig(BaseModel):
    router: RouterConfig
    routing: RoutingConfig
    database: DatabaseConfig
    trainer: TrainerConfig
    task_classifier: TaskClassifierConfig


def load_config():
    with open("/config/config.yml") as f:
        data = yaml.safe_load(f)

    return AppConfig(
        router=RouterConfig(**data["router"]),
        routing=RoutingConfig(**data["routing"]),
        database=DatabaseConfig(**data["database"]),
        trainer=TrainerConfig(**data["trainer"]),
        task_classifier=TaskClassifierConfig(**data["taskClassifier"]),
    )
