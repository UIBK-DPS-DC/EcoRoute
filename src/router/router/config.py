import yaml
from pydantic import BaseModel, model_validator
from typing import Literal


class RouterConfig(BaseModel):
    model_ttl: int = 30
    router_heartbeat_interval_seconds: int = 15
    use_task_classification: bool = True
    router_log_file: str = "/data/db/app.log"
    router_logging_level: str = "info"
    query_timeout: float = 600.0


class MABRoutingConfig(BaseModel):
    mab_options: str = "--epsilon 0.1"


class HeuristicRoutingConfig(BaseModel):
    heuristic_epsilon: float = 0.1


class KNNRoutingConfig(BaseModel):
    n_neighbors: int = 50
    distance_metric: str = "cosine"
    leaf_size: int = 30


class MLPRoutingConfig(BaseModel):
    hidden_layer_sizes: list[int] = (100, 100, 100)
    activation_function: str = "relu"
    learning_rate_method: str = "constant"
    learning_rate: float = 0.001


class RoutingConfig(BaseModel):
    algorithm: Literal["mab", "heuristic", "knn", "mlp"] = "mab"
    model_path: str | None = None
    model_save_dir: str = "/data/mab"
    training_data_path: str = "/data/routerbench/training_data.pkl"
    embedding_model: str = "all-MiniLM-L12-v2"
    mab: MABRoutingConfig | None = MABRoutingConfig()
    heuristic: HeuristicRoutingConfig | None = None
    knn: KNNRoutingConfig | None = None
    mlp: MLPRoutingConfig | None = None

    @model_validator(mode="after")
    def validate_selected_algorithm(self):
        config = getattr(self, self.algorithm)

        if config is None:
            raise ValueError(
                f"Configuration for selected algorithm "
                f"'{self.algorithm}' is missing"
            )

        return self


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
    dummy_output_quality: bool = False


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
