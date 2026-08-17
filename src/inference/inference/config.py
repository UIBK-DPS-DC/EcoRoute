import yaml
from pydantic import BaseModel


class ModelConfig(BaseModel):
    model_log_file: str = "app.log"
    model_logging_level: str = "info"
    heartbeat_interval_seconds: int = 10
    max_concurrent_queries: int = 16
    query_batch_size: int = 10
    jetstream_query_fetch_timeout_seconds: int = 1
    llm_request_timeout_seconds: float = 120.0
    max_output_tokens: int = 1024


class EnergyTrackerConfig(BaseModel):
    sample_interval: float = 0.01


class AppConfig(BaseModel):
    model: ModelConfig
    energy_tracker: EnergyTrackerConfig


def load_config():
    try:
        with open("/home/cc/code/config/inference/config.yml") as f:
            data = yaml.safe_load(f)
    except FileNotFoundError:
        with open("/config/config.yml") as f:
            data = yaml.safe_load(f)

    return AppConfig(
        model=ModelConfig(**data["model"]),
        energy_tracker=EnergyTrackerConfig(**data["energyTracker"]),
    )
