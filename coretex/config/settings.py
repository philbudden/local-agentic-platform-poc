"""Runtime configuration — settings loaded from environment variables or a .env file."""

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    ollama_base_url: str = "http://host.docker.internal:11434"
    classifier_model: str = "llama3.2:3b"
    worker_model: str = "llama3.2:3b"
    classifier_timeout: int = 60   # short: classifier generates ~30 JSON tokens
    worker_timeout: int = 300      # long: worker may generate up to 256 tokens on CPU
    max_tokens: int = 256
    ingress_port: int = 8000
    log_level: str = "INFO"
    debug_router: bool = False

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}


settings = Settings()
