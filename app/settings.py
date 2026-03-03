"""Runtime configuration loaded from environment variables or a .env file."""

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    ollama_base_url: str = "http://ollama:11434"
    classifier_model: str = "llama3"
    worker_model: str = "llama3"
    request_timeout: int = 60
    max_tokens: int = 512
    ingress_port: int = 8000

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


settings = Settings()
