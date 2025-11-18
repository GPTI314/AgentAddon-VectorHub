from pydantic import BaseSettings
from functools import lru_cache

class Settings(BaseSettings):
    ENV: str = "dev"
    SERVICE_PORT: int = 8090
    DEFAULT_DIM: int = 1536
    LOG_JSON: bool = True

    class Config:
        env_file = ".env"
        extra = "ignore"

@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
