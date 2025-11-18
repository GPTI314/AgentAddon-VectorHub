from pydantic_settings import BaseSettings, SettingsConfigDict
from functools import lru_cache

class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        extra="ignore"
    )

    ENV: str = "dev"
    SERVICE_PORT: int = 8090
    DEFAULT_DIM: int = 1536
    LOG_JSON: bool = True

@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
