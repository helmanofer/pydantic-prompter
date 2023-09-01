from pathlib import Path

from pydantic import BaseSettings


class Settings(BaseSettings):
    openai_api_key: str = None

    class Config:
        env_file = Path(__file__).resolve().parent.joinpath(".env")
