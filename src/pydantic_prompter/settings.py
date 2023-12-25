from pathlib import Path
from typing import List, Optional
from pydantic import BaseSettings, BaseModel, Field

_root_path = Path(__file__).resolve().parent


class TemplatePaths(BaseModel):
    anthropic: str = Field(
        default=_root_path.joinpath("prompt_templates", "anthropic.jinja").as_posix()
    )


class Settings(BaseSettings):
    openai_api_key: str = ""
    template_paths: TemplatePaths = TemplatePaths()
    aws_default_region: str = "us-east-1"
    aws_profile: Optional[str] = None
    aws_access_key_id: str = ""
    aws_secret_access_key: str = ""

    class Config:
        env_file = _root_path.joinpath(".env")
        env_nested_delimiter = "__"
