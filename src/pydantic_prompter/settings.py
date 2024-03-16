from pathlib import Path
from typing import Optional

from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict
from dotenv import find_dotenv

_root_path = Path(__file__).resolve().parent


class TemplatePaths(BaseModel):
    anthropic: str = Field(
        default=_root_path.joinpath(
            "prompt_templates", "{prompt_paths}", "anthropic.jinja"
        ).as_posix()
    )
    cohere: str = Field(
        default=_root_path.joinpath(
            "prompt_templates", "{prompt_paths}", "cohere.jinja"
        ).as_posix()
    )
    llama2: str = Field(
        default=_root_path.joinpath(
            "prompt_templates", "{prompt_paths}", "llama2.jinja"
        ).as_posix()
    )


class Settings(BaseSettings):
    openai_api_key: Optional[str] = None
    template_paths: TemplatePaths = TemplatePaths()
    aws_default_region: str = "us-east-1"
    aws_profile: Optional[str] = None
    aws_access_key_id: Optional[str] = None
    aws_secret_access_key: Optional[str] = None
    aws_session_token: Optional[str] = None
    cohere_key: Optional[str] = None
    model_config = SettingsConfigDict(
        env_file=find_dotenv(), env_nested_delimiter="__", extra="ignore"
    )
