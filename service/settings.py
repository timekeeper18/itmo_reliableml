import os
from dotenv import load_dotenv
from functools import lru_cache
from pydantic_settings import BaseSettings
from pydantic import Extra
load_dotenv()


class Config(BaseSettings):
    class Config:
        extra = Extra.forbid
        case_sensitive = False


class LogConfig(Config):
    level: str = os.getenv('LOG_LEVEL')
    datetime_format: str = "%Y-%m-%d %H:%M:%S"

    class Config:
        case_sensitive = False


class ServiceConfig(Config):
    service_name: str = os.getenv('SERVICE_NAME')
    version: str = os.getenv('VERSION')
    log_config: LogConfig



@lru_cache()
def get_config() -> ServiceConfig:
    return ServiceConfig(
        log_config=LogConfig(),
    )
