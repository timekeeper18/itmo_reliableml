import asyncio
from concurrent.futures.thread import ThreadPoolExecutor
from typing import Any, Dict

import uvloop
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .exception_handlers import add_exception_handlers
from .middlewares import add_middlewares
from .views import add_views
from ..log import app_logger, setup_logging
from ..settings import ServiceConfig

__all__ = ("create_app",)


def setup_asyncio(thread_name_prefix: str) -> None:
    uvloop.install()

    loop = asyncio.get_event_loop()

    executor = ThreadPoolExecutor(thread_name_prefix=thread_name_prefix)
    loop.set_default_executor(executor)

    def handler(_, context: Dict[str, Any]) -> None:
        message = "Caught asyncio exception: {message}".format_map(context)
        app_logger.warning(message)

    loop.set_exception_handler(handler)


def create_app(config: ServiceConfig) -> FastAPI:
    setup_logging(config)
    setup_asyncio(thread_name_prefix=config.service_name)
    app = FastAPI(debug=False)
    origins = [
        "http://0.0.0.0:8000",
        "http://localhost",
        "http://localhost:8000",
    ]
    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.state.version = config.version
    add_views(app)
    add_middlewares(app)
    add_exception_handlers(app)

    return app
