import os
import traceback
import functools

from fastapi import APIRouter, FastAPI, Request, Depends, Security, Form
from fastapi.responses import FileResponse
from pathlib import Path
from service.response import ResponseData, CUSTOM_RESPONSES
from service.request import Data

from service.api.exceptions import NotAuthorizedError, FileNotFoundError
from service.log import app_logger

router = APIRouter(prefix=os.getenv('URL_PREFIX'))


@router.get(
    path="/health",
    tags=["Health"],
    response_model=str
)
async def health() -> str:
    app_logger.info(f"I am alive {os.getenv('SERVICE_NAME')} v{os.getenv('VERSION')}")
    return f"I am alive {os.getenv('SERVICE_NAME')} v{os.getenv('VERSION')}"


@router.post(
    path="/rate",
    tags=["RateAction"],
    response_model=ResponseData,
    responses={500: {"description": "Internal server error"},
               404: {"description": "File or data not found"}},
)
async def get_rate(
        request: Request,
        items: Data
) -> ResponseData:
    """
    Оценка моделью входных параметров
    :param request:
    :param items: данные для обработки
    :return: ResponseData
    """
    app_logger.info(items)
    status = 200
    # реализация функционала работы модели
    return ResponseData(status=status,
                        message=CUSTOM_RESPONSES.get(status),
                        data=None)


def add_views(app: FastAPI) -> None:
    app.include_router(router)
