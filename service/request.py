from typing import List, Union
from pydantic import BaseModel


class Data(BaseModel):
    name: str = None
    body: str = None



