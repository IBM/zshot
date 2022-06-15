from typing import Optional, List

from pydantic import BaseModel


class Entity(BaseModel):
    name: str
    description: str
    vocabulary: Optional[List[str]] = None
