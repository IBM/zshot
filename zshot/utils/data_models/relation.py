from typing import Optional

from pydantic import BaseModel


class Relation(BaseModel):
    name: str
    description: Optional[str]

    def __hash__(self):
        return hash(self.__repr__())
