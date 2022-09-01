import zlib
from typing import Optional, List
from pydantic import BaseModel


class Entity(BaseModel):
    name: str
    description: str
    vocabulary: Optional[List[str]] = None

    def __hash__(self):
        self_repr = f"{self.__class__.__name__}.{str(self.__dict__)}"
        return zlib.crc32(self_repr.encode())
