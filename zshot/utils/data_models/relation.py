from typing import Optional

import zlib
from pydantic import BaseModel


class Relation(BaseModel):
    name: str
    description: Optional[str] = None

    def __hash__(self):
        self_repr = f"{self.__class__.__name__}.{str(self.__dict__)}"
        return zlib.crc32(self_repr.encode())
