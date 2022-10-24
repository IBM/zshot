from pydantic import BaseModel


class Relation(BaseModel):
    name: str
    description: str

    def __hash__(self):
        return hash(self.__repr__())
