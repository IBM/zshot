from abc import ABC, abstractmethod
from typing import Iterator, List

from spacy.tokens import Doc

from zshot.entity import Entity


class Linker(ABC):

    def __init__(self):
        self._entities = None

    @classmethod
    def id(cls) -> str:
        return f"zshot.{cls.__name__}.{Linker.__name__}" \
               f".{cls.version()}"

    def set_kg(self, entities: Iterator[Entity]):
        """
        Set entities that linker can use
        :param entities: The list of entities
        """
        self._entities = entities

    @property
    def entities(self) -> List[Entity]:
        return self._entities

    @abstractmethod
    def link(self, docs: Iterator[Doc], batch_size=None):
        """
        Perform the entity linking
        :param docs: A list of spacy Document
        :param batch_size: The batch size
        :return:
        """
        pass

    @property
    def is_end2end(self) -> bool:
        return False

    @staticmethod
    def version() -> str:
        return "v1"

