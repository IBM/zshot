from abc import ABC, abstractmethod
from pydoc import Doc
from typing import List, Iterator

from zshot.entity import Entity


class MentionsExtractor(ABC):

    def __init__(self):
        self._entities = None

    @classmethod
    def id(cls) -> str:
        return f"zshot.{cls.__name__}.{MentionsExtractor.__name__}" \
               f".{cls.version()}"

    def set_kg(self, entities: Iterator[Entity]):
        """
        Set entities that mention extractor can use
        :param entities: The list of entities
        """
        self._entities = entities

    @property
    def entities(self) -> List[Entity]:
        return self._entities

    @abstractmethod
    def extract_mentions(self, docs: List[Doc], batch_size=None):
        """
        Perform the mentions extraction
        :param docs: A list of spacy Document
        :param batch_size: The batch size
        :return:
        """
        pass

    @staticmethod
    def version() -> str:
        return "v1"
