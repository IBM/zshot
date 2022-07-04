import os
import pickle as pkl
import zlib
from abc import ABC, abstractmethod
from pydoc import Doc
from typing import List, Iterator

from spacy.util import ensure_path

from zshot.entity import Entity


class MentionsExtractor(ABC):

    def __init__(self):
        self._entities = None

    def set_kg(self, entities: Iterator[Entity]):
        """
        Set entities that mention extractor can use
        :param entities: The list of entities
        """
        self._entities = entities

    @property
    def entities(self) -> List[Entity]:
        return self._entities

    def load_models(self):
        """
        Load the model
        :return:
        """
        pass

    @abstractmethod
    def extract_mentions(self, docs: Iterator[Doc], batch_size=None):
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

    @property
    def require_existing_ner(self) -> bool:
        return False

    @staticmethod
    def _get_serialize_file(path):
        return os.path.join(path, "mentions_extractor.pkl")

    @staticmethod
    def _get_config_file(path):
        path = os.path.join(path, "mentions_extractor.json")
        path = ensure_path(path)
        return path

    @classmethod
    def from_disk(cls, path, exclude=()):
        serialize_file = cls._get_serialize_file(path)
        with open(serialize_file, "rb") as f:
            return pkl.load(f)

    def to_disk(self, path):
        serialize_file = self._get_serialize_file(path)
        with open(serialize_file, "wb") as f:
            return pkl.dump(self, f)

    def __hash__(self):
        self_repr = f"{self.__class__.__name__}.{self.version()}.{str(self.__dict__)}"
        return zlib.crc32(self_repr.encode())
