from abc import ABC, abstractmethod
from typing import Iterator, List
import os
import zlib
import pickle as pkl

from spacy.util import ensure_path
from spacy.tokens import Doc

from zshot.entity import Entity


class Linker(ABC):

    def __init__(self):
        self._entities = None
        self._is_end2end = False

    def set_kg(self, entities: Iterator[Entity]):
        """
        Set entities that linker can use
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
        return self._is_end2end

    @is_end2end.setter
    def is_end2end(self, value):
        self._is_end2end = value

    @staticmethod
    def version() -> str:
        return "v1"

    @staticmethod
    def _get_serialize_file(path):
        return os.path.join(path, "linker.pkl")

    @staticmethod
    def _get_config_file(path):
        path = os.path.join(path, "linker.json")
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
