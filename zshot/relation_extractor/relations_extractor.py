import os
import pickle as pkl
import zlib
from abc import ABC, abstractmethod
from typing import List, Iterator, Optional, Union

from spacy.tokens import Doc
from spacy.util import ensure_path

from zshot.utils.data_models.relation import Relation
from zshot.utils.data_models.relation_span import RelationSpan


class RelationsExtractor(ABC):

    def __init__(self):
        self._relations = None

    def set_relations(self, relations: Iterator[Relation]):
        """
        Set relationships that the relations extractor can use
        :param relations: The list of relationship
        """
        self._relations = relations

    @property
    def relations(self) -> List[Relation]:
        return self._relations

    def load_models(self):
        """
        Load the model
        :return:
        """
        pass

    @abstractmethod
    def predict(self, docs: Iterator[Doc], batch_size: Optional[Union[int, None]] = None) -> List[List[RelationSpan]]:
        """
        Perform the relations extraction.
        :param docs: A list of spacy Document
        :param batch_size: The batch size
        :return: the predicted relations
        """
        pass

    def extract_relations(self, docs: Iterator[Doc], batch_size: Optional[Union[int, None]] = None):
        """
        Perform the relations extraction. Call the predict function and add the mentions to the Spacy Doc
        :param docs: A list of spacy Document
        :param batch_size: The batch size
        :return:
        """
        predicted_relations = self.predict(docs, batch_size)
        for d, preds in zip(docs, predicted_relations):
            d._.relations = preds

    @staticmethod
    def version() -> str:
        return "v1"

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
