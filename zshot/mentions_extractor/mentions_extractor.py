import os
import pickle as pkl
import warnings

import zlib
from abc import ABC, abstractmethod
from spacy.tokens import Doc
from typing import List, Iterator

from spacy.util import ensure_path

from zshot.utils.data_models import Entity
from zshot.utils.data_models import Span


class MentionsExtractor(ABC):

    def __init__(self):
        self._mentions = None

    def set_kg(self, mentions: Iterator[Entity]):
        """
        Set entities that mention extractor can use
        :param mentions: The list of entities
        """
        self._mentions = mentions

    @property
    def mentions(self) -> List[Entity]:
        return self._mentions

    def load_models(self):
        """
        Load the model
        :return:
        """
        pass

    def extract_mentions(self, docs: Iterator[Doc], batch_size=None):
        """
        Perform the mentions extraction. Call the predict function and add the mentions to the Spacy Doc
        :param docs: A list of spacy Document
        :param batch_size: The batch size
        :return:
        """
        predictions_spans = self.predict(docs, batch_size)
        for doc, doc_preds in zip(docs, predictions_spans):
            for pred in doc_preds:
                try:
                    doc._.mentions += (pred,)
                except TypeError or ValueError:
                    warnings.warn("Entity couldn't be added.")

    @abstractmethod
    def predict(self, docs: Iterator[Doc], batch_size=None) -> List[List[Span]]:
        """
        Perform the mentions prediction
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
