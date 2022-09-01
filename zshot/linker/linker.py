import os
import pickle as pkl
import warnings
from abc import ABC, abstractmethod
from typing import Iterator, List, Optional, Union

import zlib
from spacy.tokens import Doc
from spacy.util import ensure_path

from zshot.utils.data_models import Entity, Span
from zshot.utils.utils import filter_extended_spans


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
    def predict(self, docs: Iterator[Doc], batch_size: Optional[Union[int, None]] = None) -> List[List[Span]]:
        """
        Perform the entity prediction
        :param docs: A list of spacy Document
        :param batch_size: The batch size
        :return: List Spans for each Document in docs
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

    def link(self, docs: Iterator[Doc], batch_size: Optional[Union[int, None]] = None):
        """
        Perform the entity linking. Call the predict function and add entities to the Spacy Docs
        :param docs: A list of spacy Document
        :param batch_size: The batch size
        :return:
        """
        predictions_spans = self.predict(docs, batch_size)
        for doc, doc_preds in zip(docs, predictions_spans):
            spans_without_overlap = filter_extended_spans(doc_preds, doc=doc)
            doc_pred_spans = [
                doc.char_span(
                    pred.start,
                    pred.end,
                    label=pred.label,
                    alignment_mode="expand",
                    kb_id=pred.kb_id
                ) if pred.kb_id else doc.char_span(
                    pred.start,
                    pred.end,
                    label=pred.label,
                    alignment_mode="expand"
                )
                for pred in spans_without_overlap
            ]
            for span in doc_pred_spans:
                try:
                    doc.ents += (span,)
                except TypeError or ValueError:
                    warnings.warn("Entity couldn't be added.")

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
