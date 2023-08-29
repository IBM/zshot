import os
import pickle as pkl
from abc import ABC, abstractmethod
from typing import List, Iterator, Optional, Union, Tuple

import torch
import zlib
from spacy.tokens import Doc

from zshot.utils.alignment_utils import filter_overlapping_spans, spacy_token_offsets
from zshot.utils.data_models import Span
from zshot.utils.data_models.relation_span import RelationSpan


class KnowledgeExtractor(ABC):

    def __init__(self, device: Optional[Union[str, torch.device]] = None):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if device is None else device

    def set_device(self, device: Union[str, torch.device]):
        """
        Set the device to use
        :param device:
        :return:
        """
        self.device = device

    def load_models(self):
        """
        Load the model
        :return:
        """
        pass

    @abstractmethod
    def predict(self, docs: Iterator[Doc], batch_size: Optional[Union[int, None]] = None) \
            -> List[List[Tuple[Span, RelationSpan, Span]]]:
        """
        Perform the knowledge extraction.
        :param docs: A list of spacy Document
        :param batch_size: The batch size
        :return: the predicted triples
        """
        pass

    def parse_triples(self, preds: List[Tuple[Span, RelationSpan, Span]]) -> Tuple[List[Span], List[RelationSpan]]:
        """ Parse the triples into lists of entities and relations

        :param preds: Predicted triples
        :return: Tuple with list of entities and list of relations
        """
        entities = []
        relations = []
        for triple in preds:
            entities.append(triple[0])
            entities.append(triple[2])
            relations.append(triple[1])

        return list(set(entities)), list(set(relations))

    def extract_knowledge(self, docs: Iterator[Doc], batch_size: Optional[Union[int, None]] = None):
        """
        Perform the relations extraction. Call the predict function and add the mentions to the Spacy Doc
        :param docs: A list of spacy Document
        :param batch_size: The batch size
        :return:
        """
        predicted_triples = self.predict(docs, batch_size)
        for d, preds in zip(docs, predicted_triples):
            entities, relations = self.parse_triples(preds)
            d._.relations = relations
            d._.spans = entities
            d.ents = map(lambda p: p.to_spacy_span(d), filter_overlapping_spans(entities, list(d),
                                                                                tokens_offsets=spacy_token_offsets(d)))

    @staticmethod
    def version() -> str:
        return "v1"

    @staticmethod
    def _get_serialize_file(path):
        return os.path.join(path, "knowledge_extractor.pkl")

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
