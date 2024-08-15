from typing import Any, Dict


import zlib

import spacy
from spacy.tokens import Doc


class Span:
    def __init__(self, start: int, end: int, label: str = None, score: float = None, kb_id: str = None):
        """  Class for handling Spans with scores

        :param start: Start char idx of the span
        :param end: End char idx of the span
        :param label: Label of the span (category it belongs to, e.g.: PER)
        :param score: Score of the prediction
        :param kb_id: ID to Knowledge base (e.g.: wikipedia)
        """
        self.start = start
        self.end = end
        self.label = label
        self.score = score
        self.kb_id = kb_id

    def __repr__(self) -> str:
        return f"{self.label}, {self.start}, {self.end}, {self.score}"

    def __hash__(self):
        return zlib.crc32(self.__repr__().encode())

    def __eq__(self, other: Any):
        return (type(other) is type(self)
                and self.start == other.start
                and self.end == other.end
                and self.label == other.label
                and self.score == other.score)

    def to_spacy_span(self, doc: Doc) -> spacy.tokens.Span:
        kwargs = {
            'alignment_mode': 'expand'
        }
        if self.kb_id:
            kwargs.update({'kb_id': self.kb_id})
        if self.label:
            kwargs.update({'label': self.label})

        return doc.char_span(self.start, self.end, **kwargs)

    @staticmethod
    def from_spacy_span(spacy_span: spacy.tokens.Span, score=None) -> "Span":
        return Span(spacy_span.start_char, spacy_span.end_char, spacy_span.label_, score=score,
                    kb_id=str(spacy_span.kb_id))

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "Span":
        start = d.get('start', None) if 'start' in d else d.get('start_char', None)
        end = d.get('end', None) if 'end' in d else d.get('end_char', None)
        label = d.get('label', None)
        score = d.get('score', None)
        kb_id = d.get('kb_id', '')
        if start is None:
            raise ValueError('One of [start, start_char] must be defined in dict.')
        if end is None:
            raise ValueError('One of [end, end_char] must be defined in dict.')
        if not label:
            raise ValueError('Label must be defined in dict.')

        return Span(start, end, label=label, score=score,
                    kb_id=str(kb_id))
