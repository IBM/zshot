import zlib

from spacy.tokens import Span

from zshot.utils.data_models import Relation


class RelationSpan:
    def __init__(self, start: Span, end: Span, relation: Relation = None, score: float = None, kb_id: str = None):
        self.start = start
        self.end = end
        self.relation = relation
        self.score = score
        self.kb_id = kb_id

    def __repr__(self) -> str:
        return f"{self.relation.name}, {self.start}, {self.end}, {self.score}"

    def __hash__(self):
        return zlib.crc32(self.__repr__().encode())
