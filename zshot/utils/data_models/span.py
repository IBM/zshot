import zlib

from spacy.tokens import Doc, Span


class Span:
    def __init__(self, start: int, end: int, label: str = None, score: float = None, kb_id: str = None):
        self.start = start
        self.end = end
        self.label = label
        self.score = score
        self.kb_id = kb_id

    def __repr__(self) -> str:
        return f"{self.label}, {self.start}, {self.end}, {self.score}"

    def __hash__(self):
        return zlib.crc32(self.__repr__().encode())

    def to_spacy_span(self, doc: Doc) -> Span:
        kwargs = {
            'alignment_mode': 'expand'
        }
        if self.kb_id:
            kwargs.update({'kb_id': self.kb_id})
        if self.label:
            kwargs.update({'label': self.label})

        return doc.char_span(self.start, self.end, **kwargs)

    @staticmethod
    def from_spacy_span(spacy_span: Span, score=None):
        return Span(spacy_span.start_char, spacy_span.end_char, spacy_span.label_, score=score,
                    kb_id=str(spacy_span.kb_id))
