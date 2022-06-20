from typing import Optional, Iterator

from spacy.tokens.doc import Doc

from zshot.mentions_extractor.mentions_extractor import MentionsExtractor
from zshot.mentions_extractor.utils import ExtractorType


class SpacyMentionsExtractor(MentionsExtractor):

    ALLOWED_POS = ("NOUN", "PROPN")
    ALLOWED_DEP = ("compound", "pobj", "dobj", "nsubj", "attr", "appos")
    COMPOUND_DEP = "compound"

    EXCLUDE_NER = ("CARDINAL", "DATE", "ORDINAL", "PERCENT", "QUANTITY", "TIME")

    def __init__(self, extractor_type: Optional[ExtractorType] = ExtractorType.NER):
        super(SpacyMentionsExtractor, self).__init__()
        self.extractor_type = extractor_type

    @property
    def require_existing_ner(self) -> bool:
        return self.extractor_type == ExtractorType.NER

    def extract_pos_mentions(self, docs: Iterator[Doc], batch_size: Optional[int] = None):
        for doc in docs:
            skip = -1
            for i, tok in enumerate(doc):
                if 0 < i < skip:
                    continue

                if tok.pos_ in self.ALLOWED_POS and tok.dep_ in self.ALLOWED_DEP:
                    if tok.dep_ == self.COMPOUND_DEP:
                        doc._.mentions.append(doc.char_span(tok.idx, tok.head.idx + len(tok.head)))
                        skip = tok.head.i + 1
                    else:
                        doc._.mentions.append(doc.char_span(tok.idx, tok.idx + len(tok)))

    def extract_ner_mentions(self, docs: Iterator[Doc], batch_size: Optional[int] = None):
        for doc in docs:
            ents_filtered = [ent for ent in doc.ents if ent.label_ not in self.EXCLUDE_NER]
            for mention in ents_filtered:
                doc._.mentions.append(doc.char_span(mention.start_char, mention.end_char))
            doc.ents = []

    def extract_mentions(self, docs: Iterator[Doc], batch_size=None):
        if self.extractor_type == ExtractorType.NER:
            return self.extract_ner_mentions(docs, batch_size)
        else:
            return self.extract_pos_mentions(docs, batch_size)
