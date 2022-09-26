from typing import Optional, Iterator

from spacy.tokens.doc import Doc

from zshot.utils.data_models import Span
from zshot.mentions_extractor.mentions_extractor import MentionsExtractor
from zshot.mentions_extractor.utils import ExtractorType


class MentionsExtractorSpacy(MentionsExtractor):
    """ SpaCy mentions extractor """
    ALLOWED_POS = ("NOUN", "PROPN")
    ALLOWED_DEP = ("compound", "pobj", "dobj", "nsubj", "attr", "appos")
    COMPOUND_DEP = "compound"

    EXCLUDE_NER = ("CARDINAL", "DATE", "ORDINAL", "PERCENT", "QUANTITY", "TIME")

    def __init__(self, extractor_type: Optional[ExtractorType] = ExtractorType.NER):
        """
        :param extractor_type: Type of extractor to get mentions. One of:
            - NER: to use Named Entity Recognition model to get the mentions
            - POS: to get the mentions based on the linguistics
        """
        super(MentionsExtractorSpacy, self).__init__()
        self.extractor_type = extractor_type

    @property
    def require_existing_ner(self) -> bool:
        """ If the type of the extractor is NER the existing NER is required """
        return self.extractor_type == ExtractorType.NER

    def predict_pos_mentions(self, docs: Iterator[Doc], batch_size: Optional[int] = None):
        """ Predict mentions of docs using POS linguistics

        :param docs: Documents to get mentions of
        :param batch_size: Batch size to use
        :return: Spans of the mentions
        """
        spans = []
        for doc in docs:
            skip = -1
            spans_tmp = []
            for i, tok in enumerate(doc):
                if 0 < i < skip:
                    continue

                if tok.pos_ in self.ALLOWED_POS and tok.dep_ in self.ALLOWED_DEP:
                    if tok.dep_ == self.COMPOUND_DEP:
                        spans_tmp.append(Span(tok.idx, tok.head.idx + len(tok.head)))
                        skip = tok.head.i + 1
                    else:
                        spans_tmp.append(Span(tok.idx, tok.idx + len(tok)))
            spans.append(spans_tmp)

        return spans

    def predict_ner_mentions(self, docs: Iterator[Doc], batch_size: Optional[int] = None):
        """ Predict mentions of docs using NER model

        :param docs: Documents to get mentions of
        :param batch_size: Batch size to use
        :return: Spans of the mentions
        """
        spans = [
            [
                Span(ent.start_char, ent.end_char)
                for ent in doc.ents if ent.label_ not in self.EXCLUDE_NER
            ]
            for doc in docs
        ]
        for doc in docs:
            doc.ents = []

        return spans

    def predict(self, docs: Iterator[Doc], batch_size=None):
        """ Predict mentions of docs

        :param docs: Documents to get mentions of
        :param batch_size: Batch size to use
        :return: Spans of the mentions
        """
        if self.extractor_type == ExtractorType.NER:
            return self.predict_ner_mentions(docs, batch_size)
        else:
            return self.predict_pos_mentions(docs, batch_size)
