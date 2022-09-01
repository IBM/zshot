import pkgutil
from typing import Optional, Iterator

from spacy.tokens.doc import Doc

from zshot.utils.data_models import Span
from zshot.mentions_extractor.mentions_extractor import MentionsExtractor
from zshot.mentions_extractor.utils import ExtractorType


class MentionsExtractorFlair(MentionsExtractor):
    ALLOWED_CHUNKS = ("NP",)

    def __init__(self, extractor_type: Optional[ExtractorType] = ExtractorType.NER):
        if not pkgutil.find_loader("flair"):
            raise Exception("Flair module not installed. You need to install Flair for using this class."
                            "Install it with: pip install flair==0.11")

        super(MentionsExtractorFlair, self).__init__()

        self.extractor_type = extractor_type
        self.model = None

    def load_models(self):
        if self.model is None:
            from flair.models import SequenceTagger
            if self.extractor_type == ExtractorType.NER:
                self.model = SequenceTagger.load("ner")
            else:
                self.model = SequenceTagger.load("chunk")

    def predict_pos_mentions(self, docs: Iterator[Doc], batch_size: Optional[int] = None):
        from flair.data import Sentence
        sentences = [
            Sentence(str(doc), use_tokenizer=True) for doc in docs
        ]
        kwargs = {'mini_batch_size': batch_size} if batch_size else {}
        self.model.predict(sentences, **kwargs)

        spans = []
        for sent, doc in zip(sentences, docs):
            spans_tmp = []
            for i in range(len(sent.labels)):
                if sent.labels[i].value in self.ALLOWED_CHUNKS:
                    spans_tmp.append(Span(sent.labels[i].data_point.start_position,
                                          sent.labels[i].data_point.end_position))

            spans.append(spans_tmp)

        return spans

    def predict_ner_mentions(self, docs: Iterator[Doc], batch_size: Optional[int] = None):
        from flair.data import Sentence
        sentences = [
            Sentence(str(doc), use_tokenizer=True) for doc in docs
        ]
        kwargs = {'mini_batch_size': batch_size} if batch_size else {}
        self.model.predict(sentences, **kwargs)

        spans = []
        for sent, doc in zip(sentences, docs):
            sent_mentions = sent.get_spans('ner')
            spans_tmp = [
                Span(mention.start_position, mention.end_position, score=mention.score)
                for mention in sent_mentions
            ]
            spans.append(spans_tmp)

        return spans

    def predict(self, docs: Iterator[Doc], batch_size=None):
        self.load_models()
        if self.extractor_type == ExtractorType.NER:
            return self.predict_ner_mentions(docs, batch_size)
        else:
            return self.predict_pos_mentions(docs, batch_size)
