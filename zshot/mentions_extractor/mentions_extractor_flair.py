import pkgutil
from typing import Optional, Iterator

from spacy.tokens.doc import Doc

from zshot.utils.data_models import Span
from zshot.mentions_extractor.mentions_extractor import MentionsExtractor
from zshot.mentions_extractor.utils import ExtractorType


class MentionsExtractorFlair(MentionsExtractor):
    """ Flair Mentions extractor """
    ALLOWED_CHUNKS = ("NP",)

    def __init__(self, extractor_type: Optional[ExtractorType] = ExtractorType.NER):
        """
        * Requires flair package to be installed *

        :param extractor_type: Type of extractor to get mentions. One of:
            - NER: to use Named Entity Recognition model to get the mentions
            - POS: to get the mentions based on the linguistics
        """
        if not pkgutil.find_loader("flair"):
            raise Exception("Flair module not installed. You need to install Flair for using this class."
                            "Install it with: pip install flair==0.11")

        super(MentionsExtractorFlair, self).__init__()

        self.extractor_type = extractor_type
        self.model = None

    def load_models(self):
        """ Load Flair model to perform the mentions extraction """
        if self.model is None:
            from flair.models import SequenceTagger
            if self.extractor_type == ExtractorType.NER:
                self.model = SequenceTagger.load("ner")
            else:
                self.model = SequenceTagger.load("chunk")

    def predict_pos_mentions(self, docs: Iterator[Doc], batch_size: Optional[int] = None):
        """ Predict mentions of docs using POS linguistics

        :param docs: Documents to get mentions of
        :param batch_size: Batch size to use
        :return: Spans of the mentions
        """
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
        """ Predict mentions of docs using NER model

        :param docs: Documents to get mentions of
        :param batch_size: Batch size to use
        :return: Spans of the mentions
        """
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
        """ Predict mentions in each document

        :param docs: Documents to get mentions of
        :param batch_size: Batch size to use
        :return: Spans of the mentions
        """
        self.load_models()
        if self.extractor_type == ExtractorType.NER:
            return self.predict_ner_mentions(docs, batch_size)
        else:
            return self.predict_pos_mentions(docs, batch_size)
