import pkgutil
from typing import Optional, Iterator

from spacy.tokens.doc import Doc

from zshot.mentions_extractor.mentions_extractor import MentionsExtractor
from zshot.mentions_extractor.utils import ExtractorType


class FlairMentionsExtractor(MentionsExtractor):

    ALLOWED_CHUNKS = ("NP",)

    def __init__(self, extractor_type: Optional[ExtractorType] = ExtractorType.NER):
        if not pkgutil.find_loader("flair"):
            raise Exception("Flair module not installed. You need to install Flair for using this class."
                            "Install it with: pip install flair==0.11")

        super(FlairMentionsExtractor, self).__init__()

        self.extractor_type = extractor_type
        self.model = None

    def load_models(self):
        if self.model is None:
            from flair.models import SequenceTagger
            if self.extractor_type == ExtractorType.NER:
                self.model = SequenceTagger.load("ner")
            else:
                self.model = SequenceTagger.load("chunk")

    def extract_pos_mentions(self, docs: Iterator[Doc], batch_size: Optional[int] = None):
        from flair.data import Sentence
        for doc in docs:
            sent = Sentence(str(doc), use_tokenizer=True)
            self.model.predict(sent)
            for i in range(len(sent.labels)):
                if sent.labels[i].value in self.ALLOWED_CHUNKS:
                    doc._.mentions.append(doc.char_span(sent.labels[i].data_point.start_position,
                                                        sent.labels[i].data_point.end_position))

    def extract_ner_mentions(self, docs: Iterator[Doc], batch_size: Optional[int] = None):
        from flair.data import Sentence
        for doc in docs:
            sent = Sentence(str(doc), use_tokenizer=True)
            self.model.predict(sent)
            sent_mentions = sent.get_spans('ner')
            for mention in sent_mentions:
                doc._.mentions.append(doc.char_span(mention.start_position, mention.end_position))

    def extract_mentions(self, docs: Iterator[Doc], batch_size=None):
        self.load_models()
        if self.extractor_type == ExtractorType.NER:
            return self.extract_ner_mentions(docs, batch_size)
        else:
            return self.extract_pos_mentions(docs, batch_size)
