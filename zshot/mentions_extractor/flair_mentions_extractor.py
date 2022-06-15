import pkgutil
from typing import List, Optional

import spacy
from spacy.tokens.doc import Doc

from zshot.mentions_extractor.mentions_extractor import MentionsExtractor


class FlairMentionsExtractor(MentionsExtractor):

    def __init__(self):
        if not pkgutil.find_loader("flair"):
            raise Exception("Flair module not installed. You need to install Flair for using this class."
                            "Install it with: pip install flair==0.11")
        from flair.models import SequenceTagger
        self.model = SequenceTagger.load("ner")

    def extract_mentions(self, docs: List[Doc], batch_size: Optional[int] = None):
        from flair.data import Sentence
        for doc in docs:
            sent = Sentence(str(doc), use_tokenizer=True)
            self.model.predict(sent)
            sent_mentions = sent.get_spans('ner')
            for mention in sent_mentions:
                doc._.mentions.append(doc.char_span(mention.start_position, mention.end_position))


@spacy.registry.misc(FlairMentionsExtractor.id())
def register_mention_extractor():
    return FlairMentionsExtractor()