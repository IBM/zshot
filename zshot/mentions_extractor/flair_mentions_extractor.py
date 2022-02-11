import pkgutil
from pydoc import Doc
from typing import List

from zshot.mentions_extractor.mentions_extractor import MentionsExtractor


class FlairMentionsExtractor(MentionsExtractor):

    def __init__(self):
        if not pkgutil.find_loader("flair"):
            raise Exception("Flair module not installed. You need to install Flair for using this class."
                            "Install it with: pip install flair")
        from flair.models import SequenceTagger
        self.model = SequenceTagger.load("ner")

    def extract_mentions(self, docs: List[Doc], batch_size=None):
        from flair.data import Sentence
        for doc in docs:
            sent = Sentence(str(doc), use_tokenizer=True)
            self.model.predict(sent)
            sent_mentions = sent.to_dict(tag_type="ner")["entities"]
            for mention in sent_mentions:
                doc._.mentions.append(doc.char_span(mention['start_pos'], mention['end_pos']))
