from spacy.tokens.doc import Doc
from typing import List, Optional

from zshot.mentions_extractor.mentions_extractor import MentionsExtractor


class SpacyMentionsExtractor(MentionsExtractor):
    def __init__(self):
        pass

    def extract_mentions(self, docs: List[Doc], batch_size: Optional[int] = None):
        for doc in docs:
            sent_mentions = doc.ents
            for mention in sent_mentions:
                doc._.mentions.append(doc.char_span(mention.start_char, mention.end_char))

            doc.ents = []
