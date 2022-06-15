from spacy.tokens.doc import Doc
from typing import List, Optional

from zshot.mentions_extractor.mentions_extractor import MentionsExtractor


class SpacyMentionsExtractor(MentionsExtractor):
    
    EXCLUDE_NER = ("CARDINAL", "DATE", "ORDINAL", "PERCENT", "QUANTITY", "TIME")

    def __init__(self):
        pass

    def extract_mentions(self, docs: List[Doc], batch_size: Optional[int] = None):
        for doc in docs:
            ents_filtered = [ent for ent in doc.ents if ent.label_ not in self.EXCLUDE_NER]
            for mention in ents_filtered:
                doc._.mentions.append(doc.char_span(mention.start_char, mention.end_char))
            doc.ents = []
