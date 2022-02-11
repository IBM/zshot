from enum import Enum
from typing import Dict, Optional, List, Union

from spacy.language import Language
from spacy.tokens import Doc

from zshot.linker.linker_blink import Blink
from zshot.mentions_extractor.flair_mentions_extractor import FlairMentionsExtractor


class Linker(str, Enum):
    BLINK = "BLINK"
    NONE = "NONE"


class MentionsExtractor(str, Enum):
    FLAIR = "FLAIR"
    NONE = "NONE"


class Entity(Dict):
    def __init__(self, name: str, description: str, label: str = None, vocabulary: List[str] = None):
        super().__init__()
        self.name = name
        self.description = description
        self.label = label if label else name
        self.vocabulary = vocabulary


@Language.factory("zshot", default_config={"entities": None, "mentions_extractor": None, "linker": None})
def create_zshot_component(nlp: Language, name: str,
                           entities: Optional[Union[Dict[str, str], List[Entity]]],
                           mentions_extractor: Optional[MentionsExtractor],
                           linker: Optional[Linker]):
    return Zshot(nlp, entities, mentions_extractor, linker)


class Zshot:

    def __init__(self, nlp: Language,
                 entities: Optional[Union[Dict[str, str], List[Entity]]],
                 mentions_extractor: MentionsExtractor,
                 linker: Linker):
        if isinstance(entities, dict):
            entities = [Entity(name=name, description=description) for name, description in entities.items()]
        self.nlp = nlp
        self.entities = entities
        self.mentions_extractor = None
        self.linker = None
        if mentions_extractor == MentionsExtractor.FLAIR:
            self.mentions_extractor = FlairMentionsExtractor()
        if linker == Linker.BLINK:
            self.linker = Blink()
        if not Doc.has_extension("mentions"):
            Doc.set_extension("mentions", default=[])

    def __call__(self, doc: Doc) -> Doc:
        # Add the matched spans when doc is processed
        self.extracts_mentions([doc])
        self.link_entities([doc])
        return doc

    def pipe(self, docs: List[Doc], batch_size: int, **kwargs):
        """.
        docs: A sequence of spacy documents.
        YIELDS (Doc): A sequence of Doc objects, in order.
        """
        self.extracts_mentions(docs, batch_size=batch_size)
        self.link_entities(docs, batch_size=batch_size)
        for doc in docs:
            yield doc

    def extracts_mentions(self, docs: List[Doc], batch_size=None):
        if self.mentions_extractor:
            self.mentions_extractor.extract_mentions(docs, batch_size=batch_size)

    def link_entities(self, docs, batch_size=None):
        if self.linker:
            self.linker.link(docs, batch_size=batch_size)
