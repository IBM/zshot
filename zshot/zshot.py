import logging
from enum import Enum
from typing import Dict, Optional, List, Union

from spacy.language import Language
from spacy.tokens import Doc
from spacy.util import registry as spacy_registry

from zshot.entity import Entity
from zshot.linker import Linker
from zshot.mentions_extractor import MentionsExtractor
from zshot.mentions_extractor import SpacyMentionsExtractor


@Language.factory("zshot", default_config={
    "entities": None,
    "mentions_extractor": None,
    "linker": None
})
def create_zshot_component(nlp: Language, name: str,
                           entities: Optional[Union[Dict[str, str], List[Entity]]],
                           mentions_extractor: Optional[Union[MentionsExtractor, str]],
                           linker: Optional[Union[Linker, str]]):
    return Zshot(nlp, entities, mentions_extractor, linker)


class Zshot:

    def __init__(self, nlp: Language,
                 entities,
                 mentions_extractor,
                 linker):
        self.nlp = nlp
        self.entities = entities
        self.mentions_extractor = mentions_extractor
        self.linker = linker
        self.setup()

    def setup(self):
        if isinstance(self.entities, list) and len(self.entities) > 0 and isinstance(self.entities[0], dict):
            self.entities = list(map(lambda e: Entity(**e), self.entities))
        elif isinstance(self.entities, dict):
            self.entities = [Entity(name=name, description=description) for name, description in self.entities.items()]

        # Load Mention Extractor if registered function ID is provided
        if isinstance(self.mentions_extractor, str):
            self.mentions_extractor = spacy_registry.get(registry_name='misc', func_name=self.mentions_extractor)()

        # Load Linker if registered function ID is provided
        if isinstance(self.linker, str):
            self.linker = spacy_registry.get(registry_name='misc', func_name=self.linker)()

        if type(self.mentions_extractor) != SpacyMentionsExtractor and "ner" in self.nlp.pipe_names:
            logging.warning("Disabling Spacy NER")
            self.nlp.disable_pipes("ner")

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
        docs = list(docs)
        self.extracts_mentions(docs, batch_size=batch_size)
        self.link_entities(docs, batch_size=batch_size)
        for doc in docs:
            yield doc

    def extracts_mentions(self, docs: List[Doc], batch_size=None):
        if self.mentions_extractor and not (self.linker is not None and self.linker.is_end2end):
            self.mentions_extractor.set_kg(self.entities)
            self.mentions_extractor.extract_mentions(docs, batch_size=batch_size)

    def link_entities(self, docs, batch_size=None):
        if self.linker:
            self.linker.set_kg(self.entities)
            self.linker.link(docs, batch_size=batch_size)
