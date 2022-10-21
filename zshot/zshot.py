import json
import logging
import os
from typing import Optional, List, Union, Iterator

from spacy.language import Language
from spacy.tokens import Doc
from spacy.util import registry as spacy_registry, ensure_path

from zshot.linker import Linker
from zshot.mentions_extractor import MentionsExtractor
from zshot.pipeline_config import PipelineConfig
from zshot.utils.data_models import Entity


@Language.factory("zshot", default_config={
    "mentions": None,
    "entities": None,
    "mentions_extractor": None,
    "linker": None,
    "disable_default_ner": True
})
def create_zshot_component(nlp: Language, name: str,
                           mentions: Optional[Union[List[Entity], str]],
                           entities: Optional[Union[List[Entity], str]],
                           mentions_extractor: Optional[Union[MentionsExtractor, str]],
                           linker: Optional[Union[Linker, str]],
                           disable_default_ner: Optional[bool] = True):
    return Zshot(nlp, mentions, entities, mentions_extractor, linker, disable_default_ner)


class Zshot:

    def __init__(self, nlp: Language,
                 mentions,
                 entities,
                 mentions_extractor,
                 linker,
                 disable_default_ner: Optional[bool] = True):
        self.nlp = nlp
        self.mentions = mentions
        self.entities = entities
        self.mentions_extractor = mentions_extractor
        self.linker = linker
        self.disable_default_ner = disable_default_ner
        self.setup()

    def setup(self):
        # Load Entities from registered function ID if provided
        if isinstance(self.mentions, str):
            self.mentions = spacy_registry.get(registry_name='misc', func_name=self.mentions)()
        if isinstance(self.mentions, list) and len(self.mentions) > 0 and isinstance(self.mentions[0], dict):
            self.mentions = list(map(lambda e: Entity(**e), self.mentions))
        if isinstance(self.entities, str):
            self.entities = spacy_registry.get(registry_name='misc', func_name=self.entities)()
        if isinstance(self.entities, list) and len(self.entities) > 0 and isinstance(self.entities[0], dict):
            self.entities = list(map(lambda e: Entity(**e), self.entities))

        # Load Mention Extractor from registered function ID if provided
        if isinstance(self.mentions_extractor, str):
            self.mentions_extractor = spacy_registry.get(registry_name='misc', func_name=self.mentions_extractor)()

        # Load Linker from registered function ID if provided
        if isinstance(self.linker, str):
            self.linker = spacy_registry.get(registry_name='misc', func_name=self.linker)()

        if self.mentions_extractor and self.mentions_extractor.require_existing_ner \
                and "ner" not in self.nlp.pipe_names:
            raise ValueError(f"The pipeline you are using does not contains a NER,"
                             f" but mentions extractor {self.mentions_extractor.__class__.__name__} rely on"
                             f"an existing NER")

        if self.mentions_extractor and (self.linker is not None and self.linker.is_end2end):
            logging.warning("Using linker end2end. Disabling mentions_extractor")
            self.mentions_extractor = None

        if self.disable_default_ner and \
                (not self.mentions_extractor or not self.mentions_extractor.require_existing_ner) \
                and "ner" in self.nlp.pipe_names:
            logging.warning("Disabling default NER")
            self.nlp.disable_pipes("ner")

        if not Doc.has_extension("mentions"):
            Doc.set_extension("mentions", default=[])

        if not Doc.has_extension("spans"):
            Doc.set_extension("spans", default=[])

    def __call__(self, doc: Doc) -> Doc:
        # Add the matched spans when doc is processed
        self.extracts_mentions([doc])
        self.link_entities([doc])
        return doc

    def pipe(self, docs: Iterator[Doc], batch_size: int, **kwargs):
        """.
        docs: A sequence of spacy documents.
        YIELDS (Doc): A sequence of Doc objects, in order.
        """
        docs = list(docs)
        self.extracts_mentions(docs, batch_size=batch_size)
        self.link_entities(docs, batch_size=batch_size)
        for doc in docs:
            yield doc

    def extracts_mentions(self, docs: Iterator[Doc], batch_size=None):
        if self.mentions_extractor and not (self.linker is not None and self.linker.is_end2end):
            self.mentions_extractor.set_kg(self.mentions)
            self.mentions_extractor.extract_mentions(docs, batch_size=batch_size)

    def link_entities(self, docs, batch_size=None):
        if self.linker:
            self.linker.set_kg(self.entities)
            self.linker.link(docs, batch_size=batch_size)

    def from_disk(self, path, exclude=()):
        with open(os.path.join(path, "config.cfg"), "r") as f:
            config = json.load(f)

        try:
            mentions_extractor = MentionsExtractor.from_disk(path)
        except FileNotFoundError:
            mentions_extractor = None
        else:
            self.mentions_extractor = mentions_extractor

        try:
            linker = Linker.from_disk(path)
        except FileNotFoundError:
            linker = None
        else:
            self.linker = linker

        PipelineConfig(mentions_extractor=mentions_extractor, linker=linker)

        self.disable_default_ner = config['disable_default_ner']
        self.setup()

    def to_disk(self, path, exclude=()):
        path = ensure_path(path)
        if not os.path.exists(path):
            os.makedirs(path, exist_ok=True)

        config = {
            "disable_default_ner": self.disable_default_ner
        }

        with open(os.path.join(path, "config.cfg"), "w") as f:
            json.dump(config, f)

        if self.mentions_extractor:
            self.mentions_extractor.to_disk(path)
        if self.linker:
            self.linker.to_disk(path)
