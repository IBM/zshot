from typing import Dict, Optional, List

from spacy.language import Language
from spacy.tokens import Doc


@Language.factory("zshot", default_config={"entities": None})
def create_zshot_component(nlp: Language, name: str, entities: Optional[Dict[str, str]]):
    return Zshot(nlp, entities)


class Zshot:
    def __init__(self, nlp: Language, entities: Optional[Dict[str, str]]):
        # Register custom extension on the Doc
        self.nlp = nlp
        self.entities = entities
        if not Doc.has_extension("acronyms"):
            Doc.set_extension("acronyms", default=[])

    def __call__(self, doc: Doc) -> Doc:
        # Add the matched spans when doc is processed
        doc._.acronyms.append("test")
        return doc

    def pipe(self, docs: List[Doc], batch_size: int, **kwargs):
        """.
        docs: A sequence of spacy documents.
        YIELDS (Doc): A sequence of Doc objects, in order.
        """
        self.extracts_mentions(docs)
        self.link_entities(docs)
        for doc in docs:
            yield self(doc)

    def extracts_mentions(self, docs: List[Doc]):
        # Extract and filter mentions
        pass

    def link_entities(self, docs):
        pass
