from typing import Dict, Optional, List, Union
from spacy.language import Language
from spacy.tokens import Doc


class Entity(Dict):
    def __init__(self, name: str, description: str, label: str = None, vocabulary: List[str] = None):
        super().__init__()
        self.name = name
        self.description = description
        self.label = label if label else name
        self.vocabulary = vocabulary


@Language.factory("zshot", default_config={"entities": None})
def create_zshot_component(nlp: Language, name: str, entities: Optional[Union[Dict[str, str], List[Entity]]]):
    return Zshot(nlp, entities)


class Zshot:

    def __init__(self, nlp: Language, entities: Optional[Union[Dict[str, str], List[Entity]]]):
        # Register custom extension on the Doc
        if isinstance(entities, dict):
            entities = [Entity(name=name, description=description) for name, description in entities.items()]
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
