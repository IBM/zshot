from typing import Iterator, Optional, Union, List

import spacy
from spacy.tokens.doc import Doc

from zshot import PipelineConfig, RelationsExtractor
from zshot.relation_extractor import RelationsExtractorZSRC
from zshot.tests.config import EX_DOCS, EX_ENTITIES, EX_RELATIONS
from zshot.tests.linker.test_linker import DummyLinkerEnd2End
from zshot.utils.data_models import Relation
from zshot.utils.data_models.relation_span import RelationSpan


class DummyRelationsExtractor(RelationsExtractor):
    def predict(self, docs: Iterator[Doc], batch_size: Optional[Union[int, None]] = None) -> List[List[RelationSpan]]:
        relations_pred = []
        for doc in docs:
            relations = []
            for span in doc._.spans:
                relations.append(RelationSpan(start=span, end=span,
                                              relation=Relation(name="rel", description="desc"), score=1))
            relations_pred.append(relations)
        return relations_pred


def test_dummy_relations_extractor_with_entities_config():
    nlp = spacy.blank("en")
    config_zshot = PipelineConfig(
        linker=DummyLinkerEnd2End(),
        relations_extractor=DummyRelationsExtractor(),
        entities=EX_ENTITIES,
        relations=EX_RELATIONS,
    )
    nlp.add_pipe("zshot", config=config_zshot, last=True)
    assert "zshot" in nlp.pipe_names
    doc = nlp(EX_DOCS[0])
    assert len(doc.ents) > 0
    assert len(doc._.relations) > 0


def test_zsrc_with_entities_config_dummy_annotator():
    nlp = spacy.blank("en")
    config_zshot = PipelineConfig(
        linker=DummyLinkerEnd2End(),
        relations_extractor=RelationsExtractorZSRC(),
        entities=EX_ENTITIES,
        relations=EX_RELATIONS,
    )
    nlp.add_pipe("zshot", config=config_zshot, last=True)
    assert "zshot" in nlp.pipe_names
    doc = nlp(EX_DOCS[0])
    assert len(doc.ents) >= 0
    assert (
        len(doc._.relations) == 0
    )  # only one entity does not allow for a relation to be extracted


# def test_zsrc_with_entities_config():
#     from zshot.linker import LinkerTARS

#     nlp = spacy.blank("en")
#     config_zshot = PipelineConfig(
#         linker=LinkerTARS(),
#         relations_extractor=RelationsExtractorZSRC(),
#         entities=EX_ENTITIES,
#         relations=EX_RELATIONS,
#     )
#     nlp.add_pipe("zshot", config=config_zshot, last=True)
#     assert "zshot" in nlp.pipe_names
#     doc = nlp(EX_DOCS[0])
#     assert len(doc.ents) > 0
#     assert len(doc._.relations) > 0
