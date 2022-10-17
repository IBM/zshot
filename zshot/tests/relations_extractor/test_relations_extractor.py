from typing import Iterator

import spacy
from spacy.tokens.doc import Doc

from zshot import PipelineConfig
from zshot.relation_extractor.relations_extractor import RelationsExtractor
from zshot.relation_extractor.relation_extractor_zsrc import RelationsExtractorZSRC
from zshot.tests.config import EX_DOCS, EX_ENTITIES, EX_RELATIONS
from zshot.tests.linker.test_linker import DummyLinkerEnd2End


class DummyRelationsExtractor(RelationsExtractor):
    def extract_relations(self, docs: Iterator[Doc], batch_size=None):
        for doc in docs:
            for e in doc.ents:
                # TODO: Use Spacy relationships format: https://www.youtube.com/watch?v=8HL-Ap5_Axo
                # or follow https://github.com/jakelever/kindred
                doc._.relations += ["test"]


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
