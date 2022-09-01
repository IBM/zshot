import spacy

from zshot import PipelineConfig
from zshot.tests.config import EX_DOCS, EX_ENTITIES, EX_RELATIONS
from zshot.tests.linker.test_linker import DummyLinkerEnd2End
from zshot.tests.relations_extractor.test_relations_extractor import DummyRelationsExtractor

nlp = spacy.blank("en")
config_zshot = PipelineConfig(linker=DummyLinkerEnd2End(),
                              relations_extractor=DummyRelationsExtractor(),
                              entities=EX_ENTITIES,
                              relations=EX_RELATIONS)
nlp.add_pipe("zshot", config=config_zshot, last=True)
assert "zshot" in nlp.pipe_names
doc = nlp(EX_DOCS[0])
assert len(doc.ents) > 0
assert len(doc._.relations) > 0
