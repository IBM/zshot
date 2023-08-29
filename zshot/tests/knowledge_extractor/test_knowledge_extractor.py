import json
import os
import tempfile
from typing import Iterator, Optional, Union, List, Tuple

import spacy
from spacy.tokens.doc import Doc

from zshot import PipelineConfig
from zshot.knowledge_extractor import KnowledgeExtractor
from zshot.tests.config import EX_DOCS
from zshot.utils.data_models import Relation
from zshot.utils.data_models import Span
from zshot.utils.data_models.relation_span import RelationSpan


class DummyKnowledgeExtractor(KnowledgeExtractor):
    def predict(self, docs: Iterator[Doc], batch_size: Optional[Union[int, None]] = None) \
            -> List[List[Tuple[Span, RelationSpan, Span]]]:
        docs_preds = []
        for doc in docs:
            e1 = doc[0]
            e2 = doc[1]
            s1 = Span(e1.idx, e1.idx + len(e1.text), "subject")
            s2 = Span(e2.idx, e2.idx + len(e2.text), "object")
            preds = [(s1, RelationSpan(s1, s2, Relation(name="relation")), s2)]
            docs_preds.append(preds)
        return docs_preds


def test_dummy_knowledge_extractor():
    nlp = spacy.blank("en")
    config_zshot = PipelineConfig(
        knowledge_extractor=DummyKnowledgeExtractor(),
    )
    nlp.add_pipe("zshot", config=config_zshot, last=True)
    assert "zshot" in nlp.pipe_names
    doc = nlp(EX_DOCS[0])
    assert len(doc.ents) > 0
    assert len(doc._.spans) > 0
    assert len(doc._.relations) > 0


def test_dummy_relations_extractor_device():
    nlp = spacy.blank("en")
    device = 'cpu'
    config_zshot = PipelineConfig(
        knowledge_extractor=DummyKnowledgeExtractor(),
        device=device,
    )
    nlp.add_pipe("zshot", config=config_zshot, last=True)
    assert "zshot" in nlp.pipe_names
    assert nlp.get_pipe("zshot").device == device


def test_serialization_knowledge_extractor():
    nlp = spacy.blank("en")
    config_zshot = PipelineConfig(knowledge_extractor=DummyKnowledgeExtractor())
    nlp.add_pipe("zshot", config=config_zshot, last=True)
    assert "zshot" in nlp.pipe_names
    assert "ner" not in nlp.pipe_names
    pipes = [p for p in nlp.pipe_names if p != "zshot"]

    d = tempfile.TemporaryDirectory()
    nlp.to_disk(d.name, exclude=pipes)
    config_fn = os.path.join(d.name, "zshot", "config.cfg")
    assert os.path.exists(config_fn)
    with open(config_fn, "r") as f:
        config = json.load(f)
    assert "disable_default_ner" in config and config["disable_default_ner"]
    nlp2 = spacy.load(d.name)
    assert "zshot" in nlp2.pipe_names
    assert isinstance(nlp2.get_pipe("zshot").knowledge_extractor, DummyKnowledgeExtractor)
