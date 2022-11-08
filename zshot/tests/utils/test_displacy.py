import spacy
from spacy.tokens import Doc

from zshot import displacy, PipelineConfig
from zshot.tests.config import EX_DOCS, EX_ENTITIES
from zshot.tests.linker.test_linker import DummyLinkerEnd2End
from zshot.tests.mentions_extractor.test_mention_extractor import DummyMentionsExtractor
from zshot.utils.data_models import Span, Relation
from zshot.utils.data_models.relation_span import RelationSpan


def test_displacy_render():
    nlp = spacy.blank("en")
    nlp.add_pipe("zshot", config=PipelineConfig(
        mentions_extractor=DummyMentionsExtractor(),
        linker=DummyLinkerEnd2End(),
        entities=EX_ENTITIES), last=True)
    doc = nlp(EX_DOCS[1])
    assert len(doc.ents) > 0
    assert len(doc._.spans) > 0
    res = displacy.render(doc, style="ent", jupyter=False)
    assert res is not None


def test_displacy_rel_style():
    nlp = spacy.load("en_core_web_sm")
    doc = nlp(EX_DOCS[1])
    relations = [
        RelationSpan(start=Span(0, 43, "IBM", -0.007964816875755787), end=Span(45, 48, "IBM", -0.00017413603200111538),
                     relation=Relation(name="is_in", description="is inside"), score=0.7),
        RelationSpan(start=Span(0, 43, "IBM", -0.007964816875755787),
                     end=Span(127, 135, "New York", -2.3538105487823486),
                     relation=Relation(name="has_headquarters", description="has headquarters"), score=0.3)
    ]
    spans = [Span(0, 43, "IBM", -0.007964816875755787), Span(45, 48, "IBM", -0.00017413603200111538),
             Span(56, 64, "American", -5.8533525466918945), Span(119, 125, "Armonk", -2.1522278785705566),
             Span(127, 135, "New York", -2.3538105487823486)]
    if not Doc.has_extension("spans"):
        Doc.set_extension("spans", default=[])
    if not Doc.has_extension("relations"):
        Doc.set_extension("relations", default=[])
    doc._.relations = relations
    doc._.spans = spans
    html = displacy.render(doc, style="rel")
    assert html is not None
    assert "IBM" in html
    assert "American" in html
    assert "New York" in html
    assert "is_in" in html
    assert "has_headquarters" in html
    assert "displacy-token" in html
    assert "displacy-tag" in html
    assert "displacy-arrow" in html


def test_displacy_rel_compact_style():
    nlp = spacy.load("en_core_web_sm")
    doc = nlp(EX_DOCS[1])
    relations = [
        RelationSpan(start=Span(45, 48, "IBM", -0.00017413603200111538), end=Span(0, 43, "IBM", -0.007964816875755787),
                     relation=Relation(name="is_in", description="is inside"), score=0.7),
        RelationSpan(start=Span(0, 43, "IBM", -0.007964816875755787),
                     end=Span(127, 135, "New York", -2.3538105487823486),
                     relation=Relation(name="has_headquarters", description="has headquarters"), score=0.3)
    ]
    spans = [Span(0, 43, "IBM", -0.007964816875755787), Span(45, 48, "IBM", -0.00017413603200111538),
             Span(56, 64, "American", -5.8533525466918945), Span(119, 125, "Armonk", -2.1522278785705566),
             Span(127, 135, "New York", -2.3538105487823486)]
    if not Doc.has_extension("spans"):
        Doc.set_extension("spans", default=[])
    if not Doc.has_extension("relations"):
        Doc.set_extension("relations", default=[])
    doc._.relations = relations
    doc._.spans = spans
    html = displacy.render(doc, style="rel", options={"compact": True})
    assert html is not None
    assert "IBM" in html
    assert "American" in html
    assert "New York" in html
    assert "is_in" in html
    assert "has_headquarters" in html
    assert "displacy-token" in html
    assert "displacy-tag" in html
    assert "displacy-arrow" in html
