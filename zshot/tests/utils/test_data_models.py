import spacy

from zshot.tests.config import EX_DATASET_RELATIONS
from zshot.utils.data_models import Entity, Relation, Span
from zshot.utils.data_models.relation_span import RelationSpan


def test_span():
    # Full
    s = Span(start=0, end=10, label='E', score=1, kb_id='e1')
    assert type(s) is Span
    assert s.start == 0
    assert s.end == 10
    assert s.label == 'E'
    assert s.score == 1
    assert s.kb_id == 'e1'

    # No score/KB Id
    s = Span(start=165, end=187, label='Q5034838')
    assert type(s) is Span
    assert s.start == 165
    assert s.end == 187
    assert s.label == 'Q5034838'

    # Check hash
    assert hash(s) == 10737688

    # Check repr
    assert repr(s) == f"{s.label}, {s.start}, {s.end}, {s.score}"

    # From Dict
    s1 = Span.from_dict(EX_DATASET_RELATIONS['sentence_entities'][0][0])
    assert type(s1) is Span
    assert s1.start == 165
    assert s1.end == 187
    assert s1.label == 'Q5034838'

    # Check eq
    assert s == s1

    # From/To SpaCy Span
    nlp = spacy.blank('en')
    doc = nlp(EX_DATASET_RELATIONS['sentence_entities'][0][0]['sentence'])
    spacy_span = s.to_spacy_span(doc)
    assert type(spacy_span) == spacy.tokens.Span
    assert spacy_span.start == 26
    assert spacy_span.end == 29
    assert spacy_span.label_ == s.label

    s1 = Span.from_spacy_span(spacy_span)
    assert type(s1) is Span
    assert s1.start == 166
    assert s1.end == 187
    assert s1.label == 'Q5034838'


def test_entity():
    # Full
    e = Entity(name='E', description='Entity', vocabulary=['Vocab'])
    assert type(e) is Entity
    assert e.name == 'E'
    assert e.description == 'Entity'
    assert len(e.vocabulary) == 1 and e.vocabulary[0] == 'Vocab'

    # No description
    e = Entity(name='E', vocabulary=['Vocab'])
    assert type(e) is Entity
    assert e.name == 'E'
    assert len(e.vocabulary) == 1 and e.vocabulary[0] == 'Vocab'

    # No vocabulary
    e = Entity(name='E', description='Entity')
    assert type(e) is Entity
    assert e.name == 'E'
    assert e.description == 'Entity'

    # Check hash
    e = Entity(name='E')
    assert hash(e) == 3095248369


def test_relation_span():
    # Full
    s1 = Span.from_dict(EX_DATASET_RELATIONS['sentence_entities'][0][0])
    s2 = Span.from_dict(EX_DATASET_RELATIONS['sentence_entities'][0][1])
    rs = RelationSpan(start=s1, end=s2, relation=Relation(name=EX_DATASET_RELATIONS['labels'][0]), score=1, kb_id='P1')
    assert type(rs) == RelationSpan
    assert type(rs.start) == Span
    assert type(rs.end) == Span
    assert type(rs.relation) == Relation
    assert rs.score == 1
    assert rs.kb_id == 'P1'

    # No Score/KB Id
    rs = RelationSpan(start=s1, end=s2, relation=Relation(name=EX_DATASET_RELATIONS['labels'][0]))
    assert type(rs) == RelationSpan
    assert type(rs.start) == Span
    assert type(rs.end) == Span
    assert type(rs.relation) == Relation

    # Check hash
    assert hash(rs) == 1864423560

    # Check repr
    assert repr(rs) == f"{rs.relation.name}, {rs.start}, {rs.end}, {rs.score}"

    # Check eq
    rs2 = RelationSpan(end=s2, start=s1, relation=Relation(name=EX_DATASET_RELATIONS['labels'][0]))
    assert rs == rs2


def test_relation():
    # Full
    r = Relation(name='R', description='Relation')
    assert type(r) is Relation
    assert r.name == 'R'
    assert r.description == 'Relation'

    # No description
    r = Relation(name='R')
    assert type(r) is Relation
    assert r.name == 'R'

    # Check hash
    assert hash(r) == 3502422000
