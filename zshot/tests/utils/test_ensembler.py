from zshot.utils.data_models import Span
from zshot.utils.ensembler import Ensembler


def test_ensemble_max():
    ensembler = Ensembler(num_voters=2)
    assert ensembler.ensemble_max([
        Span(start=0, end=6, label='fruits', score=0.9),
        Span(start=0, end=6, label='NEG', score=0.1)
    ]) == Span(start=0, end=6, label='fruits', score=0.45)
    assert ensembler.ensemble_max([
        Span(start=7, end=9, label='fruits', score=0.1),
        Span(start=7, end=9, label='NEG', score=0.8),
    ]) == Span(start=7, end=9, label='NEG', score=0.4)
    assert ensembler.ensemble_max([
        Span(start=10, end=17, label='fruits', score=0.8),
        Span(start=10, end=17, label='NEG', score=0.2),
    ]) == Span(start=10, end=17, label='fruits', score=0.4)
    assert ensembler.ensemble_max([
        Span(start=40, end=41, label='fruits', score=0.4),
        Span(start=40, end=41, label='NEG', score=0.6)
    ]) == Span(start=40, end=41, label='NEG', score=0.3)


def test_ensemble_count():
    ensembler = Ensembler(num_voters=3)
    assert ensembler.ensemble_count([
        Span(start=0, end=6, label='fruits', score=0.9),
        Span(start=0, end=6, label='fruits', score=0.9),
        Span(start=0, end=6, label='NEG', score=0.1)
    ]) == Span(start=0, end=6, label='fruits', score=2 / 3)
    assert ensembler.ensemble_count([
        Span(start=7, end=9, label='fruits', score=0.1),
        Span(start=7, end=9, label='NEG', score=0.8),
        Span(start=7, end=9, label='NEG', score=0.8)
    ]) == Span(start=7, end=9, label='NEG', score=2 / 3)
    assert ensembler.ensemble_count([
        Span(start=10, end=17, label='fruits', score=0.8),
        Span(start=10, end=17, label='fruits', score=0.8),
        Span(start=10, end=17, label='NEG', score=0.2)
    ]) == Span(start=10, end=17, label='fruits', score=2 / 3)
    assert ensembler.ensemble_count([
        Span(start=40, end=41, label='fruits', score=0.4),
        Span(start=40, end=41, label='NEG', score=0.6),
        Span(start=40, end=41, label='NEG', score=0.6)
    ]) == Span(start=40, end=41, label='NEG', score=2 / 3)


def test_select_best():
    ensembler = Ensembler(num_voters=3)
    assert ensembler.select_best({
        'fruits': 0.3,
        'NEG': 0.03
    }) == (0.3, 'fruits')
    assert ensembler.select_best({
        'fruits': 0.1,
        'NEG': 0.3
    }) == (0.3, 'NEG')


def test_inclusive():
    ensembler = Ensembler(num_voters=3)
    spans = [
        Span(start=40, end=41, label='fruits', score=0.4),
        Span(start=40, end=41, label='NEG', score=0.6),
        Span(start=40, end=42, label='NEG', score=0.6)
    ]
    assert ensembler.inclusive(spans) == [
        Span(start=40, end=42, label='NEG', score=0.6)
    ]
