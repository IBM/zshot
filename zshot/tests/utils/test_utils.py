import spacy

from zshot import PipelineConfig
from zshot.tests.config import EX_ENTITIES, EX_DOCS
from zshot.tests.linker.test_linker import DummyLinkerEnd2End
from zshot.tests.mentions_extractor.test_mention_extractor import DummyMentionsExtractor
from zshot.utils import download_file
from zshot.utils.alignment_utils import align_spans, AlignmentMode, filter_overlapping_spans
from zshot.utils.data_models import Span
from zshot.utils.displacy.displacy import ents_colors


def test_download():
    path = download_file("https://raw.githubusercontent.com/IBM/zshot/main/README.md", output_dir=".")
    assert path.is_file()


def test_colors():
    nlp = spacy.blank("en")
    nlp.add_pipe("zshot", config=PipelineConfig(
        mentions_extractor=DummyMentionsExtractor(),
        linker=DummyLinkerEnd2End(),
        entities=EX_ENTITIES), last=True)
    doc = nlp(EX_DOCS[1])
    colors = ents_colors(doc)
    assert len(colors) == len(set([s.label for s in doc._.spans]))


def test_alignment_expand():
    tokens = ["I", "am", "going"]
    spans = [Span(2, 4, label="SBJ"), Span(5, 6, label="LOC")]
    tokens_offset = [(0, 1), (2, 4), (5, 10)]
    alignment = align_spans(spans, tokens, tokens_offsets=tokens_offset, alignment_mode=AlignmentMode.expand)
    assert alignment[0] == []
    assert alignment[1] == [0]
    assert alignment[2] == [1]
    assert [spans[a[0]].label if a else "O" for a in alignment] == ["O", "SBJ", "LOC"]


def test_alignment_expand_join_by():
    tokens = ["I", "am", "going"]
    spans = [Span(2, 4, label="SBJ"), Span(5, 6, label="LOC")]
    alignment = align_spans(spans, tokens, join_by=" ", alignment_mode=AlignmentMode.expand)
    assert alignment[0] == []
    assert alignment[1] == [0]
    assert alignment[2] == [1]
    assert [spans[a[0]].label if a else "O" for a in alignment] == ["O", "SBJ", "LOC"]


def test_alignment_expand_2():
    tokens = ["I", "am", "going", "home"]
    spans = [Span(0, 1, label="SBJ"), Span(11, 15, label="LOC")]
    alignment = align_spans(spans, tokens, join_by=" ", alignment_mode=AlignmentMode.expand)
    assert alignment[0] == [0]
    assert alignment[1] == []
    assert alignment[2] == []
    assert alignment[3] == [1]
    assert [spans[a[0]].label if a else "O" for a in alignment] == ["SBJ", "O", "O", "LOC"]


def test_alignment_expand_3():
    """
    text: -- -- --
    s_0:  AA A          0.7
    s_1:  BB             0.5
    s_2:     CC CC     0.9
    s_3:        DD     0.3

    i_*  [0,1] [0,2] [2,3]
    """
    tokens = ["--", "--", "--"]
    spans = [Span(0, 4, label="A", score=0.7),
             Span(0, 2, label="B", score=0.5),
             Span(3, 8, label="C", score=0.9),
             Span(6, 8, label="D", score=0.4)]

    alignment = align_spans(spans, tokens, join_by=" ", alignment_mode=AlignmentMode.expand)

    assert alignment[0] == [0, 1]
    assert alignment[1] == [0, 2]
    assert alignment[2] == [2, 3]


def test_alignment_expand_4():
    """
    text: -- -- --
    s_0:  AA           0.7
    s_1:  BB           0.5
    s_2:     CC CC     0.9
    s_3:        DD     0.3

    i_*  [0,1] [0,2] [2,3]
    """
    tokens = ["--", "--", "--"]
    spans = [Span(0, 3, label="A", score=0.7),
             Span(0, 2, label="B", score=0.5),
             Span(3, 8, label="C", score=0.9),
             Span(6, 8, label="D", score=0.4)]

    alignment = align_spans(spans, tokens, join_by=" ", alignment_mode=AlignmentMode.contract)

    assert alignment[0] == [0, 1]
    assert alignment[1] == [2]
    assert alignment[2] == [2, 3]


def test_alignment_contract():
    tokens = ["I", "am", "going", "home"]
    spans = [Span(1, 3, label="SBJ"), Span(11, 15, label="LOC")]
    alignment = align_spans(spans, tokens, join_by=" ", alignment_mode=AlignmentMode.contract)
    assert alignment[0] == []
    assert alignment[1] == []
    assert alignment[2] == []
    assert alignment[3] == [1]
    assert [spans[a[0]].label if a else "O" for a in alignment] == ["O", "O", "O", "LOC"]


def test_alignment_expand_overlaps():
    """
    text: -- -- --
    s_0:  AA A          0.7
    s_1:  BB             0.5
    s_2:     CC CC     0.9
    s_3:        DD     0.3

    i_*  [0,1] [0,2] [2,3]
    """
    tokens = ["--", "--", "--"]
    spans = [Span(0, 4, label="A", score=0.7),
             Span(0, 2, label="B", score=0.5),
             Span(3, 8, label="C", score=0.9),
             Span(6, 8, label="D", score=0.4)]

    filtered_spans = filter_overlapping_spans(spans, tokens, join_by=" ", alignment_mode=AlignmentMode.expand)

    assert len(filtered_spans) == 2
    assert filtered_spans[0].start == 0 and filtered_spans[0].end == 2
    assert filtered_spans[0].label == "A"
    assert filtered_spans[1].start == 3 and filtered_spans[1].end == 8
    assert filtered_spans[1].label == "C"


def test_alignment_expand_overlaps_2():
    """
    text: -- -- --
    s_0:  AA           0.7
    s_1:  BB           0.5
    s_2:     CC CC     0.9
    s_3:        DD     0.3

    i_*  [0,1] [0,2] [2,3]
    """
    tokens = ["--", "--", "--"]
    spans = [Span(0, 2, label="A", score=0.7),
             Span(0, 2, label="B", score=0.5),
             Span(3, 8, label="C", score=0.9),
             Span(6, 8, label="D", score=0.4)]

    filtered_spans = filter_overlapping_spans(spans, tokens, join_by=" ", alignment_mode=AlignmentMode.expand)

    assert len(filtered_spans) == 2
    assert filtered_spans[0].start == 0 and filtered_spans[0].end == 2
    assert filtered_spans[0].label == "A"
    assert filtered_spans[1].start == 3 and filtered_spans[1].end == 8
    assert filtered_spans[1].label == "C"


def test_alignment_expand_overlaps_no_score():
    """
    text: -- -- --
    s_0:  AA           0.7
    s_1:  BB           0.5
    s_2:     CC CC     0.9
    s_3:        DD     0.3

    i_*  [0,1] [0,2] [2,3]
    """
    tokens = ["--", "--", "--"]
    spans = [Span(0, 2, label="A", score=0.7),
             Span(0, 2, label="B"),
             Span(3, 8, label="C"),
             Span(6, 8, label="D", score=0.4)]

    filtered_spans = filter_overlapping_spans(spans, tokens, join_by=" ", alignment_mode=AlignmentMode.expand)

    assert len(filtered_spans) == 2
    assert filtered_spans[0].start == 0 and filtered_spans[0].end == 2
    assert filtered_spans[0].label == "A"
    assert filtered_spans[1].start == 3 and filtered_spans[1].end == 8
    assert filtered_spans[1].label == "C"
