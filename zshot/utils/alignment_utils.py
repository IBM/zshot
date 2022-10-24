from enum import Enum
from itertools import accumulate
from operator import attrgetter
from typing import List, Union, Dict, Tuple

from spacy.tokens import Doc

from zshot.utils.data_models import Span


class AlignmentMode(str, Enum):
    expand = 'expand'
    contract = 'contract'


def spacy_token_offsets(doc: Doc) -> List[Tuple[int, int]]:
    return [(t.idx, t.idx + len(t.text)) for t in doc]


def align_spans(spans: List[Span], tokens: List[str], tokens_offsets: List[Tuple[int, int]] = None,
                join_by: str = None, alignment_mode: AlignmentMode = AlignmentMode.expand,
                return_dict=False) -> Union[Dict, List[List[int]]]:
    """
    Align spans to a given list of tokens
    :param spans: the list of spans
    :param tokens: the tokens
    :param tokens_offsets: Tokens offset, spans of the tokens.
    Either tokens_offsets of join_by must be provided to compute spans.
    :param join_by: string used to join tokens. Either tokens_offsets of join_by must be provided to compute spans.
    :param alignment_mode: "contract" (span of all tokens completely within the character span),
     "expand" (span of all tokens at least partially covered by the character span).
    :param return_dict: If true, return alignment and tokens_offsets as Dict
    :return: alignment list of list of int where len(tokens) == len(alignment) and each alignment list are the
    index of the matching span or [] if no match is detected
    """
    assert join_by is not None or tokens_offsets is not None, \
        "Either tokens_offsets of join_by must be provided to compute spans."
    if not tokens_offsets:
        tokens_map = list(accumulate(map(lambda t: len(t) + len(join_by), tokens)))
        tokens_offsets = list(zip([0] + tokens_map, map(lambda x: x - len(join_by), tokens_map)))
    alignments = [[] for _ in range(len(tokens))]
    for idt, (t_start, t_end) in enumerate(tokens_offsets):
        for ids, s in enumerate(spans):
            if (t_start <= s.start < t_end or t_start < s.end <= t_end) and alignment_mode == AlignmentMode.expand:
                alignments[idt].append(ids)
            if t_start >= s.start and t_end <= s.end and alignment_mode == AlignmentMode.contract:
                alignments[idt].append(ids)
            if t_start > s.start and t_end < s.end:
                alignments[idt].append(ids)
    if return_dict:
        return {
            'tokens_offsets': tokens_offsets,
            'alignments': alignments,
        }
    return alignments


def filter_overlapping_spans(spans: List[Span], tokens: List[str],
                             tokens_offsets: List[Tuple[int, int]] = None,
                             join_by: str = None,
                             alignment_mode: AlignmentMode = AlignmentMode.expand,
                             return_dict=False) -> Union[List[Span], Dict]:
    """
    :param spans: List of spans to align
    :param tokens: List of tokens
    :param tokens_offsets: Tokens offset, spans of the tokens
    :param join_by: string used to join tokens. Either tokens_offsets of join_by must be provided to compute spans.
    :param return_dict: If true, return filtered list of spans and partial results as Dict
    :param alignment_mode: "contract" (span of all tokens completely within the character span),
     "expand" (span of all tokens at least partially covered by the character span).
    :return: the filtered list of spans
    """
    align_dict = align_spans(spans, tokens, tokens_offsets, join_by=join_by,
                             alignment_mode=alignment_mode, return_dict=True)
    alignments = align_dict['alignments']
    tokens_offsets = align_dict['tokens_offsets']
    filtered_spans = [None] * len(alignments)
    bio_token = ['O'] * len(alignments)
    for idx, alignment in enumerate(alignments):
        t_spans = [spans[a] for a in alignment]
        if not t_spans:
            continue
        try:
            best_span = max(t_spans, key=attrgetter('score'))
        except TypeError:
            best_span = t_spans[0]
        if idx > 0 and filtered_spans[idx - 1] is not None and filtered_spans[idx - 1].label == best_span.label:
            filtered_spans[idx - 1].end = tokens_offsets[idx][1]
            bio_token[idx] = f"I-{filtered_spans[idx - 1].label}"
            filtered_spans[idx], filtered_spans[idx - 1] = filtered_spans[idx - 1], None
        else:
            filtered_spans[idx] = Span(start=tokens_offsets[idx][0], end=tokens_offsets[idx][1],
                                       label=best_span.label, kb_id=best_span.kb_id, score=best_span.score)
            bio_token[idx] = f"B-{filtered_spans[idx].label}"

    filtered_spans = list(filter(None, filtered_spans))
    if return_dict:
        return {
            'bio': bio_token,
            'filtered_spans': filtered_spans,
            'alignments': alignments,
            'tokens_offsets': tokens_offsets
        }
    return filtered_spans
