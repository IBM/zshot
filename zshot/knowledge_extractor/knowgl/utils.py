from itertools import groupby, product
from typing import List, Generator, Any, Dict, Tuple

from tokenizers import Tokenizer, Encoding

from zshot.utils.data_models import Span, Relation
from zshot.utils.data_models.relation_span import RelationSpan


def ranges(lst: List[int]) -> Generator(List[int]):
    """ Get groups made by consecutive numbers in the given list

    :param lst: List to get groups from
    :yield: Group of consecutive numbers
    """
    pos = (j - i for i, j in enumerate(lst))
    t = 0
    for i, els in groupby(pos):
        lst_ = len(list(els))
        el = lst[t]
        t += lst_
        yield list(range(el, el + lst_))


def find_sub_list(sl: List[Any], lst: List[Any]):
    """ Return init and end indexes of a sublist in a list

    :param sl: Sublist
    :param lst: List
    :return: List of tuples with the init and the end indexes
    """
    results = []
    sll = len(sl)
    for ind in (i for i, e in enumerate(lst) if e == sl[0]):
        if lst[ind:ind + sll] == sl:
            results.append((ind, ind + sll - 1))

    return results


def get_spans(mention: str, label: str,
              tokenizer: Tokenizer, encodings: Encoding,
              words_mapping: Dict[int, str], char_mapping: Dict[int, List[int]]) -> List[Span]:
    """ Get spans from a mention

    :param mention: Mention text to get Spans from
    :param label: Label to assign to the Spans
    :param tokenizer: Tokenizer used for tokenization
    :param encodings: Encodings result of the tokenization
    :param words_mapping: Mapping from words indexes to words
    :param char_mapping: Mapping from words indexes to char init/end indexes
    :return: List of Spans
    """
    spans = []

    # Find tokens in the list of encodings to create the spans
    tokens = tokenizer.encode(mention)
    results = find_sub_list(tokens[1:-1], encodings.ids)
    for result in results:
        init = encodings.token_to_chars(result[0])[0]
        end = encodings.token_to_chars(result[1])[-1]
        spans.append(Span(init, end, label))

    # With some tokenizers, the result might be different depending on the surroundings of the mention
    # In this case, get the words indexes to get the span char limits
    if not spans:
        words = mention.lower().split()
        words_idxs = [k for k, v in words_mapping.items() if v.lower() in words]
        valid_groups = [group for group in list(ranges(words_idxs)) if len(group) == len(words)]
        for group in valid_groups:
            init = char_mapping[group[0]][0]
            end = char_mapping[group[-1]][-1]
            spans.append(Span(init, end, label))
    return spans


def get_words_mappings(encodings: Encoding, text: str) -> Tuple[Dict[int, str], Dict[int, List[int]]]:
    """ Get words mappings from word index to word string and char span

    :param encodings: Encodings result of tokenization
    :param text: Text to get words mappings from
    :return: Mapping from words indexes to words and Mapping from words indexes to char init/end indexes
    """
    words_mapping = {}
    char_mapping = {}
    for token_idx in range(len(encodings.ids)):
        try:
            init, end = encodings.token_to_chars(token_idx)
            word_idx: int = encodings.token_to_word(token_idx)
            if word_idx not in words_mapping:
                words_mapping[word_idx] = text[init:end].lower()
                char_mapping[word_idx] = [init, end]
            else:
                words_mapping[word_idx] += text[init:end].lower()
                char_mapping[word_idx][1] = end
        except TypeError:
            pass

    return words_mapping, char_mapping


def get_triples(subject_spans: List[Span], relation: str, object_spans: List[Span]) \
        -> List[Tuple[Span, RelationSpan, Span]]:
    """ Get all possible triples from the spans

    :param subject_spans: List of spans for the subject
    :param relation: Relation name
    :param object_spans: List of spans for the object
    :return: List of triples (subject, relation, object)
    """
    triples = []
    relation = Relation(name=relation, description="")
    # As one word might be repeated in the text, we have to relate all of them
    for comb in product(subject_spans, object_spans):
        triples.append((comb[0],
                        RelationSpan(start=comb[0], end=comb[1], relation=relation),
                        comb[1]))

    return triples
