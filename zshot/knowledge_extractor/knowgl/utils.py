from itertools import groupby, product

from zshot.utils.data_models import Span, Relation
from zshot.utils.data_models.relation_span import RelationSpan


def ranges(lst):
    pos = (j - i for i, j in enumerate(lst))
    t = 0
    for i, els in groupby(pos):
        lst_ = len(list(els))
        el = lst[t]
        t += lst_
        yield list(range(el, el + lst_))


def find_sub_list(sl, lst):
    results = []
    sll = len(sl)
    for ind in (i for i, e in enumerate(lst) if e == sl[0]):
        if lst[ind:ind + sll] == sl:
            results.append((ind, ind + sll - 1))

    return results


def get_spans(mention, label, tokenizer, encodings, words_mapping, char_mapping):
    spans = []

    tokens = tokenizer.encode(mention)
    results = find_sub_list(tokens[1:-1], encodings.ids)
    for result in results:
        init = encodings.token_to_chars(result[0])[0]
        end = encodings.token_to_chars(result[1])[-1]
        spans.append(Span(init, end, label))
    if not spans:
        words = mention.lower().split()
        words_idxs = [k for k, v in words_mapping.items() if v.lower() in words]
        valid_groups = [group for group in list(ranges(words_idxs)) if len(group) == len(words)]
        for group in valid_groups:
            init = char_mapping[group[0]][0]
            end = char_mapping[group[-1]][-1]
            spans.append(Span(init, end, label))
    return spans


def get_words_mappings(encodings, text):
    words_mapping = {}
    char_mapping = {}
    for token_idx in range(len(encodings.ids)):
        try:
            init, end = encodings.token_to_chars(token_idx)
            word_idx = encodings.token_to_word(token_idx)
            if word_idx not in words_mapping:
                words_mapping[word_idx] = text[init:end].lower()
                char_mapping[word_idx] = [init, end]
            else:
                words_mapping[word_idx] += text[init:end].lower()
                char_mapping[word_idx][1] = end
        except TypeError:
            pass

    return words_mapping, char_mapping


def get_triples(subject_spans, relation, object_spans):
    triples = []
    relation = Relation(name=relation, description="")
    for comb in product(subject_spans, object_spans):
        triples.append((comb[0],
                        RelationSpan(start=comb[0], end=comb[1], relation=relation),
                        comb[1]))

    return triples
