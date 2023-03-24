import random

from zshot.utils.data_models import Span


def sub_span_scoring(spans):
    for k in spans.keys():
        for p in spans.keys():
            if k[0] >= p[0] and k[1] <= p[1]:
                if k[0] > p[0] or k[1] < p[1]:
                    for s in spans[k]:
                        spans[p].append(Span(label=s.label, score=s.score, start=p[0], end=p[1]))


def sub_span_scoring_per_description(union_spans, spans):
    for k in union_spans.keys():
        for span in spans:
            labels = {}
            for p in span:
                if k[0] <= p.start and k[1] >= p.end:
                    if k[0] < p.start or k[1] > p.end:
                        if p.label not in labels:
                            labels[p.label] = p
                        elif labels[p.label].score < p.score:
                            labels[p.label] = p
            for p in labels.values():
                union_spans[k].append(Span(label=p.label, score=p.score, start=k[0], end=k[1]))


def normalize_group(group, require_length):
    group.extend(random.choices(group, k=require_length - len(group)))


def get_enhance_entities(entities):
    entities_groups = [[ent for ent in entities if ent.name == name] for name in set([ent.name for ent in entities])]
    max_length = max([len(group) for group in entities_groups])
    for group in entities_groups:
        normalize_group(group, max_length)

    return [list(g) for g in zip(*entities_groups)]
