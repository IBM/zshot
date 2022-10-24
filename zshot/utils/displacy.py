from typing import Dict, Union, Iterable

from spacy import displacy as s_displacy
from spacy.tokens import Doc

from zshot.utils.alignment_utils import filter_overlapping_spans, spacy_token_offsets


def ents_colors(docs: Union[Iterable[Union[Doc]], Doc]):
    """
    Can be used to derive colors for entities in a Spacy document.
    A color for each entity type in generated, using the entity label hash
    :param docs: A list of Spacy document with entities
    :return: A colors dictionary containing a color for each entity type
    """

    def color_from_label(label: str):
        hash_s = hash(label)
        r = (hash_s & 0xFF0000) >> 16
        g = (hash_s & 0x00FF00) >> 8
        b = hash_s & 0x0000FF
        return '#%02x%02x%02x' % (r, g, b)

    if isinstance(docs, Doc):
        docs = [docs]
    labels = set([ent.label_ for doc in docs for ent in doc.ents])
    colors = dict([(ent, color_from_label(ent)) for ent in labels])
    return colors


class displacy:

    @staticmethod
    def render(docs: Union[Iterable[Union[Doc]], Doc], style: str = "dep", options: Dict = None, **kwargs) -> str:
        return displacy._call_displacy(docs, style, "render", options=options, **kwargs)

    @staticmethod
    def serve(docs: Union[Iterable[Union[Doc]], Doc], style: str = "dep", options: Dict = None, **kwargs):
        return displacy._call_displacy(docs, style, "serve", options=options, **kwargs)

    @staticmethod
    def relations_to_arcs(doc: Doc) -> Dict:
        filtered_spans = filter_overlapping_spans(doc._.spans, list(doc), tokens_offsets=spacy_token_offsets(doc))
        filtered_spans.sort(key=lambda x: x.start)

        tokens_span = []
        for idx, span in enumerate(filtered_spans):
            if idx == 0:
                if span.start > 0:
                    tokens_span.append((0, span.start, None))
            elif span.start > filtered_spans[idx - 1].end:
                tokens_span.append((filtered_spans[idx - 1].end, span.start, None))
            tokens_span.append((span.start, span.end, span))
        if filtered_spans[-1].end < len(doc.text):
            tokens_span.append((filtered_spans[-1].end, len(doc.text), None))

        words = []
        span_map = {}
        for idx, (start, end, span) in enumerate(tokens_span):
            words.append({'text': doc.text[start:end], 'tag': span.label if span else "", 'lemma': None})
            if span:
                span_map[hash(span)] = idx

        arcs = [{'start': span_map[hash(r.start)], 'end': span_map[hash(r.end)],
                 'label': r.relation.name, 'dir': 'right'} for r in doc._.relations]

        return {'words': words,
                'arcs': arcs,
                'settings': {'lang': 'en', 'direction': 'ltr'}}

    @staticmethod
    def _call_displacy(docs: Union[Iterable[Union[Doc]], Doc], style: str, method: str, options: Dict = None,
                       **kwargs) -> str:
        if isinstance(docs, Doc):
            docs = [docs]
        if options is None:
            options = {}
        if style == "ent":
            options.update({'colors': ents_colors(docs)})
        if style == "rel":
            parsed = [displacy.relations_to_arcs(doc) for doc in docs]
            style = "dep"
            kwargs.update({'manual': True})
            docs = parsed

        disp = getattr(s_displacy, method)
        return disp(docs, style=style, options=options, **kwargs)
