from pydoc import Doc
from typing import Dict

from spacy import displacy as s_displacy


def ents_colors(doc: Doc):
    """
    Can be used to derive colors for entities in a Spacy document.
    A color for each entity type in generated, using the entity label hash
    :param doc: A Spacy document with entities
    :return: A colors dictionary containing a color for each entity type
    """

    def color_from_label(label: str):
        hash_s = hash(label)
        r = (hash_s & 0xFF0000) >> 16
        g = (hash_s & 0x00FF00) >> 8
        b = hash_s & 0x0000FF
        return '#%02x%02x%02x' % (r, g, b)

    labels = set(ent.label_ for ent in doc.ents)
    colors = dict([(ent, color_from_label(ent)) for ent in labels])
    return colors


class displacy:

    @staticmethod
    def render(doc, options: Dict = None, **kwargs):
        if options:
            options['colors'] = ents_colors(doc)
        else:
            options = {'colors': ents_colors(doc)}
        s_displacy.render(doc, options=options, **kwargs)

    @staticmethod
    def serve(doc, options: Dict = None, **kwargs):
        if options:
            options['colors'] = ents_colors(doc)
        else:
            options = {'colors': ents_colors(doc)}
        s_displacy.serve(doc, options=options, **kwargs)
