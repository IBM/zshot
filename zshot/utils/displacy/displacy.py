import warnings
from typing import Dict, Union, Iterable, Any

from spacy import displacy as s_displacy
from spacy.errors import Warnings
from spacy.tokens import Doc
from spacy.util import is_in_jupyter

from zshot.utils.displacy.colors import light_color_from_label
from zshot.utils.displacy.relations_render import RelationsRenderer, parse_rels


def ents_colors(docs: Union[Iterable[Union[Doc]], Doc]):
    """
    Can be used to derive colors for entities in a Spacy document.
    A color for each entity type in generated, using the entity label hash
    :param docs: A list of Spacy document with entities
    :return: A colors dictionary containing a color for each entity type
    """

    if isinstance(docs, Doc):
        docs = [docs]
    labels = set([ent.label_ for doc in docs for ent in doc.ents])
    colors = dict([(ent, light_color_from_label(ent)) for ent in labels])
    return colors


class displacy:

    @staticmethod
    def render(docs: Union[Iterable[Union[Doc]], Doc], style: str = "dep", options: Dict = None, **kwargs) -> str:
        return displacy._call_displacy(docs, style, "render", options=options, **kwargs)

    @staticmethod
    def serve(docs: Union[Iterable[Union[Doc]], Doc], style: str = "dep", options: Dict = None, **kwargs):
        return displacy._call_displacy(docs, style, "serve", options=options, **kwargs)

    @staticmethod
    def _call_displacy(docs: Union[Iterable[Union[Doc]], Doc], style: str, method: str, options: Dict[str, Any] = {},
                       port: int = 5000, host: str = "0.0.0.0", page: bool = True, minify: bool = False,
                       **kwargs) -> str:
        if isinstance(docs, Doc):
            docs = [docs]
        if options is None:
            options = {}
        if style == "ent":
            options.update({'colors': ents_colors(docs)})
        if style == "rel":
            re_renderer = RelationsRenderer(options=options)
            parsed = [parse_rels(doc) for doc in docs]
            html = re_renderer.render(parsed, page=page, minify=minify)
            s_displacy._html["parsed"] = html
            if "serve" in method:
                from wsgiref import simple_server
                if is_in_jupyter():
                    warnings.warn(Warnings.W011)
                httpd = simple_server.make_server(host=host, port=port, app=s_displacy.app)
                print(f"\nUsing the '{style}' visualizer")
                print(f"Serving on http://{host}:{port} ...\n")
                try:
                    httpd.serve_forever()
                except KeyboardInterrupt:
                    print(f"Shutting down server on port {port}.")
                finally:
                    httpd.server_close()
            return html

        disp = getattr(s_displacy, method)
        return disp(docs, style=style, options=options, **kwargs)
