import functools
import os
import pathlib
import shutil
from urllib.request import urlopen
import requests
from spacy.tokens import Doc
from tqdm.auto import tqdm


def download_file(url, output_dir="."):
    """
    Utility for downloading a file
    :param url: the file url
    :param output_dir: the output dir
    :return:
    """
    filename = url.rsplit('/', 1)[1]
    path = pathlib.Path(os.path.join(output_dir, filename)).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    with requests.get(url, stream=True) as r:
        total_length = int(urlopen(url=url).info().get('Content-Length', 0))
        if path.exists() and os.path.getsize(path) == total_length:
            return
        r.raw.read = functools.partial(r.raw.read, decode_content=True)
        with tqdm.wrapattr(r.raw, "read", total=total_length, desc=f"Downloading {filename}") as raw:
            with path.open("wb") as output:
                shutil.copyfileobj(raw, output)


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
