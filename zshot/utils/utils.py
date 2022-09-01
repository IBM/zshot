import functools
import logging
import os
import pathlib
import shutil
from typing import List
from urllib.request import urlopen

import requests
from spacy.tokens import Doc
from tqdm.auto import tqdm

from spacy.util import filter_spans as spacy_filter_spans

from zshot.utils.data_models import Span


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
        logging.info(f"Downloading {url}")
        total_length = int(urlopen(url=url).info().get('Content-Length', 0))
        if path.exists() and os.path.getsize(path) == total_length:
            return
        r.raw.read = functools.partial(r.raw.read, decode_content=True)
        with tqdm.wrapattr(r.raw, "read", total=total_length, desc=f"Downloading {filename}") as raw:
            with path.open("wb") as output:
                shutil.copyfileobj(raw, output)


def filter_extended_spans(spans: List[Span], doc: Doc = None) -> List[Span]:
    """ If a token belongs to more than one entity, will return only the one with the higher confidence.

    :param spans: List of spans to filter
    :param doc: The Spacy Document
    :return: List of spans without overlap
    """
    if not all([s.score for s in spans]):
        spans = spacy_filter_spans([span.to_spacy_span(doc) for span in spans])
        spans = [Span.from_spacy_span(span) for span in spans]
        return spans

    # Find overlaps
    final_mentions = []
    to_check = []
    for i, span in enumerate(spans):
        if any([span in pending for pending in to_check]):
            continue

        range_ = set(range(span.start,
                           span.end))
        overlaps = []
        for mention in spans[i + 1:]:
            range_2 = set(range(mention.start,
                                mention.end))
            if range_ & range_2:
                overlaps.append(mention)
        if not overlaps:
            final_mentions.append(span)
        else:
            to_check += [(span, m) for m in overlaps]

    # Fix overlaps getting the one with higher score
    for mention, mention_2, in to_check:
        if mention_2.score > mention.score:
            final_mentions.append(mention_2)
        else:
            final_mentions.append(mention)

    return final_mentions
