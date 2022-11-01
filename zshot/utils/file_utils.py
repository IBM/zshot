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


def download_file(url, output_dir=".") -> pathlib.Path:
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
            return path
        r.raw.read = functools.partial(r.raw.read, decode_content=True)
        with tqdm.wrapattr(r.raw, "read", total=total_length, desc=f"Downloading {filename}") as raw:
            with path.open("wb") as output:
                shutil.copyfileobj(raw, output)
    return path


def filter_extended_spans(spans: List[Span], doc: Doc = None) -> List[Span]:
    """ If a token belongs to more than one entity, will return only the one with the higher confidence.

    :param spans: List of spans to filter
    :param doc: The Spacy Document
    :return: List of spans without overlap
    """
    spacy_spans = [span.to_spacy_span(doc) for span in spans]
    spans = [Span.from_spacy_span(spacy_span, score=span.score) for spacy_span, span in zip(spacy_spans, spans)]
    if not all([s.score for s in spans]):
        spans = spacy_filter_spans(spacy_spans)
        spans = [Span.from_spacy_span(span) for span in spans]
        return spans

    # Find overlaps
    rs = []
    spans_grouped = []
    for i, span in enumerate(spans):
        r = range(span.start, span.end + 1)
        group = [span]
        if any([set(r) & set(r_) for r_ in rs]):
            continue

        for span_ in spans[i:]:
            r_ = range(span_.start, span_.end + 1)
            if set(r) & set(r_):
                r = range(min(r[0], r_[0]), max(r[-1], r_[-1]))
                group.append(span_)

        rs.append(r)
        spans_grouped.append(group)

    # Fix overlaps getting the one with higher score
    final_mentions = []
    for group in spans_grouped:
        final_mentions.append(max(group, key=lambda x: x.score))

    return final_mentions
