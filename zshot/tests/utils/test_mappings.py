import logging

import pytest

from zshot.utils.mappings import spans_to_wikipedia, spans_to_dbpedia
from zshot.utils.data_models import Span

logger = logging.getLogger(__name__)


@pytest.mark.skip(reason="Too expensive to run on every commit")
def test_span_to_dbpedia():  # pragma: no cover
    s = Span(label="Surfing", start=0, end=10)
    db_links = spans_to_dbpedia([s])
    assert len(db_links) > 0
    assert db_links[0].startswith("http://dbpedia.org/resource")


@pytest.mark.skip(reason="Too expensive to run on every commit")
def test_span_to_wiki():  # pragma: no cover
    s = Span(label="Surfing", start=0, end=10)
    wiki_links = spans_to_wikipedia([s])
    assert len(wiki_links) > 0
    assert wiki_links[0].startswith("https://en.wikipedia.org/wiki?curid=")
