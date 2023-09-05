import json
from typing import Dict, List

from huggingface_hub import hf_hub_download

from zshot.config import MODELS_CACHE_PATH
from zshot.utils.data_models import Span


REPO_ID = "ibm/regen-disambiguation"
WIKIPEDIA_MAP = "wikipedia_map_id.json"
DBPEDIA_MAP = "dbpedia_map_id.json"


def load_wikipedia_mapping() -> Dict[str, str]:  # pragma: no cover
    """
    Load the wikipedia trie from the HB hub
    :return: The Wikipedia trie
    """
    wikipedia_map = hf_hub_download(repo_id=REPO_ID,
                                    repo_type='model',
                                    filename=WIKIPEDIA_MAP,
                                    cache_dir=MODELS_CACHE_PATH)
    with open(wikipedia_map, "r") as f:
        wikipedia_map = json.load(f)
    return wikipedia_map


def spans_to_wikipedia(spans: List[Span]) -> List[str]:  # pragma: no cover
    """
    Generate wikipedia link for spans
    :return: The list of generated links
    """
    links = []
    wikipedia_map = load_wikipedia_mapping()
    for s in spans:
        if s.label in wikipedia_map:
            links.append(f"https://en.wikipedia.org/wiki?curid={wikipedia_map[s.label]}")
        else:
            links.append(None)
    return links


def load_dbpedia_mapping() -> Dict[str, str]:  # pragma: no cover
    """
    Load the dbpedia trie from the HB hub
    :return: The DBpedia trie
    """
    dbpedia_map = hf_hub_download(repo_id=REPO_ID,
                                  repo_type='model',
                                  filename=DBPEDIA_MAP,
                                  cache_dir=MODELS_CACHE_PATH)
    with open(dbpedia_map, "r") as f:
        dbpedia_map = json.load(f)
    return dbpedia_map


def spans_to_dbpedia(spans: List[Span]) -> List[str]:  # pragma: no cover
    """
    Generate dbpedia link for spans
    :return: The list of generated links
    """
    dbpedia_map = load_dbpedia_mapping()
    links = [dbpedia_map[s.label] for s in spans if s.label in dbpedia_map]
    return links
