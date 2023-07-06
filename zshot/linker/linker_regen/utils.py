import json
import pickle
from typing import Dict, List

from huggingface_hub import hf_hub_download

from zshot.linker.linker_regen.trie import Trie
from zshot.utils.data_models import Span

REPO_ID = "ibm/regen-disambiguation"
WIKIPEDIA_TRIE_FILE_NAME = "wikipedia_trie.pkl"
WIKIPEDIA_MAP = "wikipedia_map_id.json"
DBPEDIA_TRIE_FILE_NAME = "dbpedia_trie.pkl"
DBPEDIA_MAP = "dbpedia_map_id.json"


def create_input(sentence, max_length, start_delimiter, end_delimiter):
    sent_list = sentence.split(" ")
    if len(sent_list) < max_length:
        return sentence
    else:
        end_delimiter_index = sent_list.index(end_delimiter)
        start_delimiter_index = sent_list.index(start_delimiter)
        half_context = (max_length - (end_delimiter_index - start_delimiter_index)) // 2
        left_index = max(0, start_delimiter_index - half_context)
        right_index = min(len(sent_list), end_delimiter_index + half_context + (
                half_context - (start_delimiter_index - left_index)))
        left_index = left_index - max(0, (half_context - (right_index - end_delimiter_index)))
        return " ".join(sent_list[left_index:right_index])


def load_wikipedia_trie() -> Trie:
    """
    Load the wikipedia trie from the HB hub
    :return: The Wikipedia trie
    """
    wikipedia_trie_file = hf_hub_download(repo_id=REPO_ID,
                                          repo_type='model',
                                          filename=WIKIPEDIA_TRIE_FILE_NAME)
    with open(wikipedia_trie_file, "rb") as f:
        wikipedia_trie = pickle.load(f)
    return wikipedia_trie


def load_wikipedia_mapping() -> Dict[str, str]:
    """
    Load the wikipedia trie from the HB hub
    :return: The Wikipedia trie
    """
    wikipedia_map = hf_hub_download(repo_id=REPO_ID,
                                    repo_type='model',
                                    filename=WIKIPEDIA_MAP)
    with open(wikipedia_map, "r") as f:
        wikipedia_map = json.load(f)
    return wikipedia_map


def spans_to_wikipedia(spans: List[Span]) -> List[str]:
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


def load_dbpedia_trie() -> Trie:
    """
    Load the dbpedia trie from the HB hub
    :return: The DBpedia trie
    """
    dbpedia_trie_file = hf_hub_download(repo_id=REPO_ID,
                                        repo_type='model',
                                        filename=DBPEDIA_TRIE_FILE_NAME)
    with open(dbpedia_trie_file, "rb") as f:
        dbpedia_trie = pickle.load(f)
    return dbpedia_trie


def load_dbpedia_mapping() -> Dict[str, str]:
    """
    Load the dbpedia trie from the HB hub
    :return: The DBpedia trie
    """
    dbpedia_map = hf_hub_download(repo_id=REPO_ID,
                                  repo_type='model',
                                  filename=DBPEDIA_MAP)
    with open(dbpedia_map, "r") as f:
        dbpedia_map = json.load(f)
    return dbpedia_map


def spans_to_dbpedia(spans: List[Span]) -> List[str]:
    """
    Generate dbpedia link for spans
    :return: The list of generated links
    """
    dbpedia_map = load_dbpedia_mapping()
    links = [dbpedia_map[s.label] for s in spans if s.label in dbpedia_map]
    return links
