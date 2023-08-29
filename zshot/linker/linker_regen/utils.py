import pickle

from huggingface_hub import hf_hub_download

from zshot.config import MODELS_CACHE_PATH
from zshot.linker.linker_regen.trie import Trie

REPO_ID = "ibm/regen-disambiguation"
WIKIPEDIA_TRIE_FILE_NAME = "wikipedia_trie.pkl"
DBPEDIA_TRIE_FILE_NAME = "dbpedia_trie.pkl"


def create_input(sentence, max_length, start_delimiter, end_delimiter):
    sent_list = sentence.split(" ")
    if len(sent_list) < max_length:
        return sentence
    else:
        end_delimiter_index = sent_list.index(end_delimiter)
        start_delimiter_index = sent_list.index(start_delimiter)
        half_context = (max_length - (end_delimiter_index - start_delimiter_index)) // 2
        left_index = max(0, start_delimiter_index - half_context)
        right_index = min(len(sent_list),
                          end_delimiter_index + half_context + (half_context - (start_delimiter_index - left_index)))
        left_index = left_index - max(0, (half_context - (right_index - end_delimiter_index)))
        return " ".join(sent_list[left_index:right_index])


def load_wikipedia_trie() -> Trie:  # pragma: no cover
    """
    Load the wikipedia trie from the HB hub
    :return: The Wikipedia trie
    """
    wikipedia_trie_file = hf_hub_download(repo_id=REPO_ID,
                                          repo_type='model',
                                          filename=WIKIPEDIA_TRIE_FILE_NAME,
                                          cache_dir=MODELS_CACHE_PATH)
    with open(wikipedia_trie_file, "rb") as f:
        wikipedia_trie = pickle.load(f)
    return wikipedia_trie


def load_dbpedia_trie() -> Trie:  # pragma: no cover
    """
    Load the dbpedia trie from the HB hub
    :return: The DBpedia trie
    """
    dbpedia_trie_file = hf_hub_download(repo_id=REPO_ID,
                                        repo_type='model',
                                        filename=DBPEDIA_TRIE_FILE_NAME,
                                        cache_dir=MODELS_CACHE_PATH)
    with open(dbpedia_trie_file, "rb") as f:
        dbpedia_trie = pickle.load(f)
    return dbpedia_trie
