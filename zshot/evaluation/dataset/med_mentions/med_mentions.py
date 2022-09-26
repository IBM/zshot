import gzip
import os
import shutil

from datasets import DatasetDict, Split

from zshot.config import MODELS_CACHE_PATH
from zshot.evaluation.dataset.dataset import DatasetWithEntities
from zshot.evaluation.dataset.med_mentions.entities import MEDMENTIONS_ENTITIES, MEDMENTIONS_SPLITS, \
    MEDMENTIONS_TYPE_INV
from zshot.evaluation.dataset.med_mentions.utils import preprocess_medmentions
from zshot.utils import download_file

LABELS = MEDMENTIONS_ENTITIES

FILES = [
    "corpus_pubtator.txt",
    "corpus_pubtator.txt.gz",
    "corpus_pubtator_pmids_all.txt",
    "corpus_pubtator_pmids_dev.txt",
    "corpus_pubtator_pmids_test.txt",
    "corpus_pubtator_pmids_train.txt"
]


def _unzip(file):
    with gzip.open(file, 'rb') as f_in:
        with open(file.replace(".gz", ""), 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)


def _download_raw_data(path):
    txt_files = [
        "https://raw.githubusercontent.com/chanzuckerberg/MedMentions/master/full/data/corpus_pubtator_pmids_all.txt",
        "https://raw.githubusercontent.com/chanzuckerberg/MedMentions/master/full/data/corpus_pubtator_pmids_dev.txt",
        "https://raw.githubusercontent.com/chanzuckerberg/MedMentions/master/full/data/corpus_pubtator_pmids_test.txt",
        "https://raw.githubusercontent.com/chanzuckerberg/MedMentions/master/full/data/corpus_pubtator_pmids_trng.txt"
    ]
    for file in txt_files:
        download_file(file, path)
    shutil.move(os.path.join(path, "corpus_pubtator_pmids_trng.txt"),
                os.path.join(path, "corpus_pubtator_pmids_train.txt"))
    gz_file = "https://raw.githubusercontent.com/chanzuckerberg/MedMentions/master/st21pv/data/corpus_pubtator.txt.gz"
    download_file(gz_file, path)
    zip_file = os.path.join(path, "corpus_pubtator.txt.gz")
    _unzip(zip_file)


def _delete_temporal_files(cache_path):
    for file in FILES:
        os.remove(os.path.join(cache_path, file))


def _create_split_dataset(data, split):
    dataset = DatasetWithEntities.from_dict(
        {
            "tokens": [[tok.word for tok in sentence] for sentence in data],
            "ner_tags": [[tok.label_id for tok in sentence] for sentence in data]
        },
        split=split,
        entities=list(
            filter(lambda ent: MEDMENTIONS_TYPE_INV[ent.name] in MEDMENTIONS_SPLITS[str(split)],
                   MEDMENTIONS_ENTITIES))
    )
    return dataset


def load_medmentions() -> DatasetDict[DatasetWithEntities]:
    _download_raw_data(MODELS_CACHE_PATH)
    train_sentences, dev_sentences, test_sentences = preprocess_medmentions(MODELS_CACHE_PATH)
    _delete_temporal_files(MODELS_CACHE_PATH)

    medmentions_zs = DatasetDict()
    for split, sentences in [(Split.TRAIN, train_sentences),
                             (Split.VALIDATION, dev_sentences),
                             (Split.TEST, test_sentences)]:
        medmentions_zs[split] = _create_split_dataset(sentences, split)

    return medmentions_zs
