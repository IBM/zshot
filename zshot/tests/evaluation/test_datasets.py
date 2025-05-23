import shutil
from pathlib import Path

import pytest

from zshot.evaluation import load_ontonotes_zs, load_medmentions_zs, load_few_rel_zs, load_pile_ner_biomed_zs
from zshot.evaluation.dataset.dataset import create_dataset
from zshot.utils.data_models import Entity

ENTITIES = [
    Entity(name="FAC", description="A facility"),
    Entity(name="LOC", description="A location"),
]


@pytest.fixture(scope="module", autouse=True)
def teardown():
    yield True
    shutil.rmtree(f"{Path.home()}/.cache/huggingface", ignore_errors=True)
    shutil.rmtree(f"{Path.home()}/.cache/zshot", ignore_errors=True)


@pytest.mark.skip(reason="Too expensive to run on every commit")
def test_ontonotes_zs():
    dataset = load_ontonotes_zs()
    assert 'train' in dataset
    assert 'test' in dataset
    assert 'validation' in dataset
    assert dataset['train'].num_rows == 41475
    assert dataset['test'].num_rows == 426
    assert dataset['validation'].num_rows == 1358
    del dataset


@pytest.mark.skip(reason="Too expensive to run on every commit")
def test_ontonotes_zs_split():
    dataset = load_ontonotes_zs(split='test')
    assert dataset.num_rows == 426
    del dataset


@pytest.mark.skip(reason="Too expensive to run on every commit")
def test_ontonotes_zs_sub_split():
    dataset = load_ontonotes_zs(split='test[0:10]')
    assert dataset.num_rows > 0
    del dataset


@pytest.mark.skip(reason="Too expensive to run on every commit")
def test_medmentions_zs():
    dataset = load_medmentions_zs()
    assert 'train' in dataset
    assert 'test' in dataset
    assert 'validation' in dataset

    assert dataset['train'].num_rows == 26770
    assert dataset['test'].num_rows == 1048
    assert dataset['validation'].num_rows == 1289
    del dataset


@pytest.mark.skip(reason="Too expensive to run on every commit")
def test_medmentions_zs_split():
    dataset = load_medmentions_zs(split='test')
    assert dataset.num_rows == 1048
    del dataset


@pytest.mark.skip(reason="Too expensive to run on every commit")
def test_pile_bioner():
    dataset = load_pile_ner_biomed_zs()
    assert dataset.num_rows == 58861
    assert len(dataset.entities) == 3912
    del dataset


@pytest.mark.skip(reason="Too expensive to run on every commit")
def test_few_rel_zs():
    dataset = load_few_rel_zs()
    assert dataset.num_rows == 11200

    dataset = load_few_rel_zs("val_wiki[0:5]")
    assert dataset.num_rows == 5
    del dataset


def test_create_dataset():
    sentences = ["New York is beautiful", "New York is beautiful"]
    gt = [["B-FAC", "I-FAC", "O", "O"], ["B-FAC", "I-FAC", "O", "O"]]

    dataset = create_dataset(gt, sentences, ENTITIES)
    assert dataset.num_rows == len(sentences)
    assert dataset.entities == ENTITIES
    del dataset
