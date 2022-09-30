import shutil
from pathlib import Path
import pytest
from zshot.evaluation import load_ontonotes, load_medmentions


@pytest.fixture(scope="module", autouse=True)
def teardown():
    yield True
    shutil.rmtree(f"{Path.home()}/.cache/huggingface", ignore_errors=True)
    shutil.rmtree(f"{Path.home()}/.cache/zshot", ignore_errors=True)


def test_ontonotes():
    dataset = load_ontonotes()
    assert 'train' in dataset
    assert 'test' in dataset
    assert 'validation' in dataset
    assert dataset['train'].num_rows == 41475
    assert dataset['test'].num_rows == 426
    assert dataset['validation'].num_rows == 1358


def test_medmentions():
    dataset = load_medmentions()
    assert 'train' in dataset
    assert 'test' in dataset
    assert 'validation' in dataset
    assert dataset['train'].num_rows == 30923
    assert dataset['test'].num_rows == 10304
    assert dataset['validation'].num_rows == 10171
