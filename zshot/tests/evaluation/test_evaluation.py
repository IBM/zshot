from typing import List, Iterator, Tuple

import spacy
from datasets import Dataset
from spacy.tokens import Doc

from zshot import PipelineConfig, Linker
from zshot.evaluation.evaluator import ZeroShotTokenClassificationEvaluator
from zshot.evaluation.pipeline import LinkerPipeline
from zshot.utils.alignment_utils import AlignmentMode
from zshot.utils.data_models import Entity, Span

ENTITIES = [
    Entity(name="FAC", description="A facility"),
    Entity(name="LOC", description="A location"),
]


class DummyLinker(Linker):

    def __init__(self, predictions: List[Tuple[str, str, float]]):
        super().__init__()
        self.predictions = predictions

    def predict(self, docs: Iterator[Doc], batch_size=None):
        sentences = []
        for doc in docs:
            preds = []
            for span, label, score in self.predictions:
                if span in doc.text:
                    preds.append(Span(doc.text.find(span), doc.text.find(span) + len(span), label=label, score=score))
            sentences.append(preds)

        return sentences


def get_pipe(predictions: List[Tuple[str, str, float]]):
    nlp = spacy.blank("en")
    nlp_config = PipelineConfig(
        linker=DummyLinker(predictions),
        entities=ENTITIES
    )

    nlp.add_pipe("zshot", config=nlp_config, last=True)

    return LinkerPipeline(nlp)


def get_spans_predictions(span: str, label: str, sentence: str):
    return [{'start': sentence.find(span),
             'end': sentence.find(span) + len(span),
             'entity': label,
             'score': 1}]


def get_dataset(gt: List[List[str]], sentence: List[str]):
    data_dict = {
        'tokens': [s.split(" ") for s in sentence],
        'ner_tags': gt,
    }
    dataset = Dataset.from_dict(data_dict)
    dataset.entities = ENTITIES
    return dataset


class TestZeroShotTokenClassificationEvaluation:

    def test_preprocess(self):
        sentences = ['New York is beautiful']
        gt = [['B-FAC', 'I-FAC', 'O', 'O']]

        predictions = [get_spans_predictions('New York', 'FAC', sentences[0])]
        custom_evaluator = ZeroShotTokenClassificationEvaluator("token-classification")

        preds = custom_evaluator.predictions_processor(predictions, [['New', 'York', 'is', 'beautiful']], join_by=" ")
        assert preds['predictions'] == gt

    def test_prediction_token_based_evaluation_all_matching(self):
        sentences = ['New York is beautiful']
        gt = [['B-FAC', 'I-FAC', 'O', 'O']]

        dataset = get_dataset(gt, sentences)

        custom_evaluator = ZeroShotTokenClassificationEvaluator("token-classification")
        metrics = custom_evaluator.compute(get_pipe([('New York', 'FAC', 1)]), dataset, "seqeval")

        assert float(metrics["overall_precision"]) == 1.0
        assert float(metrics["overall_precision"]) == 1.0
        assert float(metrics["overall_f1"]) == 1.0
        assert float(metrics["overall_accuracy"]) == 1.0

    def test_prediction_token_based_evaluation_overlapping_spans(self):
        sentences = ['New York is beautiful']
        gt = [['B-FAC', 'I-FAC', 'O', 'O']]

        dataset = get_dataset(gt, sentences)

        custom_evaluator = ZeroShotTokenClassificationEvaluator("token-classification")
        metrics = custom_evaluator.compute(get_pipe([('New York', 'FAC', 1), ('York', 'LOC', 0.7)]), dataset, "seqeval")

        assert float(metrics["overall_precision"]) == 1.0
        assert float(metrics["overall_precision"]) == 1.0
        assert float(metrics["overall_f1"]) == 1.0
        assert float(metrics["overall_accuracy"]) == 1.0

    def test_prediction_token_based_evaluation_partial_match_spans_expand(self):
        sentences = ['New York is beautiful']
        gt = [['B-FAC', 'I-FAC', 'O', 'O']]

        dataset = get_dataset(gt, sentences)

        custom_evaluator = ZeroShotTokenClassificationEvaluator("token-classification",
                                                                alignment_mode=AlignmentMode.expand)
        pipe = get_pipe([('New Yo', 'FAC', 1)])
        metrics = custom_evaluator.compute(pipe, dataset, "seqeval")

        assert float(metrics["overall_precision"]) == 1.0
        assert float(metrics["overall_precision"]) == 1.0
        assert float(metrics["overall_f1"]) == 1.0
        assert float(metrics["overall_accuracy"]) == 1.0

    def test_prediction_token_based_evaluation_partial_match_spans_contract(self):
        sentences = ['New York is beautiful']
        gt = [['B-FAC', 'I-FAC', 'O', 'O']]

        dataset = get_dataset(gt, sentences)

        custom_evaluator = ZeroShotTokenClassificationEvaluator("token-classification",
                                                                alignment_mode=AlignmentMode.contract)
        pipe = get_pipe([('New York i', 'FAC', 1)])
        metrics = custom_evaluator.compute(pipe, dataset, "seqeval")

        assert float(metrics["overall_precision"]) == 1.0
        assert float(metrics["overall_precision"]) == 1.0
        assert float(metrics["overall_f1"]) == 1.0
        assert float(metrics["overall_accuracy"]) == 1.0

    def test_prediction_token_based_evaluation_partial_and_overlapping_spans(self):
        sentences = ['New York is beautiful']
        gt = [['B-FAC', 'I-FAC', 'O', 'O']]

        dataset = get_dataset(gt, sentences)

        custom_evaluator = ZeroShotTokenClassificationEvaluator("token-classification",
                                                                alignment_mode=AlignmentMode.contract)
        pipe = get_pipe([('New York i', 'FAC', 1), ('w York', 'LOC', 0.7)])
        metrics = custom_evaluator.compute(pipe, dataset, "seqeval")

        assert float(metrics["overall_precision"]) == 1.0
        assert float(metrics["overall_precision"]) == 1.0
        assert float(metrics["overall_f1"]) == 1.0
        assert float(metrics["overall_accuracy"]) == 1.0
