from typing import Iterator, List, Tuple, Optional, Union

import spacy
from datasets import Dataset
from spacy.tokens import Doc

from zshot import Linker, MentionsExtractor, PipelineConfig, RelationsExtractor
from zshot.evaluation.dataset.dataset import DatasetWithRelations
from zshot.evaluation.dataset.dataset import create_dataset
from zshot.evaluation.evaluator import (
    MentionsExtractorEvaluator,
    RelationExtractorEvaluator,
    ZeroShotTokenClassificationEvaluator,
)
from zshot.evaluation.metrics.rel_eval import RelEval
from zshot.evaluation.pipeline import (
    LinkerPipeline,
    MentionsExtractorPipeline,
    RelationExtractorPipeline,
)
from zshot.tests.config import EX_DATASET_RELATIONS, EX_RELATIONS
from zshot.utils.alignment_utils import AlignmentMode
from zshot.utils.data_models import Entity, Span, Relation
from zshot.utils.data_models.relation_span import RelationSpan

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
                    preds.append(
                        Span(
                            doc.text.find(span),
                            doc.text.find(span) + len(span),
                            label=label,
                            score=score,
                        )
                    )
            sentences.append(preds)

        return sentences


class DummyLinkerEnd2EndForEval(Linker):
    @property
    def is_end2end(self) -> bool:
        return True

    def __init__(self, predictions):
        # this dummy linker works correctly ONLY if no shuffling is done by spacy when batching documents
        super().__init__()
        self.predictions = predictions
        self.curr_idx = 0

    def predict(self, docs, batch_size=100):
        rval = []
        for _ in docs:
            rval.append(
                [Span(item["start"], item["end"], item["label"]) for item in self.predictions[self.curr_idx]]
            )
            self.curr_idx += 1
        return rval


class DummyMentionsExtractor(MentionsExtractor):
    def __init__(self, predictions: List[Tuple[str, str, float]]):
        super().__init__()
        self.predictions = predictions

    def predict(self, docs: Iterator[Doc], batch_size=None):
        sentences = []
        for doc in docs:
            preds = []
            for span, label, score in self.predictions:
                if span in doc.text:
                    preds.append(
                        Span(
                            doc.text.find(span),
                            doc.text.find(span) + len(span),
                            label="MENTION",
                            score=score,
                        )
                    )
            sentences.append(preds)

        return sentences


class DummyRelationsExtractor(RelationsExtractor):
    # this dummy relation extractor works correctly ONLY if no shuffling is done by spacy when batching documents
    def __init__(self, predictions):
        super().__init__()
        self.predictions = predictions
        self.curr_idx = 0

    def predict(self, docs: Iterator[Doc], batch_size: Optional[Union[int, None]] = None) -> List[List[RelationSpan]]:
        rval = []
        for _ in docs:
            rval.append(
                [RelationSpan(item["start"], item["end"], Relation(name=item["label"])) for item in self.predictions[self.curr_idx]]
            )
            self.curr_idx += 1
        return rval

def get_linker_pipe(predictions: List[Tuple[str, str, float]]):
    nlp = spacy.blank("en")
    nlp_config = PipelineConfig(linker=DummyLinker(predictions), entities=ENTITIES)

    nlp.add_pipe("zshot", config=nlp_config, last=True)

    return LinkerPipeline(nlp)


def get_mentions_extractor_pipe(predictions: List[Tuple[str, str, float]]):
    nlp = spacy.blank("en")
    nlp_config = PipelineConfig(
        mentions_extractor=DummyMentionsExtractor(predictions), entities=ENTITIES
    )

    nlp.add_pipe("zshot", config=nlp_config, last=True)

    return MentionsExtractorPipeline(nlp)


def get_relation_extraction_pipeline(predictions_relations):
    nlp = spacy.blank("en")
    nlp_config = PipelineConfig(
        relations_extractor=DummyRelationsExtractor(predictions_relations),
        relations=EX_RELATIONS,
    )
    nlp.add_pipe("zshot", config=nlp_config, last=True)
    return RelationExtractorPipeline(nlp)


def get_spans_predictions(span: str, label: str, sentence: str):
    return [
        {
            "start": sentence.find(span),
            "end": sentence.find(span) + len(span),
            "entity": label,
            "score": 1,
        }
    ]


class TestZeroShotTokenClassificationEvaluation:
    def test_preprocess(self):
        sentences = ["New York is beautiful"]
        gt = [["B-FAC", "I-FAC", "O", "O"]]

        predictions = [get_spans_predictions("New York", "FAC", sentences[0])]
        custom_evaluator = ZeroShotTokenClassificationEvaluator("token-classification")

        preds = custom_evaluator.predictions_processor(
            predictions, [["New", "York", "is", "beautiful"]], join_by=" "
        )
        assert preds["predictions"] == gt

    def test_prediction_span_based_evaluation(self):
        # Correct
        sentences = ["New York is beautiful"]
        gt = [["B-FAC", "I-FAC", "O", "O"]]
        dataset = create_dataset(gt, sentences, ENTITIES)

        custom_evaluator = ZeroShotTokenClassificationEvaluator("token-classification")
        metrics = custom_evaluator.compute(
            get_linker_pipe([("New York", "FAC", 1)]),
            dataset,
            metric="seqeval",
        )

        assert float(metrics["overall_precision"]) == 1.0
        assert float(metrics["overall_recall"]) == 1.0
        assert float(metrics["overall_f1"]) == 1.0
        assert float(metrics["overall_accuracy"]) == 1.0

        # Wrong
        sentences = ["New York is beautiful"]
        gt = [["B-FAC", "I-FAC", "O", "O"]]
        dataset = create_dataset(gt, sentences, ENTITIES)

        metrics = custom_evaluator.compute(
            get_linker_pipe([("York", "FAC", 1)]),
            dataset,
            metric="seqeval",
        )

        assert float(metrics["overall_precision"]) == 0.0
        assert float(metrics["overall_recall"]) == 0.0
        assert float(metrics["overall_f1"]) == 0.0
        assert float(metrics["overall_accuracy"]) == 0.5

    def test_prediction_token_based_evaluation(self):
        # Correct
        sentences = ["New York is beautiful"]
        gt = [["B-FAC", "I-FAC", "O", "O"]]

        dataset = create_dataset(gt, sentences, ENTITIES)

        custom_evaluator = ZeroShotTokenClassificationEvaluator("token-classification", mode='token')
        metrics = custom_evaluator.compute(
            get_linker_pipe([("New York", "FAC", 1)]),
            dataset,
            metric="seqeval",
        )

        assert float(metrics["overall_precision"]) == 1.0
        assert float(metrics["overall_recall"]) == 1.0
        assert float(metrics["overall_f1"]) == 1.0
        assert float(metrics["overall_accuracy"]) == 1.0

        # Partially Correct
        sentences = ["New York is beautiful"]
        gt = [["B-FAC", "I-FAC", "O", "O"]]

        dataset = create_dataset(gt, sentences, ENTITIES)

        metrics = custom_evaluator.compute(
            get_linker_pipe([("York", "FAC", 1)]),
            dataset,
            metric="seqeval",
        )

        assert float(metrics["overall_precision"]) == 1.0
        assert float(metrics["overall_recall"]) == 0.5
        assert float(metrics["overall_f1"]) == 2 / 3
        assert float(metrics["overall_accuracy"]) == 0.75

    def test_prediction_token_based_evaluation_overlapping_spans(self):
        sentences = ["New York is beautiful"]
        gt = [["B-FAC", "I-FAC", "O", "O"]]

        dataset = create_dataset(gt, sentences, ENTITIES)

        custom_evaluator = ZeroShotTokenClassificationEvaluator("token-classification")
        metrics = custom_evaluator.compute(
            get_linker_pipe([("New York", "FAC", 1), ("York", "LOC", 0.7)]),
            dataset,
            metric="seqeval",
        )

        assert float(metrics["overall_precision"]) == 1.0
        assert float(metrics["overall_recall"]) == 1.0
        assert float(metrics["overall_f1"]) == 1.0
        assert float(metrics["overall_accuracy"]) == 1.0

    def test_prediction_token_based_evaluation_partial_match_spans_expand(self):
        sentences = ["New York is beautiful"]
        gt = [["B-FAC", "I-FAC", "O", "O"]]

        dataset = create_dataset(gt, sentences, ENTITIES)

        custom_evaluator = ZeroShotTokenClassificationEvaluator(
            "token-classification", alignment_mode=AlignmentMode.expand
        )
        pipe = get_linker_pipe([("New Yo", "FAC", 1)])
        metrics = custom_evaluator.compute(pipe, dataset, metric="seqeval")

        assert float(metrics["overall_precision"]) == 1.0
        assert float(metrics["overall_recall"]) == 1.0
        assert float(metrics["overall_f1"]) == 1.0
        assert float(metrics["overall_accuracy"]) == 1.0

    def test_prediction_token_based_evaluation_partial_match_spans_contract(self):
        sentences = ["New York is beautiful"]
        gt = [["B-FAC", "I-FAC", "O", "O"]]

        dataset = create_dataset(gt, sentences, ENTITIES)

        custom_evaluator = ZeroShotTokenClassificationEvaluator(
            "token-classification", alignment_mode=AlignmentMode.contract
        )
        pipe = get_linker_pipe([("New York i", "FAC", 1)])
        metrics = custom_evaluator.compute(pipe, dataset, metric="seqeval")

        assert float(metrics["overall_precision"]) == 1.0
        assert float(metrics["overall_recall"]) == 1.0
        assert float(metrics["overall_f1"]) == 1.0
        assert float(metrics["overall_accuracy"]) == 1.0

    def test_prediction_token_based_evaluation_partial_and_overlapping_spans(self):
        sentences = ["New York is beautiful"]
        gt = [["B-FAC", "I-FAC", "O", "O"]]

        dataset = create_dataset(gt, sentences, ENTITIES)

        custom_evaluator = ZeroShotTokenClassificationEvaluator(
            "token-classification", alignment_mode=AlignmentMode.contract
        )
        pipe = get_linker_pipe([("New York i", "FAC", 1), ("w York", "LOC", 0.7)])
        metrics = custom_evaluator.compute(pipe, dataset, metric="seqeval")

        assert float(metrics["overall_precision"]) == 1.0
        assert float(metrics["overall_recall"]) == 1.0
        assert float(metrics["overall_f1"]) == 1.0
        assert float(metrics["overall_accuracy"]) == 1.0


class TestMentionsExtractorEvaluator:
    def test_prepare_data(self):
        sentences = ["New York is beautiful"]
        gt = [["B-FAC", "I-FAC", "O", "O"]]
        processed_gt = [["B-MENTION", "I-MENTION", "O", "O"]]

        dataset = create_dataset(gt, sentences, ENTITIES)

        custom_evaluator = MentionsExtractorEvaluator("token-classification")

        preds = custom_evaluator.prepare_data(
            dataset, input_column="tokens", label_column="ner_tags", join_by=" "
        )
        assert preds[0]["references"] == processed_gt

    def test_prediction_span_based_evaluation(self):
        # Correct
        sentences = ["New York is beautiful"]
        gt = [["B-FAC", "I-FAC", "O", "O"]]
        dataset = create_dataset(gt, sentences, ENTITIES)

        custom_evaluator = MentionsExtractorEvaluator("token-classification")
        metrics = custom_evaluator.compute(
            get_mentions_extractor_pipe([("New York", "FAC", 1)]),
            dataset,
            metric="seqeval",
        )

        assert float(metrics["overall_precision"]) == 1.0
        assert float(metrics["overall_recall"]) == 1.0
        assert float(metrics["overall_f1"]) == 1.0
        assert float(metrics["overall_accuracy"]) == 1.0

        # Wrong
        sentences = ["New York is beautiful"]
        gt = [["B-FAC", "I-FAC", "O", "O"]]
        dataset = create_dataset(gt, sentences, ENTITIES)

        metrics = custom_evaluator.compute(
            get_mentions_extractor_pipe([("York", "FAC", 1)]),
            dataset,
            metric="seqeval",
        )

        assert float(metrics["overall_precision"]) == 0.0
        assert float(metrics["overall_recall"]) == 0.0
        assert float(metrics["overall_f1"]) == 0.0
        assert float(metrics["overall_accuracy"]) == 0.5

    def test_prediction_token_based_evaluation(self):
        # Correct
        sentences = ["New York is beautiful"]
        gt = [["B-FAC", "I-FAC", "O", "O"]]
        dataset = create_dataset(gt, sentences, ENTITIES)

        custom_evaluator = MentionsExtractorEvaluator("token-classification", mode='token')
        metrics = custom_evaluator.compute(
            get_mentions_extractor_pipe([("New York", "FAC", 1)]),
            dataset,
            metric="seqeval",
        )

        assert float(metrics["overall_precision"]) == 1.0
        assert float(metrics["overall_recall"]) == 1.0
        assert float(metrics["overall_f1"]) == 1.0
        assert float(metrics["overall_accuracy"]) == 1.0

        # Partially Correct
        sentences = ["New York is beautiful"]
        gt = [["B-FAC", "I-FAC", "O", "O"]]
        dataset = create_dataset(gt, sentences, ENTITIES)

        metrics = custom_evaluator.compute(
            get_mentions_extractor_pipe([("York", "FAC", 1)]),
            dataset,
            metric="seqeval",
        )

        assert float(metrics["overall_precision"]) == 1.0
        assert float(metrics["overall_recall"]) == 0.5
        assert float(metrics["overall_f1"]) == 2 / 3
        assert float(metrics["overall_accuracy"]) == 0.75

    def test_prediction_span_based_evaluation_overlapping_spans(self):
        sentences = ["New York is beautiful"]
        gt = [["B-FAC", "I-FAC", "O", "O"]]

        dataset = create_dataset(gt, sentences, ENTITIES)

        custom_evaluator = MentionsExtractorEvaluator("token-classification")
        metrics = custom_evaluator.compute(
            get_mentions_extractor_pipe([("New York", "FAC", 1), ("York", "LOC", 0.7)]),
            dataset,
            metric="seqeval",
        )

        assert float(metrics["overall_precision"]) == 1.0
        assert float(metrics["overall_recall"]) == 1.0
        assert float(metrics["overall_f1"]) == 1.0
        assert float(metrics["overall_accuracy"]) == 1.0

    def test_prediction_span_based_evaluation_partial_match_spans_expand(self):
        sentences = ["New York is beautiful"]
        gt = [["B-FAC", "I-FAC", "O", "O"]]

        dataset = create_dataset(gt, sentences, ENTITIES)

        custom_evaluator = MentionsExtractorEvaluator(
            "token-classification", alignment_mode=AlignmentMode.expand
        )
        pipe = get_mentions_extractor_pipe([("New Yo", "FAC", 1)])
        metrics = custom_evaluator.compute(pipe, dataset, metric="seqeval")

        assert float(metrics["overall_precision"]) == 1.0
        assert float(metrics["overall_recall"]) == 1.0
        assert float(metrics["overall_f1"]) == 1.0
        assert float(metrics["overall_accuracy"]) == 1.0


class TestRelationExtractorEvaluator:

    def test_relation_classification_prediction(self):
        predictions_relations = [
            [{
                'start': Span(start["start"], start["end"], start["label"]),
                'end': Span(end["start"], end["end"], end["label"]),
                'label': label
            }]
            for (start, end), label in zip(EX_DATASET_RELATIONS['sentence_entities'], EX_DATASET_RELATIONS['labels'])
        ]
        dataset = DatasetWithRelations(
            Dataset.from_dict(EX_DATASET_RELATIONS).data,
            relations=EX_RELATIONS
        )

        custom_evaluator = RelationExtractorEvaluator()
        pipe = get_relation_extraction_pipeline(predictions_relations=predictions_relations)
        results = custom_evaluator.compute(
            pipe,
            dataset,
            input_column="sentences",
            label_column="labels",
            metric=RelEval(),
        )
        assert results is not None
        assert float(results["overall_precision_micro"]) == 1.0
        assert float(results["overall_recall_micro"]) == 1.0
        assert float(results["overall_f1_micro"]) == 1.0
        assert float(results["overall_precision_macro"]) == 1.0
        assert float(results["overall_recall_macro"]) == 1.0
        assert float(results["overall_f1_macro"]) == 1.0
        assert float(results["overall_accuracy"]) == 2.0
