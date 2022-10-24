import evaluate
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import accuracy_score
import datasets

_KWARGS_DESCRIPTION = """
Produces labelling scores along with its sufficient statistics
from a source against one or more references.
Args:
    predictions: List of List of predicted labels (Estimated targets as returned by a tagger)
    references: List of List of reference labels (Ground truth (correct) target values)
"""


class RelEval(evaluate.Metric):
    def _info(self):
        return evaluate.MetricInfo(
            description="RelEval is a framework for relation extraction methods evaluation.",
            inputs_description=_KWARGS_DESCRIPTION,
            citation="alp@ibm.com",
            features=datasets.Features(
                {
                    "predictions": datasets.Value("string", id="label"),
                    "references": datasets.Value("string", id="label"),
                }
            ),
        )

    def _compute(
        self,
        predictions,
        references,
    ):
        scores = {}
        p, r, f1, _ = precision_recall_fscore_support(
            references, predictions, average="micro"
        )
        scores["overall_precision_micro"] = p
        scores["overall_recall_micro"] = r
        scores["overall_f1_micro"] = f1

        p, r, f1, _ = precision_recall_fscore_support(
            references, predictions, average="macro"
        )
        scores["overall_precision_macro"] = p
        scores["overall_recall_macro"] = r
        scores["overall_f1_macro"] = f1

        acc = accuracy_score(references, predictions, normalize=False)
        scores["overall_accuracy"] = acc

        return scores
