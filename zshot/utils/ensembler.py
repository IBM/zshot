from typing import Optional

from zshot.utils.data_models import Span


class Ensembler:

    def __init__(self,
                 num_voters: int,
                 num_enhance_entities: Optional[int] = -1,
                 strategy: Optional[str] = 'max',
                 threshold: Optional[float] = 0.5):
        self.number_pipelines = num_voters
        if num_enhance_entities > 0:
            self.number_pipelines *= self.number_pipelines
        self.strategy = strategy
        self.threshold = threshold

    def ensemble(self, spans):
        if self.strategy == 'max':
            all_union_spans = [self.ensemble_max(s) for k, s in spans.items()]
        else:
            all_union_spans = [self.ensemble_count(s) for k, s in spans.items()]

        all_union_spans = [s for s in all_union_spans if s.score > self.threshold]
        all_union_spans = self.inclusive(all_union_spans)
        return all_union_spans

    def ensemble_max(self, spans):
        # when strategy = 'max', choose the label with max total vote score
        votes = {}
        for s in spans:
            if s.label not in votes:
                votes[s.label] = s.score / self.number_pipelines
            else:
                votes[s.label] += s.score / self.number_pipelines

        max_score, best_label = self.select_best(votes)
        s = spans[0]

        return Span(label=best_label, score=max_score, start=s.start, end=s.end)

    def ensemble_count(self, spans):
        # when strategy = 'count', choose the label with max total vote count
        votes = {}
        for s in spans:
            if s.label not in votes:
                votes[s.label] = 1.0 / self.number_pipelines
            else:
                votes[s.label] += 1.0 / self.number_pipelines

        max_score, best_label = self.select_best(votes)
        s = spans[0]

        return Span(label=best_label, score=max_score, start=s.start, end=s.end)

    @staticmethod
    def select_best(votes):
        max_score = -1.0
        best_label = None
        for label, score in votes.items():
            if best_label is None:
                best_label = label
                max_score = score
            elif max_score < score:
                best_label = label
                max_score = score

        return max_score, best_label

    @staticmethod
    def inclusive(spans):
        n = len(spans)
        non_redundant_spans = []
        for i in range(n):
            is_redundant = False
            for j in range(n):
                if spans[i].start >= spans[j].start and spans[i].end <= spans[j].end:
                    if spans[i].start > spans[j].start or spans[i].end < spans[j].end:
                        is_redundant = True
                        break
            if not is_redundant:
                non_redundant_spans.append(spans[i])
        return non_redundant_spans
