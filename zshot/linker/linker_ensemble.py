from typing import Iterator

from spacy.tokens import Doc

from zshot.linker import Linker
from zshot.linker import LinkerSMXM, LinkerTARS, LinkerRegen
from zshot.utils.data_models import Span, Entity


class LinkerEnsemble(Linker):
    def __init__(self, linkers=None, enhance_entities=None, strategy='max', threshold=0.5):
        super(LinkerEnsemble, self).__init__()
        if linkers is not None:
            self.linkers = linkers
        else:
            # default options
            self.linkers = [
                LinkerSMXM(),
                # LinkerTARS(),
                # LinkerRegen()
            ]
        self.enhance_entities = enhance_entities
        self.strategy = strategy
        self.threshold = threshold

    def set_smxm_model(self, smxm_model):
        for linker in self.linkers:
            if isinstance(linker, LinkerSMXM):
                linker.model_name = smxm_model

    def set_kg(self, entities: Iterator[Entity]):
        """
        Set entities that linker can use
        :param entities: The list of entities
        """
        for linker in self.linkers:
            linker.set_kg(entities)

    def predict(self, docs: Iterator[Doc], batch_size=None):
        # docs = docs[0:1]
        spans = []
        if self.enhance_entities is not None:
            for entities in self.enhance_entities:
                self.set_kg(entities)
                for linker in self.linkers:
                    span_prediction = linker.predict(docs, batch_size)
                    spans.append(span_prediction)
        else:
            for linker in self.linkers:
                span_prediction = linker.predict(docs, batch_size)
                spans.append(span_prediction)
        return self.prediction_ensemble(spans, docs)

    def prediction_ensemble(self, spans, docs):
        doc_ensemble_spans = []
        num_doc = len(spans[0])
        for doc_idx in range(num_doc):
            union_spans = {}
            span_per_descriptions = []
            for span in spans:
                span_per_descriptions.append(span[doc_idx])
                for s in span[doc_idx]:
                    span_pos = (s.start, s.end)
                    if span_pos not in union_spans:
                        union_spans[span_pos] = [s]
                    else:
                        union_spans[span_pos].append(s)
            # self.sub_span_scoring(union_spans)
            self.sub_span_scoring_per_description(union_spans, span_per_descriptions)
            if self.strategy == 'max':
                all_union_spans = [self.ensemble_max(s) for k, s in union_spans.items()]
            else:
                all_union_spans = [self.ensemble_count(s) for k, s in union_spans.items()]
            all_union_spans = [s for s in all_union_spans if s.score > self.threshold]
            all_union_spans = self.inclusive(all_union_spans)
            print("---------------------------------------------------------------------------")
            print(str(docs[doc_idx]))
            for span in all_union_spans:
                print(str(docs[doc_idx])[span.start:span.end], span)
            doc_ensemble_spans.append(all_union_spans)
        return doc_ensemble_spans

    def ensemble_max(self, spans):
        # when strategy = 'max', choose the label with max total vote score
        votes = {}
        number_pipelines = len(self.linkers)
        if self.enhance_entities is not None:
            number_pipelines *= len(self.enhance_entities)
        for s in spans:
            if s.label not in votes:
                votes[s.label] = s.score / number_pipelines
            else:
                votes[s.label] += s.score / number_pipelines

        max_score = -1.0
        best_label = None
        for label, score in votes.items():
            if best_label is None:
                best_label = label
                max_score = score
            elif max_score < score:
                best_label = label
                max_score = score
        s = spans[0]
        num_classes = len(votes) + 1
        # corrected_score = (number_pipelines - len(spans)) / (num_classes * number_pipelines + 0.0)
        # max_score += corrected_score
        return Span(label=best_label, score=max_score, start=s.start, end=s.end)

    def ensemble_count(self, spans):
        # when strategy = 'count', choose the label with max total vote count
        votes = {}
        number_pipelines = len(self.linkers)
        if self.enhance_entities is not None:
            number_pipelines *= len(self.enhance_entities)
        for s in spans:
            if s.label not in votes:
                votes[s.label] = 1.0 / number_pipelines
            else:
                votes[s.label] += 1.0 / number_pipelines

        max_score = -1.0
        best_label = None
        for label, score in votes.items():
            if best_label is None:
                best_label = label
                max_score = score
            elif max_score < score:
                best_label = label
                max_score = score
        s = spans[0]
        num_classes = len(votes) + 1
        # corrected_score = (number_pipelines - len(spans)) / (num_classes * number_pipelines + 0.0)
        # max_score += corrected_score
        return Span(label=best_label, score=max_score, start=s.start, end=s.end)

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

    @staticmethod
    def sub_span_scoring(spans):
        for k in spans.keys():
            for p in spans.keys():
                if k[0] >= p[0] and k[1] <= p[1]:
                    if k[0] > p[0] or k[1] < p[1]:
                        for s in spans[k]:
                            spans[p].append(Span(label=s.label, score=s.score, start=p[0], end=p[1]))

    @staticmethod
    def sub_span_scoring_per_description(union_spans, spans):
        for k in union_spans.keys():
            for span in spans:
                labels = {}
                for p in span:
                    if k[0] <= p.start and k[1] >= p.end:
                        if k[0] < p.start or k[1] > p.end:
                            if p.label not in labels:
                                labels[p.label] = p
                            elif labels[p.label].score < p.score:
                                labels[p.label] = p
                for p in labels.values():
                    union_spans[k].append(Span(label=p.label, score=p.score, start=k[0], end=k[1]))
