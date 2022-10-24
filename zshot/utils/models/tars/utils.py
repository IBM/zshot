from zshot.utils.data_models import Span


def tars_predict(model, sentences, batch_size):
    kwargs = {'mini_batch_size': batch_size} if batch_size else {}
    model.predict(sentences, **kwargs)

    spans_annotations = []
    for sent in sentences:
        sent_mentions = sent.get_spans('ner')
        spans = [
            Span(mention.start_position, mention.end_position, mention.tag, mention.score)
            for mention in sent_mentions
        ]
        spans_annotations.append(spans)

    return spans_annotations
