class LinkerPipeline:
    def __init__(self, nlp, batch_size=100):
        self.nlp = nlp
        self.task = 'token-classification'
        self.batch_size = batch_size

    def __call__(self, *args, **kwargs):
        res = []
        docs = self.nlp.pipe(args[0], batch_size=self.batch_size)
        for doc in docs:
            res_doc = []
            for span in doc._.spans:
                label = {
                    'entity': span.label, 'score': span.score,
                    'word': doc.text[span.start:span.end],
                    'start': span.start, 'end': span.end
                }
                res_doc.append(label)
            res.append(res_doc)

        return res


class MentionsExtractorPipeline:
    def __init__(self, nlp, batch_size=100):
        self.nlp = nlp
        self.task = 'token-classification'
        self.batch_size = batch_size

    def __call__(self, *args, **kwargs):
        res = []
        docs = self.nlp.pipe(args[0], batch_size=self.batch_size)
        for doc in docs:
            res_doc = []
            for span in doc._.mentions:
                label = {
                    'entity': "MENTION",
                    'word': doc.text[span.start_char:span.end_char],
                    'start': span.start_char, 'end': span.end_char
                }
                res_doc.append(label)
            res.append(res_doc)

        return res
