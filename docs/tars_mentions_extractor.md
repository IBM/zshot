# TARS Mentions Extractor

When the mentions to be extracted are known by the user they can be specified in order to improve the performance of the system. Based on the TARS linker, the TARS **mentions extractor** uses the labels of the mentions to give the model information about the mentions to be extracted. TARS doesn't need the descriptions of the entities, so if you can't provide the descriptions of the entities maybe this is the approach you're looking for.

The TARS **mentions extractor** will use the **mentions** specified in the `zshot.PipelineConfig`.

- [Paper](https://kishaloyhalder.github.io/pdfs/tars_coling2020.pdf)
- [Original Source Code](https://github.com/flairNLP/flair)

::: zshot.mentions_extractor.MentionsExtractorTARS