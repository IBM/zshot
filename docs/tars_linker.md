# TARS Linker

Task-aware representation of sentences (TARS), is a simple and effective method for few-shot and even zero-shot learning for text classification. However, it was extended to perform Zero-Shot NERC. 

Basically, TARS tries to convert the problem to a binary classification problem, predicting if a given text belongs to a specific class.

TARS doesn't need the descriptions of the entities, so if you can't provide the descriptions of the entities maybe this is the approach you're looking for.

The TARS **linker** will use the **entities** specified in the `zshot.PipelineConfig`.


- [Paper](https://kishaloyhalder.github.io/pdfs/tars_coling2020.pdf)
- [Original Source Code](https://github.com/flairNLP/flair)

::: zshot.linker.LinkerTARS