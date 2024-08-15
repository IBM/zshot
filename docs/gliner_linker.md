# GLiNER Linker

GLiNER is a Named Entity Recognition (NER) model capable of identifying any entity type using a bidirectional transformer encoder (BERT-like). It provides a practical alternative to traditional NER models, which are limited to predefined entities, and Large Language Models (LLMs) that, despite their flexibility, are costly and large for resource-constrained scenarios.

The GLiNER **linker** will use the **entities** specified in the `zshot.PipelineConfig`, it just uses the names of the entities, it doesn't use the descriptions of the entities.


- [Paper](https://arxiv.org/abs/2311.08526)
- [Original Source Code](https://github.com/urchade/GLiNER)

::: zshot.linker.LinkerGLINER