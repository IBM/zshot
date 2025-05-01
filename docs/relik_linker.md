# ReLiK Linker
ReLiK is a lightweight and fast model for Entity Linking and Relation Extraction. It is composed of two main components: a retriever and a reader. The retriever is responsible for retrieving relevant documents from a large collection, while the reader is responsible for extracting entities and relations from the retrieved documents. ReLiK can be used with the from_pretrained method to load a pre-trained pipeline.

In **Zshot**, we created a linker to use ReLiK, and it works both providing entities or without providing entities, and with descriptions.

This is an *end-to-end* model, so there is no need to use a **mentions extractor** before.

The ReLiK **linker** will use the **entities** specified in the `zshot.PipelineConfig`, if any.

- [Paper](https://arxiv.org/abs/2408.00103)
- [Original Source Code](https://github.com/SapienzaNLP/relik)

::: zshot.linker.LinkerRelik