# ReLiK Knowledge Extractor

ReLiK is a lightweight and fast model for Entity Linking and Relation Extraction. It is composed of two main components: a retriever and a reader. The retriever is responsible for retrieving relevant documents from a large collection, while the reader is responsible for extracting entities and relations from the retrieved documents. 

In **Zshot**, we created a Knowledge Extractor to use ReLiK and extract relations directly, without having to specify any entities or relation names.

- [Paper](https://arxiv.org/abs/2408.00103)
- [Original Source Code](https://github.com/SapienzaNLP/relik)

::: zshot.knowledge_extractor.KnowledgeExtractorRelik