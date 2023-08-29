# KnowGL Knowledge Extractor

The knowgl-large model is trained by combining Wikidata with an extended version of the training data in the REBEL dataset. Given a sentence, KnowGL generates triple(s) in the following format: 
```
[(subject mention # subject label # subject type) | relation label | (object mention # object label # object type)]
```
If there are more than one triples generated, they are separated by $ in the output. The model achieves state-of-the-art results for relation extraction on the REBEL dataset. The generated labels (for the subject, relation, and object) and their types can be directly mapped to Wikidata IDs associated with them.

This `KnowledgeExtractor` does not use any entity/relation pre-defined.

- [Paper Rossiello et al. (AAAI 2023)](https://arxiv.org/pdf/2210.13952.pdf)
- [Paper Mihindukulasooriya et al. (ISWC 2022)](https://arxiv.org/pdf/2207.05188.pdf)
- [Original Model](https://huggingface.co/ibm/knowgl-large)

::: zshot.knowledge_extractor.KnowGL