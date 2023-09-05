# ZS-BERT Relations Extractor

The ZS-BERT model is a novel multi-task learning model to directly predict unseen relations without hand-crafted attribute labeling and multiple pairwise classifications. Given training instances consisting of input sentences and the descriptions of their relations, ZS-BERT learns two functions that project sentences and relation descriptions into an embedding space by jointly minimizing the distances between them and classifying seen relations. By generating the embeddings of unseen relations and new-coming sentences based on such two functions, we use nearest neighbor search to obtain the prediction of unseen relations.

This `RelationsExtractor` uses relations pre-defined along with their descriptions, added into the `PipelineConfig` using the `Relation` data model.

- [Paper](https://arxiv.org/abs/2104.04697)
- [Original Repo](https://github.com/dinobby/ZS-BERT)

::: zshot.relations_extractor.RelationsExtractorZSRC