# SMXM Mentions Extractor
When the mentions to be extracted are known by the user they can be specified in order to improve the performance of the system. Based on the SMXM linker, the SMXM **mentions extractor** uses the description of the mentions to give the model information about the mentions to be extracted. By using the descriptions, the SMXM model is able to understand the mention.

The SMXM **mentions extractor** will use the **mentions** specified in the `zshot.PipelineConfig`.

- [Paper](https://aclanthology.org/2021.acl-long.120/)
- [Original Source Code](https://github.com/Raldir/Zero-shot-NERC)

::: zshot.mentions_extractor.MentionsExtractorSMXM