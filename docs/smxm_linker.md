# SMXM Linker
When there is no labelled data (i.e.: Zero-Shot approaches) the performance usually decreases due to the fact that the model doesn't really know what does the entity represent. To address this problem the SMXM model uses the description of the entities to give the model information about the entities.

By using the descriptions, the SMXM model is able to understand the entity. Although this approach is Zero-Shot, as it doesn't need to have seen the entities during training, the user still have to specify the descriptions of the entities.

This is an *end-to-end* model, so there is no need to use a **mentions extractor** before.

The SMXM **linker** will use the **entities** specified in the `zshot.PipelineConfig`.

- [Paper](https://aclanthology.org/2021.acl-long.120/)
- [Original Source Code](https://github.com/Raldir/Zero-shot-NERC)

::: zshot.linker.LinkerSMXM