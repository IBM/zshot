# Linker

The **linker** will link the detected entities to a existing set of labels. Some of the **linkers**, however, are *end-to-end*, i.e. they don't need the **mentions extractor**, as they detect and link the entities at the same time.  

There are 6 **linkers** available currently, 4 of them are *end-to-end* and 2 are not.

|                     Linker Name                      | end-to-end | Source Code                                              | Paper                                                              |
|:----------------------------------------------------:|:----------:|----------------------------------------------------------|--------------------------------------------------------------------|
|  [Blink](https://ibm.github.io/zshot/blink_linker/)  |      X     | [Source Code](https://github.com/facebookresearch/BLINK) | [Paper](https://arxiv.org/pdf/1911.03814.pdf)                      |
|  [GENRE](https://ibm.github.io/zshot/genre_linker/)  |      X     | [Source Code](https://github.com/facebookresearch/GENRE) | [Paper](https://arxiv.org/pdf/2010.00904.pdf)                      |
|   [SMXM](https://ibm.github.io/zshot/smxm_linker/)   |   &check;  | [Source Code](https://github.com/Raldir/Zero-shot-NERC)  | [Paper](https://aclanthology.org/2021.acl-long.120/)               |
|   [TARS](https://ibm.github.io/zshot/tars_linker/)   |   &check;  | [Source Code](https://github.com/flairNLP/flair)         | [Paper](https://kishaloyhalder.github.io/pdfs/tars_coling2020.pdf) |
| [GLINER](https://ibm.github.io/zshot/gliner_linker/) |   &check;  | [Source Code](https://github.com/urchade/GLiNER)         | [Paper](https://arxiv.org/abs/2311.08526) |
|  [RELIK](https://ibm.github.io/zshot/relik_linker/)  |   &check;  | [Source Code](https://github.com/SapienzaNLP/relik)         | [Paper](https://arxiv.org/abs/2408.00103) |


::: zshot.Linker