# GENRE
GENRE is also an entity linking model released by Facebook, but in this case it uses a different approach by conseidering the NERC task as a sequence-to-sequence problem, and retrieves the entities by using a constrained beam search to force the model to generate the entities.

In a nutshell, (m)GENRE uses a sequence-to-sequence approach to entity retrieval (e.g., linking), based on fine-tuned [BART](https://arxiv.org/abs/1910.13461). GENRE performs retrieval generating the unique entity name conditioned on the input text using constrained beam search to only generate valid identifiers.
Although there is a version *end-to-end* of GENRE, it is not currently supported on ZShot (but it will). 

- [Paper](https://arxiv.org/pdf/2010.00904.pdf)
- [Original Source Code](https://github.com/facebookresearch/GENRE)

::: zshot.linker.LinkerRegen