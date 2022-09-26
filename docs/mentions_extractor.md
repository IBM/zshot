# MentionsExtractor
The **mentions extractor** will detect the possible entities (a.k.a. mentions), that will be then linked to a data source (e.g.: Wikidata) by the **linker**. 

Currently, there are 4 different **mentions extractors** supported, 2 of them are based on *SpaCy*, and 2 of them are based on *Flair*. The two different versions for each library are similar, one is based on NERC and the other one is based on the linguistics (i.e.: using PoS and DP).

The NERC approach will use NERC models to detect all the entities that have to be linked. This approach depends on the model that is being used, and the entities the model has been trained on, so depending on the use case and the target entities it may be not the best approach, as the entities may be not recognized by the NERC model and thus won't be linked.

The linguistic approach relies on the idea that mentions will usually be a syntagma or a noun. Therefore, this approach detects nouns that are included in a syntagma and that act like objects, subjects, etc. This approach do not depend on the model (although the performance does), but a noun in a text should be always a noun, it doesn't depend on the dataset the model has been trained on.


::: zshot.MentionsExtractor