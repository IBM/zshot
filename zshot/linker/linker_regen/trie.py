from typing import Collection


class Trie(object):
    def __init__(self, sequences: Collection[Collection[int]] = []):
        self.trie_dict = {}
        for sequence in sequences:
            self.add(sequence)

    def add(self, sequence: Collection[int]):
        trie = self.trie_dict
        for idx in sequence:
            if idx not in trie:
                trie[idx] = {}
            trie = trie[idx]

    def postfix(self, prefix_sequence: Collection[int]):
        if len(prefix_sequence) == 1:
            return list(self.trie_dict.keys())
        trie = self.trie_dict
        for pfx in prefix_sequence[1:]:
            if pfx not in trie:
                return []
            trie = trie[pfx]
        return list(trie.keys())
