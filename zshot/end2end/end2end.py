from abc import ABC, abstractmethod
from typing import List

from spacy.tokens import Doc


class End2End(ABC):
    @abstractmethod
    def link(self, docs: List[Doc], batch_size=None):
        pass
