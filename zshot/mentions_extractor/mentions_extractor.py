from abc import ABC, abstractmethod
from pydoc import Doc
from typing import List


class MentionsExtractor(ABC):
    @abstractmethod
    def extract_mentions(self, docs: List[Doc], batch_size=None):
        pass
