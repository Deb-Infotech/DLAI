from abc import ABC, abstractmethod


class AIModel(ABC):
    def __init__(self, model_name: str):
        self.model_name = model_name
    @abstractmethod
    def execute(self, prompt: str) -> str:
        raise NotImplementedError("Subclasses must implement this method")