from src.llm.AIModel import AIModel


class OllamaAIModel(AIModel):
    def __init__(self, model_name: str):
        super().__init__(model_name=model_name)

    def execute(self, prompt: str) -> str:
        # Placeholder implementation
        return f"Ollama model '{self.model_name}' received prompt: {prompt}"
