import os

import dotenv
from ollama import Client

from src.llm.AIModel import AIModel


class OllamaAIModel(AIModel):
    def __init__(self, model_name: str):
        super().__init__(model_name=model_name)
        # load environment variables from a .env file
        dotenv.load_dotenv()
        # Initialize Ollama client here (placeholder)
        self.client = Client(host='https://ollama.com',
                             headers={'Authorization': 'Bearer ' + os.getenv('OLLAMA_API_KEY')})

    def execute(self, prompt: str) -> str:
        resp = self.client.generate(self.model_name, prompt=prompt)
        return resp['response']
