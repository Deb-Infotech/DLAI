import os

import dotenv
from openai import OpenAI

from src.llm.AIModel import AIModel


class OpenAIModel(AIModel):
    def __init__(self, model_name: str):
        dotenv.load_dotenv()
        api_key = os.getenv("OPENAI_API_KEY")
        self.client = OpenAI(api_key=api_key)
        super().__init__(model_name=model_name)

    def execute(self, prompt: str) -> str:
        response = self.client.responses.create(
            model=self.model_name,
            input="Write a one-sentence bedtime story about a unicorn."
        )

        return response.output_text
