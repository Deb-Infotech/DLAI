import os

import dotenv
from openai import OpenAI

from src.llm.AIModel import AIModel


class OpenAIModel(AIModel):
    def __init__(self, model_name: str):
        # Load environment variables from a .env file
        dotenv.load_dotenv()
        # by default, the OpenAI library reads the API key from the OPENAI_API_KEY environment variable
        #api_key = os.getenv("OPENAI_API_KEY")
        self.client = OpenAI()
        super().__init__(model_name=model_name)

    def execute(self, prompt: str) -> str:
        response = self.client.responses.create(
            model=self.model_name,
            input="who is prime minister of India?"
        )

        return response.output_text
