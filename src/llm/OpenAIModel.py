import dotenv
from openai import OpenAI

from src.llm.AIModel import AIModel


class OpenAIModel(AIModel):
    def __init__(self, model_name: str):
        super().__init__(model_name=model_name)
        # Load environment variables from a .env file
        dotenv.load_dotenv()
        # by default, the OpenAI library reads the API key from the OPENAI_API_KEY environment variable
        # api_key = os.getenv("OPENAI_API_KEY")
        self.client = OpenAI()

    def execute(self, prompt: str) -> str:
        resp = self.client.responses.create(
            model=self.model_name,
            input=prompt,
        )
        print("Response object:", resp)
        return resp.output_text
        # resp2 = self.client.chat.completions.create(model=self.model_name, messages=
        # [
        #     {"role": "system", "content": "You are a helpful assistant to provide General knowledge answers."},
        #     {"role": "user", "content": prompt}
        # ]
        #                                             )
        # print("Chat Completion Response object:", resp2)
        # return resp2.choices[0].message.content
