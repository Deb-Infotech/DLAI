import json

from src.llm.AIModel import AIModel
from src.llm.OpenAIModel import OpenAIModel


def test_model_response():
    ai_model: AIModel = OpenAIModel(model_name="gpt-oss-120b")
    response = ai_model.execute(
        prompt="Write a one-sentence bedtime story about a unicorn" )
    assert isinstance(json.loads(response), dict )