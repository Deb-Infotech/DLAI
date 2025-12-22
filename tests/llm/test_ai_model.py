import pytest

from src.llm.AIModel import AIModel
from src.llm.OllamaModel import OllamaAIModel
from src.llm.OpenAIModel import OpenAIModel

prompt = "who is prime minister of India?"


@pytest.mark.parametrize("aimodel, prompt", [
    (OpenAIModel(model_name="gpt-4.1-mini"), prompt),
    (OllamaAIModel(model_name="gpt-oss:120b"), prompt)
])
def test_model_response(aimodel, prompt):
    ai_model: AIModel = aimodel
    response = ai_model.execute(prompt)
    print(response)
    assert 'Narendra' in response, f"Unexpected response: {response}"
    assert 'Modi' in response, f"Unexpected response: {response}"
