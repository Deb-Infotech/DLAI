import pytest

from src.llm.AIModel import AIModel
from src.llm.OllamaModel import OllamaAIModel
from src.llm.OpenAIModel import OpenAIModel

prompt = "who is prime minister of India?"
exp_1 = "Narendra"
exp_2 = "Modi"


@pytest.mark.parametrize("aimodel, prompt, exp_1, exp_2", [
    (OpenAIModel(model_name="gpt-4.1-mini"), prompt, exp_1, exp_2),
    (OllamaAIModel(model_name="gpt-oss:120b"), prompt, exp_1, exp_2)
])
def test_model_response(aimodel, prompt, exp_1, exp_2):
    ai_model: AIModel = aimodel
    response = ai_model.execute(prompt)
    print(response)
    assert exp_1 in response, f"Unexpected response: {response}"
    assert exp_2 in response, f"Unexpected response: {response}"
