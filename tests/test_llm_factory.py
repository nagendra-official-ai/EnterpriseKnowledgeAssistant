from langchain_core.messages import HumanMessage

from llm.llm_factory import LLMFactory


def main() -> None:
    factory = LLMFactory()

    chat_model = factory.create_chat_model(
        provider="ollama",
        model_name="qwen2.5:3b",
        temperature=0.0,
    )

    response = chat_model.invoke(
        [HumanMessage(content=("Respond with exactly: " "Provider test successful"))]
    )

    print(response.content)

    assert response.content


if __name__ == "__main__":
    main()
