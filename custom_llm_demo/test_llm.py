from nemoguardrails import LLMRails, RailsConfig
from llm_providers.custom_llm import MyCustomLLM

def main():
    # 1. 首先创建 LLM 实例
    llm = MyCustomLLM(
        endpoint_url="your-llm-endpoint-url",
        model_name="your-llm-model-name",
        headers={
            "Content-Type": "your-content-type",
            "Authorization": "your-llm-api-key"
        }
    )

    # 2. 创建 Rails 实例
    config = RailsConfig.from_path("./config")
    rails = LLMRails(config, llm=llm)

    # 3. 测试对话
    messages = [{
        "role": "user",
        "content": "Hello!"
    }]
    
    response = rails.generate(messages=messages)
    print("Response:", response)

if __name__ == "__main__":
    main() 