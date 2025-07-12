from nemoguardrails import LLMRails, RailsConfig
from llm_providers.custom_llm import MyCustomLLM
import os

def main():
    # 1. 首先创建 LLM 实例
    llm = MyCustomLLM(
        model_name="qwen-plus-2025-01-25",
        endpoint_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        api_key=os.getenv("DASHSCOPE_API_KEY"),
        temperature=0.7,
        extra_body={
            "enable_thinking": False
        }
    )

    # 2. 创建 Rails 实例
    config = RailsConfig.from_path("./custom_llm_demo/config")
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