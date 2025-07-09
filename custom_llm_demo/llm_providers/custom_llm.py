from typing import Any, Dict, List, Optional, Sequence
import aiohttp
from openai import OpenAI, AsyncOpenAI
from pydantic import Field, ConfigDict

from langchain.llms.base import BaseLLM
from langchain_core.callbacks.manager import (
    CallbackManagerForLLMRun,
    AsyncCallbackManagerForLLMRun,
)
from langchain_core.language_models.base import LanguageModelInput
from langchain_core.outputs import LLMResult, Generation


class MyCustomLLM(BaseLLM):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    model_name: str = Field(default="your-llm-model-name")
    endpoint_url: str = Field(default="your-llm-endpoint-url")
    api_key: str = Field(default="your-llm-api-key")
    max_tokens: int = Field(default=2048)
    temperature: float = Field(default=0.6)
    client: OpenAI = Field(default=None)
    async_client: AsyncOpenAI = Field(default=None)

    def __init__(
        self,
        model_name: str = None,
        endpoint_url: str = None,
        api_key: str = None,
        max_tokens: int = None,
        temperature: float = None,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.model_name = model_name or self.model_name
        self.endpoint_url = endpoint_url or self.endpoint_url
        api_key = api_key or self.api_key
        self.max_tokens = max_tokens or self.max_tokens
        self.temperature = temperature or self.temperature
        
        # 初始化 OpenAI 客户端
        self.client = OpenAI(
            api_key=api_key,
            base_url=endpoint_url
        )
        self.async_client = AsyncOpenAI(
            api_key=api_key,
            base_url=endpoint_url
        )

    @property
    def _llm_type(self) -> str:
        return "custom_llm"

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs,
    ) -> str:
        try:
            completion = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                stop=stop,
                stream=False,
                extra_body={"enable_thinking": False}  # DashScope 特定参数
            )
            
            return completion.choices[0].message.content
            
        except Exception as e:
            raise RuntimeError(f"API request failed: {str(e)}")

    async def _acall(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
        **kwargs,
    ) -> str:
        try:
            completion = await self.async_client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                stop=stop,
                stream=False,
                extra_body={"enable_thinking": False}  # DashScope 特定参数
            )
            
            return completion.choices[0].message.content
            
        except Exception as e:
            raise RuntimeError(f"API request failed: {str(e)}")

    def _generate(
        self,
        prompts: List[str],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> LLMResult:
        generations = []
        for prompt in prompts:
            text = self._call(prompt, stop=stop, run_manager=run_manager, **kwargs)
            generations.append([Generation(text=text)])
            
        return LLMResult(generations=generations)