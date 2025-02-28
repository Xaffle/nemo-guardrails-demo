from typing import Any, Dict, List, Optional, Sequence
import aiohttp
import requests
from pydantic import Field, ConfigDict

from langchain.llms.base import BaseLLM
from langchain_core.callbacks.manager import (
    CallbackManagerForLLMRun,
    AsyncCallbackManagerForLLMRun,
)
from langchain_core.language_models.base import LanguageModelInput
from langchain_core.outputs import LLMResult, Generation

from nemoguardrails.llm.providers import register_llm_provider

class MyCustomLLM(BaseLLM):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    model_name: str = Field(default="your-llm-model-name")
    endpoint_url: str = Field(default="your-llm-endpoint-url")
    api_key: str = Field(default="your-llm-api-key")
    max_tokens: int = Field(default=2048)
    temperature: float = Field(default=0.6)
    headers: Dict[str, str] = Field(default_factory=dict)

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
        self.headers = {
            'Content-Type': 'application/json',
            'Authorization': f'{api_key}'
        }
        self.max_tokens = max_tokens or self.max_tokens
        self.temperature = temperature or self.temperature

    @property
    def _llm_type(self) -> str:
        return "custom_llm"

    def _generate(
        self,
        prompts: List[str],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> LLMResult:
        stop = self.stop if stop is None else stop
        generations = []
        for prompt in prompts:
            text = self._call(prompt, stop=stop, run_manager=run_manager, **kwargs)
            generations.append(
                [Generation(text=text, generation_info={"prompt": prompt})]
            )
        return LLMResult(
            generations=generations,
            llm_output={
                "url": self._get_request_url(),
                "headers": {
                    k: v
                    for k, v in self._get_request_headers().items()
                    # We make sure the Authorization header is not returned as this
                    # can lean the authorization key.
                    if k != "Authorization"
                },
                "model_name": self.model,
            },
        )
    
    def invoke(self, input: LanguageModelInput, **kwargs) -> str:
        """调用模型"""
        prompt = self.generate_prompt(input)
        return self.predict(prompt, **kwargs)

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs,
    ) -> str:
        try:
            payload = {
                "model": self.model_name,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": self.max_tokens,
                "stream": False
            }
            
            if stop:
                payload["stop"] = stop

            response = requests.post(
                self.endpoint_url,
                json=payload,
                headers=self.headers
            )
            response.raise_for_status()
            
            result = response.json()
            if not result.get("choices") or len(result["choices"]) == 0:
                raise ValueError("Invalid response format: no choices found")
                
            return result["choices"][0]["message"]["content"]
            
        except requests.RequestException as e:
            raise RuntimeError(f"API request failed: {str(e)}")
        except (KeyError, IndexError) as e:
            raise ValueError(f"Invalid response format: {str(e)}")

    async def _acall(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
        **kwargs,
    ) -> str:
        try:
            payload = {
                "model": self.model_name,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": self.max_tokens,
                "stream": False
            }
            
            if stop:
                payload["stop"] = stop

            async with aiohttp.ClientSession() as session:
                async with session.post(
                    self.endpoint_url,
                    json=payload,
                    headers=self.headers
                ) as response:
                    if response.status != 200:
                        error_text = await response.text()
                        raise RuntimeError(f"API returned status code {response.status}: {error_text}")
                    
                    result = await response.json()
                    if not result.get("choices") or len(result["choices"]) == 0:
                        raise ValueError("Invalid response format: no choices found")
                    
                    return result["choices"][0]["message"]["content"]
                    
        except aiohttp.ClientError as e:
            raise RuntimeError(f"API request failed: {str(e)}")
        except (KeyError, IndexError) as e:
            raise ValueError(f"Invalid response format: {str(e)}")

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