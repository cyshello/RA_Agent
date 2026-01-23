import dotenv
from dataclasses import dataclass
from typing import Literal, Optional
import PIL.Image
import os
from openai import AsyncOpenAI
from google import genai
import json

# .env 파일 경로 설정
env_path = os.path.join(os.path.dirname(__file__), ".env")
GEMINI_KEY = dotenv.get_key(env_path, "GEMINI_KEY")
OPENAI_KEY = dotenv.get_key(env_path, "OPENAI_KEY")

@dataclass
class ChatRequest():
    provider: Literal["openai", "gemini"]
    model: str
    messages: list
    input: Literal["text-only", "with-image"]
    image: Optional[PIL.Image.Image] = None
    output: Literal["text", "json"] = "text"
    web: Optional[bool] = False
    #schema: dict | None = None


### Provider classes
class Provider():
    async def chat(self, request: ChatRequest):
        raise NotImplementedError

class OpenAIProvider(Provider):
    def __init__(self, client):
        super().__init__()
        self.client = client
    
    async def chat(self, request: ChatRequest):
        return await self.client.chat(request)

class GeminiProvider(Provider):
    def __init__(self, client):
        super().__init__()
        self.client = client
    
    async def chat(self, request: ChatRequest):
        return await self.client.chat(request)
        

### async api clients

class AsyncOpenAIClient:
    def __init__(self, api_key: str):
        self.client = AsyncOpenAI(api_key=api_key)
    
    async def chat(self, request: ChatRequest):
        # o1, o3 시리즈는 chat.completions API를 사용하지만 제약사항이 있음
        # - system 메시지를 지원하지 않음
        # - response_format (JSON 모드) 지원하지 않음
        # - temperature, top_p 등 파라미터 지원하지 않음
        is_reasoning_model = request.model.startswith("o1") or request.model.startswith("o3")
        
        if is_reasoning_model:
            # o1/o3 모델: system 메시지를 user 메시지로 변환
            messages = []
            for msg in request.messages:
                role = msg.get("role", "user")
                content = msg.get("content", "")
                
                if role == "system":
                    # system 메시지를 user 메시지로 변환
                    messages.append({
                        "role": "user",
                        "content": f"Instructions: {content}"
                    })
                else:
                    messages.append({
                        "role": role,
                        "content": content
                    })
            
            # o1/o3는 JSON 모드를 지원하지 않으므로 일반 호출
            # 이미지도 지원하지 않음
            response = await self.client.chat.completions.create(
                model=request.model,
                messages=messages
            )
            return response.choices[0].message.content
        
        # 일반 chat completion 모델 (gpt-4o, gpt-4-turbo 등)
        # messages 구성
        if request.input == "with-image":
            # 이미지가 있는 경우
            import base64
            import io
            
            # PIL Image를 base64로 변환
            buffered = io.BytesIO()
            request.image.save(buffered, format="PNG")
            img_str = base64.b64encode(buffered.getvalue()).decode()
            
            messages = []
            for msg in request.messages:
                if msg.get("role") == "user":
                    messages.append({
                        "role": "user",
                        "content": [
                            {"type": "text", "text": msg["content"]},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/png;base64,{img_str}"
                                }
                            }
                        ]
                    })
                else:
                    messages.append(msg)
        else:
            messages = request.messages
        
        # output 타입에 따라 요청 구성
        if request.output == "json":
            response = await self.client.chat.completions.create(
                model=request.model,
                messages=messages,
                response_format={"type": "json_object"}
            )
        else:
            response = await self.client.chat.completions.create(
                model=request.model,
                messages=messages
            )
        
        return response.choices[0].message.content

class AsyncGeminiClient:
    def __init__(self, api_key: str):
        self.client = genai.Client(api_key=api_key)
    
    async def chat(self, request: ChatRequest):
        # messages를 Gemini 형식으로 변환
        # Gemini는 contents 형식을 사용
        contents = []
        
        # 마지막 사용자 메시지 추출
        user_message = None
        for msg in reversed(request.messages):
            if msg.get("role") == "user":
                user_message = msg["content"]
                break
        
        if request.input == "with-image":
            # 이미지와 텍스트 함께 전달
            contents = [user_message, request.image]
        else:
            # 텍스트만 전달
            contents = user_message
        
        # output 타입과 웹 검색 설정을 GenerateContentConfig로 구성
        from google.genai import types
        
        config_params = {}
        
        # 웹 검색(grounding) 설정
        # 주의: Gemini는 tools와 response_mime_type을 동시에 사용할 수 없음
        if request.web:
            grounding_tool = types.Tool(google_search=types.GoogleSearch())
            config_params["tools"] = [grounding_tool]
        # JSON 출력 설정 (웹 검색과 동시 사용 불가)
        elif request.output == "json":
            config_params["response_mime_type"] = "application/json"
        
        # config 생성 (파라미터가 있을 때만)
        config = types.GenerateContentConfig(**config_params) if config_params else None
        
        # Gemini API는 동기 함수이므로 executor에서 실행하여 비동기로 처리
        import asyncio
        loop = asyncio.get_event_loop()
        
        response = await loop.run_in_executor(
            None,
            lambda: self.client.models.generate_content(
                model=request.model,
                contents=contents,
                config=config
            )
        )
        
        return response.text

### Dispatcher
class Dispatcher:
    def __init__(self, max_rps: float = 1.0):
        """
        Args:
            max_rps: 초당 최대 LLM API 요청 수 (기본값: 1.0)
        """
        self.providers = {
            "openai": OpenAIProvider(AsyncOpenAIClient(OPENAI_KEY)),
            "gemini": GeminiProvider(AsyncGeminiClient(GEMINI_KEY))
        }
        self.max_rps = max_rps
        self.min_interval = 1.0 / max_rps
        self.last_request_time = 0
        import asyncio
        self.lock = asyncio.Lock()
    
    async def dispatch(self, request: ChatRequest):
        """
        요청을 적절한 provider로 라우팅 (rate limiting 적용)
        """
        # Rate limiting: 최소 간격 보장
        async with self.lock:
            import time
            elapsed = time.time() - self.last_request_time
            if elapsed < self.min_interval:
                import asyncio
                await asyncio.sleep(self.min_interval - elapsed)
            self.last_request_time = time.time()
        
        provider = self.providers.get(request.provider)
        if not provider:
            raise ValueError(f"Unknown provider: {request.provider}")
        
        return await provider.chat(request)
