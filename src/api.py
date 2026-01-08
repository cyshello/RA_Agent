import dotenv
from dataclasses import dataclass
from typing import Literal
import PIL
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
    image: PIL.Image | None = None
    output: Literal["text", "json"] = "text"
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
        
        # output 타입에 따라 요청 구성
        generation_config = {}
        if request.output == "json":
            generation_config = {
                "response_mime_type": "application/json"
            }
        
        response = self.client.models.generate_content(
            model=request.model,
            contents=contents,
            config=generation_config if generation_config else None
        )
        
        return response.text

### Dispatcher
class Dispatcher:
    def __init__(self):
        self.providers = {
            "openai": OpenAIProvider(AsyncOpenAIClient(OPENAI_KEY)),
            "gemini": GeminiProvider(AsyncGeminiClient(GEMINI_KEY))
        }
    
    async def dispatch(self, request: ChatRequest):
        """
        요청을 적절한 provider로 라우팅
        """
        provider = self.providers.get(request.provider)
        if not provider:
            raise ValueError(f"Unknown provider: {request.provider}")
        
        return await provider.chat(request)
