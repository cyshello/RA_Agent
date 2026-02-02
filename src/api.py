"""
LangChain 기반 API 클라이언트 모듈

이 모듈은 OpenAI와 Google Gemini API를 LangChain 인터페이스로 래핑하여
통일된 방식으로 LLM 호출을 처리합니다. Rate limiting과 멀티모달 지원을 포함합니다.
"""

import dotenv
from dataclasses import dataclass
from typing import Literal, Optional, Any
import PIL.Image
import os
import asyncio
import time
import base64
import io

from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.runnables import RunnableParallel, RunnableLambda, Runnable

# .env 파일 경로 설정
env_path = os.path.join(os.path.dirname(__file__), ".env")
GEMINI_KEY = dotenv.get_key(env_path, "GEMINI_KEY")
OPENAI_KEY = dotenv.get_key(env_path, "OPENAI_KEY")


@dataclass
class ChatRequest:
    """LLM API 요청을 위한 데이터 클래스"""
    provider: Literal["openai", "gemini"]
    model: str
    messages: list
    input: Literal["text-only", "with-image"]
    image: Optional[PIL.Image.Image] = None
    output: Literal["text", "json"] = "text"
    web: Optional[bool] = False


class ModelFactory:
    """LangChain ChatModel 인스턴스를 생성하는 팩토리 클래스"""
    
    @staticmethod
    def create_model(
        provider: str,
        model_name: str,
        output_format: str = "text",
        web_search: bool = False
    ) -> BaseChatModel:
        """
        LangChain ChatModel 인스턴스 생성
        
        Args:
            provider: "openai" 또는 "gemini"
            model_name: 모델 이름
            output_format: "text" 또는 "json"
            web_search: 웹 검색 활성화 (Gemini만 지원)
        
        Returns:
            LangChain BaseChatModel 인스턴스
        """
        if provider == "openai":
            # OpenAI 모델 설정
            base_model = ChatOpenAI(
                model=model_name,
                api_key=OPENAI_KEY,
                temperature=0 if not model_name.startswith("o1") else 1  # o1 모델은 temperature 고정
            )
            
            # o1 모델은 response_format을 지원하지 않음
            if model_name.startswith("o1"):
                return base_model
            
            # JSON 모드가 필요하면 bind로 추가
            if output_format == "json":
                return base_model.bind(response_format={"type": "json_object"})
            else:
                return base_model
        
        elif provider == "gemini":
            # Gemini 모델 설정
            model_kwargs = {}
            
            # 웹 검색과 JSON 모드는 동시 사용 불가
            if web_search:
                # Grounding with Google Search
                model_kwargs["tools"] = ["google_search_retrieval"]
            elif output_format == "json":
                model_kwargs["response_mime_type"] = "application/json"
            
            return ChatGoogleGenerativeAI(
                model=model_name,
                google_api_key=GEMINI_KEY,
                temperature=0,
                **model_kwargs
            )
        
        else:
            raise ValueError(f"Unknown provider: {provider}")
    
    @staticmethod
    def create_model_chain(
        provider: str,
        model_name: str,
        output_format: str = "text",
        web_search: bool = False,
        max_rps: float = 2.0
    ) -> BaseChatModel:
        """
        Rate limiting이 적용된 LangChain 모델 체인 생성 (LCEL용)
        
        Args:
            provider: "openai" 또는 "gemini"
            model_name: 모델 이름
            output_format: "text" 또는 "json"
            web_search: 웹 검색 활성화 (Gemini만 지원)
            max_rps: 초당 최대 요청 수 (rate limiting)
        
        Returns:
            Rate limiting이 적용된 LangChain 모델
        """
        base_model = ModelFactory.create_model(
            provider=provider,
            model_name=model_name,
            output_format=output_format,
            web_search=web_search
        )
        
        # Rate limiting 래퍼로 감싸기
        return RateLimitedModel(base_model, max_rps=max_rps)


class RateLimitedModel(Runnable):
    """
    Rate limiting을 적용한 LangChain Runnable 래퍼
    
    LCEL 체인에서 사용할 수 있도록 모델을 래핑하여 rate limiting을 적용합니다.
    """
    
    def __init__(self, model: BaseChatModel, max_rps: float = 1.0):
        """
        Args:
            model: LangChain BaseChatModel 인스턴스
            max_rps: 초당 최대 요청 수
        """
        self.model = model
        self.max_rps = max_rps
        self.min_interval = 1.0 / max_rps
        self.last_request_time = 0
        self.lock = asyncio.Lock()
    
    async def ainvoke(self, input, config=None, **kwargs):
        """Rate limiting을 적용한 비동기 호출"""
        async with self.lock:
            elapsed = time.time() - self.last_request_time
            if elapsed < self.min_interval:
                await asyncio.sleep(self.min_interval - elapsed)
            self.last_request_time = time.time()
        
        return await self.model.ainvoke(input, config=config, **kwargs)
    
    def invoke(self, input, config=None, **kwargs):
        """동기 호출 (사용하지 않음)"""
        return self.model.invoke(input, config=config, **kwargs)


class Dispatcher:
    """
    LangChain 모델을 관리하고 rate limiting을 적용하는 디스패처
    
    여러 LLM 호출을 조율하고, API rate limit을 준수하며,
    멀티모달 입력(이미지+텍스트)을 처리합니다.
    """
    
    def __init__(self, max_rps: float = 1.0):
        """
        Args:
            max_rps: 초당 최대 LLM API 요청 수 (기본값: 1.0)
        """
        self.max_rps = max_rps
        self.min_interval = 1.0 / max_rps
        self.last_request_time = 0
        self.lock = asyncio.Lock()
    
    async def dispatch(self, request: ChatRequest) -> str:
        """
        요청을 적절한 LangChain 모델로 라우팅하고 응답 반환
        
        Args:
            request: ChatRequest 객체
        
        Returns:
            LLM 응답 텍스트
        """
        # Rate limiting: 최소 간격 보장
        async with self.lock:
            elapsed = time.time() - self.last_request_time
            if elapsed < self.min_interval:
                await asyncio.sleep(self.min_interval - elapsed)
            self.last_request_time = time.time()
        
        # 모델 생성
        model = ModelFactory.create_model(
            provider=request.provider,
            model_name=request.model,
            output_format=request.output,
            web_search=request.web
        )
        
        # 메시지 변환 (LangChain 형식)
        messages = self._convert_messages(request)
        
        # LangChain 호출 (async)
        response = await model.ainvoke(messages)
        
        return response.content
    
    def _convert_messages(self, request: ChatRequest) -> list:
        """
        ChatRequest의 messages를 LangChain Message 객체로 변환
        
        Args:
            request: ChatRequest 객체
        
        Returns:
            LangChain Message 객체 리스트
        """
        lc_messages = []
        
        for msg in request.messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            
            if role == "system":
                lc_messages.append(SystemMessage(content=content))
            
            elif role == "user":
                # 이미지가 있는 경우 멀티모달 처리
                if request.input == "with-image" and request.image:
                    if request.provider == "openai":
                        # OpenAI: base64 인코딩
                        buffered = io.BytesIO()
                        request.image.save(buffered, format="PNG")
                        img_base64 = base64.b64encode(buffered.getvalue()).decode()
                        
                        lc_messages.append(HumanMessage(
                            content=[
                                {"type": "text", "text": content},
                                {
                                    "type": "image_url",
                                    "image_url": {"url": f"data:image/png;base64,{img_base64}"}
                                }
                            ]
                        ))
                    
                    elif request.provider == "gemini":
                        # Gemini: PIL Image 직접 전달
                        lc_messages.append(HumanMessage(
                            content=[
                                {"type": "text", "text": content},
                                {"type": "image_url", "image_url": request.image}
                            ]
                        ))
                else:
                    # 텍스트만
                    lc_messages.append(HumanMessage(content=content))
        
        return lc_messages

