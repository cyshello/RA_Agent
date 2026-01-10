# 필요한 util 함수들
import json
import PIL
import dotenv
from google import genai
import os
import logging
import time
from .api import ChatRequest, Dispatcher

logger = logging.getLogger(__name__)

def parse_json(response: str | dict | list) -> dict:
    """
    JSON 문자열/리스트를 dict로 파싱하고 구조화
    """
    try:
        # set으로 감싸진 경우 처리
        if isinstance(response, set):
            response = list(response)[0]
        
        # 문자열이면 JSON으로 파싱
        if isinstance(response, str):
            parsed = json.loads(response)
        elif isinstance(response, (dict, list)):
            parsed = response
        else:
            raise ValueError(f"Cannot parse response of type {type(response)}")
        
        # 리스트가 온 경우 딕셔너리로 변환
        if isinstance(parsed, list):
            result = {}
            for item in parsed:
                if isinstance(item, dict):
                    # text와 다른 정보가 있으면 병합
                    if "text" in item:
                        # bounding_box 같은 메타데이터는 제외하고 text만 추출
                        text = item["text"]
                        # 텍스트를 키-값으로 변환 (첫 줄을 키로 사용)
                        lines = text.split('\n')
                        if lines:
                            key = lines[0].strip()
                            value = '\n'.join(lines).strip()
                            result[key] = value
                    else:
                        result.update(item)
            return result
        
        # 이미 dict면 그대로 반환
        elif isinstance(parsed, dict):
            return parsed
        
        return {}
        
    except json.JSONDecodeError as e:
        raise ValueError(f"Failed to parse JSON: {e}\nResponse: {response}")
    except Exception as e:
        raise ValueError(f"Error parsing response: {e}\nResponse: {response}")

def generateSkeleton(arg : ChatRequest):
    """
    슬라이드별로 분석한 JSON 데이터를 받아 맥락 파악, 전체적인 구조를 맞추어 skeleton prompt 생성
    """
    pass

async def extractJSON(image: PIL.Image, dispatcher: Dispatcher, page_num: int = None, start_time_base: float = None) -> dict:
    """
    extract JSON from an image
    """
    page_start_time = time.time()
    page_label = f"페이지 {page_num}" if page_num is not None else "페이지"
    
    if start_time_base:
        elapsed_from_main = page_start_time - start_time_base
        logger.info(f"{page_label} 분석 시작 (전체 시작 후 {elapsed_from_main:.2f}초)")
    else:
        logger.info(f"{page_label} 분석 시작")

    # 우선 image에 OCR 진행 (별도 스레드에서 실행하여 블로킹 방지)
    import asyncio
    loop = asyncio.get_event_loop()
    lines = await loop.run_in_executor(None, extract_text_from_image, image)
    # lines = CLOVA_ocr(image)
    ocr_text = "\n".join(lines)

    systemPrompt = "당신은 주어진 기업 소개 문서 중 한 페이지를 보고, 해당 페이지 안의 정보를 추출하는 전문가입니다. 페이지 안의 텍스트들은 모두 추출되어 주어지며, 이미지를 보고 해당 텍스트들이 어떤 역할을 하는지 유추하여 아래와 같이 JSON 형식으로 반환하세요.\n" + \
    """
    JSON 스키마:
    {
        "텍스트의 역할" : "텍스트 내용"
    }
    예를 들어서 다음과 같을 수 있습니다.
    {
        "회사명": "ABC 주식회사",
        "설립연도": "1995년",
        "주요 제품": "스마트폰, 태블릿",
        "시장 점유율": "25%",
        "CEO 이름": "홍길동"
    }
    """ + \
    "이때, 반드시 JSON 형식으로만 응답해 주세요. 추가적인 설명이나 다른 텍스트는 포함하지 마세요.\n 또한, 이미지에서 추출한 텍스트가 같이 주어지는데, 대답에는 반드시 모든 추출한 텍스트를 포함해주세요."
    extractPrompt = "다음은 이미지에서 추출한 텍스트입니다:\n" + ocr_text + "\n이 텍스트를 바탕으로 아래의 JSON 스키마에 맞추어 정보를 추출해 주세요.\n"
    
    request = ChatRequest(
        provider="gemini",
        model="gemini-2.0-flash-exp",
        messages=[
            {
                "role": "system",
                "content": systemPrompt
            },
            {
                "role": "user",
                "content": extractPrompt
            }
        ],
        input="with-image",
        image=image,
        output="json"
    )

    response = await dispatcher.dispatch(request)
    
    # JSON 문자열을 dict로 변환
    response = parse_json(response)

    assert isinstance(response, dict), "Response is not a valid JSON"
    
    # 소요 시간 및 결과 로깅
    page_elapsed = time.time() - page_start_time
    if start_time_base:
        total_elapsed = time.time() - start_time_base
        logger.info(f"{page_label} 분석 완료 (전체 시작 후 {total_elapsed:.2f}초, 페이지 소요: {page_elapsed:.2f}초)")
    else:
        logger.info(f"{page_label} 분석 완료 - 소요 시간: {page_elapsed:.2f}초")
    logger.debug(f"{page_label} 결과: {response}")
    
    return response, ocr_text

    
#### OCR 

import pytesseract

def extract_text_from_image(image: PIL.Image) -> list[str]:
    """
    이미지에서 텍스트 추출
    """
    text = pytesseract.image_to_string(image, lang='eng+kor')
    ## NOTE : 한국어, 영어만 됨
    lines = text.split('\n')
    lines = [line.strip() for line in lines if line.strip()]
    return lines

def CLOVA_ocr(image: PIL.Image) -> list[str]:
    """
    CLOVA OCR API를 사용하여 이미지에서 텍스트 추출
    """
    #TODO : Try OCR API
    pass