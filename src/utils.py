# 필요한 util 함수들
import json
import PIL
import dotenv
from google import genai
import os
from RA_Agent.src.api import ChatRequest, Dispatcher

def generateSkeleton(arg : ChatRequest):
    """
    슬라이드별로 분석한 JSON 데이터를 받아 맥락 파악, 전체적인 구조를 맞추어 skeleton prompt 생성
    """
    pass

async def extractJSON(image: PIL.Image, dispatcher: Dispatcher) -> dict:
    """
    extract JSON from an image
    """

    # 우선 image에 OCR 진행
    lines = extract_text_from_image(image)
    # lines = CLOVA_ocr(image)
    ocr_text = "\n".join(lines)

    systemPrompt = ""
    extractPrompt = "다음은 이미지에서 추출한 텍스트입니다:\n" + ocr_text + "\n이 텍스트를 바탕으로 아래의 JSON 스키마에 맞추어 정보를 추출해 주세요.\n"
    
    request = ChatRequest(
        provider="gemini",
        model="gemini-1.5-flash",
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

    response = dispatcher.dispatch(request)

    assert isinstance(response, dict), "Response is not a valid JSON"
    return response

    
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