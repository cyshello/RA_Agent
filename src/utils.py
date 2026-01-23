# 필요한 util 함수들
import json
import PIL
import dotenv
from google import genai
import os
import logging
import time
from .api import ChatRequest, Dispatcher
import requests
import uuid
import time
import json
import io

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
            # Markdown 코드 블록 제거 (```json ... ``` 또는 ``` ... ```)
            response = response.strip()
            if response.startswith("```"):
                # 첫 번째 줄 제거 (```json 또는 ```)
                lines = response.split('\n')
                if lines[0].strip().startswith("```"):
                    lines = lines[1:]
                # 마지막 줄 제거 (```)
                if lines and lines[-1].strip() == "```":
                    lines = lines[:-1]
                response = '\n'.join(lines).strip()
            
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

def parse_upstage(response: str | dict | list) -> list[str]:
    """
    Upstage OCR 결과에서 HTML을 추출하고 유니코드를 디코딩
    
    Args:
        response: Upstage OCR 결과 (JSON 문자열, dict, 또는 list)
        
    Returns:
        디코딩된 텍스트 리스트
        
    Example:
        >>> response = {"elements": [{"content": {"html": "<p>캐시닥</p>", ...}, ...}]}
        >>> parse_upstage(response)
        ['캐시닥', ...]
    """
    try:
        from html.parser import HTMLParser
        
        class HTMLTextExtractor(HTMLParser):
            def __init__(self):
                super().__init__()
                self.text = []
            
            def handle_data(self, data):
                if data.strip():
                    self.text.append(data.strip())
        
        # 문자열이면 JSON으로 파싱
        if isinstance(response, str):
            parsed = json.loads(response)
        elif isinstance(response, (dict, list)):
            parsed = response
        else:
            raise ValueError(f"Cannot parse response of type {type(response)}")
        
        text_list = []
        
        # Upstage OCR 응답 구조: {"elements": [...], ...}
        if isinstance(parsed, dict):
            elements = parsed.get("elements", [])
            if not elements and "content" in parsed:
                # 단일 요소인 경우
                elements = [parsed]
        elif isinstance(parsed, list):
            elements = parsed
        else:
            elements = []
        
        for item in elements:
            if isinstance(item, dict):
                # content.html 필드 찾기
                if "content" in item and isinstance(item["content"], dict):
                    html = item["content"].get("html", "")
                    if html:
                        # HTML에서 텍스트 추출
                        parser = HTMLTextExtractor()
                        try:
                            parser.feed(html)
                            extracted = " ".join(parser.text)
                            if extracted:
                                text_list.append(extracted)
                        except:
                            # HTML 파싱 실패시 content.text 사용
                            text = item["content"].get("text", "")
                            if text:
                                text_list.append(text)
        
        return text_list
        
    except json.JSONDecodeError as e:
        raise ValueError(f"Failed to parse JSON: {e}\nResponse: {response}")
    except Exception as e:
        raise ValueError(f"Error parsing Upstage response: {e}\nResponse: {response}")

def parse_upstage_to_html(response: str | dict | list, output_file: str = None) -> str:
    """
    Upstage OCR 결과에서 HTML을 추출하여 하나의 완전한 HTML 문서로 생성
    
    Args:
        response: Upstage OCR 결과 (JSON 문자열, dict, 또는 list)
        output_file: 저장할 HTML 파일 경로 (선택사항)
        
    Returns:
        완전한 HTML 문서 문자열
        
    Example:
        >>> response = {"elements": [{"content": {"html": "<p>캐시닥</p>", ...}, ...}]}
        >>> html = parse_upstage_to_html(response, "output.html")
    """
    try:
        # 문자열이면 JSON으로 파싱
        if isinstance(response, str):
            parsed = json.loads(response)
        elif isinstance(response, (dict, list)):
            parsed = response
        else:
            raise ValueError(f"Cannot parse response of type {type(response)}")
        
        # Upstage OCR 응답 구조: {"elements": [...], ...}
        if isinstance(parsed, dict):
            elements = parsed.get("elements", [])
            if not elements and "content" in parsed:
                # 단일 요소인 경우
                elements = [parsed]
        elif isinstance(parsed, list):
            elements = parsed
        else:
            elements = []
        
        # HTML 조각들을 모으기
        html_parts = []
        for item in elements:
            if isinstance(item, dict):
                # content.html 필드 찾기
                if "content" in item and isinstance(item["content"], dict):
                    html = item["content"].get("html", "")
                    if html:
                        html_parts.append(html)
        
        # 완전한 HTML 문서 생성
        full_html = """<!DOCTYPE html>
<html lang="ko">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Upstage OCR Result</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            margin: 20px;
            line-height: 1.6;
        }
        p {
            margin: 10px 0;
        }
    </style>
</head>
<body>
"""
        full_html += "\n".join(html_parts)
        full_html += """
</body>
</html>"""
        
        # 파일로 저장 (선택사항)
        if output_file:
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(full_html)
            logger.info(f"HTML 파일이 저장되었습니다: {output_file}")
        
        return full_html
        
    except json.JSONDecodeError as e:
        raise ValueError(f"Failed to parse JSON: {e}\nResponse: {response}")
    except Exception as e:
        raise ValueError(f"Error parsing Upstage response to HTML: {e}\nResponse: {response}")

def generateSkeleton(arg : ChatRequest):
    """
    슬라이드별로 분석한 JSON 데이터를 받아 맥락 파악, 전체적인 구조를 맞추어 skeleton prompt 생성
    """
    pass

async def extractJSON(image: PIL.Image, dispatcher: Dispatcher, page_num: int = None, start_time_base: float = None, debug: bool = False, ocr_provider: str = "CLOVA", model_provider: str = "openai", model_name: str = "gpt-4o", company=None) -> dict:
    """
    extract JSON from an image
    
    Args:
        image: 분석할 이미지
        dispatcher: API 디스패처
        page_num: 페이지 번호
        start_time_base: 전체 분석 시작 시간
        debug: 디버그 모드 여부
        ocr_provider: OCR API 종류 ("CLOVA" 또는 "Upstage")
        model_provider: AI 모델 제공자 ("openai" 또는 "gemini")
        model_name: 사용할 구체적인 모델명
        company: Company 객체 (API 호출 카운터 증가용)
    """
    page_start_time = time.time()
    page_label = f"페이지 {page_num}" if page_num is not None else "페이지"
    
    if debug:
        if start_time_base:
            elapsed_from_main = page_start_time - start_time_base
            logger.info(f"{page_label} 분석 시작 (전체 시작 후 {elapsed_from_main:.2f}초)")
        else:
            logger.info(f"{page_label} 분석 시작")

    # 우선 image에 OCR 진행 (별도 스레드에서 실행하여 블로킹 방지)
    import asyncio
    loop = asyncio.get_event_loop()
    img_byte_arr = io.BytesIO()
    image.save(img_byte_arr, format='PNG')
    img_byte_arr = img_byte_arr.getvalue()
    
    # OCR 제공자 선택
    if ocr_provider == "Upstage":
        lines = await loop.run_in_executor(None, Upstage_ocr, img_byte_arr)
    else:  # CLOVA (기본값)
        lines = await loop.run_in_executor(None, CLOVA_ocr, img_byte_arr)
    
    # OCR 호출 카운터 증가
    if company:
        company.ocr_call_count += 1

    ocr_text = "\n".join(lines)
    
    # OCR 완료 시간 로깅 (debug 모드일 때만)
    if debug:
        ocr_elapsed = time.time() - page_start_time
        if start_time_base:
            total_elapsed_ocr = time.time() - start_time_base
            logger.info(f"{page_label} OCR 완료 (전체 시작 후 {total_elapsed_ocr:.2f}초, OCR 소요: {ocr_elapsed:.2f}초)")
        else:
            logger.info(f"{page_label} OCR 완료 - 소요 시간: {ocr_elapsed:.2f}초")

    systemPrompt = """당신은 주어진 기업 소개 문서 중 한 페이지를 보고, 해당 페이지 안의 정보를 추출하는 전문가입니다. 페이지 안의 텍스트들은 모두 추출되어 주어지며, 이미지를 보고 슬라이드를 최대한 자세하게 설명하여 아래와 같이 JSON 형식으로 반환하세요. \
            이때, 다음과 같은 사항을 꼭 지켜주세요. \n
            1, 추출된 텍스트는 이미지의 정보를 담고 있지 못한 순수한 텍스트이므로 이미지를 우선 관찰하여 이미지가 담고 있는 정보를 최대한 많이, 정확하게 반환해야 합니다. \n \
            2. 우선 이미지에 포함된 시각적 구성요소들을 구분하고, 각 구성요소들이 담고 있는 텍스트들을 추출된 텍스트에서 참고하여 해당 구성요소가 의미하는 바를 최대한 유추하여 JSON 형식으로 표현하세요. \n\
            3. 표, 그래프, 다이어그램 등의 시각적 요소를 설명하기 위해서는 반드시 우선 해당 요소를 설명하고, 그 다음에 구성 요소들을 키-값 쌍으로 표현하세요. \n\
            예를 들어, 원형 그래프의 여러 요소들을 크기의 차이로 표현하는 그래프가 있다면 다음과 같이 표현합니다.\n \
                "원형 그래프의 이름 또는 역할" : {
                    "설명" : "이 그래프는 ~을 나타내며, 각 요소는 다음과 같습니다.",
                    "요소1 이름": "요소1 값(숫자가 주어져 있지 않다면 상대적인 크기 표현)",
                    "요소2 이름": "요소2 값",
                    ...
                } \
            4. 각 페이지의 중심 아이디어를 반드시 포함하세요. \n\
    """ + \
    """
    JSON 스키마:
    {
        "구성요소" : "구성 내용"
    }
    예를 들어서 다음과 같을 수 있습니다.
    {
        "중심 아이디어": "이 슬라이드는 ABC 주식회사의 개요를 설명합니다.", 
        "회사명": "ABC 주식회사",
        "설립연도": "1995년",
        "주요 제품": "스마트폰, 태블릿",
        "시장 점유율": "25%",
        "CEO 이름": "홍길동",
        "주요 경쟁사 사용시간" : {
            "경쟁사 A": "40%",
            "경쟁사 B": "35%",
            "자사": "25%"
        }
    }
    """ + \
    "이때, 반드시 JSON 형식으로만 응답해 주세요. 추가적인 설명이나 다른 텍스트는 포함하지 마세요.\n 또한, 대답에는 반드시 모든 추출한 텍스트를 포함하며, 추출한 텍스트 이외의 글자를 읽거나 유추하지 마세요."
    extractPrompt = "다음은 이미지에서 추출한 텍스트입니다:\n" + ocr_text + "\n이 텍스트를 바탕으로 아래의 JSON 스키마에 맞추어 정보를 추출해 주세요.\n"

    request = ChatRequest(
        provider=model_provider,
        model=model_name,
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
    
    # LLM 호출 카운터 증가
    if company:
        company.llm_call_count += 1
    
    # JSON 문자열을 dict로 변환
    response = parse_json(response)

    assert isinstance(response, dict), "Response is not a valid JSON"
    
    # 소요 시간 및 결과 로깅 (debug 모드일 때만)
    if debug:
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

def pyt_ocr(image: PIL.Image) -> list[str]:
    """
    이미지에서 텍스트 추출
    """
    text = pytesseract.image_to_string(image, lang='eng+kor')
    ## NOTE : 한국어, 영어만 됨
    lines = text.split('\n')
    lines = [line.strip() for line in lines if line.strip()]
    return lines

def paddle_ocr(image: PIL.Image, use_gpu: bool = True) -> list[str]:
    """
    PaddleOCR을 사용하여 이미지에서 텍스트 추출
    
    Args:
        image: PIL Image 객체
        use_gpu: GPU 사용 여부 (기본값: True)
    
    Returns:
        추출된 텍스트 라인 리스트
    """
    from paddleocr import PaddleOCR
    import numpy as np
    
    # PaddleOCR 초기화 (한국어+영어)
    # device='gpu' 또는 'cpu'로 지정
    device = 'gpu' if use_gpu else 'cpu'
    ocr = PaddleOCR(use_angle_cls=True, lang='korean', device=device)
    
    # PIL Image를 numpy array로 변환
    img_array = np.array(image)
    
    # OCR 수행
    result = ocr.ocr(img_array, cls=True)
    
    # 결과 파싱
    lines = []
    if result and result[0]:
        for line in result[0]:
            if line and len(line) >= 2:
                text = line[1][0]  # (text, confidence)에서 text 추출
                if text.strip():
                    lines.append(text.strip())
    
    return lines

def paddle_ocr_with_bbox(image: PIL.Image, use_gpu: bool = True) -> list[dict]:
    """
    PaddleOCR을 사용하여 이미지에서 텍스트와 bounding box 추출
    
    Args:
        image: PIL Image 객체
        use_gpu: GPU 사용 여부 (기본값: True)
    
    Returns:
        추출된 텍스트 정보 리스트. 각 항목은 다음 형식의 dict:
        {
            'text': str,           # 인식된 텍스트
            'confidence': float,   # 신뢰도 (0~1)
            'bbox': list,          # bounding box 좌표 [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
        }
    """
    from paddleocr import PaddleOCR
    import numpy as np
    
    # PaddleOCR 초기화 (한국어+영어)
    device = 'gpu' if use_gpu else 'cpu'
    ocr = PaddleOCR(use_angle_cls=True, lang='korean', device=device)
    
    # PIL Image를 numpy array로 변환
    img_array = np.array(image)
    
    # OCR 수행
    result = ocr.ocr(img_array, cls=True)
    
    # 결과 파싱
    ocr_results = []
    if result and result[0]:
        for line in result[0]:
            if line and len(line) >= 2:
                bbox = line[0]  # bounding box 좌표
                text_info = line[1]  # (text, confidence)
                text = text_info[0]
                confidence = text_info[1]
                
                if text.strip():
                    ocr_results.append({
                        'text': text.strip(),
                        'confidence': confidence,
                        'bbox': bbox
                    })
    
    return ocr_results

def CLOVA_ocr(image) -> list[str]:
    """
    CLOVA OCR API를 사용하여 이미지에서 텍스트 추출
    """
    dotenv.load_dotenv()
    from dotenv import find_dotenv
    env_path = find_dotenv()

    api_url = dotenv.get_key(env_path, "CLOVA_api_url")
    secret_key = dotenv.get_key(env_path, "CLOVA_secret_key")

    request_json = {
        'images': [
            {
                'format': 'png',
                'name': 'demo'
            }
        ],
        'requestId': str(uuid.uuid4()),
        'version': 'V2',
        'timestamp': int(round(time.time() * 1000))
    }

    payload = {'message': json.dumps(request_json).encode('UTF-8')}
    files = [
        ('file', ('image.png', image, 'image/png'))
    ]
    headers = {
        'X-OCR-SECRET': secret_key
    }

    response = requests.request("POST", api_url, headers=headers, data=payload, files=files)

    results = []
    if response.json().get('images') is not None:
        for i in response.json()['images'][0]['fields']:
            text = i['inferText']
            results.append(text)

    return results


def Upstage_ocr(image, model="ocr") -> list[str]:
    """
    Upstage OCR API를 사용하여 이미지에서 텍스트 추출
    """

    dotenv.load_dotenv()
    from dotenv import find_dotenv
    env_path = find_dotenv()

    api_key = dotenv.get_key(env_path, "UPSTAGE_api_key")
     
    url = "https://api.upstage.ai/v1/document-digitization"
    headers = {"Authorization": f"Bearer {api_key}"}
    files = {"document": image}
    if model == "document-parse":
        data = {"ocr": "force", "base64_encoding": "['table']", "model": "document-parse"} # NOTE: document parsing 모델
    else:
        data = {"model": "ocr"} # NOTE: OCR만
    response = requests.post(url, headers=headers, files=files, data=data)

    response_data = response.json()
    
    # 에러 체크
    if "error" in response_data:
        error_msg = response_data["error"].get("message", "Unknown error")
        error_code = response_data["error"].get("code", "")
        raise RuntimeError(f"Upstage OCR API Error ({error_code}): {error_msg}")
    
    texts = []
    # pages[0].words에서 텍스트 추출
    for w in response_data["pages"][0]["words"]:
        texts.append(w["text"])
    
    return texts