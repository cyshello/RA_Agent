"""
유틸리티 함수 모듈 - LangChain 기반

OCR 처리, JSON 파싱, 페이지 추출 Chain을 포함합니다.
"""

import json
import PIL
import PIL.Image
import dotenv
import os
import logging
import time
import requests
import uuid
import io
import asyncio

from langchain_core.output_parsers import JsonOutputParser
from langchain_core.runnables import RunnableLambda
from langchain_core.messages import HumanMessage

from .api import ChatRequest, Dispatcher, ModelFactory
from .prompts import EXTRACTION_PROMPT

OUTPUT_JSON_SCHEMA = {
    "section1" : {
        "scores": {
            "management_eval": {
                "score": 0,
                "reason": "",
                "needs_more_data": []
            },
            "national_agenda": {
                "score": 0,
                "reason": "",
                "needs_more_data": []
            },
            "co_growth": {
                "score": 0,
                "reason": "",
                "needs_more_data": []
            }
        },
        "radar": [
            { "axis": "정책정합성", "score": 0 },
            { "axis": "공공기관적합도", "score": 0 },
            { "axis": "시장성장성", "score": 0 },
            { "axis": "동반성장·협력", "score": 0 },
            { "axis": "신뢰도·레퍼런스", "score": 0 }
        ],
        "overall": {
            "overall_score": 0,
            "grade": "",
            "grade_rule": "",
            "reason_summary": "",
            "needs_more_data_global": [],
            "keywords": []
        }
    },
    "section2" : {
        "finance" : {
            "revenue" : {
                "year" : 0,
                "amount" : ""
            },
            "profit" : {
                "year" : 0,
                "amount" : ""
            },
            "invest" : {
                "year" : 0,
                "amount" : ""
            }
        },
        "performance" : {
            "contents" : []
        },
        "BM" : {
            "contents" : []
        },
        "competencies" : {
            "b2g_keywords" : [],
            "evidences" : []
        }
    },
    "section3" : {
        "market_growth" : 0.0, # 이 부분은 아래 data로부터 계산됨.
        "market_size" : {
            "unit" : "",
            "market_name" : "",
            "reference" : "",
            "data" : {
            }
        },
        "competition" : {
            "competitors" : [],
            "details" : [],
            "differentiation" : []
        },
        "tech_policy_trends" : {
            "keywords" : [],
            "evidences" : [
                {
                    "content" : "",
                    "source" : ""
                },
                {
                    "content" : "",
                    "source" : ""
                }
            ]
        }
    },
    "section4" : {
        "presidential_agenda" : {
            "top10" : [
            {
                "rank" : "",
                "name" : "",
                "description" : ""
            }],
            "analysis" : {
                "rank" : "",
                "insight" : {
                    "title" : "",
                    "details" : []
                },
                "risk" : {
                    "title" : "",
                    "details" : []
                },
                "consider" : []
            }
        },
        "management_eval" : {
            "top10" : [
                {
                    "rank" : "",
                    "name" : "",
                    "description" : ""
                }
            ],
           "analysis" : {
                "rank" : "",
                "insight" : {
                    "title" : "",
                    "details" : []
                },
                "risk" : {
                    "title" : "",
                    "details" : []
                },
                "consider" : []
            }
        },
        "inclusive_growth" : {
            "top10" : [
                {
                    "rank" : "",
                    "name" : "",
                    "description" : ""
                }
            ],
            "analysis" : {
                "rank" : "",
                "insight" : {
                    "title" : "",
                    "details" : []
                },
                "risk" : {
                    "title" : "",
                    "details" : []
                },
                "consider" : []
            }
        },
        "overall" : {
            "rank" : "",
            "expect" : []
        }
    },
    "section5" : {
        "weakness_analysis" : {
            "keyword" : "",
            "evidences" : []
        },
        "strategy" : {
            "keyword" : "",
            "strategy" : "",
            "details" : []
        },
        "to_do_list" : {
            "keyword" : "",
            "tasks" : [
                {
                    "content" : "",
                    "details" : []
                },
                {
                    "content" : "",
                    "details" : []
                }
            ]
        }
    }
}

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

### Upstage 사용하지 않음.
# def parse_upstage(response: str | dict | list) -> list[str]:
#     """
#     Upstage OCR 결과에서 HTML을 추출하고 유니코드를 디코딩
    
#     Args:
#         response: Upstage OCR 결과 (JSON 문자열, dict, 또는 list)
        
#     Returns:
#         디코딩된 텍스트 리스트
        
#     Example:
#         >>> response = {"elements": [{"content": {"html": "<p>캐시닥</p>", ...}, ...}]}
#         >>> parse_upstage(response)
#         ['캐시닥', ...]
#     """
#     try:
#         from html.parser import HTMLParser
        
#         class HTMLTextExtractor(HTMLParser):
#             def __init__(self):
#                 super().__init__()
#                 self.text = []
            
#             def handle_data(self, data):
#                 if data.strip():
#                     self.text.append(data.strip())
        
#         # 문자열이면 JSON으로 파싱
#         if isinstance(response, str):
#             parsed = json.loads(response)
#         elif isinstance(response, (dict, list)):
#             parsed = response
#         else:
#             raise ValueError(f"Cannot parse response of type {type(response)}")
        
#         text_list = []
        
#         # Upstage OCR 응답 구조: {"elements": [...], ...}
#         if isinstance(parsed, dict):
#             elements = parsed.get("elements", [])
#             if not elements and "content" in parsed:
#                 # 단일 요소인 경우
#                 elements = [parsed]
#         elif isinstance(parsed, list):
#             elements = parsed
#         else:
#             elements = []
        
#         for item in elements:
#             if isinstance(item, dict):
#                 # content.html 필드 찾기
#                 if "content" in item and isinstance(item["content"], dict):
#                     html = item["content"].get("html", "")
#                     if html:
#                         # HTML에서 텍스트 추출
#                         parser = HTMLTextExtractor()
#                         try:
#                             parser.feed(html)
#                             extracted = " ".join(parser.text)
#                             if extracted:
#                                 text_list.append(extracted)
#                         except:
#                             # HTML 파싱 실패시 content.text 사용
#                             text = item["content"].get("text", "")
#                             if text:
#                                 text_list.append(text)
        
#         return text_list
        
#     except json.JSONDecodeError as e:
#         raise ValueError(f"Failed to parse JSON: {e}\nResponse: {response}")
#     except Exception as e:
#         raise ValueError(f"Error parsing Upstage response: {e}\nResponse: {response}")

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

async def extractJSON(image: PIL.Image.Image, dispatcher: Dispatcher, page_num: int = None, start_time_base: float = None, debug: bool = False, model_provider: str = "openai", model_name: str = "gpt-4o", company=None) -> tuple[dict, str]:
    """
    이미지에서 JSON 데이터를 추출하는 LCEL 체인 기반 함수
    
    Args:
        image: 분석할 PIL Image 객체
        dispatcher: API 디스패처 (rate limiting 용도)
        page_num: 페이지 번호
        start_time_base: 전체 분석 시작 시간
        debug: 디버그 모드 여부
        model_provider: AI 모델 제공자 ("openai" 또는 "gemini")
        model_name: 사용할 구체적인 모델명
        company: Company 객체 (API 호출 카운터 증가용)
    
    Returns:
        tuple: (추출된 JSON dict, OCR 텍스트)
    """
    page_start_time = time.time()
    page_label = f"페이지 {page_num}" if page_num is not None else "페이지"
    
    if debug:
        if start_time_base:
            elapsed_from_main = page_start_time - start_time_base
            logger.info(f"{page_label} 분석 시작 (전체 시작 후 {elapsed_from_main:.2f}초)")
        else:
            logger.info(f"{page_label} 분석 시작")

    # 1. OCR 실행 (별도 스레드에서 실행하여 블로킹 방지)
    loop = asyncio.get_event_loop()
    img_byte_arr = io.BytesIO()
    image.save(img_byte_arr, format='PNG')
    img_byte_arr = img_byte_arr.getvalue()
    
    # CLOVA OCR 사용
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

    # 2. LCEL 체인 구성: 멀티모달 입력 준비 | 프롬프트 | 모델 | 파서
    
    # 2-1. 멀티모달 메시지 생성 함수
    def create_multimodal_message(inputs: dict) -> list:
        """OCR 텍스트와 이미지를 멀티모달 메시지로 변환"""
        ocr_text = inputs.get("ocr_text", "")
        image = inputs.get("image")
        
        # 프롬프트 템플릿에서 메시지 추출
        messages = EXTRACTION_PROMPT.format_messages(ocr_text=ocr_text)
        
        # 멀티모달 처리: user 메시지에 이미지 추가
        import base64
        buffered = io.BytesIO()
        image.save(buffered, format="PNG")
        img_base64 = base64.b64encode(buffered.getvalue()).decode()
        
        if model_provider == "openai":
            # OpenAI: base64 URL 형식
            for msg in messages:
                if msg.type == "human":
                    msg.content = [
                        {"type": "text", "text": msg.content},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/png;base64,{img_base64}"}
                        }
                    ]
        elif model_provider == "gemini":
            # Gemini: base64 문자열 형식
            for msg in messages:
                if msg.type == "human":
                    msg.content = [
                        {"type": "text", "text": msg.content},
                        {"type": "image_url", "image_url": f"data:image/png;base64,{img_base64}"}
                    ]
        
        return messages
    
    # 2-2. Rate limiting이 적용된 모델 생성
    model = ModelFactory.create_model_chain(
        provider=model_provider,
        model_name=model_name,
        output_format="json",
        max_rps=dispatcher.max_rps
    )
    
    # 2-3. LCEL 체인 구성
    extraction_chain = (
        RunnableLambda(create_multimodal_message)
        | model
        | JsonOutputParser()
    )
    
    # 3. 체인 실행
    response = await extraction_chain.ainvoke({
        "ocr_text": ocr_text,
        "image": image
    })
    
    # LLM 호출 카운터 증가
    if company:
        company.llm_call_count += 1
    
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


def CLOVA_ocr_with_table(image) -> dict:
    """
    CLOVA OCR API를 사용하여 이미지에서 텍스트와 표(Table) 추출
    
    Args:
        image: 이미지 바이트 데이터
        
    Returns:
        dict: {
            'text': str,  # 전체 텍스트 (줄바꿈으로 구분)
            'fields': list[str],  # 일반 텍스트 필드 리스트
            'tables': list[dict],  # 표 데이터 리스트
            'raw_response': dict  # 원본 응답
        }
    """
    import os
    
    # 먼저 환경변수에서 찾기
    api_url = os.environ.get("CLOVA_api_url")
    secret_key = os.environ.get("CLOVA_secret_key")
    
    # 없으면 .env 파일에서 찾기
    if not api_url or not secret_key:
        dotenv.load_dotenv()
        from dotenv import find_dotenv
        env_path = find_dotenv()
        
        if env_path:
            api_url = dotenv.get_key(env_path, "CLOVA_api_url")
            secret_key = dotenv.get_key(env_path, "CLOVA_secret_key")
    
    # 여전히 없으면 src/.env에서 직접 로드
    if not api_url or not secret_key:
        src_env_path = os.path.join(os.path.dirname(__file__), ".env")
        if os.path.exists(src_env_path):
            dotenv.load_dotenv(src_env_path)
            api_url = dotenv.get_key(src_env_path, "CLOVA_api_url")
            secret_key = dotenv.get_key(src_env_path, "CLOVA_secret_key")

    # API 설정 확인
    if not api_url:
        raise Exception("CLOVA OCR API URL이 설정되지 않았습니다. .env 파일에 CLOVA_api_url을 설정하세요.")
    if not secret_key:
        raise Exception("CLOVA OCR Secret Key가 설정되지 않았습니다. .env 파일에 CLOVA_secret_key를 설정하세요.")

    request_json = {
        'images': [
            {
                'format': 'png',
                'name': 'demo'
            }
        ],
        'requestId': str(uuid.uuid4()),
        'version': 'V2',
        'timestamp': int(round(time.time() * 1000)),
        'enableTableDetection': True  # 표 인식 활성화
    }

    payload = {'message': json.dumps(request_json).encode('UTF-8')}
    files = [
        ('file', ('image.png', image, 'image/png'))
    ]
    headers = {
        'X-OCR-SECRET': secret_key
    }

    response = requests.request("POST", api_url, headers=headers, data=payload, files=files)
    
    # API 응답 상태 확인
    if response.status_code != 200:
        raise Exception(f"CLOVA OCR API 오류: HTTP {response.status_code} - {response.text[:200]}")
    
    # 응답이 비어있는지 확인
    if not response.text or response.text.strip() == '':
        raise Exception(f"CLOVA OCR API 응답이 비어있습니다. API URL이나 Secret Key를 확인하세요. (api_url: {api_url[:30] if api_url else 'None'}...)")
    
    try:
        response_json = response.json()
    except Exception as e:
        raise Exception(f"CLOVA OCR API 응답 JSON 파싱 실패: {e}. 응답 내용: {response.text[:200]}")
    
    result = {
        'text': '',
        'fields': [],
        'tables': [],
        'raw_response': response_json
    }
    
    if response_json.get('images') is not None:
        image_result = response_json['images'][0]
        
        # 일반 텍스트 필드 추출
        if 'fields' in image_result:
            for field in image_result['fields']:
                text = field.get('inferText', '')
                if text:
                    result['fields'].append(text)
        
        # 표(Table) 데이터 추출
        if 'tables' in image_result:
            for table in image_result['tables']:
                table_data = {
                    'cells': [],
                    'text': table.get('inferText', ''),
                    'rows': {},  # row_index -> {col_index: cell_text}
                    'markdown': ''  # 마크다운 형식 표
                }
                
                if 'cells' in table:
                    max_row = 0
                    max_col = 0
                    
                    for cell in table['cells']:
                        row_idx = cell.get('rowIndex', 0)
                        col_idx = cell.get('columnIndex', 0)
                        row_span = cell.get('rowSpan', 1)
                        col_span = cell.get('columnSpan', 1)
                        
                        # 셀 텍스트 추출
                        cell_text = ''
                        if 'cellTextLines' in cell:
                            cell_texts = []
                            for line in cell['cellTextLines']:
                                if 'cellWords' in line:
                                    line_text = ' '.join([
                                        word.get('inferText', '') 
                                        for word in line['cellWords']
                                    ])
                                    cell_texts.append(line_text)
                            cell_text = ' '.join(cell_texts)
                        
                        cell_info = {
                            'row': row_idx,
                            'col': col_idx,
                            'row_span': row_span,
                            'col_span': col_span,
                            'text': cell_text
                        }
                        table_data['cells'].append(cell_info)
                        
                        # rows 딕셔너리에 저장
                        if row_idx not in table_data['rows']:
                            table_data['rows'][row_idx] = {}
                        table_data['rows'][row_idx][col_idx] = cell_text
                        
                        max_row = max(max_row, row_idx)
                        max_col = max(max_col, col_idx)
                    
                    # 마크다운 표 생성
                    md_lines = []
                    for r in range(max_row + 1):
                        row_cells = []
                        for c in range(max_col + 1):
                            cell_val = table_data['rows'].get(r, {}).get(c, '')
                            row_cells.append(cell_val)
                        md_lines.append('| ' + ' | '.join(row_cells) + ' |')
                        if r == 0:  # 헤더 구분선
                            md_lines.append('|' + '---|' * (max_col + 1))
                    
                    table_data['markdown'] = '\n'.join(md_lines)
                
                result['tables'].append(table_data)
        
        # 전체 텍스트 조합 (필드 + 표)
        all_text_parts = result['fields'].copy()
        for table in result['tables']:
            if table['markdown']:
                all_text_parts.append(f"\n[표]\n{table['markdown']}\n")
        
        result['text'] = '\n'.join(all_text_parts)
    
    return result


