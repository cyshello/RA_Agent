# main pipeline
from src.api import (
    ChatRequest,
    Dispatcher,
)
import os
import json
import logging
import time
from src.utils import extractJSON, parse_json
import asyncio

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s - %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)


class Document():
    """
    이 클래스는 하나의 문서를 나타냅니다.
    file_path: 문서(PDF) 파일 경로

    """
    def __init__(self, file_path: str):
        self.file_path = file_path
        self.pages = [] 
        self.analysis = [] # 모든 페이지를 각자 분석한 결과
        self.ocr_texts = [] # 모든 페이지의 OCR 텍스트
        self.organized = {} # 전체 문서의 종합 구조화 (minor detail, 필요없는 슬라이드 등 제외)
    
    def convert_images(self):
        """
        PDF 문서를 페이지 단위로 나누어 이미지(PIL.Image 객체) 리스트로 변환
        """
        from pdf2image import convert_from_path
        self.pages = convert_from_path(self.file_path)
        return self.pages
    
    def get_pages(self):
        return self.pages
    
    async def analyze_doc(self, dispatcher: Dispatcher, start_time: float = None, debug: bool = False, max_rps: float = 1.0):
        """
        문서의 각 페이지 이미지를 분석하여 JSON 데이터 추출 (병렬 처리)
        
        Args:
            max_rps: 초당 최대 요청 수 (Upstage API rate limit 고려, 기본값: 1 RPS)
                    Tier 0: 1 RPS, Tier 1: 3 RPS, Tier 2: 10 RPS
        """
        # 요청 간 최소 간격 계산 (초)
        min_interval = 1.0 / max_rps
        last_request_time = [0]  # 리스트로 감싸서 내부 함수에서 수정 가능하게
        lock = asyncio.Lock()
        
        async def rate_limited_extract(page, i):
            async with lock:
                # 마지막 요청 이후 경과 시간 확인
                elapsed = time.time() - last_request_time[0]
                if elapsed < min_interval:
                    # 최소 간격을 만족하지 않으면 대기
                    await asyncio.sleep(min_interval - elapsed)
                last_request_time[0] = time.time()
            
            return await extractJSON(page, dispatcher, i, start_time, debug)
        
        # 모든 페이지를 rate limit을 지키며 분석
        tasks = [asyncio.create_task(rate_limited_extract(page, i)) 
                 for i, page in enumerate(self.pages)]
        results = await asyncio.gather(*tasks)
        
        # 결과를 페이지 번호와 함께 저장
        self.analysis = [{i: result[0]} for i, result in enumerate(results)]
        self.ocr_texts = [{i:ocr_text} for i, (_, ocr_text) in enumerate(results)]
        
        return self.analysis

class Company():
    """
    여러 문서를 넣어서 요청한 회사 단위
    """
    def __init__(self, name: str):
        self.name = name
        self.documents = {}
        self.dispatcher = Dispatcher()
        self.reports = {}
        self.report_types = ["competencies", "b2g_strategy","market"] #우선 3개만
    
    def add_document(self, name: str, document: Document):
        """문서를 이름과 함께 추가"""
        self.documents[name] = document

    async def process_documents(self, debug: bool = False, max_rps: float = 0.5):
        """
        모든 문서를 병렬로 처리하고 결과를 저장
        
        Args:
            debug: True일 경우 각 페이지별 시간과 전체 소요 시간을 로깅
            max_rps: 초당 최대 요청 수 (Upstage API rate limit 고려, 기본값: 1 RPS)
                    Tier 0: 1 RPS, Tier 1: 3 RPS, Tier 2: 10 RPS
        """
        # 타이머 시작
        start_time = time.time()
        
        # 모든 문서를 이미지로 변환
        for name, document in self.documents.items():
            document.convert_images()
            logger.info(f"문서 '{name}' ({document.file_path}) - 총 {len(document.pages)}페이지")
        
        if debug:
            logger.info(f"분석 시작 (debug 모드, RPS: {max_rps})")
        else:
            logger.info(f"분석 시작 (RPS: {max_rps})")
        
        # 모든 문서를 병렬로 분석
        tasks = [asyncio.create_task(document.analyze_doc(self.dispatcher, start_time, debug, max_rps))
                 for document in self.documents.values()]
        await asyncio.gather(*tasks)
        
        # 전체 소요 시간
        total_time = time.time() - start_time
        if debug:
            logger.info(f"전체 분석 완료 - 소요 시간: {total_time:.2f}초")
        else:
            logger.info(f"전체 분석 완료")
        
        # results 디렉토리 생성 (없으면)
        results_dir = "src/results"
        os.makedirs(results_dir, exist_ok=True)
        
        # 각 문서의 결과를 저장
        for name, document in self.documents.items():
            output_path = os.path.join(results_dir, f"{name}.json")
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(document.analysis, f, ensure_ascii=False, indent=2)
            
            ocr_path = os.path.join(results_dir, f"{name}_ocr.json")
            with open(ocr_path, 'w', encoding='utf-8') as f:
                json.dump(document.ocr_texts, f, ensure_ascii=False, indent=2)
            
            logger.info(f"분석 결과 저장: {output_path}")
        
        return [doc.analysis for doc in self.documents.values()]

    async def create_report(self, model:str = "openai", web: bool = False, report_type: str = "competencies"):
        """
        모든 문서를 참고하여 각 부분의 보고서 생성.
        """
        from src.prompts import PROMPTS

        sys_prompt = PROMPTS[report_type]["system"]
        user_prompt = PROMPTS[report_type]["user"]
        json_schema = PROMPTS[report_type]["json"]

        request = ChatRequest(
            provider=model,
            model="gpt-4o" if model=="openai" else "gemini-2.0-flash-exp",
            messages=[
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": user_prompt + f"\n\n문서 데이터: {json.dumps([doc.analysis for doc in self.documents.values()], ensure_ascii=False)}"} #모든 문서의 분석 결과 포함
            ],
            input="text-only",
            output="json",
            web=web
        )
        response = await self.dispatcher.dispatch(request)

        ## TODO : JSON 파싱 필요!!

        response = parse_json(response)

        return response    
    
    async def generate_all_reports(self, model: str = "openai", web: bool = False):
        """
        모든 종류의 보고서를 병렬로 생성하여 self.reports에 저장
        
        Args:
            model: 사용할 AI 모델 ("openai" 또는 "gemini")
            web: 웹 검색 활성화 여부
        """
        logger.info(f"보고서 생성 시작 - 총 {len(self.report_types)}개 유형")
        
        # 모든 보고서 타입을 병렬로 생성
        tasks = [
            asyncio.create_task(self.create_report(model=model, web=web, report_type=report_type))
            for report_type in self.report_types
        ]
        
        results = await asyncio.gather(*tasks)
        
        # 결과를 self.reports에 저장
        for report_type, result in zip(self.report_types, results):
            self.reports[report_type] = result
            logger.info(f"'{report_type}' 보고서 생성 완료")
        
        # results 디렉토리에 저장
        results_dir = "src/results"
        os.makedirs(results_dir, exist_ok=True)
        
        reports_path = os.path.join(results_dir, f"{self.name}_reports.json")
        with open(reports_path, 'w', encoding='utf-8') as f:
            json.dump(self.reports, f, ensure_ascii=False, indent=2)
        
        logger.info(f"모든 보고서 저장 완료: {reports_path}")
        
        return self.reports

async def main_async(debug: bool = False):
    """
    회사 단위로 여러 문서를 병렬 처리 및 보고서 생성
    
    Args:
        debug: True일 경우 각 페이지별 시간과 전체 소요 시간을 로깅
    """
    com = Company("example")
    name = "IR1"
    doc = "data/IR1.pdf"
    com.add_document(name, Document(doc))
    
    # 결과 파일 경로 확인
    results_dir = "src/results"
    result_file = os.path.join(results_dir, f"{name}.json")
    ocr_file = os.path.join(results_dir, f"{name}_ocr.json")
    
    # 이미 분석 결과가 있으면 로드, 없으면 분석 실행
    if os.path.exists(result_file) and os.path.exists(ocr_file):
        logger.info(f"기존 분석 결과 로드: {result_file}")
        with open(result_file, 'r', encoding='utf-8') as f:
            com.documents[name].analysis = json.load(f)
        with open(ocr_file, 'r', encoding='utf-8') as f:
            com.documents[name].ocr_texts = json.load(f)
    else:
        # 모든 문서를 병렬로 처리
        await com.process_documents(debug=debug)
    
    # TODO : 보고서 생성 이전에 skeleton prompt 작성 (JSON에서 좀 더 자연화된 프롬프트, 또는 minor detail 제거 등) 
    # 정말 필요할지 의문. hallucination 문제가 발생할 위험이 있다.
    # TODO : 외부 지식 tool 활용 기능


    # 모든 보고서 생성
    #await com.generate_all_reports(model="openai", web=False)
    await com.generate_all_reports(model="gemini", web=True) #웹 검색

def main(debug: bool = False):
    """
    main_async 함수를 실행하는 래퍼
    """
    asyncio.run(main_async(debug=debug))

main(debug=True)

#### test function for one page extraction ####
async def test():
    """extractJSON 함수 단일 테스트"""
    import PIL.Image
    
    img_path = "data/image4.png"
    image = PIL.Image.open(img_path)
    
    dispatcher = Dispatcher()
    result = await extractJSON(image, dispatcher, 0)

    print("분석 결과:", result[0])
    print("\nOCR 텍스트:", result[1])
    
    return result

# test() 실행하려면:
# asyncio.run(test())