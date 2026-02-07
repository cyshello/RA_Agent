"""
LangChain 기반 기업 분석 및 보고서 생성 파이프라인

이 모듈은 PDF 문서를 분석하고 기업 보고서를 생성하는 메인 파이프라인입니다.
LangChain을 활용하여 OCR, 문서 분석, 보고서 생성을 Chain 구조로 처리합니다.
"""

# main pipeline
from src.api import ChatRequest, Dispatcher, ModelFactory
from src.prompts import PROMPTS
from src.utils import extractJSON, parse_json, OUTPUT_JSON_SCHEMA
from src.db_main import SCHEMA_REGISTRY
from langchain_core.runnables import RunnableParallel
from langchain_core.output_parsers import JsonOutputParser
import os
import json
import logging
import time
import asyncio
import argparse
import sys

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
    
    async def analyze_doc(self, dispatcher: Dispatcher, start_time: float = None, debug: bool = False, max_rps: float = 1.0, model_provider: str = "openai", model_name: str = "gpt-4o", company=None):
        """
        문서의 각 페이지 이미지를 분석하여 JSON 데이터 추출 (병렬 처리)
        
        Args:
            max_rps: 초당 최대 요청 수 (기본값: 1 RPS)
            model_provider: 사용할 AI 모델 제공자 ("openai" 또는 "gemini")
            model_name: 사용할 AI 모델명
            company: Company 객체 (API 호출 카운터 증가용)
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
            
            return await extractJSON(page, dispatcher, i, start_time, debug, model_provider, model_name, company)
        
        # 모든 페이지를 rate limit을 지키며 분석
        tasks = [asyncio.create_task(rate_limited_extract(page, i)) 
                 for i, page in enumerate(self.pages)]
        results = await asyncio.gather(*tasks)

        # 모든 페이지 분석 이후에 재무현황만 추출 (여러 개 있다면 기준년도가 최신인 것, 기준년도가 같다면 임의로 마지막 값 사용)
        # NOTE : 재무현황 추출
        import re

        for i, (result, ocr_text) in enumerate(results):
            # 재무현황 키가 없으면 건너뛰기
            if "재무현황" not in result:
                print(f"재무현황 디버깅: 페이지 {i}에 '재무현황' 키가 없음")
                continue
                
            for k in [("revenue","매출"), ("profit","영업이익"), ("invest","누적투자")]:  
                # 해당 재무 항목이 없거나 빈 값인 경우 체크
                if k[1] not in result["재무현황"]:
                    print(f"재무현황 디버깅: 페이지 {i}에 '{k[1]}' 키가 없음")
                    continue
                    
                finance_item = result["재무현황"][k[1]]
                
                # 금액 또는 기준년도 키가 없거나 빈 값인 경우 체크
                if not isinstance(finance_item, dict):
                    print(f"재무현황 디버깅: 페이지 {i}의 '{k[1]}'가 dict가 아님: {finance_item}")
                    continue
                    
                amount = finance_item.get("금액", "")
                year_raw = finance_item.get("기준년도", "")
                
                if amount == "" or amount is None or year_raw == "" or year_raw is None:
                    # 둘 중 하나라도 값이 없으면 패스 -> 둘 다 있어야지만 저장됨.
                    print(f"재무현황 디버깅: 페이지 {i}의 '{k[1]}' - 금액({amount}) 또는 기준년도({year_raw})가 비어있음")
                    continue
                    
                year = 0
                
                if year_raw is None:
                    year = 0
                elif isinstance(year_raw, int):
                    year = year_raw
                else:
                    # 기준년도에서 숫자만 추출
                    year_match = re.search(r'\d{4}', str(year_raw))
                    if year_match:
                        year = int(year_match.group(0))
                    else:
                        year = 0

                print(f"재무현황 디버깅: 페이지 {i}, {k[1]}, 금액={amount}, 기준년도={year}")

                if company.result_json["section2"]["finance"][k[0]]["year"] == 0 and (company.result_json["section2"]["finance"][k[0]]["amount"] == "" or company.result_json["section2"]["finance"][k[0]]["amount"] is None):
                    company.result_json["section2"]["finance"][k[0]]["year"] = year #반드시 정수로 저장
                    company.result_json["section2"]["finance"][k[0]]["amount"] = amount
                elif company.result_json["section2"]["finance"][k[0]]["year"] != 0: #이미 값이 존재한다면 기준 년도 비교해서 최신 값으로 대체
                    if company.result_json["section2"]["finance"][k[0]]["year"] < year:
                        company.result_json["section2"]["finance"][k[0]]["year"] = year
                        company.result_json["section2"]["finance"][k[0]]["amount"] = amount
            
            # 재무현황 키가 있을 때만 제거
            if "재무현황" in result:
                result.pop("재무현황")  # 페이지별 결과에서 재무현황 제거

        # 결과를 페이지 번호와 함께 저장
        self.analysis = [{i: result[0]} for i, result in enumerate(results)]
        self.ocr_texts = [{i:ocr_text} for i, (_, ocr_text) in enumerate(results)]
        
        return self.analysis

class Company():
    """
    여러 문서를 넣어서 요청한 회사 단위
    """
    def __init__(self, name: str, max_rps: float = 1.0):
        """
        Args:
            name: 회사 이름
            max_rps: LLM API 초당 최대 요청 수 (기본값: 1.0)
        """
        self.name = name
        self.documents = {}
        self.dispatcher = Dispatcher(max_rps=max_rps)
        self.reports = {}
        self.report_types = ["overall_competency", "competencies", "market"]  # b2g_strategy는 section4 이후 별도 생성
        self.result_dir = None  # 결과 저장 폴더 경로
        self.result_json = OUTPUT_JSON_SCHEMA.copy() if OUTPUT_JSON_SCHEMA else {} # OUTPUT_JSON_SCHEMA를 초기값으로 사용
        # API 호출 카운터
        self.ocr_call_count = 0
        self.llm_call_count = 0
    
    def add_document(self, name: str, document: Document):
        """문서를 이름과 함께 추가"""
        self.documents[name] = document
    
    def setup_result_directory(self, web_search: bool = False, max_rps: float = 2.0, debug: bool = False):
        """
        결과 저장 폴더 생성: 회사명_문서명1_문서명2_..._web_debug_rps
        
        모델 설정 (하드코딩):
            - 문서 추출 및 Section 1, 2: OpenAI GPT-4o
            - Section 3, 4, 5: Google Gemini 2.0 Pro
        """
        doc_names = "_".join(self.documents.keys())
        
        # 폴더명 구성 요소
        parts = [
            self.name.replace(" ", "_"),  # 공백 제거
            doc_names,
        ]
        
        # 선택적 파라미터 추가
        if web_search:
            parts.append("web")
        if debug:
            parts.append("debug")
        
        # RPS 추가 (소수점 제거)
        rps_str = f"rps{max_rps:.1f}".replace(".", "_")
        parts.append(rps_str)
        
        folder_name = "_".join(parts)
        self.result_dir = os.path.join("src/results", folder_name)
        os.makedirs(self.result_dir, exist_ok=True)
        return self.result_dir

    def _load_b2g_criteria_data(self) -> dict:
        """
        B2G 기준 데이터 로드 (Section 1 overall_competency용)
        
        로드하는 파일:
            - script/output/extracted_projects.json (국정과제)
            - script/output_inclusive_growth/indicators_final.json (동반성장 평가지표)
            - script/output_management_eval/indicators_final.json (경영평가 지표)
        
        Returns:
            dict: 각 평가 유형별 요약 데이터
        """
        result = {
            "national_agenda": "",
            "management_eval": "",
            "inclusive_growth": ""
        }
        
        # 국정과제 데이터 로드
        national_agenda_path = "script/output/extracted_projects.json"
        if os.path.exists(national_agenda_path):
            try:
                with open(national_agenda_path, 'r', encoding='utf-8') as f:
                    projects = json.load(f)
                # 주요 과제명과 목표만 추출하여 요약
                summaries = []
                for proj in projects[:30]:  # 상위 30개만
                    summary = f"[{proj.get('과제번호', '')}] {proj.get('과제명', '')}"
                    goals = proj.get('과제 목표', [])
                    if goals:
                        summary += f" - 목표: {'; '.join(goals[:2])}"
                    summaries.append(summary)
                result["national_agenda"] = "\n".join(summaries)
                logger.info(f"국정과제 데이터 로드 완료: {len(projects)}개 과제")
            except Exception as e:
                logger.warning(f"국정과제 데이터 로드 실패: {e}")
        
        # 경영평가 지표 데이터 로드
        management_eval_path = "script/output_management_eval/indicators_final.json"
        if os.path.exists(management_eval_path):
            try:
                with open(management_eval_path, 'r', encoding='utf-8') as f:
                    indicators = json.load(f)
                # 지표명과 평가기준 요약
                summaries = []
                for ind in indicators:
                    name = ind.get('지표명', '')
                    criteria = ind.get('평가기준', [])
                    if name:
                        summary = f"- {name}"
                        if criteria:
                            summary += f": {'; '.join(criteria[:2])}"
                        summaries.append(summary)
                result["management_eval"] = "\n".join(summaries)
                logger.info(f"경영평가 지표 데이터 로드 완료: {len(indicators)}개 지표")
            except Exception as e:
                logger.warning(f"경영평가 지표 데이터 로드 실패: {e}")
        
        # 동반성장 평가지표 데이터 로드
        inclusive_growth_path = "script/output_inclusive_growth/indicators_final.json"
        if os.path.exists(inclusive_growth_path):
            try:
                with open(inclusive_growth_path, 'r', encoding='utf-8') as f:
                    indicators = json.load(f)
                # 지표명과 평가기준 요약
                summaries = []
                for ind in indicators:
                    name = ind.get('지표명', '')
                    criteria = ind.get('평가기준', [])
                    if name:
                        summary = f"- {name}"
                        if criteria:
                            summary += f": {'; '.join(criteria[:2])}"
                        summaries.append(summary)
                result["inclusive_growth"] = "\n".join(summaries)
                logger.info(f"동반성장 평가지표 데이터 로드 완료: {len(indicators)}개 지표")
            except Exception as e:
                logger.warning(f"동반성장 평가지표 데이터 로드 실패: {e}")
        
        return result

    async def process_documents(self, debug: bool = False, max_rps: float = 2.0, ocr_provider: str = "CLOVA", web_search: bool = False):
        """
        모든 문서를 병렬로 처리하고 결과를 저장
        
        모델: OpenAI GPT-4o (하드코딩)
        
        Args:
            debug: True일 경우 각 페이지별 시간과 전체 소요 시간을 로깅
            max_rps: 초당 최대 요청 수 (기본값: 1 RPS)
            ocr_provider: OCR API 종류 ("CLOVA" 또는 "Upstage")
            web_search: 웹 검색 활성화 여부
        """
        # 결과 저장 폴더 설정
        if self.result_dir is None:
            self.setup_result_directory(
                web_search=web_search,
                max_rps=max_rps,
                debug=debug
            )
        
        # 타이머 시작
        start_time = time.time()
        
        # 모든 문서를 이미지로 변환
        for name, document in self.documents.items():
            document.convert_images()
            logger.info(f"문서 '{name}' ({document.file_path}) - 총 {len(document.pages)}페이지")
        
        if debug:
            logger.info(f"분석 시작 (debug 모드, RPS: {max_rps}, Extract Model: openai/gpt-4o)")
        else:
            logger.info(f"분석 시작 (RPS: {max_rps})")
        
        # 모든 문서를 병렬로 분석 (OpenAI GPT-4o 사용)
        tasks = [asyncio.create_task(document.analyze_doc(self.dispatcher, start_time, debug, max_rps, "openai", "gpt-4o", self))
                 for document in self.documents.values()]
        await asyncio.gather(*tasks)
        
        # 전체 소요 시간
        total_time = time.time() - start_time
        if debug:
            logger.info(f"전체 분석 완료 - 소요 시간: {total_time:.2f}초")
        else:
            logger.info(f"전체 분석 완료")
        
        # 각 문서의 결과를 결과 폴더에 저장 (debug 모드일 때만)
        if debug:
            for name, document in self.documents.items():
                # 분석 결과 저장
                output_path = os.path.join(self.result_dir, f"{name}.json")
                
                with open(output_path, 'w', encoding='utf-8') as f:
                    json.dump(document.analysis, f, ensure_ascii=False, indent=2)
                
                # OCR 결과 저장 (페이지 번호 포함 형식)
                ocr_path = os.path.join(self.result_dir, f"{name}_ocr.json")
                # OCR 결과를 페이지별로 구조화
                ocr_by_page = {}
                for ocr_item in document.ocr_texts:
                    for page_num, text in ocr_item.items():
                        ocr_by_page[f"page_{page_num}"] = text
                
                with open(ocr_path, 'w', encoding='utf-8') as f:
                    json.dump(ocr_by_page, f, ensure_ascii=False, indent=2)
                
                logger.info(f"분석 결과 저장: {output_path}")
                logger.info(f"OCR 결과 저장: {ocr_path}")
        
        return [doc.analysis for doc in self.documents.values()]

    async def create_report(self, model: str = "openai", model_name: str = "gpt-4o", web: bool = False, report_type: str = "competencies"):
        """
        LCEL 체인을 사용하여 특정 타입의 보고서를 생성
        
        Args:
            model: 사용할 AI 모델 제공자 ("openai" 또는 "gemini")
            model_name: 사용할 구체적인 모델명
            web: 웹 검색 활성화 여부
            report_type: 보고서 유형 ("competencies", "market", "b2g_strategy")
        
        Returns:
            파싱된 JSON 보고서 dict
        """
        # 문서 데이터 준비
        document_data = json.dumps(
            [doc.analysis for doc in self.documents.values()],
            ensure_ascii=False
        )
        
        # LCEL 체인 구성: 프롬프트 | 모델 | 파서
        prompt_template = PROMPTS[report_type]
        
        # Rate limiting이 적용된 모델 생성
        rate_limited_model = ModelFactory.create_model_chain(
            provider=model,
            model_name=model_name,
            output_format="json",
            web_search=web if report_type == "market" else False,  # NOTE: market 보고서에만 web_search 적용
            max_rps=self.dispatcher.max_rps
        )
        
        # LCEL 체인 구성
        report_chain = (
            prompt_template
            | rate_limited_model
            | JsonOutputParser()
        )
        
        # 체인 실행
        response = await report_chain.ainvoke({"document_data": document_data})
        
        return response    
    
    async def generate_all_reports(self, web: bool = False, debug: bool = False):
        """
        RunnableParallel을 사용하여 모든 보고서를 병렬로 생성
        
        모델 설정 (하드코딩):
            - Section 1, 2 (overall_competency, competencies): OpenAI GPT-4o
            - Section 3 (market): Google Gemini 2.0 Pro
        
        Args:
            web: 웹 검색 활성화 여부
            debug: 디버그 모드
        """
        report_start_time = time.time()
        logger.info(f"보고서 생성 시작 - 총 {len(self.report_types)}개 유형")
        logger.info(f"  Section 1, 2: openai/gpt-4o | Section 3: gemini/gemini-3-pro-preview | Web: {web}")
        
        # 문서 데이터 준비
        document_data = json.dumps(
            [doc.analysis for doc in self.documents.values()],
            ensure_ascii=False
        )
        
        # B2G 기준 데이터 로드 (Section 1용)
        b2g_criteria_data = self._load_b2g_criteria_data()
        
        # Section 1, 2용 모델 (OpenAI GPT-4o)
        gpt4o_model = ModelFactory.create_model_chain(
            provider="openai",
            model_name="gpt-4o",
            output_format="json",
            web_search=False,
            max_rps=self.dispatcher.max_rps
        )
        
        # Section 3용 모델 (Gemini 2.0 Pro)
        gemini_model = ModelFactory.create_model_chain(
            provider="gemini",
            model_name="gemini-3-pro-preview",
            output_format="json",
            web_search=web,  # market 보고서에만 web_search 적용
            max_rps=self.dispatcher.max_rps
        )
        
        # 각 보고서 타입별로 LCEL 체인 구성 (섹션에 따라 다른 모델 사용)
        report_chains = {}
        for report_type in self.report_types:
            if report_type == "overall_competency":
                # Section 1: GPT-4o + B2G 기준 데이터
                report_chains[report_type] = PROMPTS[report_type] | gpt4o_model | JsonOutputParser()
            elif report_type == "competencies":
                # Section 2: GPT-4o
                report_chains[report_type] = PROMPTS[report_type] | gpt4o_model | JsonOutputParser()
            else:
                # Section 3 (market): Gemini 2.0 Pro
                report_chains[report_type] = PROMPTS[report_type] | gemini_model | JsonOutputParser()
        
        # RunnableParallel로 병렬 실행
        parallel_chain = RunnableParallel(report_chains)
        
        # 입력 데이터 준비 (Section 1용 B2G 기준 데이터 포함)
        input_data = {
            "document_data": document_data,
            "national_agenda_data": b2g_criteria_data.get("national_agenda", "데이터 없음"),
            "management_eval_data": b2g_criteria_data.get("management_eval", "데이터 없음"),
            "inclusive_growth_data": b2g_criteria_data.get("inclusive_growth", "데이터 없음"),
        }
        
        # 모든 보고서를 동시에 생성
        all_reports = await parallel_chain.ainvoke(input_data)
        
        # LLM 호출 카운터 증가 (보고서 타입별로 1회씩)
        self.llm_call_count += len(self.report_types)
        
        # 결과를 self.reports에 저장하고 result_json에 매핑
        for report_type, result in all_reports.items():
            self.reports[report_type] = result
            
            # result_json에 매핑
            if report_type == "overall_competency":
                # section1의 scores, radar, overall 업데이트
                self.result_json["section1"]["scores"] = result.get("scores", {})
                self.result_json["section1"]["radar"] = result.get("radar", [])
                self.result_json["section1"]["overall"] = result.get("overall", {})
            elif report_type == "competencies":
                # section2의 performance, BM, competencies 업데이트
                self.result_json["section2"]["performance"] = result.get("performance", {})
                self.result_json["section2"]["BM"] = result.get("BM", {})
                self.result_json["section2"]["competencies"] = result.get("competencies", {})
            elif report_type == "market":
                # section3의 market_size, competition, tech_policy_trends 업데이트
                self.result_json["section3"]["market_size"] = result.get("market_size", {})
                self.result_json["section3"]["competition"] = result.get("competition", {})
                self.result_json["section3"]["tech_policy_trends"] = result.get("tech_policy_trends", {})
                
                # market_growth 계산 (market_size의 data로부터)
                market_data = result.get("market_size", {}).get("data", {})
                if market_data and len(market_data) >= 2:
                    years = sorted(map(int, market_data.keys()))
                    if len(years) >= 2:
                        first_year, last_year = years[0], years[-1]
                        try:
                            first_value = float(str(market_data[str(first_year)]).replace(",", ""))
                            last_value = float(str(market_data[str(last_year)]).replace(",", ""))
                            growth_rate = ((last_value / first_value) ** (1 / (last_year - first_year)) - 1) * 100
                            self.result_json["section3"]["market_growth"] = round(growth_rate, 2)
                        except (ValueError, ZeroDivisionError):
                            pass
            elif report_type == "b2g_strategy":
                # section5의 weakness_analysis, strategy, to_do_list 업데이트
                self.result_json["section5"]["weakness_analysis"] = result.get("weakness_analysis", {})
                self.result_json["section5"]["strategy"] = result.get("strategy", {})
                self.result_json["section5"]["to_do_list"] = result.get("to_do_list", {})
            
            if debug:
                report_elapsed = time.time() - report_start_time
                logger.info(f"'{report_type}' 보고서 생성 완료 (보고서 생성 시작 후 {report_elapsed:.2f}초)")
            else:
                logger.info(f"'{report_type}' 보고서 생성 완료")
        
        # 결과 저장 폴더 설정 (아직 설정되지 않았다면)
        if self.result_dir is None:
            self.setup_result_directory(
                web_search=web,
                max_rps=2.0,  # 기본값
                debug=debug
            )
        
        # 보고서를 결과 폴더에 저장 (debug 모드일 때만)
        if debug:
            doc_names = "_".join(self.documents.keys())
            reports_filename = f"{self.name}_{doc_names}.json"
            reports_path = os.path.join(self.result_dir, reports_filename)
            with open(reports_path, 'w', encoding='utf-8') as f:
                json.dump(self.reports, f, ensure_ascii=False, indent=2)
            
            logger.info(f"모든 보고서 저장 완료: {reports_path}")
            
            # result_json도 함께 저장
            result_json_filename = f"{self.name}_{doc_names}_result.json"
            result_json_path = os.path.join(self.result_dir, result_json_filename)
            with open(result_json_path, 'w', encoding='utf-8') as f:
                json.dump(self.result_json, f, ensure_ascii=False, indent=2)
            
            logger.info(f"통합 결과 JSON 저장 완료: {result_json_path}")
        
        return self.reports

    def _calculate_rank_from_scores(self, ranks: list) -> str:
        """
        S/A/B/C/D 등급 리스트에서 평균을 계산하여 최종 등급 반환
        
        점수 매핑: S=100, A=80, B=60, C=40, D=20
        평균을 구한 후 10의 자리로 올림하여 등급 산정
        
        Args:
            ranks: S/A/B/C/D 등급 문자열 리스트
            
        Returns:
            계산된 최종 등급 (S/A/B/C/D)
        """
        import math
        
        rank_to_score = {'S': 100, 'A': 80, 'B': 60, 'C': 40, 'D': 20}
        score_to_rank = {100: 'S', 80: 'A', 60: 'B', 40: 'C', 20: 'D'}
        
        if not ranks:
            return 'D'
        
        # 등급을 점수로 변환하여 평균 계산
        scores = []
        for rank in ranks:
            # rank가 문자열인 경우 첫 글자만 추출 (예: "S/A" -> "S")
            if isinstance(rank, str):
                rank_char = rank.strip().upper()[0] if rank.strip() else 'D'
                if rank_char in rank_to_score:
                    scores.append(rank_to_score[rank_char])
                else:
                    scores.append(20)  # 알 수 없는 등급은 D로 처리
        
        if not scores:
            return 'D'
        
        # 평균 계산
        avg_score = sum(scores) / len(scores)
        
        # 10의 자리로 올림
        rounded_score = math.ceil(avg_score / 10) * 10
        
        # 범위 제한 (20~100)
        rounded_score = max(20, min(100, rounded_score))
        
        # 가장 가까운 등급 찾기
        if rounded_score >= 100:
            return 'S'
        elif rounded_score >= 80:
            return 'A'
        elif rounded_score >= 60:
            return 'B'
        elif rounded_score >= 40:
            return 'C'
        else:
            return 'D'

    async def search_b2g_evaluations(self, debug: bool = False):
        """
        3가지 B2G 평가 DB에서 기업에 적합한 항목을 검색하고 3단계 분석 수행
        
        파이프라인 (모두 Gemini 2.0 Pro 사용):
        1. 기업 문서 분석 결과에서 검색 쿼리 추출
        2. 각 DB에서 관련 항목 검색 (국정과제, 경영평가, 동반성장)
        3. Step A: B2G_EVALUATION_RANK_PROMPT로 top10 랭킹 생성
        4. Step B: B2G_EVALUATION_ANALYSIS_PROMPT로 analysis (insight, risk, consider) 생성 + rank 계산
        5. Step C: B2G_EVALUATION_SUMMARY_PROMPT로 overall (expect) 생성 + rank 계산
        6. 결과를 section4에 저장
        
        Args:
            debug: 디버그 모드 여부
        """
        from src.db_main import MySQLStore
        from src.prompts import (
            COMPANY_FEATURE_EXTRACTION_PROMPT, 
            B2G_EVALUATION_RANK_PROMPT,
            B2G_EVALUATION_ANALYSIS_PROMPT,
            B2G_EVALUATION_SUMMARY_PROMPT
        )
        
        logger.info("B2G 평가 검색 및 분석 시작 (국정과제, 경영평가, 동반성장) - MySQL 임베딩 검색")
        start_time = time.time()
        
        # 문서 데이터 준비
        document_data = json.dumps(
            [doc.analysis for doc in self.documents.values()],
            ensure_ascii=False
        )

        # 1. 기업 특징 추출 (검색 쿼리 생성) - GPT-4o-mini 사용
        logger.info("Step 1: 기업 특징 추출 중...")
        
        feature_model = ModelFactory.create_model_chain(
            provider="openai",
            model_name="gpt-4o-mini",
            output_format="json",
            max_rps=self.dispatcher.max_rps
        )
        
        feature_chain = (
            COMPANY_FEATURE_EXTRACTION_PROMPT
            | feature_model
            | JsonOutputParser()
        )
        
        company_features = await feature_chain.ainvoke({
            "company_name": self.name,
            "document_data": document_data
        })
        
        self.llm_call_count += 1
        
        if debug:
            logger.info(f"추출된 기업 특징: {json.dumps(company_features, ensure_ascii=False, indent=2)}")
        
        # MySQL DB 연결
        mysql_store = MySQLStore(
            host="localhost",
            port=3306,
            database="b2g_data",
            user="root",
            password=""
        )
        
        # DB 통계 가져오기 (전체 개수 확인용)
        try:
            db_stats = mysql_store.get_stats()
        except Exception as e:
            logger.warning(f"DB 통계 조회 실패: {e}")
            db_stats = {}
        
        # 평가 유형별 설정 (SCHEMA_REGISTRY 기반 동적 생성)
        eval_configs = {}
        # result_json 키 매핑 (기존 호환성 유지)
        type_mapping = {
            "project": "presidential_agenda",
            "management_eval": "management_eval",
            "inclusive_growth": "inclusive_growth"
        }
        
        for data_type, schema in SCHEMA_REGISTRY.items():
            eval_type = type_mapping.get(data_type, data_type)
            eval_configs[eval_type] = {
                "name": schema["type_display"],
                "data_type": data_type,
                "name_field": schema["name_field"],
                "id_field": schema["number_field"] if data_type == "project" else schema["name_field"],
                "schema": schema
            }
        
        search_queries = company_features.get("search_queries", [])
        core_technologies = company_features.get("core_technologies", [])
        
        # Gemini 2.0 Pro 모델 생성 (Section 4 전체에서 사용)
        gemini_model = ModelFactory.create_model_chain(
            provider="gemini",
            model_name="gemini-3-pro-preview",
            output_format="json",
            web_search=True,  # 웹서칭 활성화
            max_rps=self.dispatcher.max_rps
        )
        
        # 각 평가 유형별 랭킹 결과 저장 (Summary용)
        all_rank_results = {}
        all_analysis_results = {}
        
        # 2. 각 평가 유형별로 검색 및 3단계 분석
        for eval_type, config in eval_configs.items():
            logger.info(f"\n{'='*50}")
            logger.info(f"[{config['name']}] 처리 시작")
            logger.info(f"{'='*50}")
            
            # 2-1. DB 검색 (MySQL 임베딩 기반)
            logger.info(f"Step 2-{eval_type}: {config['name']} 임베딩 검색 중...")
            
            data_type = config["data_type"]
            table_name = config["schema"]["table"]
            total_count = db_stats.get(table_name, 0)
            all_items = {}
            
            # 1. DB 전체 개수가 20개 이하인 경우 전체 로드
            if total_count > 0 and total_count <= 20:
                logger.info(f"  - 전체 데이터 개수({total_count}개)가 20개 이하이므로 전체 로드")
                try:
                    results = mysql_store.get_records(data_type, limit=50)
                    for item in results:
                        item_id = item.get(config["id_field"], "") or item.get(config["name_field"], "")
                        if item_id:
                            all_items[item_id] = item
                except Exception as e:
                    logger.warning(f"전체 데이터 로드 실패: {e}")
            else:
                # 2. 20개 이상이면 k를 늘리면서 검색 (최소 20개 확보 시도)
                k_steps = [5, 10, 15, 20]
                
                # 검색 쿼리 + 핵심 기술 모두 활용
                search_targets = search_queries + core_technologies
                
                for k in k_steps:
                    if len(all_items) >= 20:
                        break
                        
                    if debug:
                        logger.info(f"  - 검색 시도 (k={k})... 현재 확보된 항목: {len(all_items)}개")
                    
                    for query in search_targets:
                        if len(all_items) >= 20:
                            break
                            
                        try:
                            results = mysql_store.search_by_embedding(data_type, query, k=k)
                            for item in results:
                                item_id = item.get(config["id_field"], "") or item.get(config["name_field"], "")
                                if item_id and item_id not in all_items:
                                    all_items[item_id] = item
                        except Exception as e:
                            logger.warning(f"검색 오류: {e}")
                            continue

            # 3. 여전히 20개 미만이면 Fallback (전체 DB에서 추가 로드)
            if len(all_items) < 20:
                try:
                    logger.info(f"  - 검색 결과 부족({len(all_items)}개) -> Fallback 데이터 로드")
                    additional_items = mysql_store.get_records(data_type, limit=50)
                    for item in additional_items:
                        if len(all_items) >= 20:
                            break
                        
                        item_id = item.get(config["id_field"], "") or item.get(config["name_field"], "")
                        if item_id and item_id not in all_items:
                            all_items[item_id] = item
                            if debug:
                                logger.info(f"  - Fallback 추가: {item.get(config['name_field'], '')}")
                except Exception as e:
                    logger.warning(f"Fallback 데이터 로드 실패 ({config['name']}): {e}")

            items_list = list(all_items.values())[:20]  # 20개 후보 항목 선정
            
            if debug:
                logger.info(f"검색된 {config['name']} 수: {len(items_list)}")
            
            if not items_list:
                logger.warning(f"{config['name']} 검색 결과 없음 - 기본값 설정")
                self.result_json["section4"][eval_type] = {
                    "top10": [],
                    "analysis": {
                        "rank": "",
                        "insight": {"title": "", "details": []},
                        "risk": {"title": "", "details": []},
                        "consider": []
                    }
                }
                continue
            
            # 2-2. 항목 리스트를 텍스트로 변환 (MySQL 스키마에 맞게 동적 생성)
            items_text = ""
            schema = config["schema"]
            for i, item in enumerate(items_list, 1):
                name = item.get(config["name_field"], "")
                score = item.get("score", 0)
                item_id = item.get(config["id_field"], "")
                
                details_text = ""
                for field_name, field_spec in schema["fields"].items():
                    if field_spec.get("extract_detail", False):
                        val = item.get(field_name, [])
                        if isinstance(val, list):
                            val_str = ', '.join([str(v) for v in val[:2]]) if val else '정보 없음'
                        else:
                            val_str = str(val) if val else '정보 없음'
                        details_text += f"   - {field_name}: {val_str}\n"
                
                items_text += f"\n{i}. [{item_id}] {name} (유사도: {score:.3f})\n{details_text}"
            
            # ============================================================
            # Step A: 랭킹 생성 (B2G_EVALUATION_RANK_PROMPT)
            # ============================================================
            logger.info(f"Step 3A-{eval_type}: {config['name']} 랭킹 생성 중...")
            
            rank_chain = (
                B2G_EVALUATION_RANK_PROMPT
                | gemini_model
                | JsonOutputParser()
            )
            
            try:
                rank_result = await rank_chain.ainvoke({
                    "company_name": self.name,
                    "company_summary": company_features.get("company_summary", ""),
                    "core_technologies": ", ".join(company_features.get("core_technologies", [])),
                    "target_sectors": ", ".join(company_features.get("target_sectors", [])),
                    "eval_type_name": config["name"],
                    "items_list": items_text,
                    "document_data": document_data
                })
                
                self.llm_call_count += 1
                
                # top10 결과 추출 (LLM이 10개 이상 반환할 수 있으므로 슬라이싱)
                top10_result = rank_result.get("top10", [])[:10]
                all_rank_results[eval_type] = top10_result
                
                if debug:
                    logger.info(f"  - 랭킹 완료: {len(top10_result)}개 항목")
                    
            except Exception as e:
                logger.error(f"{config['name']} 랭킹 오류: {e}")
                top10_result = []
                all_rank_results[eval_type] = []
            
            # ============================================================
            # Step B: 분석 생성 (B2G_EVALUATION_ANALYSIS_PROMPT)
            # ============================================================
            logger.info(f"Step 3B-{eval_type}: {config['name']} 분석 생성 중...")
            
            analysis_chain = (
                B2G_EVALUATION_ANALYSIS_PROMPT
                | gemini_model
                | JsonOutputParser()
            )
            
            # 랭킹 결과를 텍스트로 변환
            rank_result_text = json.dumps({"top10": top10_result}, ensure_ascii=False, indent=2)
            
            try:
                analysis_result = await analysis_chain.ainvoke({
                    "eval_type_name": config["name"],
                    "b2g_evaluation_rank_result": rank_result_text,
                    "document_data": document_data
                })
                
                self.llm_call_count += 1
                
                # analysis 결과 추출
                analysis_data = analysis_result.get("analysis", {})
                
                # Step B: top10의 rank 평균으로 analysis rank 계산
                top10_ranks = [item.get("rank", "D") for item in top10_result if item.get("rank")]
                calculated_rank = self._calculate_rank_from_scores(top10_ranks)
                
                # 계산된 rank를 analysis_data에 추가
                analysis_data["rank"] = calculated_rank
                all_analysis_results[eval_type] = analysis_data
                
                if debug:
                    logger.info(f"  - 분석 완료: rank={calculated_rank} (top10 평균에서 계산됨)")
                
                # section4에 저장 (top10 + analysis)
                self.result_json["section4"][eval_type] = {
                    "top10": top10_result,
                    "analysis": {
                        "rank": calculated_rank,
                        "insight": analysis_data.get("insight", {"title": "", "details": []}),
                        "risk": analysis_data.get("risk", {"title": "", "details": []}),
                        "consider": analysis_data.get("consider", [])
                    }
                }
                    
            except Exception as e:
                logger.error(f"{config['name']} 분석 오류: {e}")
                # 오류 발생 시에도 top10에서 rank 계산 시도
                top10_ranks = [item.get("rank", "D") for item in top10_result if item.get("rank")]
                calculated_rank = self._calculate_rank_from_scores(top10_ranks) if top10_ranks else ""
                
                self.result_json["section4"][eval_type] = {
                    "top10": top10_result,
                    "analysis": {
                        "rank": calculated_rank,
                        "insight": {"title": "", "details": []},
                        "risk": {"title": "", "details": []},
                        "consider": []
                    }
                }
                all_analysis_results[eval_type] = {"rank": calculated_rank}
        
        # ============================================================
        # Step C: 종합 요약 생성 (B2G_EVALUATION_SUMMARY_PROMPT)
        # ============================================================
        logger.info(f"\n{'='*50}")
        logger.info("Step 4: B2G 평가 종합 요약 생성 중...")
        logger.info(f"{'='*50}")
        
        summary_chain = (
            B2G_EVALUATION_SUMMARY_PROMPT
            | gemini_model
            | JsonOutputParser()
        )
        
        # 랭킹 결과 텍스트 생성
        rank_results_text = ""
        for eval_type, config in eval_configs.items():
            rank_results_text += f"\n### {config['name']} Top10 랭킹:\n"
            rank_results_text += json.dumps(all_rank_results.get(eval_type, []), ensure_ascii=False, indent=2)
        
        # 분석 결과 텍스트 생성
        analysis_results_text = ""
        for eval_type, config in eval_configs.items():
            analysis_results_text += f"\n### {config['name']} 분석 결과:\n"
            analysis_results_text += json.dumps(all_analysis_results.get(eval_type, {}), ensure_ascii=False, indent=2)
        
        try:
            summary_result = await summary_chain.ainvoke({
                "b2g_evaluation_rank_result": rank_results_text,
                "b2g_analysis": analysis_results_text,
                "document_data": document_data
            })
            
            self.llm_call_count += 1
            
            # Step C: 3개 평가 유형의 analysis rank 평균으로 overall rank 계산
            analysis_ranks = [
                all_analysis_results.get("presidential_agenda", {}).get("rank", ""),
                all_analysis_results.get("management_eval", {}).get("rank", ""),
                all_analysis_results.get("inclusive_growth", {}).get("rank", "")
            ]
            # 빈 문자열 제거
            analysis_ranks = [r for r in analysis_ranks if r]
            overall_rank = self._calculate_rank_from_scores(analysis_ranks)
            
            # overall 결과 저장
            self.result_json["section4"]["overall"] = {
                "rank": overall_rank,
                "expect": summary_result.get("expect", [])
            }
            
            if debug:
                logger.info(f"  - 종합 등급: {overall_rank} (3개 평가 rank 평균에서 계산됨)")
                logger.info(f"  - 기대효과: {len(summary_result.get('expect', []))}개")
                
        except Exception as e:
            logger.error(f"종합 요약 생성 오류: {e}")
            # 오류 발생 시에도 analysis ranks에서 overall rank 계산 시도
            analysis_ranks = [
                all_analysis_results.get("presidential_agenda", {}).get("rank", ""),
                all_analysis_results.get("management_eval", {}).get("rank", ""),
                all_analysis_results.get("inclusive_growth", {}).get("rank", "")
            ]
            analysis_ranks = [r for r in analysis_ranks if r]
            overall_rank = self._calculate_rank_from_scores(analysis_ranks) if analysis_ranks else ""
            
            self.result_json["section4"]["overall"] = {
                "rank": overall_rank,
                "expect": []
            }
        
        elapsed = time.time() - start_time
        logger.info(f"\nB2G 평가 검색 및 분석 완료 - 소요 시간: {elapsed:.2f}초")
        
        return self.result_json["section4"]

    async def generate_b2g_strategy(
        self,
        web: bool = False,
        debug: bool = False
    ):
        """
        B2G 전략 방향 수립 (Section 5)
        
        모델: Google Gemini 2.0 Pro (하드코딩)
        
        Section 4의 분석 결과 (레이더 차트 점수, 국정과제/경영평가/동반성장 분석)를
        input으로 받아 전략을 수립합니다.
        
        Args:
            web: 웹 검색 활성화 여부
            debug: 디버그 모드
        """
        from src.prompts import B2G_STRATEGY_PROMPT
        
        logger.info("B2G 전략 방향 수립 시작 (Section 5) - Model: gemini/gemini-3-pro-preview")
        start_time = time.time()
        
        # 문서 데이터 준비
        document_data = json.dumps(
            [doc.analysis for doc in self.documents.values()],
            ensure_ascii=False
        )
        
        # Section 1의 레이더 차트 및 점수 데이터
        section1_data = self.result_json.get("section1", {})
        radar_data = section1_data.get("radar", [])
        scores_data = section1_data.get("scores", {})
        overall_data = section1_data.get("overall", {})
        
        # Section 4의 B2G 평가 분석 결과
        section4_data = self.result_json.get("section4", {})
        
        # B2G 평가 분석 요약 생성
        b2g_analysis_summary = self._format_b2g_analysis_for_strategy(section4_data)
        
        # Gemini 2.0 Pro 모델 생성
        rate_limited_model = ModelFactory.create_model_chain(
            provider="gemini",
            model_name="gemini-3-pro-preview",
            output_format="json",
            web_search=web,
            max_rps=self.dispatcher.max_rps
        )
        
        # B2G 전략 체인 구성
        strategy_chain = (
            B2G_STRATEGY_PROMPT
            | rate_limited_model
            | JsonOutputParser()
        )
        
        # 전략 생성
        try:
            result = await strategy_chain.ainvoke({
                "document_data": document_data,
                "radar_scores": json.dumps(radar_data, ensure_ascii=False),
                "evaluation_scores": json.dumps(scores_data, ensure_ascii=False),
                "overall_assessment": json.dumps(overall_data, ensure_ascii=False),
                "b2g_analysis": b2g_analysis_summary
            })
            
            self.llm_call_count += 1
            
            # result가 리스트인 경우 첫 번째 요소 사용
            if isinstance(result, list):
                result = result[0] if result else {}
            
            # Section 5에 저장
            self.result_json["section5"]["weakness_analysis"] = result.get("weakness_analysis", {})
            self.result_json["section5"]["strategy"] = result.get("strategy", {})
            self.result_json["section5"]["to_do_list"] = result.get("to_do_list", {})
            
            self.reports["b2g_strategy"] = result
            
            if debug:
                logger.info("B2G 전략 생성 완료:")
                weakness = result.get('weakness_analysis', {})
                strategy = result.get('strategy', {})
                weakness_keyword = weakness.get('keyword', '') if isinstance(weakness, dict) else ''
                strategy_keyword = strategy.get('keyword', '') if isinstance(strategy, dict) else ''
                logger.info(f"  - 약점 분석: {weakness_keyword[:50]}...")
                logger.info(f"  - 전략 방향: {strategy_keyword[:50]}...")
                
        except Exception as e:
            logger.error(f"B2G 전략 생성 오류: {e}")
            self.result_json["section5"] = {
                "weakness_analysis": {"keyword": "", "evidences": []},
                "strategy": {"keyword": "", "strategy": "", "details": []},
                "to_do_list": {"keyword": "", "tasks": []}
            }
        
        elapsed = time.time() - start_time
        logger.info(f"B2G 전략 방향 수립 완료 - 소요 시간: {elapsed:.2f}초")
        
        return self.result_json["section5"]
    
    def _format_b2g_analysis_for_strategy(self, section4_data: dict) -> str:
        """
        Section 4의 B2G 평가 분석 결과를 전략 수립용 텍스트로 포맷팅
        """
        parts = []
        
        # Overall 종합 정보 추가
        overall = section4_data.get("overall", {})
        if overall:
            overall_part = "\n### B2G 종합 평가\n"
            overall_part += f"종합 등급: {overall.get('rank', 'N/A')}\n"
            expect = overall.get("expect", [])
            if expect:
                overall_part += "기대 효과:\n"
                for e in expect[:3]:
                    overall_part += f"  - {e}\n"
            parts.append(overall_part)
        
        eval_names = {
            "presidential_agenda": "국정과제",
            "management_eval": "공공기관 경영평가",
            "inclusive_growth": "동반성장 평가"
        }
        
        for eval_type, eval_name in eval_names.items():
            data = section4_data.get(eval_type, {})
            if not data:
                continue
            
            part = f"\n### {eval_name} 분석 결과\n"
            
            # 분석 등급
            analysis = data.get("analysis", {})
            rank = analysis.get("rank", "")
            if rank:
                part += f"등급: {rank}\n"
            
            # Top 3 항목
            top10 = data.get("top10", [])[:3]
            if top10:
                part += "주요 관련 항목:\n"
                for item in top10:
                    part += f"  - [{item.get('rank', '')}] {item.get('name', '')}: {item.get('description', '')}\n"
            
            insight = analysis.get("insight", {})
            if insight.get("title"):
                part += f"\n핵심 인사이트: {insight.get('title')}\n"
                for detail in insight.get("details", [])[:2]:
                    part += f"  - {detail}\n"
            
            risk = analysis.get("risk", {})
            if risk.get("title"):
                part += f"\n주요 위험: {risk.get('title')}\n"
                for detail in risk.get("details", [])[:2]:
                    part += f"  - {detail}\n"
            
            consider = analysis.get("consider", [])[:2]
            if consider:
                part += f"\n검토 필요 사항:\n"
                for q in consider:
                    part += f"  - {q}\n"
            
            parts.append(part)
        
        return "\n".join(parts) if parts else "B2G 평가 분석 결과 없음"

async def main_async(
    company_name: str,
    documents: list[tuple[str, str]],  # [(name1, path1), (name2, path2), ...]
    web_search: bool = False,
    max_rps: float = 2.0,
    debug: bool = False
):
    """
    회사 단위로 여러 문서를 병렬 처리 및 보고서 생성
    
    모델 설정 (하드코딩):
        - 문서 추출 및 Section 1, 2: OpenAI GPT-4o
        - Section 3, 4, 5: Google Gemini 2.0 Pro
    
    Args:
        company_name: 회사 이름
        documents: 문서 이름과 경로 리스트 [(name1, path1), (name2, path2), ...]
        web_search: 웹 검색 활성화 여부
        max_rps: 초당 최대 요청 수
        debug: True일 경우 각 페이지별 시간과 전체 소요 시간을 로깅
    """
    com = Company(company_name, max_rps=max_rps)
    
    # 모든 문서 추가
    for name, doc_path in documents:
        com.add_document(name, Document(doc_path))
    
    # 결과 저장 폴더 설정
    result_dir = com.setup_result_directory(
        web_search=web_search,
        max_rps=max_rps,
        debug=debug
    )
    logger.info(f"결과 저장 폴더: {result_dir}")
    
    # 디버그 모드일 때 파일 핸들러 추가
    file_handler = None
    if debug:
        debug_log_path = os.path.join(result_dir, "debug.txt")
        file_handler = logging.FileHandler(debug_log_path, mode='w', encoding='utf-8')
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(logging.Formatter('[%(asctime)s] %(levelname)s - %(message)s', datefmt='%H:%M:%S'))
        
        # root logger에 핸들러 추가하여 모든 로그 캡처
        root_logger = logging.getLogger()
        root_logger.addHandler(file_handler)
        
        # 하이퍼파라미터 정보를 파일에 먼저 기록
        with open(debug_log_path, 'a', encoding='utf-8') as f:
            f.write("=" * 60 + "\n")
            f.write("HYPERPARAMETERS\n")
            f.write("=" * 60 + "\n")
            f.write(f'COMPANY_NAME="{company_name}"\n')
            
            # 문서 정보
            doc_strings = []
            for name, path in documents:
                doc_strings.append(f'"{name}:{path}"')
            f.write(f'DOCUMENTS={" ".join(doc_strings)}\n')
            
            f.write(f'EXTRACT_MODEL="openai"\n')
            f.write(f'EXTRACT_MODEL_NAME="gpt-4o"\n')
            f.write(f'SECTION_1_2_MODEL="openai/gpt-4o"\n')
            f.write(f'SECTION_3_4_5_MODEL="gemini/gemini-3-pro-preview"\n')
            f.write(f'OCR_PROVIDER="CLOVA"\n')
            f.write(f'WEB_SEARCH={"--web" if web_search else ""}\n')
            f.write(f'MAX_RPS="{max_rps}"\n')
            f.write(f'DEBUG={"--debug" if debug else ""}\n')
            f.write("=" * 60 + "\n\n")
        
        logger.info(f"디버그 로그 저장 경로: {debug_log_path}")
    
    # 캐시 확인 및 로드
    all_cached = True
    
    for name, _ in documents:
        result_file = os.path.join(result_dir, f"{name}.json")
        ocr_file = os.path.join(result_dir, f"{name}_ocr.json")
        
        # 이미 분석 결과가 있으면 로드
        if os.path.exists(result_file) and os.path.exists(ocr_file):
            logger.info(f"기존 분석 결과 로드: {result_file}")
            with open(result_file, 'r', encoding='utf-8') as f:
                com.documents[name].analysis = json.load(f)
            with open(ocr_file, 'r', encoding='utf-8') as f:
                ocr_by_page = json.load(f)
                # OCR 결과를 원래 형식으로 변환
                com.documents[name].ocr_texts = [
                    {int(k.replace("page_", "")): v} for k, v in ocr_by_page.items()
                ]
            
            # 캐시된 분석 결과에서 재무현황 추출
            import re
            for page_data in com.documents[name].analysis:
                for page_num, result in page_data.items():
                    if "재무현황" in result:
                        for k in [("revenue","매출"), ("profit","영업이익"), ("invest","누적투자")]:
                            if result["재무현황"].get(k[1], {}).get("금액", "") == "" or result["재무현황"].get(k[1], {}).get("기준년도", "") == "":
                                continue
                            amount = result["재무현황"][k[1]]["금액"]
                            year = 0
                            year_raw = result["재무현황"][k[1]].get("기준년도")
                            if year_raw is None:
                                year = 0
                            elif type(year_raw) == int:
                                year = year_raw
                            else:
                                year_match = re.search(r'\d{4}', str(year_raw))
                                if year_match:
                                    year = int(year_match.group(0))
                            
                            current_year = com.result_json["section2"]["finance"][k[0]]["year"]
                            current_amount = com.result_json["section2"]["finance"][k[0]]["amount"]
                            if current_year == 0 and (current_amount == "" or current_amount is None):
                                com.result_json["section2"]["finance"][k[0]]["year"] = year #반드시 정수로 저장
                                com.result_json["section2"]["finance"][k[0]]["amount"] = amount
                            elif current_year != 0 and current_year < year:
                                com.result_json["section2"]["finance"][k[0]]["year"] = year
                                com.result_json["section2"]["finance"][k[0]]["amount"] = amount
        else:
            all_cached = False
            break
    
    # 캐시되지 않은 문서가 있으면 분석 실행
    if not all_cached:
        await com.process_documents(
            debug=debug,
            max_rps=max_rps,
            web_search=web_search
        )
    
    # TODO : 보고서 생성 이전에 skeleton prompt 작성 (JSON에서 좀 더 자연화된 프롬프트, 또는 minor detail 제거 등) 
    # 정말 필요할지 의문. hallucination 문제가 발생할 위험이 있다.
    # TODO : 외부 지식 tool 활용 기능

    # Section 1~3 보고서 생성 (overall_competency, competencies, market)
    await com.generate_all_reports(
        web=web_search,
        debug=debug
    )
    
    # Section 4: B2G 평가 검색 및 분석 (국정과제, 경영평가, 동반성장)
    await com.search_b2g_evaluations(debug=debug)
    
    # Section 5: B2G 전략 방향 수립 (section4 결과를 input으로 사용)
    await com.generate_b2g_strategy(
        web=web_search,
        debug=debug
    )
    
    # 최종 result_json 저장 (debug 모드일 때만)
    if debug:
        doc_names = "_".join(com.documents.keys())
        result_json_filename = f"{com.name}_{doc_names}_result.json"
        result_json_path = os.path.join(result_dir, result_json_filename)
        with open(result_json_path, 'w', encoding='utf-8') as f:
            json.dump(com.result_json, f, ensure_ascii=False, indent=2)
        logger.info(f"최종 결과 JSON 저장 완료: {result_json_path}")
    
    # 디버그 모드일 때 API 통계 기록 및 파일 핸들러 제거
    if debug and file_handler:
        logger.info(f"\nAPI 호출 통계 - OCR: {com.ocr_call_count}회, LLM: {com.llm_call_count}회")
        
        debug_log_path = os.path.join(result_dir, "debug.txt")
        with open(debug_log_path, 'a', encoding='utf-8') as f:
            f.write("\n" + "=" * 60 + "\n")
            f.write("API CALL STATISTICS\n")
            f.write("=" * 60 + "\n")
            f.write(f"OCR API 호출: {com.ocr_call_count}회\n")
            f.write(f"LLM API 호출: {com.llm_call_count}회\n")
            f.write(f"총 API 호출: {com.ocr_call_count + com.llm_call_count}회\n")
            f.write("=" * 60 + "\n")
        
        # root logger에서 핸들러 제거
        root_logger = logging.getLogger()
        root_logger.removeHandler(file_handler)
        file_handler.close()

def main():
    """
    명령줄 인자를 파싱하고 main_async 함수를 실행하는 래퍼
    """
    parser = argparse.ArgumentParser(
        description='문서 분석 및 보고서 생성 파이프라인',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
예시:
  python main.py -c "Example Corp" -d instruction1:data/instruction1.pdf --web --max-rps 2.0 --debug
  python main.py -c "Example Corp" -d doc1:data/doc1.pdf doc2:data/doc2.pdf

모델 설정 (하드코딩):
  - 문서 추출 및 Section 1, 2: OpenAI GPT-4o
  - Section 3, 4, 5: Google Gemini 2.0 Pro
        """
    )
    
    # 필수 인자
    parser.add_argument(
        '-c', '--company',
        type=str,
        required=True,
        help='회사 이름 (예: "Example Corp")'
    )
    
    parser.add_argument(
        '-d', '--documents',
        type=str,
        nargs='+',
        required=True,
        help='문서 이름:경로 형식 (예: "doc1:data/doc1.pdf doc2:data/doc2.pdf")'
    )
    
    parser.add_argument(
        '--web',
        action='store_true',
        help='웹 검색 활성화 (기본값: False)'
    )
    
    parser.add_argument(
        '--max-rps',
        type=float,
        default=2.0,
        help='초당 최대 요청 수 (기본값: 2.0)'
    )
    
    parser.add_argument(
        '--debug',
        action='store_true',
        help='디버그 모드 활성화 (기본값: False)'
    )
    
    args = parser.parse_args()
    
    # 문서 파싱 (name:path 형식)
    documents = []
    for doc_str in args.documents:
        if ':' not in doc_str:
            logger.error(f"잘못된 문서 형식: {doc_str}. 'name:path' 형식이어야 합니다.")
            sys.exit(1)
        name, path = doc_str.split(':', 1)
        if not os.path.exists(path):
            logger.error(f"문서 파일을 찾을 수 없습니다: {path}")
            sys.exit(1)
        documents.append((name, path))
    
    # 설정 정보 출력
    logger.info("=" * 60)
    logger.info("문서 분석 파이프라인 시작")
    logger.info("=" * 60)
    logger.info(f"회사 이름: {args.company}")
    logger.info(f"문서 개수: {len(documents)}")
    for name, path in documents:
        logger.info(f"  - {name}: {path}")
    logger.info(f"문서 추출 및 Section 1, 2: OpenAI/gpt-4o")
    logger.info(f"Section 3, 4, 5: Google/gemini-3-pro-preview")
    logger.info(f"웹 검색: {args.web}")
    logger.info(f"Max RPS: {args.max_rps}")
    logger.info(f"디버그 모드: {args.debug}")
    logger.info("=" * 60)
    
    # main_async 실행
    asyncio.run(main_async(
        company_name=args.company,
        documents=documents,
        web_search=args.web,
        max_rps=args.max_rps,
        debug=args.debug
    ))

if __name__ == "__main__":
    main()

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