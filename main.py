"""
LangChain 기반 기업 분석 및 보고서 생성 파이프라인

이 모듈은 PDF 문서를 분석하고 기업 보고서를 생성하는 메인 파이프라인입니다.
LangChain을 활용하여 OCR, 문서 분석, 보고서 생성을 Chain 구조로 처리합니다.
"""

# main pipeline
from src.api import ChatRequest, Dispatcher, ModelFactory
from src.prompts import PROMPTS
from src.utils import extractJSON, parse_json, OUTPUT_JSON_SCHEMA
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
            for k in [("revenue","매출"), ("profit","영업이익"), ("invest","누적투자")]:  
                if result["재무현황"][k[1]]["금액"] == "" or result["재무현황"][k[1]]["기준년도"] == "":
                    # 둘 중 하나라도 값이 없으면 패스 -> 둘 다 있어야지만 저장됨.
                    continue
                amount = result["재무현황"][k[1]]["금액"]
                year = 0
                
                if result["재무현황"][k[1]]["기준년도"] == None:
                    year = 0
                elif type(result["재무현황"][k[1]]["기준년도"]) == int:
                    year = result["재무현황"][k[1]]["기준년도"]
                else:
                    # 기준년도에서 숫자만 추출
                    year_match = re.search(r'\d{4}', str(result["재무현황"][k[1]]["기준년도"]))
                    if year_match:
                        year = int(year_match.group(0))
                    else:
                        year = 0

                print("재무현황 디버깅",i, amount, year)

                if company.result_json["section2"]["finance"][k[0]]["year"] == 0 and (company.result_json["section2"]["finance"][k[0]]["amount"] == "" or company.result_json["section2"]["finance"][k[0]]["amount"] is None):
                    company.result_json["section2"]["finance"][k[0]]["year"] = year #반드시 정수로 저장
                    company.result_json["section2"]["finance"][k[0]]["amount"] = amount
                elif company.result_json["section2"]["finance"][k[0]]["year"] != 0: #이미 값이 존재한다면 기준 년도 비교해서 최신 값으로 대체
                    if company.result_json["section2"]["finance"][k[0]]["year"] < year:
                        company.result_json["section2"]["finance"][k[0]]["year"] = year
                        company.result_json["section2"]["finance"][k[0]]["amount"] = amount
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
        self.report_types = ["overall_competency", "competencies", "b2g_strategy", "market"]
        self.result_dir = None  # 결과 저장 폴더 경로
        self.result_json = OUTPUT_JSON_SCHEMA.copy() if OUTPUT_JSON_SCHEMA else {} # OUTPUT_JSON_SCHEMA를 초기값으로 사용
        # API 호출 카운터
        self.ocr_call_count = 0
        self.llm_call_count = 0
    
    def add_document(self, name: str, document: Document):
        """문서를 이름과 함께 추가"""
        self.documents[name] = document
    
    def setup_result_directory(self, extract_model_provider: str = "openai", extract_model_name: str = "gpt-4o", report_model_provider: str = "openai", report_model_name: str = "gpt-4o", web_search: bool = False, max_rps: float = 2.0, debug: bool = False):
        """
        결과 저장 폴더 생성: 회사명_문서명1_문서명2_..._extract모델_report모델_web_debug_rps
        """
        doc_names = "_".join(self.documents.keys())
        
        # 모델명에서 특수문자 제거 (파일명으로 사용 불가한 문자)
        extract_model_safe = f"{extract_model_provider}-{extract_model_name.replace('/', '-').replace(':', '-')}"
        report_model_safe = f"{report_model_provider}-{report_model_name.replace('/', '-').replace(':', '-')}"
        
        # 폴더명 구성 요소
        parts = [
            self.name.replace(" ", "_"),  # 공백 제거
            doc_names,
            f"extract_{extract_model_safe}",
            f"report_{report_model_safe}",
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

    async def process_documents(self, debug: bool = False, max_rps: float = 2.0, ocr_provider: str = "CLOVA", extract_model_provider: str = "openai", extract_model_name: str = "gpt-4o", web_search: bool = False):
        """
        모든 문서를 병렬로 처리하고 결과를 저장
        
        Args:
            debug: True일 경우 각 페이지별 시간과 전체 소요 시간을 로깅
            max_rps: 초당 최대 요청 수 (기본값: 1 RPS)
            ocr_provider: OCR API 종류 ("CLOVA" 또는 "Upstage")
            extract_model_provider: 문서 분석에 사용할 AI 모델 제공자 ("openai" 또는 "gemini")
            extract_model_name: 문서 분석에 사용할 AI 모델명
            web_search: 웹 검색 활성화 여부
        """
        # 결과 저장 폴더 설정
        if self.result_dir is None:
            self.setup_result_directory(
                extract_model_provider=extract_model_provider,
                extract_model_name=extract_model_name,
                report_model_provider="openai",  # 기본값 - 보고서는 아직 생성 전
                report_model_name="gpt-4o",      # 기본값
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
            logger.info(f"분석 시작 (debug 모드, RPS: {max_rps}, Extract Model: {extract_model_provider}/{extract_model_name})")
        else:
            logger.info(f"분석 시작 (RPS: {max_rps})")
        
        # 모든 문서를 병렬로 분석
        tasks = [asyncio.create_task(document.analyze_doc(self.dispatcher, start_time, debug, max_rps, extract_model_provider, extract_model_name, self))
                 for document in self.documents.values()]
        await asyncio.gather(*tasks)
        
        # 전체 소요 시간
        total_time = time.time() - start_time
        if debug:
            logger.info(f"전체 분석 완료 - 소요 시간: {total_time:.2f}초")
        else:
            logger.info(f"전체 분석 완료")
        
        # 각 문서의 결과를 결과 폴더에 저장
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
    
    async def generate_all_reports(self, report_model_provider: str = "openai", report_model_name: str = "gpt-4o", web: bool = False, debug: bool = False):
        """
        RunnableParallel을 사용하여 모든 보고서를 병렬로 생성
        
        Args:
            report_model_provider: 보고서 생성에 사용할 AI 모델 제공자 ("openai" 또는 "gemini")
            report_model_name: 보고서 생성에 사용할 구체적인 모델명
            web: 웹 검색 활성화 여부
            debug: 디버그 모드
        """
        report_start_time = time.time()
        logger.info(f"보고서 생성 시작 - 총 {len(self.report_types)}개 유형 (Report Model: {report_model_provider}/{report_model_name}, Web: {web})")
        
        # 문서 데이터 준비
        document_data = json.dumps(
            [doc.analysis for doc in self.documents.values()],
            ensure_ascii=False
        )
        
        # Rate limiting이 적용된 모델 생성
        rate_limited_model = ModelFactory.create_model_chain(
            provider=report_model_provider,
            model_name=report_model_name,
            output_format="json",
            web_search=web,
            max_rps=self.dispatcher.max_rps
        )
        
        # 각 보고서 타입별로 LCEL 체인 구성
        report_chains = {
            report_type: PROMPTS[report_type] | rate_limited_model | JsonOutputParser()
            for report_type in self.report_types
        }
        
        # RunnableParallel로 병렬 실행
        parallel_chain = RunnableParallel(report_chains)
        
        # 모든 보고서를 동시에 생성
        all_reports = await parallel_chain.ainvoke({"document_data": document_data})
        
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
                extract_model_provider="openai",  # 기본값
                extract_model_name="gpt-4o",  # 기본값
                report_model_provider=report_model_provider,
                report_model_name=report_model_name,
                web_search=web,
                max_rps=2.0,  # 기본값
                debug=debug
            )
        
        # 보고서를 결과 폴더에 저장
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

    async def search_b2g_evaluations(self, debug: bool = False):
        """
        3가지 B2G 평가 DB에서 기업에 적합한 항목을 검색하고 분석
        
        1. 기업 문서 분석 결과에서 검색 쿼리 추출
        2. 각 DB에서 관련 항목 검색 (국정과제, 경영평가, 동반성장)
        3. 검색된 항목과 기업 정보를 LLM에 전달하여 분석 (top10, insight, risk, consider)
        4. 결과를 section4에 저장
        
        Args:
            debug: 디버그 모드 여부
        """
        from src.db import B2GVectorStore, InclusiveGrowthVectorStore
        from src.prompts import COMPANY_FEATURE_EXTRACTION_PROMPT, B2G_EVALUATION_ANALYSIS_PROMPT
        
        logger.info("B2G 평가 검색 및 분석 시작 (국정과제, 경영평가, 동반성장)")
        start_time = time.time()
        
        # 문서 데이터 준비
        document_data = json.dumps(
            [doc.analysis for doc in self.documents.values()],
            ensure_ascii=False
        )

        # 1. 기업 특징 추출 (검색 쿼리 생성)
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
        
        # DB 연결 설정
        connection_string = "postgresql://youngseocho:@localhost:5432/b2g_data"
        
        # 평가 유형별 설정
        eval_configs = {
            "presidential_agenda": {
                "name": "국정과제",
                "collection": "b2g_projects",
                "store_class": B2GVectorStore,
                "search_method": "search_unique_projects",
                "name_field": "과제명",
                "id_field": "과제번호",
            },
            "management_eval": {
                "name": "공공기관 경영평가",
                "collection": "management_eval_indicators",
                "store_class": InclusiveGrowthVectorStore,
                "search_method": "search_unique_indicators",
                "name_field": "지표명",
                "id_field": "지표명",
            },
            "inclusive_growth": {
                "name": "동반성장 평가지표",
                "collection": "inclusive_growth_indicators",
                "store_class": InclusiveGrowthVectorStore,
                "search_method": "search_unique_indicators",
                "name_field": "지표명",
                "id_field": "지표명",
            }
        }
        
        search_queries = company_features.get("search_queries", [])
        core_technologies = company_features.get("core_technologies", [])
        
        # 2. 각 평가 유형별로 검색 및 분석
        for eval_type, config in eval_configs.items():
            logger.info(f"\nStep 2-{eval_type}: {config['name']} 검색 중...")
            
            # 벡터 스토어 생성
            vector_store = config["store_class"](
                connection_string=connection_string,
                collection_name=config["collection"]
            )
            
            # 검색 메서드 가져오기
            search_func = getattr(vector_store, config["search_method"])
            
            # 검색 쿼리로 항목 검색
            all_items = {}
            
            for query in search_queries:
                try:
                    results = search_func(query, k=5)
                    for item in results:
                        item_id = item.get(config["id_field"], "") or item.get("full_data", {}).get(config["id_field"], "")
                        if item_id and item_id not in all_items:
                            all_items[item_id] = item
                except Exception as e:
                    logger.warning(f"검색 오류 ({config['name']}): {e}")
                    continue
            
            # 부족하면 추가 검색
            if len(all_items) < 10:
                for tech in core_technologies:
                    if len(all_items) >= 10:
                        break
                    try:
                        results = search_func(tech, k=3)
                        for item in results:
                            item_id = item.get(config["id_field"], "") or item.get("full_data", {}).get(config["id_field"], "")
                            if item_id and item_id not in all_items:
                                all_items[item_id] = item
                    except Exception as e:
                        continue
            
            items_list = list(all_items.values())[:10]
            
            if debug:
                logger.info(f"검색된 {config['name']} 수: {len(items_list)}")
                for item in items_list[:3]:
                    name = item.get(config["name_field"], "") or item.get("full_data", {}).get(config["name_field"], "")
                    logger.info(f"  - {name[:40]}...")
            
            if not items_list:
                logger.warning(f"{config['name']} 검색 결과 없음")
                continue
            
            # 3. 항목 리스트를 텍스트로 변환
            items_text = ""
            for i, item in enumerate(items_list, 1):
                name = item.get(config["name_field"], "") or item.get("full_data", {}).get(config["name_field"], "")
                matched_text = item.get("matched_text", "")[:200]
                
                if eval_type == "presidential_agenda":
                    # 국정과제용 형식
                    goals = item.get("과제 목표", [])[:2]
                    contents = item.get("주요내용", [])[:2]
                    items_text += f"""
{i}. [{item.get(config["id_field"], "")}] {name}
   - 과제 목표: {', '.join(goals) if goals else '정보 없음'}
   - 주요내용: {', '.join(contents) if contents else matched_text}
"""
                else:
                    # 경영평가/동반성장 지표용 형식
                    full_data = item.get("full_data", {})
                    eval_criteria = full_data.get("평가기준", [])[:2]
                    eval_method = full_data.get("평가방법", [])[:1]
                    items_text += f"""
{i}. {name}
   - 평가기준: {', '.join(eval_criteria) if eval_criteria else matched_text}
   - 평가방법: {', '.join(eval_method) if eval_method else ''}
"""
            
            # 4. LLM으로 분석 (insight, risk, consider 생성)
            logger.info(f"Step 3-{eval_type}: {config['name']} 분석 중...")
            
            analysis_model = ModelFactory.create_model_chain(
                provider="openai",
                model_name="gpt-4o",
                output_format="json",
                max_rps=self.dispatcher.max_rps
            )
            
            analysis_chain = (
                B2G_EVALUATION_ANALYSIS_PROMPT
                | analysis_model
                | JsonOutputParser()
            )
            
            try:
                analysis_result = await analysis_chain.ainvoke({
                    "company_name": self.name,
                    "company_summary": company_features.get("company_summary", ""),
                    "core_technologies": ", ".join(company_features.get("core_technologies", [])),
                    "target_sectors": ", ".join(company_features.get("target_sectors", [])),
                    "eval_type_name": config["name"],
                    "items_list": items_text
                })
                
                self.llm_call_count += 1
                
                # 5. 결과를 section4에 저장
                self.result_json["section4"][eval_type] = {
                    "top10": analysis_result.get("top10", []),
                    "analysis": analysis_result.get("analysis", {
                        "insight": {"title": "", "details": []},
                        "risk": {"title": "", "details": []},
                        "consider": []
                    })
                }
                
                if debug:
                    logger.info(f"{config['name']} 분석 완료:")
                    logger.info(f"  - Top10 항목 수: {len(analysis_result.get('top10', []))}")
                    logger.info(f"  - Insight: {analysis_result.get('analysis', {}).get('insight', {}).get('title', '')[:50]}...")
                    
            except Exception as e:
                logger.error(f"{config['name']} 분석 오류: {e}")
                self.result_json["section4"][eval_type] = {
                    "top10": [],
                    "analysis": {
                        "insight": {"title": "", "details": []},
                        "risk": {"title": "", "details": []},
                        "consider": []
                    }
                }
        
        elapsed = time.time() - start_time
        logger.info(f"\nB2G 평가 검색 및 분석 완료 - 소요 시간: {elapsed:.2f}초")
        
        return self.result_json["section4"]

async def main_async(
    company_name: str,
    documents: list[tuple[str, str]],  # [(name1, path1), (name2, path2), ...]
    extract_model_provider: str = "openai",
    extract_model_name: str = "gpt-4o",
    report_model_provider: str = "openai",
    report_model_name: str = "gpt-4o",
    web_search: bool = False,
    max_rps: float = 2.0,
    debug: bool = False
):
    """
    회사 단위로 여러 문서를 병렬 처리 및 보고서 생성
    
    Args:
        company_name: 회사 이름
        documents: 문서 이름과 경로 리스트 [(name1, path1), (name2, path2), ...]
        extract_model_provider: 문서 추출용 AI 모델 제공자 ("openai" 또는 "gemini")
        extract_model_name: 문서 추출용 구체적인 모델명
        report_model_provider: 보고서 생성용 AI 모델 제공자 ("openai" 또는 "gemini")
        report_model_name: 보고서 생성용 구체적인 모델명
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
        extract_model_provider=extract_model_provider,
        extract_model_name=extract_model_name,
        report_model_provider=report_model_provider,
        report_model_name=report_model_name,
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
            
            f.write(f'EXTRACT_MODEL="{extract_model_provider}"\n')
            f.write(f'EXTRACT_MODEL_NAME="{extract_model_name}"\n')
            f.write(f'REPORT_MODEL="{report_model_provider}"\n')
            f.write(f'REPORT_MODEL_NAME="{report_model_name}"\n')
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
        else:
            all_cached = False
            break
    
    # 캐시되지 않은 문서가 있으면 분석 실행
    if not all_cached:
        await com.process_documents(
            debug=debug,
            max_rps=max_rps,
            extract_model_provider=extract_model_provider,
            extract_model_name=extract_model_name,
            web_search=web_search
        )
    
    # TODO : 보고서 생성 이전에 skeleton prompt 작성 (JSON에서 좀 더 자연화된 프롬프트, 또는 minor detail 제거 등) 
    # 정말 필요할지 의문. hallucination 문제가 발생할 위험이 있다.
    # TODO : 외부 지식 tool 활용 기능

    # 모든 보고서 생성
    await com.generate_all_reports(
        report_model_provider=report_model_provider,
        report_model_name=report_model_name,
        web=web_search,
        debug=debug
    )
    
    # B2G 평가 검색 및 분석 (section4: 국정과제, 경영평가, 동반성장)
    await com.search_b2g_evaluations(debug=debug)
    
    # 최종 result_json 저장
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
  python main.py -c "Example Corp" -d instruction1:data/instruction1.pdf -em openai -emn gpt-4o -rm openai -rmn gpt-4o --web --max-rps 2.0 --debug
  python main.py -c "Example Corp" -d doc1:data/doc1.pdf doc2:data/doc2.pdf -em gemini -emn gemini-2.0-flash-exp -rm openai -rmn gpt-4o
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
    
    # 선택 인자 - 추출용 모델
    parser.add_argument(
        '-em', '--extract-model',
        type=str,
        choices=['openai', 'gemini'],
        default='openai',
        help='문서 추출용 AI 모델 제공자 (기본값: openai)'
    )
    
    parser.add_argument(
        '-emn', '--extract-model-name',
        type=str,
        default=None,
        help='문서 추출용 구체적인 모델명 (기본값: openai=gpt-4o, gemini=gemini-2.0-flash-exp)'
    )
    
    # 선택 인자 - 보고서용 모델
    parser.add_argument(
        '-rm', '--report-model',
        type=str,
        choices=['openai', 'gemini'],
        default='openai',
        help='보고서 생성용 AI 모델 제공자 (기본값: openai)'
    )
    
    parser.add_argument(
        '-rmn', '--report-model-name',
        type=str,
        default=None,
        help='보고서 생성용 구체적인 모델명 (기본값: openai=gpt-4o, gemini=gemini-2.0-flash-exp)'
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
    
    # 모델명 기본값 설정
    extract_model_name = args.extract_model_name
    if extract_model_name is None:
        extract_model_name = "gpt-4o" if args.extract_model == "openai" else "gemini-2.0-flash-exp"
    
    report_model_name = args.report_model_name
    if report_model_name is None:
        report_model_name = "gpt-4o" if args.report_model == "openai" else "gemini-2.0-flash-exp"
    
    # 설정 정보 출력
    logger.info("=" * 60)
    logger.info("문서 분석 파이프라인 시작")
    logger.info("=" * 60)
    logger.info(f"회사 이름: {args.company}")
    logger.info(f"문서 개수: {len(documents)}")
    for name, path in documents:
        logger.info(f"  - {name}: {path}")
    logger.info(f"추출 모델: {args.extract_model}/{extract_model_name}")
    logger.info(f"보고서 모델: {args.report_model}/{report_model_name}")
    logger.info(f"웹 검색: {args.web}")
    logger.info(f"Max RPS: {args.max_rps}")
    logger.info(f"디버그 모드: {args.debug}")
    logger.info("=" * 60)
    
    # main_async 실행
    asyncio.run(main_async(
        company_name=args.company,
        documents=documents,
        extract_model_provider=args.extract_model,
        extract_model_name=extract_model_name,
        report_model_provider=args.report_model,
        report_model_name=report_model_name,
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