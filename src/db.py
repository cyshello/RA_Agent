"""
B2G 기준데이터 DB 모듈 - LangChain + PGVector 기반

이 모듈은 정부 과제 PDF를 분석하여 PostgreSQL의 PGVector에 저장하고
검색 기능을 제공합니다.

주요 기능:
1. PDF를 페이지별 이미지로 변환 후 CLOVA OCR로 텍스트/표 추출
2. LLM을 사용하여 섹션 경계 감지
3. 섹션별 OCR 결과를 LLM으로 구조화
4. 구조화된 데이터를 PGVector에 임베딩과 함께 저장
5. 시맨틱 검색 및 하이브리드 검색 지원
"""

import os
import json
import uuid
import logging
import asyncio
import dotenv
from typing import Optional, List, Dict, Any
from dataclasses import dataclass, field
from datetime import datetime

import fitz  # PyMuPDF for PDF processing
import PIL.Image
import io

from langchain_openai import OpenAIEmbeddings
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.documents import Document
from langchain_postgres import PGVector
from langchain_postgres.vectorstores import PGVector as PGVectorStore

from sqlalchemy import create_engine, Column, Integer, String, Text, JSON, DateTime, Float
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

from .utils import CLOVA_ocr_with_table
from .api import ModelFactory, Dispatcher
from .prompts import (
    SECTION_END_DETECTION_PROMPT, 
    SECTION_STRUCTURING_PROMPT,
    INCLUSIVE_BULK_STRUCTURING_PROMPT,
    INCLUSIVE_INDEX_EXTRACTION_PROMPT,
    INCLUSIVE_DETAIL_EXTRACTION_PROMPT
)

# .env에서 API 키 로드
env_path = os.path.join(os.path.dirname(__file__), ".env")
OPENAI_KEY = dotenv.get_key(env_path, "OPENAI_KEY")

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ============================================================================
# 데이터 모델 정의
# ============================================================================

@dataclass
class PageOCRResult:
    """페이지별 OCR 결과"""
    page_num: int
    text: str
    fields: List[str]
    tables: List[Dict]
    raw_response: Dict


@dataclass
class Section:
    """문서 섹션"""
    section_id: str
    section_type: str  # 서론, 목차, 과제상세, 부록, 기타
    start_page: int
    end_page: int
    ocr_results: List[PageOCRResult] = field(default_factory=list)
    structured_data: Optional[Dict] = None
    
    def get_full_text(self) -> str:
        """섹션의 전체 텍스트 반환"""
        return "\n\n---페이지 구분---\n\n".join([
            f"[페이지 {r.page_num}]\n{r.text}" 
            for r in self.ocr_results
        ])

@dataclass
class StructuredProject:
    """
    정부 과제 데이터 스키마 (새 스키마 - prompts.py SECTION_STRUCTURING_PROMPT 기반)
    
    JSON 스키마:
    {
        "과제명": "",
        "과제번호": "",
        "과제 목표": [],
        "주요내용": [],
        "기대효과": []
    }
    """
    과제명: str
    과제번호: str
    과제_목표: List[str] = field(default_factory=list)
    주요내용: List[str] = field(default_factory=list)
    기대효과: List[str] = field(default_factory=list)
    
    # 메타데이터
    source_document: Optional[str] = None
    page_range: Optional[str] = None
    extraction_date: Optional[str] = None
    
    @classmethod
    def from_dict(cls, data: Dict, source_document: str = "", page_range: str = "") -> "StructuredProject":
        """딕셔너리에서 객체 생성"""
        return cls(
            과제명=data.get("과제명", ""),
            과제번호=str(data.get("과제번호", "")),
            과제_목표=data.get("과제 목표", []) if isinstance(data.get("과제 목표"), list) else [],
            주요내용=data.get("주요내용", []) if isinstance(data.get("주요내용"), list) else [],
            기대효과=data.get("기대효과", []) if isinstance(data.get("기대효과"), list) else [],
            source_document=source_document,
            page_range=page_range,
            extraction_date=datetime.now().isoformat()
        )
    
    def to_dict(self) -> Dict:
        """딕셔너리로 변환 (원본 JSON 형식)"""
        return {
            "과제명": self.과제명,
            "과제번호": self.과제번호,
            "과제 목표": self.과제_목표,
            "주요내용": self.주요내용,
            "기대효과": self.기대효과,
            "source_document": self.source_document,
            "page_range": self.page_range,
            "extraction_date": self.extraction_date
        }
    
    def to_embedding_items(self) -> List[Dict]:
        """
        각 리스트 항목을 개별 임베딩 항목으로 변환
        
        Returns:
            [
                {
                    "text": "임베딩할 텍스트",
                    "field_type": "과제 목표|주요내용|기대효과",
                    "field_index": 0,
                    "metadata": {...원본 전체 JSON...}
                },
                ...
            ]
        """
        items = []
        base_metadata = self.to_dict()
        
        # 과제 목표
        for i, text in enumerate(self.과제_목표):
            if text.strip():
                items.append({
                    "text": text.strip(),
                    "field_type": "과제 목표",
                    "field_index": i,
                    "metadata": base_metadata
                })
        
        # 주요내용
        for i, text in enumerate(self.주요내용):
            if text.strip():
                items.append({
                    "text": text.strip(),
                    "field_type": "주요내용",
                    "field_index": i,
                    "metadata": base_metadata
                })
        
        # 기대효과
        for i, text in enumerate(self.기대효과):
            if text.strip():
                items.append({
                    "text": text.strip(),
                    "field_type": "기대효과",
                    "field_index": i,
                    "metadata": base_metadata
                })
        
        return items


@dataclass
class StructuredIndicator:
    """
    동반성장 평가 지표 데이터 스키마 (prompts.py INCLUSIVE_BULK_STRUCTURING_PROMPT 기반)
    
    JSON 스키마:
    {
        "지표명": "",
        "평가기준": [],
        "평가방법": [],
        "참고사항": [],
        "증빙자료": []
    }
    """
    지표명: str
    평가기준: List[str] = field(default_factory=list)
    평가방법: List[str] = field(default_factory=list)
    참고사항: List[str] = field(default_factory=list)
    증빙자료: List[str] = field(default_factory=list)
    
    # 메타데이터
    source_document: Optional[str] = None
    page_range: Optional[str] = None
    extraction_date: Optional[str] = None
    
    @classmethod
    def from_dict(cls, data: Dict, source_document: str = "", page_range: str = "") -> "StructuredIndicator":
        """딕셔너리에서 객체 생성"""
        return cls(
            지표명=data.get("지표명", ""),
            평가기준=data.get("평가기준", []) if isinstance(data.get("평가기준"), list) else [],
            평가방법=data.get("평가방법", []) if isinstance(data.get("평가방법"), list) else [],
            참고사항=data.get("참고사항", []) if isinstance(data.get("참고사항"), list) else [],
            증빙자료=data.get("증빙자료", data.get("증빙사료", [])) if isinstance(data.get("증빙자료", data.get("증빙사료")), list) else [],
            source_document=source_document,
            page_range=page_range,
            extraction_date=datetime.now().isoformat()
        )
    
    def to_dict(self) -> Dict:
        """딕셔너리로 변환 (원본 JSON 형식)"""
        return {
            "지표명": self.지표명,
            "평가기준": self.평가기준,
            "평가방법": self.평가방법,
            "참고사항": self.참고사항,
            "증빙자료": self.증빙자료,
            "source_document": self.source_document,
            "page_range": self.page_range,
            "extraction_date": self.extraction_date
        }
    
    def to_embedding_items(self) -> List[Dict]:
        """
        각 리스트 항목을 개별 임베딩 항목으로 변환
        
        Returns:
            [
                {
                    "text": "임베딩할 텍스트",
                    "field_type": "평가기준|평가방법|참고사항|증빙자료",
                    "field_index": 0,
                    "metadata": {...원본 전체 JSON...}
                },
                ...
            ]
        """
        items = []
        base_metadata = self.to_dict()
        
        # 평가기준
        for i, text in enumerate(self.평가기준):
            if text.strip():
                items.append({
                    "text": text.strip(),
                    "field_type": "평가기준",
                    "field_index": i,
                    "metadata": base_metadata
                })
        
        # 평가방법
        for i, text in enumerate(self.평가방법):
            if text.strip():
                items.append({
                    "text": text.strip(),
                    "field_type": "평가방법",
                    "field_index": i,
                    "metadata": base_metadata
                })
        
        # 참고사항
        for i, text in enumerate(self.참고사항):
            if text.strip():
                items.append({
                    "text": text.strip(),
                    "field_type": "참고사항",
                    "field_index": i,
                    "metadata": base_metadata
                })
        
        # 증빙자료
        for i, text in enumerate(self.증빙자료):
            if text.strip():
                items.append({
                    "text": text.strip(),
                    "field_type": "증빙자료",
                    "field_index": i,
                    "metadata": base_metadata
                })
        
        return items


# ============================================================================
# PDF 처리 클래스
# ============================================================================

class PDFProcessor:
    """PDF를 페이지별 이미지로 변환하고 OCR 수행"""
    
    def __init__(self, dpi: int = 200):
        """
        Args:
            dpi: 이미지 변환 해상도 (기본값: 200)
        """
        self.dpi = dpi
        self.zoom = dpi / 72  # PDF 기본 해상도는 72 DPI
    
    def pdf_to_images(self, pdf_path: str) -> List[PIL.Image.Image]:
        """
        PDF를 페이지별 이미지로 변환
        
        Args:
            pdf_path: PDF 파일 경로
            
        Returns:
            PIL Image 객체 리스트
        """
        images = []
        doc = fitz.open(pdf_path)
        
        mat = fitz.Matrix(self.zoom, self.zoom)
        
        for page_num in range(len(doc)):
            page = doc[page_num]
            pix = page.get_pixmap(matrix=mat)
            
            # Pixmap을 PIL Image로 변환
            img_data = pix.tobytes("png")
            img = PIL.Image.open(io.BytesIO(img_data))
            images.append(img)
            
            logger.info(f"페이지 {page_num + 1}/{len(doc)} 이미지 변환 완료")
        
        doc.close()
        return images
    
    def process_page(self, image: PIL.Image.Image, page_num: int) -> PageOCRResult:
        """
        단일 페이지 OCR 처리 (표 인식 포함)
        
        Args:
            image: PIL Image 객체
            page_num: 페이지 번호 (1-based)
            
        Returns:
            PageOCRResult 객체
        """
        # 이미지를 바이트로 변환
        img_byte_arr = io.BytesIO()
        image.save(img_byte_arr, format='PNG')
        img_bytes = img_byte_arr.getvalue()
        
        # CLOVA OCR 호출 (표 인식 포함)
        ocr_result = CLOVA_ocr_with_table(img_bytes)
        
        return PageOCRResult(
            page_num=page_num,
            text=ocr_result['text'],
            fields=ocr_result['fields'],
            tables=ocr_result['tables'],
            raw_response=ocr_result['raw_response']
        )


# ============================================================================
# 섹션 분석기 클래스
# ============================================================================

class SectionAnalyzer:
    """LLM을 사용하여 문서의 섹션 경계를 감지 (국정과제 PDF용)"""
    
    def __init__(
        self,
        model_provider: str = "openai",
        model_name: str = "gpt-4o-mini",
        max_rps: float = 2.0
    ):
        self.model_provider = model_provider
        self.model_name = model_name
        self.max_rps = max_rps
        
        # LangChain 모델 생성
        self.model = ModelFactory.create_model_chain(
            provider=model_provider,
            model_name=model_name,
            output_format="json",
            max_rps=max_rps
        )
        self.parser = JsonOutputParser()
    
    async def detect_section_end(
        self,
        current_page_text: str,
        next_page_text: str,
        current_page_num: int
    ) -> Dict:
        """
        현재 페이지가 섹션의 끝인지 감지
        
        Args:
            current_page_text: 현재 페이지 OCR 텍스트
            next_page_text: 다음 페이지 OCR 텍스트
            current_page_num: 현재 페이지 번호
            
        Returns:
            {
                "is_section_end": bool,
                "reason": str,
                "current_section_type": str,
                "next_section_type": str
            }
        """
        # 프롬프트 생성 (주입된 프롬프트 사용)
        messages = self.section_detection_prompt.format_messages(
            current_page_text=current_page_text[:3000],  # 토큰 제한
            next_page_text=next_page_text[:3000],
            current_page_num=current_page_num
        )
        
        # LLM 호출
        response = await self.model.ainvoke(messages)
        
        # JSON 파싱
        try:
            result = self.parser.parse(response.content)
        except Exception as e:
            logger.warning(f"JSON 파싱 실패: {e}")
            result = {
                "is_section_end": False,
                "reason": "파싱 실패",
                "current_section_type": "기타",
                "next_section_type": "동일섹션계속"
            }
        
        return result
    
    async def analyze_document(
        self,
        ocr_results: List[PageOCRResult]
    ) -> List[Section]:
        """
        전체 문서를 분석하여 섹션으로 분리
        
        Args:
            ocr_results: 페이지별 OCR 결과 리스트
            
        Returns:
            Section 객체 리스트
        """
        sections = []
        current_section_pages = []
        current_section_type = None
        section_start_page = 1
        
        for i, ocr_result in enumerate(ocr_results):
            current_page_num = ocr_result.page_num
            current_section_pages.append(ocr_result)
            
            # 마지막 페이지가 아닌 경우 섹션 끝 감지
            if i < len(ocr_results) - 1:
                next_ocr_result = ocr_results[i + 1]
                
                detection_result = await self.detect_section_end(
                    current_page_text=ocr_result.text,
                    next_page_text=next_ocr_result.text,
                    current_page_num=current_page_num
                )
                
                if current_section_type is None:
                    current_section_type = detection_result.get("current_section_type", "기타")
                
                if detection_result.get("is_section_end", False):
                    # 현재 섹션 저장
                    section = Section(
                        section_id=str(uuid.uuid4()),
                        section_type=current_section_type,
                        start_page=section_start_page,
                        end_page=current_page_num,
                        ocr_results=current_section_pages.copy()
                    )
                    sections.append(section)
                    
                    logger.info(
                        f"섹션 감지: {current_section_type} "
                        f"(페이지 {section_start_page}-{current_page_num})"
                    )
                    
                    # 새 섹션 시작
                    current_section_pages = []
                    current_section_type = detection_result.get("next_section_type", "기타")
                    if current_section_type == "동일섹션계속":
                        current_section_type = "기타"
                    section_start_page = current_page_num + 1
            
            else:
                # 마지막 페이지: 남은 페이지들로 섹션 생성
                if current_section_pages:
                    section = Section(
                        section_id=str(uuid.uuid4()),
                        section_type=current_section_type or "기타",
                        start_page=section_start_page,
                        end_page=current_page_num,
                        ocr_results=current_section_pages.copy()
                    )
                    sections.append(section)
                    
                    logger.info(
                        f"마지막 섹션: {current_section_type} "
                        f"(페이지 {section_start_page}-{current_page_num})"
                    )
        
        return sections


# ============================================================================
# 데이터 구조화 클래스
# ============================================================================

class DataStructurer:
    """LLM을 사용하여 섹션 데이터를 구조화"""
    
    def __init__(
        self,
        model_provider: str = "openai",
        model_name: str = "gpt-4o",
        max_rps: float = 2.0
    ):
        self.model_provider = model_provider
        self.model_name = model_name
        
        self.model = ModelFactory.create_model_chain(
            provider=model_provider,
            model_name=model_name,
            output_format="json",
            max_rps=max_rps
        )
        self.parser = JsonOutputParser()
    
    async def structure_section(
        self,
        section: Section,
        source_document: str = ""
    ) -> Optional[StructuredProject]:
        """
        섹션을 구조화된 정부 과제 데이터로 변환 (새 스키마)
        
        Args:
            section: Section 객체
            source_document: 원본 문서명
            
        Returns:
            StructuredProject 객체 또는 None (과제 정보가 아닌 경우)
        """
        # 서론, 목차 등은 과제 정보로 변환하지 않음
        if section.section_type in ["서론", "목차", "부록"]:
            logger.info(f"섹션 유형 '{section.section_type}'은 과제 정보로 변환하지 않음")
            return None
        
        section_text = section.get_full_text()
        page_range = f"{section.start_page}-{section.end_page}"
        
        # SECTION_STRUCTURING_PROMPT 사용
        messages = SECTION_STRUCTURING_PROMPT.format_messages(
            section_text=section_text[:8000],  # 토큰 제한
            section_type=section.section_type,
            page_range=page_range
        )
        
        response = await self.model.ainvoke(messages)
        
        try:
            structured_data = self.parser.parse(response.content)
        except Exception as e:
            logger.warning(f"JSON 파싱 실패: {e}")
            # 파싱 실패 시 기본값 반환
            structured_data = {
                "과제명": f"섹션 {page_range}",
                "과제번호": section.section_id[:8],
                "과제 목표": [],
                "주요내용": [section_text[:500]],
                "기대효과": []
            }
        
        # StructuredProject 객체 생성
        project = StructuredProject.from_dict(
            data=structured_data,
            source_document=source_document,
            page_range=page_range
        )
        
        section.structured_data = project.to_dict()
        return project
    


# ============================================================================
# PGVector 저장소 클래스
# ============================================================================

class B2GVectorStore:
    """
    PGVector를 사용한 정부 과제 데이터 저장 및 검색
    
    저장 방식:
    - 각 과제의 리스트 필드(과제 목표, 주요내용, 기대효과)의 각 항목을 개별 row로 저장
    - page_content: 해당 항목 텍스트 (임베딩 대상)
    - metadata: 원본 JSON 전체 + field_type, field_index
    
    검색 방식:
    - embedding similarity로 검색
    - 중복 과제번호 제거하여 서로 다른 과제 반환
    """
    
    def __init__(
        self,
        connection_string: str,
        collection_name: str = "b2g_projects",
        embedding_model: str = "text-embedding-3-small"
    ):
        """
        Args:
            connection_string: PostgreSQL 연결 문자열
                예: "postgresql://user:password@localhost:5432/dbname"
            collection_name: 벡터 컬렉션 이름
            embedding_model: OpenAI 임베딩 모델
        """
        self.connection_string = connection_string
        self.collection_name = collection_name
        
        # OpenAI 임베딩 모델 (API 키 명시적 전달)
        self.embeddings = OpenAIEmbeddings(
            model=embedding_model,
            openai_api_key=OPENAI_KEY
        )
        
        # PGVector 저장소 초기화
        self.vector_store = PGVectorStore(
            embeddings=self.embeddings,
            collection_name=collection_name,
            connection=connection_string,
            use_jsonb=True  # 메타데이터 검색 최적화
        )
        
        logger.info(f"B2GVectorStore 초기화 완료: {collection_name}")
    
    # ========================================================================
    # 새 스키마용 메서드 (StructuredProject)
    # ========================================================================
    
    def add_structured_project(self, project: StructuredProject) -> List[str]:
        """
        구조화된 과제를 벡터 저장소에 추가 (각 리스트 항목별로 개별 row)
        
        Args:
            project: StructuredProject 객체
            
        Returns:
            저장된 문서 ID 리스트
        """
        embedding_items = project.to_embedding_items()
        
        if not embedding_items:
            logger.warning(f"과제 '{project.과제명}'에 임베딩할 항목이 없습니다.")
            return []
        
        docs = []
        for item in embedding_items:
            # metadata에 field_type과 field_index 추가
            metadata = item["metadata"].copy()
            metadata["field_type"] = item["field_type"]
            metadata["field_index"] = item["field_index"]
            
            doc = Document(
                page_content=item["text"],
                metadata=metadata
            )
            docs.append(doc)
        
        ids = self.vector_store.add_documents(docs)
        logger.info(f"과제 '{project.과제명}' 저장 완료: {len(ids)}개 항목")
        
        return ids
    
    def add_structured_projects(self, projects: List[StructuredProject]) -> List[str]:
        """
        여러 구조화된 과제를 벡터 저장소에 배치 추가
        
        Args:
            projects: StructuredProject 객체 리스트
            
        Returns:
            저장된 문서 ID 리스트
        """
        all_docs = []
        
        for project in projects:
            embedding_items = project.to_embedding_items()
            
            for item in embedding_items:
                metadata = item["metadata"].copy()
                metadata["field_type"] = item["field_type"]
                metadata["field_index"] = item["field_index"]
                
                doc = Document(
                    page_content=item["text"],
                    metadata=metadata
                )
                all_docs.append(doc)
        
        if not all_docs:
            logger.warning("저장할 문서가 없습니다.")
            return []
        
        ids = self.vector_store.add_documents(all_docs)
        logger.info(f"{len(projects)}개 과제에서 총 {len(ids)}개 항목 저장 완료")
        
        return ids
    
    def search_unique_projects(
        self,
        query: str,
        k: int = 10,
        max_candidates: int = 100
    ) -> List[Dict]:
        """
        중복 과제번호를 제외하고 서로 다른 과제 검색
        
        Args:
            query: 검색 쿼리
            k: 반환할 서로 다른 과제 수 (기본값: 10)
            max_candidates: 검색할 최대 후보 수 (기본값: 100)
            
        Returns:
            서로 다른 과제번호를 가진 과제 리스트
            [
                {
                    "score": 0.95,
                    "과제번호": "1",
                    "과제명": "...",
                    "matched_text": "검색에 매칭된 텍스트",
                    "matched_field": "과제 목표|주요내용|기대효과",
                    "full_data": {...원본 전체 JSON...}
                },
                ...
            ]
        """
        # 충분한 후보를 가져옴
        results = self.vector_store.similarity_search_with_score(
            query=query,
            k=max_candidates
        )
        
        seen_project_numbers = set()
        unique_results = []
        
        for doc, score in results:
            project_number = doc.metadata.get("과제번호", "")
            
            # 이미 본 과제번호면 스킵
            if project_number in seen_project_numbers:
                continue
            
            seen_project_numbers.add(project_number)
            
            unique_results.append({
                "score": float(score),
                "과제번호": project_number,
                "과제명": doc.metadata.get("과제명", ""),
                "matched_text": doc.page_content,
                "matched_field": doc.metadata.get("field_type", ""),
                "full_data": {
                    "과제명": doc.metadata.get("과제명", ""),
                    "과제번호": project_number,
                    "과제 목표": doc.metadata.get("과제 목표", []),
                    "주요내용": doc.metadata.get("주요내용", []),
                    "기대효과": doc.metadata.get("기대효과", []),
                    "source_document": doc.metadata.get("source_document", ""),
                    "page_range": doc.metadata.get("page_range", "")
                }
            })
            
            # k개 찾으면 종료
            if len(unique_results) >= k:
                break
        
        logger.info(f"검색 완료: {len(unique_results)}개 서로 다른 과제 반환")
        return unique_results
    
    def search_by_field_type(
        self,
        query: str,
        field_type: str,
        k: int = 10
    ) -> List[Dict]:
        """
        특정 필드 유형에서만 검색
        
        Args:
            query: 검색 쿼리
            field_type: "과제 목표" | "주요내용" | "기대효과"
            k: 반환할 결과 수
            
        Returns:
            검색 결과 리스트
        """
        results = self.vector_store.similarity_search_with_score(
            query=query,
            k=k,
            filter={"field_type": field_type}
        )
        
        return [{
            "score": float(score),
            "과제번호": doc.metadata.get("과제번호", ""),
            "과제명": doc.metadata.get("과제명", ""),
            "matched_text": doc.page_content,
            "full_data": doc.metadata
        } for doc, score in results]
    
    
    def search(
        self,
        query: str,
        k: int = 5,
        filter: Optional[Dict] = None
    ) -> List[Document]:
        """
        시맨틱 검색 수행
        
        Args:
            query: 검색 쿼리
            k: 반환할 결과 수
            filter: 메타데이터 필터 (예: {"ministry": "과학기술정보통신부"})
            
        Returns:
            검색 결과 Document 리스트
        """
        results = self.vector_store.similarity_search(
            query=query,
            k=k,
            filter=filter
        )
        return results
    
    def search_with_score(
        self,
        query: str,
        k: int = 5,
        filter: Optional[Dict] = None
    ) -> List[tuple]:
        """
        유사도 점수와 함께 시맨틱 검색 수행
        
        Args:
            query: 검색 쿼리
            k: 반환할 결과 수
            filter: 메타데이터 필터
            
        Returns:
            (Document, score) 튜플 리스트
        """
        results = self.vector_store.similarity_search_with_score(
            query=query,
            k=k,
            filter=filter
        )
        return results
    
    def search_by_ministry(
        self,
        query: str,
        ministry: str,
        k: int = 5
    ) -> List[Document]:
        """
        특정 부처의 과제만 검색
        
        Args:
            query: 검색 쿼리
            ministry: 담당 부처명
            k: 반환할 결과 수
            
        Returns:
            검색 결과 Document 리스트
        """
        return self.search(query, k=k, filter={"ministry": ministry})
    
    def get_all_projects(self, limit: int = 100) -> List[Document]:
        """
        저장된 모든 과제 조회 (페이지네이션)
        
        Args:
            limit: 최대 반환 수
            
        Returns:
            Document 리스트
        """
        # 빈 쿼리로 전체 조회
        return self.vector_store.similarity_search(
            query="정부 과제",
            k=limit
        )
    
    def delete_by_document_id(self, doc_id: str) -> bool:
        """
        문서 ID로 삭제
        
        Args:
            doc_id: 문서 ID
            
        Returns:
            삭제 성공 여부
        """
        try:
            self.vector_store.delete([doc_id])
            logger.info(f"문서 삭제 완료: {doc_id}")
            return True
        except Exception as e:
            logger.error(f"문서 삭제 실패: {e}")
            return False
    
    def delete_collection(self) -> bool:
        """
        전체 컬렉션 삭제
        
        Returns:
            삭제 성공 여부
        """
        try:
            self.vector_store.delete_collection()
            logger.info(f"컬렉션 삭제 완료: {self.collection_name}")
            return True
        except Exception as e:
            logger.error(f"컬렉션 삭제 실패: {e}")
            return False


# ============================================================================
# 동반성장 평가지표 저장소 클래스
# ============================================================================

class InclusiveGrowthVectorStore:
    """
    PGVector를 사용한 동반성장 평가지표 데이터 저장 및 검색
    
    저장 방식:
    - 각 지표의 리스트 필드(평가기준, 평가방법, 참고사항, 증빙자료)의 각 항목을 개별 row로 저장
    - page_content: 해당 항목 텍스트 (임베딩 대상)
    - metadata: 원본 JSON 전체 + field_type, field_index
    
    검색 방식:
    - embedding similarity로 검색
    - 중복 지표명 제거하여 서로 다른 지표 반환
    """
    
    def __init__(
        self,
        connection_string: str,
        collection_name: str = "inclusive_growth_indicators",
        embedding_model: str = "text-embedding-3-small"
    ):
        """
        Args:
            connection_string: PostgreSQL 연결 문자열
                예: "postgresql://user:password@localhost:5432/dbname"
            collection_name: 벡터 컬렉션 이름
            embedding_model: OpenAI 임베딩 모델
        """
        self.connection_string = connection_string
        self.collection_name = collection_name
        
        # OpenAI 임베딩 모델 (API 키 명시적 전달)
        self.embeddings = OpenAIEmbeddings(
            model=embedding_model,
            openai_api_key=OPENAI_KEY
        )
        
        # PGVector 저장소 초기화
        self.vector_store = PGVectorStore(
            embeddings=self.embeddings,
            collection_name=collection_name,
            connection=connection_string,
            use_jsonb=True  # 메타데이터 검색 최적화
        )
        
        logger.info(f"InclusiveGrowthVectorStore 초기화 완료: {collection_name}")
    
    def add_indicator(self, indicator: StructuredIndicator) -> List[str]:
        """
        구조화된 지표를 벡터 저장소에 추가 (각 리스트 항목별로 개별 row)
        
        Args:
            indicator: StructuredIndicator 객체
            
        Returns:
            저장된 문서 ID 리스트
        """
        embedding_items = indicator.to_embedding_items()
        
        if not embedding_items:
            logger.warning(f"지표 '{indicator.지표명}'에 임베딩할 항목이 없습니다.")
            return []
        
        docs = []
        for item in embedding_items:
            # metadata에 field_type과 field_index 추가
            metadata = item["metadata"].copy()
            metadata["field_type"] = item["field_type"]
            metadata["field_index"] = item["field_index"]
            
            doc = Document(
                page_content=item["text"],
                metadata=metadata
            )
            docs.append(doc)
        
        ids = self.vector_store.add_documents(docs)
        logger.info(f"지표 '{indicator.지표명}' 저장 완료: {len(ids)}개 항목")
        
        return ids
    
    def add_indicators(self, indicators: List[StructuredIndicator]) -> List[str]:
        """
        여러 구조화된 지표를 벡터 저장소에 배치 추가
        
        Args:
            indicators: StructuredIndicator 객체 리스트
            
        Returns:
            저장된 문서 ID 리스트
        """
        all_docs = []
        
        for indicator in indicators:
            embedding_items = indicator.to_embedding_items()
            
            for item in embedding_items:
                metadata = item["metadata"].copy()
                metadata["field_type"] = item["field_type"]
                metadata["field_index"] = item["field_index"]
                
                doc = Document(
                    page_content=item["text"],
                    metadata=metadata
                )
                all_docs.append(doc)
        
        if not all_docs:
            logger.warning("저장할 문서가 없습니다.")
            return []
        
        ids = self.vector_store.add_documents(all_docs)
        logger.info(f"{len(indicators)}개 지표에서 총 {len(ids)}개 항목 저장 완료")
        
        return ids
    
    def search_unique_indicators(
        self,
        query: str,
        k: int = 10,
        max_candidates: int = 100
    ) -> List[Dict]:
        """
        중복 지표명을 제외하고 서로 다른 지표 검색
        
        Args:
            query: 검색 쿼리
            k: 반환할 서로 다른 지표 수 (기본값: 10)
            max_candidates: 검색할 최대 후보 수 (기본값: 100)
            
        Returns:
            서로 다른 지표명을 가진 지표 리스트
            [
                {
                    "score": 0.95,
                    "지표명": "...",
                    "matched_text": "검색에 매칭된 텍스트",
                    "matched_field": "평가기준|평가방법|참고사항|증빙자료",
                    "full_data": {...원본 전체 JSON...}
                },
                ...
            ]
        """
        # 충분한 후보를 가져옴
        results = self.vector_store.similarity_search_with_score(
            query=query,
            k=max_candidates
        )
        
        seen_indicator_names = set()
        unique_results = []
        
        for doc, score in results:
            indicator_name = doc.metadata.get("지표명", "")
            
            # 이미 본 지표명이면 스킵
            if indicator_name in seen_indicator_names:
                continue
            
            seen_indicator_names.add(indicator_name)
            
            unique_results.append({
                "score": float(score),
                "지표명": indicator_name,
                "matched_text": doc.page_content,
                "matched_field": doc.metadata.get("field_type", ""),
                "full_data": {
                    "지표명": indicator_name,
                    "평가기준": doc.metadata.get("평가기준", []),
                    "평가방법": doc.metadata.get("평가방법", []),
                    "참고사항": doc.metadata.get("참고사항", []),
                    "증빙자료": doc.metadata.get("증빙자료", []),
                    "source_document": doc.metadata.get("source_document", ""),
                    "page_range": doc.metadata.get("page_range", "")
                }
            })
            
            # k개 찾으면 종료
            if len(unique_results) >= k:
                break
        
        logger.info(f"검색 완료: {len(unique_results)}개 서로 다른 지표 반환")
        return unique_results
    
    def search_by_field_type(
        self,
        query: str,
        field_type: str,
        k: int = 10
    ) -> List[Dict]:
        """
        특정 필드 유형에서만 검색
        
        Args:
            query: 검색 쿼리
            field_type: "평가기준" | "평가방법" | "참고사항" | "증빙자료"
            k: 반환할 결과 수
            
        Returns:
            검색 결과 리스트
        """
        results = self.vector_store.similarity_search_with_score(
            query=query,
            k=k,
            filter={"field_type": field_type}
        )
        
        return [{
            "score": float(score),
            "지표명": doc.metadata.get("지표명", ""),
            "matched_text": doc.page_content,
            "full_data": doc.metadata
        } for doc, score in results]
    
    def delete_collection(self) -> bool:
        """
        전체 컬렉션 삭제
        
        Returns:
            삭제 성공 여부
        """
        try:
            self.vector_store.delete_collection()
            logger.info(f"컬렉션 삭제 완료: {self.collection_name}")
            return True
        except Exception as e:
            logger.error(f"컬렉션 삭제 실패: {e}")
            return False


# ============================================================================
# 메인 파이프라인 클래스
# ============================================================================

class B2GDataPipeline:
    """
    PDF에서 PGVector까지의 전체 파이프라인 관리
    
    처리 흐름:
    1. PDF → 이미지 변환
    2. 이미지 → OCR (표 포함)
    3. OCR 결과 → 섹션 분리
    4. 섹션 → 구조화된 데이터 (새 스키마: StructuredProject)
    5. 구조화된 데이터 → PGVector 저장 (각 리스트 항목별 개별 row)
    """
    
    def __init__(
        self,
        connection_string: str,
        collection_name: str = "b2g_projects",
        model_provider: str = "openai",
        extraction_model: str = "gpt-4o-mini",
        structuring_model: str = "gpt-4o",
        max_rps: float = 2.0
    ):
        """
        Args:
            connection_string: PostgreSQL 연결 문자열
            collection_name: 벡터 컬렉션 이름
            model_provider: LLM 제공자 ("openai" 또는 "gemini")
            extraction_model: 섹션 감지용 모델
            structuring_model: 데이터 구조화용 모델
            max_rps: API 요청 제한 (초당)
        """
        self.pdf_processor = PDFProcessor()
        self.section_analyzer = SectionAnalyzer(
            model_provider=model_provider,
            model_name=extraction_model,
            max_rps=max_rps
        )
        self.data_structurer = DataStructurer(
            model_provider=model_provider,
            model_name=structuring_model,
            max_rps=max_rps
        )
        self.vector_store = B2GVectorStore(
            connection_string=connection_string,
            collection_name=collection_name
        )
        
        logger.info("B2GDataPipeline 초기화 완료")
    
    async def process_pdf(
        self,
        pdf_path: str,
        save_intermediate: bool = False,
        output_dir: Optional[str] = None
    ) -> List[StructuredProject]:
        """
        PDF 파일을 처리하여 구조화된 데이터를 벡터 저장소에 저장
        
        Args:
            pdf_path: PDF 파일 경로
            save_intermediate: 중간 결과 저장 여부
            output_dir: 중간 결과 저장 디렉토리
            
        Returns:
            처리된 StructuredProject 리스트
        """
        source_document = os.path.basename(pdf_path)
        logger.info(f"PDF 처리 시작: {source_document}")
        
        # 1. PDF를 이미지로 변환
        logger.info("Step 1: PDF → 이미지 변환")
        images = self.pdf_processor.pdf_to_images(pdf_path)
        logger.info(f"총 {len(images)} 페이지 변환 완료")
        
        # 2. 페이지별 OCR 수행 (직렬 처리)
        logger.info("Step 2: 페이지별 OCR 수행")
        ocr_results = []
        for i, image in enumerate(images):
            page_num = i + 1
            logger.info(f"OCR 처리 중: {page_num}/{len(images)}")
            result = self.pdf_processor.process_page(image, page_num)
            ocr_results.append(result)
            
            # 중간 결과 저장
            if save_intermediate and output_dir:
                os.makedirs(output_dir, exist_ok=True)
                ocr_file = os.path.join(output_dir, f"ocr_page_{page_num}.json")
                with open(ocr_file, 'w', encoding='utf-8') as f:
                    json.dump({
                        'page_num': result.page_num,
                        'text': result.text,
                        'fields': result.fields,
                        'tables': result.tables
                    }, f, ensure_ascii=False, indent=2)
        
        # 3. 섹션 분석 (직렬 처리 - 다음 페이지 참조 필요)
        logger.info("Step 3: 섹션 분석")
        sections = await self.section_analyzer.analyze_document(ocr_results)
        logger.info(f"총 {len(sections)}개 섹션 감지")
        
        # 중간 결과 저장
        if save_intermediate and output_dir:
            sections_file = os.path.join(output_dir, "sections.json")
            with open(sections_file, 'w', encoding='utf-8') as f:
                sections_data = [{
                    'section_id': s.section_id,
                    'section_type': s.section_type,
                    'start_page': s.start_page,
                    'end_page': s.end_page,
                    'text_preview': s.get_full_text()[:500]
                } for s in sections]
                json.dump(sections_data, f, ensure_ascii=False, indent=2)
        
        # 4. 섹션 데이터 구조화 (새 스키마)
        logger.info("Step 4: 데이터 구조화")
        projects: List[StructuredProject] = []
        for section in sections:
            project = await self.data_structurer.structure_section(
                section, source_document
            )
            if project:
                projects.append(project)
        
        logger.info(f"총 {len(projects)}개 과제 데이터 생성")
        
        # 중간 결과 저장
        if save_intermediate and output_dir:
            projects_file = os.path.join(output_dir, "projects.json")
            with open(projects_file, 'w', encoding='utf-8') as f:
                json.dump([p.to_dict() for p in projects], f, ensure_ascii=False, indent=2)
        
        # 5. PGVector에 저장 (각 리스트 항목별 개별 row)
        logger.info("Step 5: PGVector 저장")
        if projects:
            self.vector_store.add_structured_projects(projects)
        
        logger.info(f"PDF 처리 완료: {source_document}")
        return projects
    
    def search_unique_projects(
        self,
        query: str,
        k: int = 10,
        max_candidates: int = 100
    ) -> List[Dict]:
        """
        중복 과제번호를 제외하고 서로 다른 과제 검색
        
        Args:
            query: 검색 쿼리
            k: 반환할 서로 다른 과제 수 (기본값: 10)
            max_candidates: 검색할 최대 후보 수 (기본값: 100)
            
        Returns:
            서로 다른 과제번호를 가진 과제 리스트
        """
        return self.vector_store.search_unique_projects(
            query=query,
            k=k,
            max_candidates=max_candidates
        )
    
    def search_by_field(
        self,
        query: str,
        field_type: str,
        k: int = 10
    ) -> List[Dict]:
        """
        특정 필드 유형에서만 검색
        
        Args:
            query: 검색 쿼리
            field_type: "과제 목표" | "주요내용" | "기대효과"
            k: 반환할 결과 수
        """
        return self.vector_store.search_by_field_type(
            query=query,
            field_type=field_type,
            k=k
        )
    
    def search(
        self,
        query: str,
        k: int = 5,
        filter: Optional[Dict] = None
    ) -> List[Dict]:
        """
        저장된 과제 검색 (레거시 - 모든 row 반환)
        
        Args:
            query: 검색 쿼리
            k: 반환할 결과 수
            filter: 메타데이터 필터
            
        Returns:
            검색 결과 (메타데이터 딕셔너리 리스트)
        """
        results = self.vector_store.search_with_score(query, k=k, filter=filter)
        
        return [{
            'score': score,
            '과제명': doc.metadata.get('과제명', ''),
            '과제번호': doc.metadata.get('과제번호', ''),
            'matched_text': doc.page_content,
            'field_type': doc.metadata.get('field_type', ''),
            'full_metadata': doc.metadata
        } for doc, score in results]


# ============================================================================
# 동반성장 평가지표 파이프라인 클래스
# ============================================================================

class InclusiveGrowthPipeline:
    """
    동반성장 평가지표 PDF 처리 파이프라인
    
    처리 흐름:
    1. PDF → 이미지 변환
    2. 이미지 → OCR (표 포함)
    3. OCR 결과 → 섹션 분리
    4. 섹션 → 구조화된 데이터 (StructuredIndicator)
    5. 구조화된 데이터 → PGVector 저장 (각 리스트 항목별 개별 row)
    
    문서 구조: 목차, 개요, 평가그룹, 평가방법, 결과활용, 세부평가
    - 세부평가 부분만 구조화하여 저장
    """
    
    def __init__(
        self,
        connection_string: str,
        collection_name: str = "inclusive_growth_indicators",
        model_provider: str = "openai",
        extraction_model: str = "gpt-4o-mini",
        structuring_model: str = "gpt-4o",
        max_rps: float = 2.0
    ):
        """
        Args:
            connection_string: PostgreSQL 연결 문자열
            collection_name: 벡터 컬렉션 이름
            model_provider: LLM 제공자 ("openai" 또는 "gemini")
            extraction_model: 섹션 감지용 모델
            structuring_model: 데이터 구조화용 모델
            max_rps: API 요청 제한 (초당)
        """
        self.pdf_processor = PDFProcessor()
        # 섹션 분석기 (페이지 범위 기반 처리에서는 직접 사용하지 않음)
        self.section_analyzer = SectionAnalyzer(
            model_provider=model_provider,
            model_name=extraction_model,
            max_rps=max_rps
        )
        self.data_structurer = DataStructurer(
            model_provider=model_provider,
            model_name=structuring_model,
            max_rps=max_rps
        )
        self.vector_store = InclusiveGrowthVectorStore(
            connection_string=connection_string,
            collection_name=collection_name
        )
        
        logger.info("InclusiveGrowthPipeline 초기화 완료")
    
    def _tables_to_text(self, tables: List[Dict]) -> str:
        """
        OCR 결과의 tables를 텍스트 테이블 형태로 변환
        
        Args:
            tables: OCR 결과의 tables 리스트
            
        Returns:
            텍스트 테이블 문자열
        """
        if not tables:
            return ""
        
        table_texts = []
        for i, table in enumerate(tables):
            # markdown 필드가 있으면 그대로 사용
            if "markdown" in table and table["markdown"]:
                table_texts.append(f"[표 {i+1}]\n{table['markdown']}")
            # rows 필드가 있으면 텍스트 테이블로 변환
            elif "rows" in table and table["rows"]:
                rows = table["rows"]
                lines = []
                for row_idx in sorted(rows.keys(), key=int):
                    row = rows[row_idx]
                    cells = [row.get(str(col_idx), "") for col_idx in sorted(int(k) for k in row.keys())]
                    lines.append(" | ".join(cells))
                table_texts.append(f"[표 {i+1}]\n" + "\n".join(lines))
        
        return "\n\n".join(table_texts)
    
    def _load_ocr_from_json(self, ocr_dir: str, page_num: int) -> Optional[Dict]:
        """
        저장된 OCR JSON 파일 로드
        
        Args:
            ocr_dir: OCR JSON 파일이 있는 디렉토리
            page_num: 페이지 번호
            
        Returns:
            OCR 결과 딕셔너리 또는 None
        """
        ocr_file = os.path.join(ocr_dir, f"ocr_page_{page_num}.json")
        if os.path.exists(ocr_file):
            with open(ocr_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        return None
    
    async def process_pdf_by_page_range(
        self,
        pdf_path: str,
        start_page: int,
        end_page: int,
        save_intermediate: bool = False,
        output_dir: Optional[str] = None,
        reuse_ocr_dir: Optional[str] = None
    ) -> List[StructuredIndicator]:
        """
        지정된 페이지 범위의 OCR 결과를 모아서 한번에 LLM으로 구조화
        
        Args:
            pdf_path: PDF 파일 경로
            start_page: 시작 페이지 (1-based)
            end_page: 끝 페이지 (1-based, inclusive)
            save_intermediate: 중간 결과 저장 여부
            output_dir: 중간 결과 저장 디렉토리
            reuse_ocr_dir: 기존 OCR 결과를 재사용할 디렉토리 (지정하면 OCR 스킵)
            
        Returns:
            처리된 StructuredIndicator 리스트
        """
        source_document = os.path.basename(pdf_path)
        logger.info(f"동반성장 평가지표 PDF 처리 시작: {source_document} (페이지 {start_page}-{end_page})")
        
        ocr_data_list = []
        
        # 기존 OCR 결과 재사용
        if reuse_ocr_dir and os.path.exists(reuse_ocr_dir):
            logger.info(f"Step 1-2: 기존 OCR 결과 재사용 ({reuse_ocr_dir})")
            for page_num in range(start_page, end_page + 1):
                ocr_data = self._load_ocr_from_json(reuse_ocr_dir, page_num)
                if ocr_data:
                    ocr_data_list.append(ocr_data)
                    logger.info(f"OCR 로드: 페이지 {page_num}")
                else:
                    logger.warning(f"OCR 파일 없음: 페이지 {page_num}")
        else:
            # 새로 OCR 수행
            logger.info("Step 1: PDF → 이미지 변환")
            images = self.pdf_processor.pdf_to_images(pdf_path)
            total_pages = len(images)
            logger.info(f"총 {total_pages} 페이지 변환 완료")
            
            # 페이지 범위 검증
            if start_page < 1 or end_page > total_pages or start_page > end_page:
                raise ValueError(f"잘못된 페이지 범위: {start_page}-{end_page} (총 {total_pages} 페이지)")
            
            logger.info(f"Step 2: 페이지 {start_page}-{end_page} OCR 수행")
            for page_num in range(start_page, end_page + 1):
                idx = page_num - 1  # 0-based index
                logger.info(f"OCR 처리 중: {page_num}/{end_page}")
                result = self.pdf_processor.process_page(images[idx], page_num)
                
                ocr_data = {
                    'page_num': result.page_num,
                    'text': result.text,
                    'fields': result.fields,
                    'tables': result.tables
                }
                ocr_data_list.append(ocr_data)
                
                # 중간 결과 저장
                if save_intermediate and output_dir:
                    os.makedirs(output_dir, exist_ok=True)
                    ocr_file = os.path.join(output_dir, f"ocr_page_{page_num}.json")
                    with open(ocr_file, 'w', encoding='utf-8') as f:
                        json.dump(ocr_data, f, ensure_ascii=False, indent=2)
        
        # 3. 모든 OCR 결과를 테이블 중심 텍스트로 변환
        logger.info("Step 3: OCR 결과 통합 (테이블 형식)")
        combined_parts = []
        for ocr_data in ocr_data_list:
            page_num = ocr_data['page_num']
            tables = ocr_data.get('tables', [])
            
            # 테이블을 텍스트로 변환
            table_text = self._tables_to_text(tables)
            
            if table_text:
                combined_parts.append(f"=== 페이지 {page_num} ===\n{table_text}")
            else:
                # 테이블이 없으면 일반 텍스트 사용
                text = ocr_data.get('text', '')
                if text:
                    combined_parts.append(f"=== 페이지 {page_num} ===\n{text}")
        
        combined_text = "\n\n".join(combined_parts)
        
        # 중간 결과 저장
        if save_intermediate and output_dir:
            os.makedirs(output_dir, exist_ok=True)
            combined_file = os.path.join(output_dir, f"combined_tables_{start_page}_{end_page}.txt")
            with open(combined_file, 'w', encoding='utf-8') as f:
                f.write(combined_text)
        
        # 4. LLM으로 한번에 모든 지표 추출
        logger.info("Step 4: LLM으로 평가지표 일괄 추출")
        indicators = await self._extract_indicators_bulk(
            ocr_text=combined_text,
            start_page=start_page,
            end_page=end_page,
            source_document=source_document
        )
        
        logger.info(f"총 {len(indicators)}개 평가지표 데이터 생성")
        
        # 중간 결과 저장
        if save_intermediate and output_dir:
            indicators_file = os.path.join(output_dir, "indicators.json")
            with open(indicators_file, 'w', encoding='utf-8') as f:
                json.dump([i.to_dict() for i in indicators], f, ensure_ascii=False, indent=2)
        
        # 5. PGVector에 저장
        logger.info("Step 5: PGVector 저장")
        if indicators:
            self.vector_store.add_indicators(indicators)
        
        logger.info(f"동반성장 평가지표 PDF 처리 완료: {source_document}")
        return indicators
    
    async def _extract_indicators_bulk(
        self,
        ocr_text: str,
        start_page: int,
        end_page: int,
        source_document: str
    ) -> List[StructuredIndicator]:
        """
        OCR 텍스트에서 여러 평가지표를 한번에 추출
        
        Args:
            ocr_text: 통합된 OCR 텍스트
            start_page: 시작 페이지
            end_page: 끝 페이지
            source_document: 원본 문서명
            
        Returns:
            StructuredIndicator 리스트
        """
        # LLM 모델 생성
        model = ModelFactory.create_model_chain(
            provider="openai",
            model_name="gpt-4o",
            output_format="json",
            max_rps=2.0
        )
        parser = JsonOutputParser()
        
        # 프롬프트 생성
        messages = INCLUSIVE_BULK_STRUCTURING_PROMPT.format_messages(
            ocr_text=ocr_text[:30000],  # 토큰 제한
            start_page=start_page,
            end_page=end_page
        )
        
        # LLM 호출
        response = await model.ainvoke(messages)
        
        try:
            result = parser.parse(response.content)
            indicators_data = result.get("indicators", [])
        except Exception as e:
            logger.warning(f"JSON 파싱 실패: {e}")
            indicators_data = []
        
        # StructuredIndicator 객체로 변환
        indicators = []
        for data in indicators_data:
            indicator = StructuredIndicator.from_dict(
                data=data,
                source_document=source_document,
                page_range=f"{start_page}-{end_page}"
            )
            if indicator.지표명:  # 지표명이 있는 경우만 추가
                indicators.append(indicator)
        
        return indicators
    
    async def process_two_stage(
        self,
        index_start: int,
        index_end: int,
        detail_start: int,
        detail_end: int,
        save_intermediate: bool = False,
        output_dir: Optional[str] = None,
        reuse_ocr_dir: Optional[str] = None,
        source_document: str = "evaluation.pdf",
        pdf_path: Optional[str] = None
    ) -> List[StructuredIndicator]:
        """
        2단계 처리: 목록 페이지에서 지표명 추출 → 세부 페이지에서 상세 정보 채우기
        
        Args:
            index_start: 목록(목차) 시작 페이지
            index_end: 목록(목차) 끝 페이지
            detail_start: 세부 내용 시작 페이지
            detail_end: 세부 내용 끝 페이지
            save_intermediate: 중간 결과 저장 여부
            output_dir: 중간 결과 저장 디렉토리
            reuse_ocr_dir: 기존 OCR 결과 재사용 디렉토리 (없으면 새로 OCR 수행)
            source_document: 원본 문서명
            pdf_path: PDF 파일 경로 (OCR 수행시 필요)
            
        Returns:
            처리된 StructuredIndicator 리스트
        """
        logger.info(f"2단계 처리 시작")
        logger.info(f"  - 목록 페이지: {index_start}-{index_end}")
        logger.info(f"  - 세부 페이지: {detail_start}-{detail_end}")
        
        # 필요한 모든 페이지 목록
        all_pages = sorted(set(
            list(range(index_start, index_end + 1)) + 
            list(range(detail_start, detail_end + 1))
        ))
        
        # OCR 데이터 로드 또는 수행
        ocr_data_map = {}  # page_num -> ocr_data
        missing_pages = []
        
        # 1. 기존 OCR 결과가 있으면 로드
        if reuse_ocr_dir and os.path.exists(reuse_ocr_dir):
            logger.info(f"기존 OCR 결과 확인: {reuse_ocr_dir}")
            for page_num in all_pages:
                ocr_data = self._load_ocr_from_json(reuse_ocr_dir, page_num)
                if ocr_data:
                    ocr_data_map[page_num] = ocr_data
                else:
                    missing_pages.append(page_num)
            
            if ocr_data_map:
                logger.info(f"기존 OCR 로드: {len(ocr_data_map)}개 페이지")
            if missing_pages:
                logger.info(f"OCR 필요: {len(missing_pages)}개 페이지")
        else:
            missing_pages = all_pages
            logger.info(f"OCR 결과 없음, 전체 {len(missing_pages)}개 페이지 OCR 필요")
        
        # 2. 누락된 페이지가 있으면 OCR 수행
        if missing_pages:
            if not hasattr(self, '_pdf_images') or self._pdf_images is None:
                # PDF 이미지 로드 (한번만)
                if source_document:
                    # output_dir에서 PDF 경로 추정
                    pdf_candidates = [
                        os.path.join(os.path.dirname(output_dir or reuse_ocr_dir or '.'), source_document),
                        os.path.join(os.path.dirname(output_dir or reuse_ocr_dir or '.'), '..', 'data', source_document),
                    ]
                    pdf_path = None
                    for candidate in pdf_candidates:
                        if os.path.exists(candidate):
                            pdf_path = candidate
                            break
                
                if not pdf_path or not os.path.exists(pdf_path):
                    raise ValueError(f"OCR을 위한 PDF 파일이 필요합니다. pdf_path를 지정하거나 source_document 경로를 확인하세요.")
                
                logger.info(f"PDF 로드: {pdf_path}")
                self._pdf_images = self.pdf_processor.pdf_to_images(pdf_path)
            
            # 누락된 페이지만 OCR 수행
            logger.info(f"OCR 수행 중: {len(missing_pages)}개 페이지")
            for page_num in missing_pages:
                idx = page_num - 1  # 0-based index
                if idx < 0 or idx >= len(self._pdf_images):
                    logger.warning(f"페이지 {page_num}이 PDF 범위를 벗어남")
                    continue
                
                logger.info(f"  OCR: 페이지 {page_num}")
                result = self.pdf_processor.process_page(self._pdf_images[idx], page_num)
                
                ocr_data = {
                    'page_num': result.page_num,
                    'text': result.text,
                    'fields': result.fields,
                    'tables': result.tables
                }
                ocr_data_map[page_num] = ocr_data
                
                # 중간 결과 저장
                if save_intermediate and output_dir:
                    os.makedirs(output_dir, exist_ok=True)
                    ocr_file = os.path.join(output_dir, f"ocr_page_{page_num}.json")
                    with open(ocr_file, 'w', encoding='utf-8') as f:
                        json.dump(ocr_data, f, ensure_ascii=False, indent=2)
        
        # ========== 1단계: 목록 페이지에서 지표명 추출 ==========
        logger.info("Step 1: 목록 페이지에서 평가지표 목록 추출")
        
        index_text_parts = []
        for page_num in range(index_start, index_end + 1):
            if page_num in ocr_data_map:
                table_text = self._tables_to_text(ocr_data_map[page_num].get('tables', []))
                if table_text:
                    index_text_parts.append(f"=== 페이지 {page_num} ===\n{table_text}")
        
        index_text = "\n\n".join(index_text_parts)
        
        indicator_names = await self._extract_indicator_index(
            ocr_text=index_text,
            start_page=index_start,
            end_page=index_end
        )
        
        logger.info(f"추출된 평가지표 목록: {len(indicator_names)}개")
        for item in indicator_names:
            logger.info(f"  - {item.get('번호', '')} {item.get('지표명', '')}")
        
        # 중간 결과 저장
        if save_intermediate and output_dir:
            os.makedirs(output_dir, exist_ok=True)
            with open(os.path.join(output_dir, "indicator_index.json"), 'w', encoding='utf-8') as f:
                json.dump(indicator_names, f, ensure_ascii=False, indent=2)
        
        # ========== 2단계: 세부 페이지에서 상세 정보 추출 ==========
        logger.info("Step 2: 세부 페이지에서 상세 정보 추출")
        
        # 지표별 상세 정보를 누적할 딕셔너리
        indicator_details = {item.get('지표명', ''): {
            '번호': item.get('번호', ''),
            '지표명': item.get('지표명', ''),
            '배점': item.get('배점', ''),
            '평가기준': [],
            '평가방법': [],
            '참고사항': [],
            '증빙자료': []
        } for item in indicator_names if item.get('지표명')}
        
        # 이전 페이지 텍스트 (첫 페이지는 빈 문자열)
        prev_page_text = ""
        
        for page_num in range(detail_start, detail_end + 1):
            if page_num not in ocr_data_map:
                logger.warning(f"페이지 {page_num} OCR 데이터 없음, 스킵")
                continue
            
            current_table_text = self._tables_to_text(ocr_data_map[page_num].get('tables', []))
            if not current_table_text:
                current_table_text = ocr_data_map[page_num].get('text', '')
            
            logger.info(f"페이지 {page_num} 처리 중...")
            
            # LLM으로 현재 페이지 분석
            page_indicators = await self._extract_page_details(
                indicator_list=indicator_names,
                prev_page=page_num - 1 if page_num > detail_start else 0,
                prev_page_text=prev_page_text,
                current_page=page_num,
                current_page_text=current_table_text
            )
            
            # 결과를 누적
            for pi in page_indicators:
                name = pi.get('지표명', '')
                if name in indicator_details:
                    for field in ['평가기준', '평가방법', '참고사항', '증빙자료']:
                        items = pi.get(field, [])
                        if items:
                            indicator_details[name][field].extend(items)
                else:
                    # 새로운 지표명이면 추가 (유사 매칭 시도)
                    matched = False
                    for known_name in indicator_details.keys():
                        if name in known_name or known_name in name:
                            for field in ['평가기준', '평가방법', '참고사항', '증빙자료']:
                                items = pi.get(field, [])
                                if items:
                                    indicator_details[known_name][field].extend(items)
                            matched = True
                            break
                    if not matched:
                        logger.warning(f"알 수 없는 지표명: {name}")
            
            # 현재 페이지를 다음 반복의 이전 페이지로
            prev_page_text = current_table_text
        
        # 중간 결과 저장
        if save_intermediate and output_dir:
            with open(os.path.join(output_dir, "indicator_details_raw.json"), 'w', encoding='utf-8') as f:
                json.dump(indicator_details, f, ensure_ascii=False, indent=2)
        
        # ========== 3단계: StructuredIndicator 객체 생성 ==========
        logger.info("Step 3: StructuredIndicator 객체 생성")
        
        indicators = []
        for name, details in indicator_details.items():
            # 중복 제거
            details['평가기준'] = list(dict.fromkeys(details['평가기준']))
            details['평가방법'] = list(dict.fromkeys(details['평가방법']))
            details['참고사항'] = list(dict.fromkeys(details['참고사항']))
            details['증빙자료'] = list(dict.fromkeys(details['증빙자료']))
            
            indicator = StructuredIndicator.from_dict(
                data=details,
                source_document=source_document,
                page_range=f"{detail_start}-{detail_end}"
            )
            if indicator.지표명:
                indicators.append(indicator)
        
        logger.info(f"총 {len(indicators)}개 평가지표 생성 완료")
        
        # 중간 결과 저장
        if save_intermediate and output_dir:
            with open(os.path.join(output_dir, "indicators_final.json"), 'w', encoding='utf-8') as f:
                json.dump([i.to_dict() for i in indicators], f, ensure_ascii=False, indent=2)
        
        # ========== 4단계: PGVector에 저장 ==========
        logger.info("Step 4: PGVector 저장")
        if indicators:
            self.vector_store.add_indicators(indicators)
        
        return indicators
    
    async def _extract_indicator_index(
        self,
        ocr_text: str,
        start_page: int,
        end_page: int
    ) -> List[Dict]:
        """
        목록 페이지에서 평가지표 이름 목록 추출
        """
        model = ModelFactory.create_model_chain(
            provider="openai",
            model_name="gpt-4o",
            output_format="json",
            max_rps=2.0
        )
        parser = JsonOutputParser()
        
        messages = INCLUSIVE_INDEX_EXTRACTION_PROMPT.format_messages(
            ocr_text=ocr_text[:20000],
            start_page=start_page,
            end_page=end_page
        )
        
        response = await model.ainvoke(messages)
        
        try:
            result = parser.parse(response.content)
            return result.get("indicators", [])
        except Exception as e:
            logger.warning(f"목록 추출 JSON 파싱 실패: {e}")
            return []
    
    async def _extract_page_details(
        self,
        indicator_list: List[Dict],
        prev_page: int,
        prev_page_text: str,
        current_page: int,
        current_page_text: str
    ) -> List[Dict]:
        """
        현재 페이지에서 평가지표 상세 정보 추출 (이전 페이지 컨텍스트 포함)
        """
        model = ModelFactory.create_model_chain(
            provider="openai",
            model_name="gpt-4o",
            output_format="json",
            max_rps=2.0
        )
        parser = JsonOutputParser()
        
        # 지표 목록을 문자열로 변환
        indicator_list_str = "\n".join([
            f"- {item.get('번호', '')} {item.get('지표명', '')} ({item.get('배점', '')})"
            for item in indicator_list
        ])
        
        messages = INCLUSIVE_DETAIL_EXTRACTION_PROMPT.format_messages(
            indicator_list=indicator_list_str,
            prev_page=prev_page,
            prev_page_text=prev_page_text[:5000] if prev_page_text else "(없음)",
            current_page=current_page,
            current_page_text=current_page_text[:10000]
        )
        
        response = await model.ainvoke(messages)
        
        try:
            result = parser.parse(response.content)
            return result.get("page_indicators", [])
        except Exception as e:
            logger.warning(f"페이지 {current_page} 상세 추출 실패: {e}")
            return []
    
    def search_unique_indicators(
        self,
        query: str,
        k: int = 10,
        max_candidates: int = 100
    ) -> List[Dict]:
        """
        중복 지표명을 제외하고 서로 다른 지표 검색
        """
        return self.vector_store.search_unique_indicators(
            query=query,
            k=k,
            max_candidates=max_candidates
        )
    
    def search_by_field(
        self,
        query: str,
        field_type: str,
        k: int = 10
    ) -> List[Dict]:
        """
        특정 필드 유형에서만 검색
        
        Args:
            query: 검색 쿼리
            field_type: "평가기준" | "평가방법" | "참고사항" | "증빙자료"
            k: 반환할 결과 수
        """
        return self.vector_store.search_by_field_type(
            query=query,
            field_type=field_type,
            k=k
        )


# ============================================================================
# 편의 함수
# ============================================================================

def create_pipeline(
    db_host: str = "localhost",
    db_port: int = 5432,
    db_name: str = "b2g_data",
    db_user: str = "postgres",
    db_password: str = "",
    collection_name: str = "b2g_projects"
) -> B2GDataPipeline:
    """
    파이프라인 인스턴스 생성 헬퍼 함수
    
    Args:
        db_host: PostgreSQL 호스트
        db_port: PostgreSQL 포트
        db_name: 데이터베이스 이름
        db_user: 사용자명
        db_password: 비밀번호
        collection_name: 벡터 컬렉션 이름
        
    Returns:
        B2GDataPipeline 인스턴스
    """
    connection_string = (
        f"postgresql://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}"
    )
    
    return B2GDataPipeline(
        connection_string=connection_string,
        collection_name=collection_name
    )


def create_vector_store(
    db_host: str = "localhost",
    db_port: int = 5432,
    db_name: str = "b2g_data",
    db_user: str = "postgres",
    db_password: str = "",
    collection_name: str = "b2g_projects"
) -> B2GVectorStore:
    """
    벡터 저장소만 생성하는 헬퍼 함수 (검색 전용)
    
    Args:
        db_host: PostgreSQL 호스트
        db_port: PostgreSQL 포트
        db_name: 데이터베이스 이름
        db_user: 사용자명
        db_password: 비밀번호
        collection_name: 벡터 컬렉션 이름
        
    Returns:
        B2GVectorStore 인스턴스
    """
    connection_string = (
        f"postgresql://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}"
    )
    
    return B2GVectorStore(
        connection_string=connection_string,
        collection_name=collection_name
    )


def create_inclusive_growth_pipeline(
    db_host: str = "localhost",
    db_port: int = 5432,
    db_name: str = "b2g_data",
    db_user: str = "postgres",
    db_password: str = "",
    collection_name: str = "inclusive_growth_indicators"
) -> InclusiveGrowthPipeline:
    """
    동반성장 평가지표 파이프라인 생성 헬퍼 함수
    
    Args:
        db_host: PostgreSQL 호스트
        db_port: PostgreSQL 포트
        db_name: 데이터베이스 이름
        db_user: 사용자명
        db_password: 비밀번호
        collection_name: 벡터 컬렉션 이름
        
    Returns:
        InclusiveGrowthPipeline 인스턴스
    """
    connection_string = (
        f"postgresql://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}"
    )
    
    return InclusiveGrowthPipeline(
        connection_string=connection_string,
        collection_name=collection_name
    )


def create_inclusive_growth_vector_store(
    db_host: str = "localhost",
    db_port: int = 5432,
    db_name: str = "b2g_data",
    db_user: str = "postgres",
    db_password: str = "",
    collection_name: str = "inclusive_growth_indicators"
) -> InclusiveGrowthVectorStore:
    """
    동반성장 평가지표 벡터 저장소 생성 헬퍼 함수 (검색 전용)
    
    Args:
        db_host: PostgreSQL 호스트
        db_port: PostgreSQL 포트
        db_name: 데이터베이스 이름
        db_user: 사용자명
        db_password: 비밀번호
        collection_name: 벡터 컬렉션 이름
        
    Returns:
        InclusiveGrowthVectorStore 인스턴스
    """
    connection_string = (
        f"postgresql://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}"
    )
    
    return InclusiveGrowthVectorStore(
        connection_string=connection_string,
        collection_name=collection_name
    )


async def process_and_search_example():
    """
    사용 예시
    """
    # 1. 파이프라인 생성
    pipeline = create_pipeline(
        db_host="localhost",
        db_port=5432,
        db_name="b2g_data",
        db_user="postgres",
        db_password="your_password"
    )
    
    # 2. PDF 처리 (새 스키마로 저장)
    # projects = await pipeline.process_pdf(
    #     pdf_path="/path/to/government_projects.pdf",
    #     save_intermediate=True,
    #     output_dir="./output"
    # )
    # 
    # print(f"처리된 과제 수: {len(projects)}")
    # for p in projects:
    #     print(f"  - {p.과제번호}: {p.과제명}")
    
    # 3. 검색 예시 - 서로 다른 과제 10개 검색
    # results = pipeline.search_unique_projects(
    #     query="인공지능 기술 개발",
    #     k=10
    # )
    # 
    # print("\n=== 검색 결과 (서로 다른 과제 10개) ===")
    # for i, r in enumerate(results, 1):
    #     print(f"\n{i}. [{r['score']:.3f}] {r['과제명']}")
    #     print(f"   과제번호: {r['과제번호']}")
    #     print(f"   매칭 필드: {r['matched_field']}")
    #     print(f"   매칭 텍스트: {r['matched_text'][:100]}...")
    
    # 4. 특정 필드에서만 검색
    # results = pipeline.search_by_field(
    #     query="탄소중립",
    #     field_type="기대효과",
    #     k=5
    # )
    
    pass


if __name__ == "__main__":
    asyncio.run(process_and_search_example())
