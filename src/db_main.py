"""
B2G 기준데이터 DB 모듈 - MySQL/MariaDB 기반

이 모듈은 정부 과제 PDF를 분석하여 MySQL에 저장하고 검색 기능을 제공합니다.

주요 기능:
1. PDF를 페이지별 이미지로 변환 후 CLOVA OCR로 텍스트/표 추출
2. LLM을 사용하여 지표 목록 추출 및 세부내용 채우기
3. 구조화된 데이터를 MySQL에 저장
4. 키워드 기반 검색 지원 (20개 후보 중 LLM이 10개 선별)

지원 스키마:
- 국정과제 (StructuredProject)
- 경영평가 (StructuredManagementEval)  
- 동반성장 (StructuredInclusiveGrowth)
"""

import os
import io
import json
import logging
import asyncio
import base64
from typing import Optional, List, Dict, Any
from dataclasses import dataclass, field
from datetime import datetime

import dotenv
import fitz  # PyMuPDF
import PIL.Image

from langchain_core.output_parsers import JsonOutputParser
from langchain_core.messages import HumanMessage, SystemMessage

from .utils import CLOVA_ocr_with_table
from .api import ModelFactory
from .prompts import (
    INCLUSIVE_INDEX_EXTRACTION_PROMPT,
    INCLUSIVE_DETAIL_EXTRACTION_PROMPT
)

# MySQL 연결
import pymysql
from pymysql.cursors import DictCursor

# 임베딩 관련
import struct
import numpy as np
from openai import OpenAI

# .env에서 API 키 로드
env_path = os.path.join(os.path.dirname(__file__), ".env")
OPENAI_KEY = dotenv.get_key(env_path, "OPENAI_KEY")

# 임베딩 설정
EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_DIM = 1536

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
class StructuredProject:
    """
    국정과제 데이터 스키마
    
    JSON 스키마:
    {
        "과제명": "세계에서 AI를 가장 잘 쓰는 나라 구현",
        "과제번호": "21",
        "과제 목표": ["AI 기술 선도국 도약", "AI 산업 생태계 육성"],
        "주요내용": ["AI 인재 양성", "AI 규제 혁신"],
        "기대효과": ["글로벌 AI 경쟁력 확보"],
        "source_document": "presidential_agenda.pdf",
        "page_range": "15-18"
    }
    """
    과제명: str
    과제번호: str
    과제_목표: List[str] = field(default_factory=list)
    주요내용: List[str] = field(default_factory=list)
    기대효과: List[str] = field(default_factory=list)
    source_document: Optional[str] = None
    page_range: Optional[str] = None
    extraction_date: Optional[str] = None
    
    @classmethod
    def from_dict(cls, data: Dict, source_document: str = "", page_range: str = "") -> "StructuredProject":
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
    
    def get_searchable_text(self) -> str:
        """검색용 통합 텍스트 생성"""
        parts = [self.과제명]
        parts.extend(self.과제_목표)
        parts.extend(self.주요내용)
        parts.extend(self.기대효과)
        return " ".join(filter(None, parts))


@dataclass
class StructuredManagementEval:
    """
    경영평가 지표 데이터 스키마
    
    JSON 스키마:
    {
        "지표명": "리더십 및 전략기획",
        "평가기준": ["경영진의 전략적 비전 제시", "조직 목표 달성 노력"],
        "평가방법": ["정성평가", "계량평가 병행"],
        "참고사항": ["전년도 대비 개선 여부 확인"],
        "증빙자료": ["이사회 회의록", "전략 수립 보고서"],
        "source_document": "management_eval.pdf",
        "page_range": "26-28"
    }
    """
    지표명: str
    평가기준: List[str] = field(default_factory=list)
    평가방법: List[str] = field(default_factory=list)
    참고사항: List[str] = field(default_factory=list)
    증빙자료: List[str] = field(default_factory=list)
    source_document: Optional[str] = None
    page_range: Optional[str] = None
    extraction_date: Optional[str] = None
    
    @classmethod
    def from_dict(cls, data: Dict, source_document: str = "", page_range: str = "") -> "StructuredManagementEval":
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
    
    def get_searchable_text(self) -> str:
        """검색용 통합 텍스트 생성"""
        parts = [self.지표명]
        parts.extend(self.평가기준)
        parts.extend(self.평가방법)
        parts.extend(self.참고사항)
        return " ".join(filter(None, parts))


@dataclass
class StructuredInclusiveGrowth:
    """
    동반성장 세부추진과제 데이터 스키마
    
    JSON 스키마:
    {
        "세부추진과제명": "상생금융 프로그램 확대",
        "세부내용": ["기업･금융권이 출연하고...", "대기업 등이 상생협력을 위하여..."],
        "source_document": "inclusive_growth.pdf",
        "page_range": "26-28"
    }
    """
    세부추진과제명: str
    세부내용: List[str] = field(default_factory=list)
    source_document: Optional[str] = None
    page_range: Optional[str] = None
    extraction_date: Optional[str] = None
    
    @classmethod
    def from_dict(cls, data: Dict, source_document: str = "", page_range: str = "") -> "StructuredInclusiveGrowth":
        return cls(
            세부추진과제명=data.get("세부추진과제명", ""),
            세부내용=data.get("세부내용", []) if isinstance(data.get("세부내용"), list) else [],
            source_document=source_document,
            page_range=page_range,
            extraction_date=datetime.now().isoformat()
        )
    
    def to_dict(self) -> Dict:
        return {
            "세부추진과제명": self.세부추진과제명,
            "세부내용": self.세부내용,
            "source_document": self.source_document,
            "page_range": self.page_range,
            "extraction_date": self.extraction_date
        }
    
    def get_searchable_text(self) -> str:
        """검색용 통합 텍스트 생성"""
        parts = [self.세부추진과제명]
        parts.extend(self.세부내용)
        return " ".join(filter(None, parts))


# ============================================================================
# PDF 처리 클래스
# ============================================================================

class PDFProcessor:
    """PDF를 페이지별 이미지로 변환하고 OCR 수행"""
    
    def __init__(self, dpi: int = 200):
        self.dpi = dpi
        self.zoom = dpi / 72
    
    def pdf_to_images(self, pdf_path: str) -> List[PIL.Image.Image]:
        """PDF를 페이지별 이미지로 변환"""
        images = []
        doc = fitz.open(pdf_path)
        mat = fitz.Matrix(self.zoom, self.zoom)
        
        for page_num in range(len(doc)):
            page = doc[page_num]
            pix = page.get_pixmap(matrix=mat)
            img_data = pix.tobytes("png")
            img = PIL.Image.open(io.BytesIO(img_data))
            images.append(img)
            logger.info(f"페이지 {page_num + 1}/{len(doc)} 이미지 변환 완료")
        
        doc.close()
        return images
    
    def process_page(self, image: PIL.Image.Image, page_num: int) -> PageOCRResult:
        """단일 페이지 OCR 처리"""
        img_byte_arr = io.BytesIO()
        image.save(img_byte_arr, format='PNG')
        img_bytes = img_byte_arr.getvalue()
        
        ocr_result = CLOVA_ocr_with_table(img_bytes)
        
        return PageOCRResult(
            page_num=page_num,
            text=ocr_result['text'],
            fields=ocr_result['fields'],
            tables=ocr_result['tables'],
            raw_response=ocr_result['raw_response']
        )


# ============================================================================
# MySQL 저장소 클래스
# ============================================================================

class MySQLStore:
    """
    MySQL을 사용한 B2G 데이터 저장 및 검색
    
    테이블 구조:
    - national_projects: 국정과제
    - management_evals: 경영평가 지표
    - inclusive_growth: 동반성장 세부추진과제
    """
    
    def __init__(
        self,
        host: str = "localhost",
        port: int = 3306,
        database: str = "b2g_data",
        user: str = "root",
        password: str = ""
    ):
        self.connection_params = {
            "host": host,
            "port": port,
            "database": database,
            "user": user,
            "password": password,
            "charset": "utf8mb4",
            "cursorclass": DictCursor
        }
        self._init_tables()
        logger.info(f"MySQLStore 초기화 완료: {database}")
    
    def _get_connection(self):
        """MySQL 연결 반환"""
        return pymysql.connect(**self.connection_params)
    
    def _init_tables(self):
        """테이블 생성"""
        conn = self._get_connection()
        try:
            with conn.cursor() as cursor:
                # 국정과제 테이블
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS national_projects (
                        id INT AUTO_INCREMENT PRIMARY KEY,
                        과제명 VARCHAR(500) NOT NULL,
                        과제번호 VARCHAR(50),
                        과제_목표 JSON,
                        주요내용 JSON,
                        기대효과 JSON,
                        source_document VARCHAR(255),
                        page_range VARCHAR(50),
                        extraction_date DATETIME,
                        searchable_text TEXT,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        FULLTEXT INDEX ft_searchable (searchable_text) WITH PARSER ngram
                    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
                """)
                
                # 경영평가 테이블
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS management_evals (
                        id INT AUTO_INCREMENT PRIMARY KEY,
                        지표명 VARCHAR(500) NOT NULL,
                        평가기준 JSON,
                        평가방법 JSON,
                        참고사항 JSON,
                        증빙자료 JSON,
                        source_document VARCHAR(255),
                        page_range VARCHAR(50),
                        extraction_date DATETIME,
                        searchable_text TEXT,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        FULLTEXT INDEX ft_searchable (searchable_text) WITH PARSER ngram
                    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
                """)
                
                # 동반성장 테이블
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS inclusive_growth (
                        id INT AUTO_INCREMENT PRIMARY KEY,
                        세부추진과제명 VARCHAR(500) NOT NULL,
                        세부내용 JSON,
                        source_document VARCHAR(255),
                        page_range VARCHAR(50),
                        extraction_date DATETIME,
                        searchable_text TEXT,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        FULLTEXT INDEX ft_searchable (searchable_text) WITH PARSER ngram
                    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
                """)
                
            conn.commit()
        finally:
            conn.close()
    
    # ========================================================================
    # 국정과제 CRUD
    # ========================================================================
    
    def add_project(self, project: StructuredProject) -> int:
        """국정과제 추가"""
        conn = self._get_connection()
        try:
            with conn.cursor() as cursor:
                cursor.execute("""
                    INSERT INTO national_projects 
                    (과제명, 과제번호, 과제_목표, 주요내용, 기대효과, 
                     source_document, page_range, extraction_date, searchable_text)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                """, (
                    project.과제명,
                    project.과제번호,
                    json.dumps(project.과제_목표, ensure_ascii=False),
                    json.dumps(project.주요내용, ensure_ascii=False),
                    json.dumps(project.기대효과, ensure_ascii=False),
                    project.source_document,
                    project.page_range,
                    project.extraction_date,
                    project.get_searchable_text()
                ))
                conn.commit()
                return cursor.lastrowid
        finally:
            conn.close()
    
    def add_projects(self, projects: List[StructuredProject]) -> List[int]:
        """여러 국정과제 배치 추가"""
        ids = []
        for project in projects:
            ids.append(self.add_project(project))
        logger.info(f"{len(ids)}개 국정과제 저장 완료")
        return ids
    
    def search_projects(self, query: str, k: int = 20) -> List[Dict]:
        """국정과제 검색 (FULLTEXT + LIKE)"""
        conn = self._get_connection()
        try:
            with conn.cursor() as cursor:
                # FULLTEXT 검색 시도
                cursor.execute("""
                    SELECT *, MATCH(searchable_text) AGAINST(%s IN NATURAL LANGUAGE MODE) as score
                    FROM national_projects
                    WHERE MATCH(searchable_text) AGAINST(%s IN NATURAL LANGUAGE MODE)
                    ORDER BY score DESC
                    LIMIT %s
                """, (query, query, k))
                results = cursor.fetchall()
                
                # FULLTEXT 결과가 없으면 LIKE 검색
                if not results:
                    keywords = query.split()
                    like_conditions = " OR ".join(["searchable_text LIKE %s" for _ in keywords])
                    like_params = [f"%{kw}%" for kw in keywords]
                    
                    cursor.execute(f"""
                        SELECT *, 1.0 as score
                        FROM national_projects
                        WHERE {like_conditions}
                        LIMIT %s
                    """, (*like_params, k))
                    results = cursor.fetchall()
                
                return self._format_project_results(results)
        finally:
            conn.close()
    
    def _format_project_results(self, results: List[Dict]) -> List[Dict]:
        """국정과제 결과 포맷팅"""
        formatted = []
        for row in results:
            formatted.append({
                "id": row["id"],
                "score": float(row.get("score", 0)),
                "과제명": row["과제명"],
                "과제번호": row["과제번호"],
                "과제 목표": json.loads(row["과제_목표"]) if row["과제_목표"] else [],
                "주요내용": json.loads(row["주요내용"]) if row["주요내용"] else [],
                "기대효과": json.loads(row["기대효과"]) if row["기대효과"] else [],
                "source_document": row["source_document"],
                "page_range": row["page_range"]
            })
        return formatted
    
    # ========================================================================
    # 경영평가 CRUD
    # ========================================================================
    
    def add_management_eval(self, eval_item: StructuredManagementEval) -> int:
        """경영평가 지표 추가"""
        conn = self._get_connection()
        try:
            with conn.cursor() as cursor:
                cursor.execute("""
                    INSERT INTO management_evals 
                    (지표명, 평가기준, 평가방법, 참고사항, 증빙자료,
                     source_document, page_range, extraction_date, searchable_text)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                """, (
                    eval_item.지표명,
                    json.dumps(eval_item.평가기준, ensure_ascii=False),
                    json.dumps(eval_item.평가방법, ensure_ascii=False),
                    json.dumps(eval_item.참고사항, ensure_ascii=False),
                    json.dumps(eval_item.증빙자료, ensure_ascii=False),
                    eval_item.source_document,
                    eval_item.page_range,
                    eval_item.extraction_date,
                    eval_item.get_searchable_text()
                ))
                conn.commit()
                return cursor.lastrowid
        finally:
            conn.close()
    
    def add_management_evals(self, evals: List[StructuredManagementEval]) -> List[int]:
        """여러 경영평가 지표 배치 추가"""
        ids = []
        for eval_item in evals:
            ids.append(self.add_management_eval(eval_item))
        logger.info(f"{len(ids)}개 경영평가 지표 저장 완료")
        return ids
    
    def search_management_evals(self, query: str, k: int = 20) -> List[Dict]:
        """경영평가 지표 검색"""
        conn = self._get_connection()
        try:
            with conn.cursor() as cursor:
                cursor.execute("""
                    SELECT *, MATCH(searchable_text) AGAINST(%s IN NATURAL LANGUAGE MODE) as score
                    FROM management_evals
                    WHERE MATCH(searchable_text) AGAINST(%s IN NATURAL LANGUAGE MODE)
                    ORDER BY score DESC
                    LIMIT %s
                """, (query, query, k))
                results = cursor.fetchall()
                
                if not results:
                    keywords = query.split()
                    like_conditions = " OR ".join(["searchable_text LIKE %s" for _ in keywords])
                    like_params = [f"%{kw}%" for kw in keywords]
                    
                    cursor.execute(f"""
                        SELECT *, 1.0 as score
                        FROM management_evals
                        WHERE {like_conditions}
                        LIMIT %s
                    """, (*like_params, k))
                    results = cursor.fetchall()
                
                return self._format_management_eval_results(results)
        finally:
            conn.close()
    
    def _format_management_eval_results(self, results: List[Dict]) -> List[Dict]:
        """경영평가 결과 포맷팅"""
        formatted = []
        for row in results:
            formatted.append({
                "id": row["id"],
                "score": float(row.get("score", 0)),
                "지표명": row["지표명"],
                "평가기준": json.loads(row["평가기준"]) if row["평가기준"] else [],
                "평가방법": json.loads(row["평가방법"]) if row["평가방법"] else [],
                "참고사항": json.loads(row["참고사항"]) if row["참고사항"] else [],
                "증빙자료": json.loads(row["증빙자료"]) if row["증빙자료"] else [],
                "source_document": row["source_document"],
                "page_range": row["page_range"]
            })
        return formatted
    
    # ========================================================================
    # 동반성장 CRUD
    # ========================================================================
    
    def add_inclusive_growth(self, item: StructuredInclusiveGrowth) -> int:
        """동반성장 세부추진과제 추가"""
        conn = self._get_connection()
        try:
            with conn.cursor() as cursor:
                cursor.execute("""
                    INSERT INTO inclusive_growth 
                    (세부추진과제명, 세부내용, source_document, page_range, 
                     extraction_date, searchable_text)
                    VALUES (%s, %s, %s, %s, %s, %s)
                """, (
                    item.세부추진과제명,
                    json.dumps(item.세부내용, ensure_ascii=False),
                    item.source_document,
                    item.page_range,
                    item.extraction_date,
                    item.get_searchable_text()
                ))
                conn.commit()
                return cursor.lastrowid
        finally:
            conn.close()
    
    def add_inclusive_growths(self, items: List[StructuredInclusiveGrowth]) -> List[int]:
        """여러 동반성장 세부추진과제 배치 추가"""
        ids = []
        for item in items:
            ids.append(self.add_inclusive_growth(item))
        logger.info(f"{len(ids)}개 동반성장 세부추진과제 저장 완료")
        return ids
    
    def search_inclusive_growth(self, query: str, k: int = 20) -> List[Dict]:
        """동반성장 세부추진과제 검색"""
        conn = self._get_connection()
        try:
            with conn.cursor() as cursor:
                cursor.execute("""
                    SELECT *, MATCH(searchable_text) AGAINST(%s IN NATURAL LANGUAGE MODE) as score
                    FROM inclusive_growth
                    WHERE MATCH(searchable_text) AGAINST(%s IN NATURAL LANGUAGE MODE)
                    ORDER BY score DESC
                    LIMIT %s
                """, (query, query, k))
                results = cursor.fetchall()
                
                if not results:
                    keywords = query.split()
                    like_conditions = " OR ".join(["searchable_text LIKE %s" for _ in keywords])
                    like_params = [f"%{kw}%" for kw in keywords]
                    
                    cursor.execute(f"""
                        SELECT *, 1.0 as score
                        FROM inclusive_growth
                        WHERE {like_conditions}
                        LIMIT %s
                    """, (*like_params, k))
                    results = cursor.fetchall()
                
                return self._format_inclusive_growth_results(results)
        finally:
            conn.close()
    
    def _format_inclusive_growth_results(self, results: List[Dict]) -> List[Dict]:
        """동반성장 결과 포맷팅"""
        formatted = []
        for row in results:
            formatted.append({
                "id": row["id"],
                "score": float(row.get("score", 0)),
                "세부추진과제명": row["세부추진과제명"],
                "세부내용": json.loads(row["세부내용"]) if row["세부내용"] else [],
                "source_document": row["source_document"],
                "page_range": row["page_range"]
            })
        return formatted
    
    # ========================================================================
    # 유틸리티
    # ========================================================================
    
    def delete_all_data(self, table: str = None):
        """데이터 삭제 (테이블 지정 또는 전체)"""
        conn = self._get_connection()
        try:
            with conn.cursor() as cursor:
                if table:
                    cursor.execute(f"TRUNCATE TABLE {table}")
                else:
                    cursor.execute("TRUNCATE TABLE national_projects")
                    cursor.execute("TRUNCATE TABLE management_evals")
                    cursor.execute("TRUNCATE TABLE inclusive_growth")
                conn.commit()
                logger.info(f"데이터 삭제 완료: {table or '전체'}")
        finally:
            conn.close()
    
    def get_stats(self) -> Dict[str, int]:
        """각 테이블의 레코드 수 반환"""
        conn = self._get_connection()
        try:
            with conn.cursor() as cursor:
                cursor.execute("SELECT COUNT(*) as cnt FROM national_projects")
                projects = cursor.fetchone()["cnt"]
                cursor.execute("SELECT COUNT(*) as cnt FROM management_evals")
                evals = cursor.fetchone()["cnt"]
                cursor.execute("SELECT COUNT(*) as cnt FROM inclusive_growth")
                growth = cursor.fetchone()["cnt"]
                return {
                    "national_projects": projects,
                    "management_evals": evals,
                    "inclusive_growth": growth
                }
        finally:
            conn.close()
    
    # ========================================================================
    # 임베딩 기반 벡터 검색
    # ========================================================================
    
    def _init_openai_client(self):
        """OpenAI 클라이언트 초기화 (lazy loading)"""
        if not hasattr(self, '_openai_client') or self._openai_client is None:
            if not OPENAI_KEY:
                raise ValueError("OPENAI_KEY not found in .env")
            self._openai_client = OpenAI(api_key=OPENAI_KEY)
        return self._openai_client
    
    def _get_query_embedding(self, text: str) -> List[float]:
        """쿼리 텍스트의 임베딩 벡터 생성"""
        client = self._init_openai_client()
        response = client.embeddings.create(
            model=EMBEDDING_MODEL,
            input=text
        )
        return response.data[0].embedding
    
    def _blob_to_vector(self, blob: bytes) -> np.ndarray:
        """BLOB 데이터를 numpy 벡터로 변환"""
        if blob is None:
            return None
        return np.array(struct.unpack(f'{EMBEDDING_DIM}f', blob))
    
    def _cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """코사인 유사도 계산"""
        if vec1 is None or vec2 is None:
            return 0.0
        dot = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        if norm1 == 0 or norm2 == 0:
            return 0.0
        return dot / (norm1 * norm2)
    
    def _safe_json_parse(self, data):
        """JSON 파싱 (문자열 또는 이미 파싱된 데이터 처리)"""
        if data is None:
            return []
        if isinstance(data, (list, dict)):
            return data
        if isinstance(data, str):
            data = data.strip()
            if not data:
                return []
            try:
                return json.loads(data)
            except json.JSONDecodeError:
                # JSON이 아닌 경우 문자열 그대로 반환
                return data
        return []
    
    def search_projects_by_embedding(self, query: str, k: int = 10) -> List[Dict]:
        """국정과제 임베딩 기반 검색"""
        query_vec = np.array(self._get_query_embedding(query))
        
        conn = self._get_connection()
        try:
            with conn.cursor() as cursor:
                cursor.execute("""
                    SELECT id, 과제명, 과제번호, 과제_목표, 주요내용, 기대효과,
                           source_document, page_range, embedding
                    FROM national_projects
                    WHERE embedding IS NOT NULL
                """)
                rows = cursor.fetchall()
            
            # 코사인 유사도 계산
            results = []
            for row in rows:
                db_vec = self._blob_to_vector(row['embedding'])
                if db_vec is not None:
                    score = self._cosine_similarity(query_vec, db_vec)
                    results.append({
                        'id': row['id'],
                        'score': float(score),
                        '과제명': row['과제명'],
                        '과제번호': row['과제번호'],
                        '과제 목표': self._safe_json_parse(row['과제_목표']),
                        '주요내용': self._safe_json_parse(row['주요내용']),
                        '기대효과': self._safe_json_parse(row['기대효과']),
                        'source_document': row['source_document'],
                        'page_range': row['page_range']
                    })
            
            # 유사도 기준 정렬 후 상위 k개 반환
            results.sort(key=lambda x: x['score'], reverse=True)
            return results[:k]
        finally:
            conn.close()
    
    def search_management_evals_by_embedding(self, query: str, k: int = 10) -> List[Dict]:
        """경영평가 임베딩 기반 검색"""
        query_vec = np.array(self._get_query_embedding(query))
        
        conn = self._get_connection()
        try:
            with conn.cursor() as cursor:
                cursor.execute("""
                    SELECT id, 지표명, 평가기준, 평가방법, 참고사항, 증빙자료,
                           source_document, page_range, embedding
                    FROM management_evals
                    WHERE embedding IS NOT NULL
                """)
                rows = cursor.fetchall()
            
            results = []
            for row in rows:
                db_vec = self._blob_to_vector(row['embedding'])
                if db_vec is not None:
                    score = self._cosine_similarity(query_vec, db_vec)
                    results.append({
                        'id': row['id'],
                        'score': float(score),
                        '지표명': row['지표명'],
                        '평가기준': self._safe_json_parse(row['평가기준']),
                        '평가방법': self._safe_json_parse(row['평가방법']),
                        '참고사항': self._safe_json_parse(row['참고사항']),
                        '증빙자료': self._safe_json_parse(row['증빙자료']),
                        'source_document': row['source_document'],
                        'page_range': row['page_range']
                    })
            
            results.sort(key=lambda x: x['score'], reverse=True)
            return results[:k]
        finally:
            conn.close()
    
    def search_inclusive_growth_by_embedding(self, query: str, k: int = 10) -> List[Dict]:
        """동반성장 임베딩 기반 검색"""
        query_vec = np.array(self._get_query_embedding(query))
        
        conn = self._get_connection()
        try:
            with conn.cursor() as cursor:
                # 동반성장 테이블은 load_json_to_db.py 스키마와 일치 (지표명, 평가기준 등)
                cursor.execute("""
                    SELECT id, 지표명, 평가기준, 평가방법, 참고사항, 증빙자료,
                           source_document, page_range, embedding
                    FROM inclusive_growth
                    WHERE embedding IS NOT NULL
                """)
                rows = cursor.fetchall()
            
            results = []
            for row in rows:
                db_vec = self._blob_to_vector(row['embedding'])
                if db_vec is not None:
                    score = self._cosine_similarity(query_vec, db_vec)
                    results.append({
                        'id': row['id'],
                        'score': float(score),
                        '지표명': row['지표명'],
                        '평가기준': self._safe_json_parse(row['평가기준']),
                        '평가방법': self._safe_json_parse(row['평가방법']),
                        '참고사항': self._safe_json_parse(row['참고사항']),
                        '증빙자료': self._safe_json_parse(row['증빙자료']),
                        'source_document': row['source_document'],
                        'page_range': row['page_range']
                    })
            
            results.sort(key=lambda x: x['score'], reverse=True)
            return results[:k]
        finally:
            conn.close()
    
    def search_all_by_embedding(self, query: str, k: int = 10) -> Dict[str, List[Dict]]:
        """전체 테이블 임베딩 기반 검색"""
        return {
            'projects': self.search_projects_by_embedding(query, k),
            'management_evals': self.search_management_evals_by_embedding(query, k),
            'inclusive_growth': self.search_inclusive_growth_by_embedding(query, k)
        }


# ============================================================================
# 파이프라인 클래스
# ============================================================================

class B2GPipeline:
    """
    PDF에서 MySQL까지의 전체 파이프라인 관리
    
    처리 흐름:
    1. PDF → 이미지 변환
    2. 이미지 → OCR (표 포함)
    3. 목록 페이지에서 리스트 추출
    4. 세부 페이지에서 각 항목의 상세정보 채우기
    5. 구조화된 JSON을 MySQL에 저장
    6. 키워드 검색 (20개 후보 → LLM 선별)
    """
    
    def __init__(
        self,
        db_host: str = "localhost",
        db_port: int = 3306,
        db_name: str = "b2g_data",
        db_user: str = "root",
        db_password: str = "",
        model_provider: str = "openai",
        extraction_model: str = "gpt-4o-mini",
        structuring_model: str = "gpt-4o",
        max_rps: float = 2.0
    ):
        self.pdf_processor = PDFProcessor()
        self.store = MySQLStore(
            host=db_host,
            port=db_port,
            database=db_name,
            user=db_user,
            password=db_password
        )
        self.model_provider = model_provider
        self.extraction_model = extraction_model
        self.structuring_model = structuring_model
        self.max_rps = max_rps
        self._pdf_images = None
        
        logger.info("B2GPipeline 초기화 완료")
    
    def _tables_to_text(self, tables: List[Dict]) -> str:
        """OCR 결과의 tables를 텍스트로 변환"""
        if not tables:
            return ""
        
        table_texts = []
        for i, table in enumerate(tables):
            if "markdown" in table and table["markdown"]:
                table_texts.append(f"[표 {i+1}]\n{table['markdown']}")
            elif "rows" in table and table["rows"]:
                rows = table["rows"]
                lines = []
                for row_idx in sorted(rows.keys(), key=int):
                    cols = rows[row_idx]
                    line = " | ".join([cols.get(str(c), "") for c in sorted(cols.keys(), key=int)])
                    lines.append(line)
                table_texts.append(f"[표 {i+1}]\n" + "\n".join(lines))
        
        return "\n\n".join(table_texts)
    
    def _load_ocr_from_json(self, ocr_dir: str, page_num: int) -> Optional[Dict]:
        """저장된 OCR JSON 파일 로드"""
        ocr_file = os.path.join(ocr_dir, f"ocr_page_{page_num}.json")
        if os.path.exists(ocr_file):
            with open(ocr_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        return None
    
    async def process_pdf(
        self,
        pdf_path: str,
        data_type: str,  # "project" | "management_eval" | "inclusive_growth"
        index_pages: tuple,  # (start, end) - 목록 페이지 범위
        detail_pages: tuple,  # (start, end) - 세부내용 페이지 범위
        save_intermediate: bool = False,
        output_dir: Optional[str] = None,
        reuse_ocr_dir: Optional[str] = None
    ) -> List[Any]:
        """
        PDF 파일을 처리하여 구조화된 데이터를 DB에 저장
        
        Args:
            pdf_path: PDF 파일 경로
            data_type: 데이터 유형 ("project", "management_eval", "inclusive_growth")
            index_pages: 목록 페이지 범위 (start, end)
            detail_pages: 세부내용 페이지 범위 (start, end)
            save_intermediate: 중간 결과 저장 여부
            output_dir: 중간 결과 저장 디렉토리
            reuse_ocr_dir: 기존 OCR 결과 재사용 디렉토리
            
        Returns:
            처리된 구조화 데이터 리스트
        """
        source_document = os.path.basename(pdf_path)
        logger.info(f"PDF 처리 시작: {source_document} (타입: {data_type})")
        
        index_start, index_end = index_pages
        detail_start, detail_end = detail_pages
        
        # 필요한 모든 페이지
        all_pages = sorted(set(
            list(range(index_start, index_end + 1)) + 
            list(range(detail_start, detail_end + 1))
        ))
        
        # OCR 데이터 로드/수행
        ocr_data_map = {}
        missing_pages = []
        
        # OCR 재사용 디렉토리 결정 (명시적 지정 > output_dir)
        effective_ocr_dir = reuse_ocr_dir or output_dir
        
        if effective_ocr_dir and os.path.exists(effective_ocr_dir):
            logger.info(f"기존 OCR 결과 확인 중: {effective_ocr_dir}")
            loaded_count = 0
            for page_num in all_pages:
                ocr_data = self._load_ocr_from_json(effective_ocr_dir, page_num)
                if ocr_data:
                    ocr_data_map[page_num] = ocr_data
                    loaded_count += 1
                else:
                    missing_pages.append(page_num)
            logger.info(f"기존 OCR 로드: {loaded_count}개, 누락: {len(missing_pages)}개")
        else:
            missing_pages = all_pages
            if effective_ocr_dir:
                logger.info(f"OCR 캐시 디렉토리 없음: {effective_ocr_dir}")
        
        # 누락된 페이지 OCR 수행
        if missing_pages:
            logger.info(f"OCR 수행 필요: {len(missing_pages)}개 페이지")
            if self._pdf_images is None:
                logger.info(f"PDF 로드: {pdf_path}")
                self._pdf_images = self.pdf_processor.pdf_to_images(pdf_path)
            
            for page_num in missing_pages:
                idx = page_num - 1
                if idx < 0 or idx >= len(self._pdf_images):
                    logger.warning(f"페이지 {page_num} 범위 초과")
                    continue
                
                logger.info(f"OCR 수행: 페이지 {page_num}")
                result = self.pdf_processor.process_page(self._pdf_images[idx], page_num)
                
                ocr_data = {
                    'page_num': result.page_num,
                    'text': result.text,
                    'fields': result.fields,
                    'tables': result.tables
                }
                ocr_data_map[page_num] = ocr_data
                
                if save_intermediate and output_dir:
                    os.makedirs(output_dir, exist_ok=True)
                    with open(os.path.join(output_dir, f"ocr_page_{page_num}.json"), 'w', encoding='utf-8') as f:
                        json.dump(ocr_data, f, ensure_ascii=False, indent=2)
        
        # 1단계: 목록 페이지에서 항목 리스트 추출
        logger.info("Step 1: 목록 페이지에서 항목 리스트 추출")
        index_text_parts = []
        for page_num in range(index_start, index_end + 1):
            if page_num in ocr_data_map:
                table_text = self._tables_to_text(ocr_data_map[page_num].get('tables', []))
                if table_text:
                    index_text_parts.append(f"=== 페이지 {page_num} ===\n{table_text}")
                elif ocr_data_map[page_num].get('text'):
                    index_text_parts.append(f"=== 페이지 {page_num} ===\n{ocr_data_map[page_num]['text']}")
        
        index_text = "\n\n".join(index_text_parts)
        item_list = await self._extract_item_list(index_text, index_start, index_end, data_type)
        logger.info(f"추출된 항목: {len(item_list)}개")
        
        if save_intermediate and output_dir:
            with open(os.path.join(output_dir, "item_list.json"), 'w', encoding='utf-8') as f:
                json.dump(item_list, f, ensure_ascii=False, indent=2)
        
        # 2단계: 세부 페이지에서 상세 정보 추출
        logger.info("Step 2: 세부 페이지에서 상세 정보 추출")
        item_details = {item.get(self._get_name_field(data_type), ''): item.copy() for item in item_list}
        
        # 멀티모달 처리를 위해 PDF 이미지 로드 (아직 로드 안 됐으면)
        if self._pdf_images is None:
            logger.info(f"PDF 이미지 로드 (멀티모달 처리용): {pdf_path}")
            self._pdf_images = self.pdf_processor.pdf_to_images(pdf_path)
        
        prev_page_text = ""
        for page_num in range(detail_start, detail_end + 1):
            if page_num not in ocr_data_map:
                continue
            
            # OCR text 필드 사용 (tables가 없는 경우 대비)
            current_page_text = ocr_data_map[page_num].get('text', '')
            
            # 현재 페이지의 원본 이미지 가져오기
            current_page_image = None
            if self._pdf_images and page_num - 1 < len(self._pdf_images):
                current_page_image = self._pdf_images[page_num - 1]
            
            logger.info(f"페이지 {page_num} 처리 중... (이미지: {'있음' if current_page_image else '없음'})")
            
            page_items = await self._extract_page_details(
                item_list=item_list,
                prev_page=page_num - 1 if page_num > detail_start else 0,
                prev_page_text=prev_page_text,
                current_page=page_num,
                current_page_text=current_page_text,
                data_type=data_type,
                current_page_image=current_page_image
            )
            
            # 결과 누적
            for pi in page_items:
                name = pi.get(self._get_name_field(data_type), '')
                if name in item_details:
                    self._merge_item_details(item_details[name], pi, data_type)
            
            prev_page_text = current_page_text
        
        if save_intermediate and output_dir:
            with open(os.path.join(output_dir, "item_details.json"), 'w', encoding='utf-8') as f:
                json.dump(item_details, f, ensure_ascii=False, indent=2)
        
        # 3단계: 구조화 데이터 생성 및 저장
        logger.info("Step 3: 구조화 데이터 생성 및 DB 저장")
        structured_items = self._create_structured_items(
            item_details, data_type, source_document, f"{detail_start}-{detail_end}"
        )
        
        if structured_items:
            self._save_to_db(structured_items, data_type)
        
        if save_intermediate and output_dir:
            with open(os.path.join(output_dir, "structured_items.json"), 'w', encoding='utf-8') as f:
                json.dump([item.to_dict() for item in structured_items], f, ensure_ascii=False, indent=2)
        
        logger.info(f"PDF 처리 완료: {len(structured_items)}개 항목")
        return structured_items
    
    def _get_name_field(self, data_type: str) -> str:
        """데이터 타입별 이름 필드 반환"""
        if data_type == "project":
            return "과제명"
        elif data_type == "management_eval":
            return "지표명"
        else:
            return "세부추진과제명"
    
    def _merge_item_details(self, target: Dict, source: Dict, data_type: str):
        """항목 상세정보 병합"""
        if data_type == "project":
            fields = ["과제 목표", "주요내용", "기대효과"]
        elif data_type == "management_eval":
            fields = ["평가기준", "평가방법", "참고사항", "증빙자료"]
        else:
            fields = ["세부내용"]
        
        for field in fields:
            if field not in target:
                target[field] = []
            if field in source and isinstance(source[field], list):
                for item in source[field]:
                    if item and item not in target[field]:
                        target[field].append(item)
    
    def _create_structured_items(
        self, item_details: Dict, data_type: str, 
        source_document: str, page_range: str
    ) -> List[Any]:
        """구조화 데이터 객체 생성"""
        items = []
        for name, details in item_details.items():
            if not name:
                continue
            
            if data_type == "project":
                item = StructuredProject.from_dict(details, source_document, page_range)
            elif data_type == "management_eval":
                item = StructuredManagementEval.from_dict(details, source_document, page_range)
            else:
                item = StructuredInclusiveGrowth.from_dict(details, source_document, page_range)
            
            items.append(item)
        return items
    
    def _save_to_db(self, items: List[Any], data_type: str):
        """DB에 저장"""
        if data_type == "project":
            self.store.add_projects(items)
        elif data_type == "management_eval":
            self.store.add_management_evals(items)
        else:
            self.store.add_inclusive_growths(items)
    
    async def _extract_item_list(
        self, ocr_text: str, start_page: int, end_page: int, data_type: str
    ) -> List[Dict]:
        """목록 페이지에서 항목 리스트 추출 (데이터 타입별 프롬프트)"""
        model = ModelFactory.create_model_chain(
            provider=self.model_provider,
            model_name=self.structuring_model,
            output_format="json",
            max_rps=self.max_rps
        )
        parser = JsonOutputParser()
        
        # 데이터 타입별 프롬프트 생성
        if data_type == "project":
            system_prompt = """당신은 국정과제 문서를 분석하는 전문가입니다.
주어진 OCR 텍스트에서 국정과제의 목록을 추출합니다.

중요 원칙:
- 주어진 텍스트는 국정과제 목록/목차 페이지입니다.
- 각 과제의 번호와 과제명만 추출하세요.
- 세부 내용(목표, 주요내용 등)은 이 단계에서 추출하지 않습니다.
- 목록에 나타난 과제명을 정확히 그대로 추출하세요.

과제 식별 방법:
- 번호 체계: "국정과제01" 등의 형식. 반드시 국정과제만 추출하세요.
- 대분류와 소분류가 있을 수 있음

출력은 반드시 JSON 형식으로만 응답하세요."""
            user_template = """OCR 텍스트 (국정과제 목록 페이지 {start_page} ~ {end_page}):
{ocr_text}

위 텍스트에서 모든 국정과제의 번호와 이름을 추출하세요.

{{
    "indicators": [
        {{
            "번호": "1",
            "과제명": "디지털 플랫폼 정부 구현"
        }},
        {{
            "번호": "1-1",
            "과제명": "모든 데이터가 연결되는 디지털 플랫폼"
        }}
    ]
}}"""
        elif data_type == "management_eval":
            system_prompt = """당신은 공공기관 경영평가 지표 문서를 분석하는 전문가입니다.
주어진 OCR 텍스트에서 경영평가 지표의 목록을 추출합니다.

중요 원칙:
- 주어진 텍스트는 경영평가 지표 목록/목차 페이지입니다.
- 각 지표의 번호와 지표명만 추출하세요.
- 세부 내용(평가기준, 평가방법 등)은 이 단계에서 추출하지 않습니다.
- 목록에 나타난 지표명을 정확히 그대로 추출하세요.

지표 식별 방법:
- 번호 체계: "1", "1-1", "2-3" 등의 형식
- 대분류(범주)와 소분류가 있을 수 있음
- 배점 정보가 함께 있을 수 있음

출력은 반드시 JSON 형식으로만 응답하세요."""
            user_template = """OCR 텍스트 (경영평가 지표 목록 페이지 {start_page} ~ {end_page}):
{ocr_text}

위 텍스트에서 모든 경영평가 지표의 번호와 이름을 추출하세요.

{{
    "indicators": [
        {{
            "번호": "1",
            "지표명": "경영전략 및 리더십",
            "배점": "5점"
        }},
        {{
            "번호": "1-1",
            "지표명": "전략기획",
            "배점": "3점"
        }}
    ]
}}"""
        else:  # inclusive_growth
            system_prompt = """당신은 동반성장 평가지표 문서를 분석하는 전문가입니다.
주어진 OCR 텍스트에서 동반성장 평가지표의 목록을 추출합니다.

중요 원칙:
- 주어진 텍스트는 동반성장 평가지표 목록/목차 페이지입니다.
- 각 평가지표의 번호와 이름만 추출하세요.
- 세부 내용(평가기준, 평가방법 등)은 이 단계에서 추출하지 않습니다.
- 표에 나타난 지표명을 정확히 그대로 추출하세요.

평가지표 식별 방법:
- 번호 체계: "1", "1-1", "2-3" 등의 형식
- 대분류와 소분류가 있을 수 있음
- 배점 정보가 함께 있을 수 있음

출력은 반드시 JSON 형식으로만 응답하세요."""
            user_template = """OCR 텍스트 (동반성장 평가지표 목록 페이지 {start_page} ~ {end_page}):
{ocr_text}

위 텍스트에서 모든 평가지표의 번호와 이름을 추출하세요.

{{
    "indicators": [
        {{
            "번호": "1",
            "세부추진과제명": "동반성장 전략수립 및 체계",
            "배점": "8점"
        }},
        {{
            "번호": "1-1",
            "세부추진과제명": "추진목표·전략 등의 적정성·구체성",
            "배점": "2.0점"
        }}
    ]
}}"""
        
        user_text = user_template.format(
            start_page=start_page,
            end_page=end_page,
            ocr_text=ocr_text[:20000]
        )
        
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_text)
        ]
        
        response = await model.ainvoke(messages)
        
        try:
            result = parser.parse(response.content)
            items = result.get("indicators", [])
            
            # 데이터 타입에 맞게 키 이름 정규화
            name_field = self._get_name_field(data_type)
            normalized_items = []
            for item in items:
                # 다양한 이름 필드를 해당 타입의 필드명으로 통일
                for key in ["지표명", "과제명", "세부추진과제명"]:
                    if key in item and key != name_field:
                        item[name_field] = item.pop(key)
                normalized_items.append(item)
            
            return normalized_items
        except Exception as e:
            logger.warning(f"목록 추출 실패: {e}")
            return []
    
    def _image_to_base64(self, image: PIL.Image.Image) -> str:
        """PIL Image를 base64 문자열로 변환"""
        buffered = io.BytesIO()
        image.save(buffered, format="PNG")
        return base64.b64encode(buffered.getvalue()).decode("utf-8")
    
    async def _extract_page_details(
        self,
        item_list: List[Dict],
        prev_page: int,
        prev_page_text: str,
        current_page: int,
        current_page_text: str,
        data_type: str,
        current_page_image: Optional[PIL.Image.Image] = None
    ) -> List[Dict]:
        """페이지에서 상세 정보 추출 (텍스트 + 이미지 멀티모달, 데이터 타입별 프롬프트)"""
        model = ModelFactory.create_model_chain(
            provider=self.model_provider,
            model_name=self.structuring_model,
            output_format="json",
            max_rps=self.max_rps
        )
        parser = JsonOutputParser()
        
        name_field = self._get_name_field(data_type)
        item_list_str = "\n".join([
            f"- {item.get('번호', '')} {item.get(name_field, '')}"
            for item in item_list
        ])
        
        # 데이터 타입별 시스템 프롬프트
        if data_type == "project":
            system_prompt = """당신은 국정과제 문서 상세정보 추출 전문가입니다.

주어진 페이지의 텍스트와 이미지에서 국정과제의 상세 정보를 추출합니다.

## 규칙
1. 현재 페이지에 내용이 있는 과제만 추출하세요
2. 이전 페이지의 연속 내용인 경우도 포함하세요
3. 표 형태의 데이터가 있으면 이미지를 참고하여 정확히 추출하세요
4. OCR 텍스트에서 누락된 내용은 이미지를 보고 보완하세요

## 출력 형식 (JSON)
{
    "page_indicators": [
        {
            "번호": "과제 번호",
            "과제명": "과제 이름",
            "과제_목표": "과제의 목표/비전",
            "주요내용": "주요 추진 내용 상세",
            "기대효과": "기대되는 효과/성과"
        }
    ]
}"""
            user_template = """## 추출 대상 국정과제 목록
{item_list}

## 이전 페이지 ({prev_page}페이지) 텍스트
{prev_text}

## 현재 페이지 ({current_page}페이지) OCR 텍스트
{current_text}

위 과제 목록 중 현재 페이지에서 확인되는 과제의 상세 정보를 추출해주세요.
이미지를 참고하여 OCR에서 누락된 표나 내용을 보완해주세요."""

        elif data_type == "management_eval":
            system_prompt = """당신은 공공기관 경영평가 지표 상세정보 추출 전문가입니다.

주어진 페이지의 텍스트와 이미지에서 경영평가 지표의 상세 정보를 추출합니다.

## 규칙
1. 현재 페이지에 내용이 있는 지표만 추출하세요
2. 이전 페이지의 연속 내용인 경우도 포함하세요
3. 표 형태의 데이터가 있으면 이미지를 참고하여 정확히 추출하세요
4. OCR 텍스트에서 누락된 내용은 이미지를 보고 보완하세요

## 출력 형식 (JSON)
{
    "page_indicators": [
        {
            "번호": "지표 번호",
            "지표명": "지표 이름",
            "평가기준": "평가 기준 상세",
            "평가방법": "평가 방법 상세",
            "참고사항": "참고할 내용",
            "증빙자료": "필요한 증빙자료"
        }
    ]
}"""
            user_template = """## 추출 대상 경영평가 지표 목록
{item_list}

## 이전 페이지 ({prev_page}페이지) 텍스트
{prev_text}

## 현재 페이지 ({current_page}페이지) OCR 텍스트
{current_text}

위 지표 목록 중 현재 페이지에서 확인되는 지표의 상세 정보를 추출해주세요.
이미지를 참고하여 OCR에서 누락된 표나 내용을 보완해주세요."""

        else:  # inclusive_growth
            system_prompt = """당신은 동반성장 평가지표 상세정보 추출 전문가입니다.

주어진 페이지의 텍스트와 이미지에서 동반성장 평가지표의 상세 정보를 추출합니다.

## 규칙
1. 현재 페이지에 내용이 있는 지표만 추출하세요
2. 이전 페이지의 연속 내용인 경우도 포함하세요
3. 표 형태의 데이터가 있으면 이미지를 참고하여 정확히 추출하세요
4. OCR 텍스트에서 누락된 내용은 이미지를 보고 보완하세요

## 출력 형식 (JSON)
{
    "page_indicators": [
        {
            "번호": "지표 번호",
            "세부추진과제명": "세부추진과제 이름",
            "세부내용": "세부 내용 상세"
        }
    ]
}"""
            user_template = """## 추출 대상 동반성장 평가지표 목록
{item_list}

## 이전 페이지 ({prev_page}페이지) 텍스트
{prev_text}

## 현재 페이지 ({current_page}페이지) OCR 텍스트
{current_text}

위 지표 목록 중 현재 페이지에서 확인되는 지표의 상세 정보를 추출해주세요.
이미지를 참고하여 OCR에서 누락된 표나 내용을 보완해주세요."""

        # 사용자 프롬프트 생성
        user_text = user_template.format(
            item_list=item_list_str,
            prev_page=prev_page,
            prev_text=prev_page_text[:5000] if prev_page_text else "(없음)",
            current_page=current_page,
            current_text=current_page_text[:10000]
        )

        # 멀티모달 메시지 구성
        if current_page_image:
            # 이미지가 있으면 멀티모달로 호출
            image_base64 = self._image_to_base64(current_page_image)
            
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=[
                    {"type": "text", "text": user_text},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{image_base64}",
                            "detail": "high"
                        }
                    }
                ])
            ]
        else:
            # 이미지 없으면 텍스트만
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_text)
            ]
        
        response = await model.ainvoke(messages)
        
        try:
            result = parser.parse(response.content)
            page_items = result.get("page_indicators", [])
            
            # 데이터 타입에 맞게 키 이름 정규화
            normalized_items = []
            for item in page_items:
                # 다양한 이름 필드를 해당 타입의 필드명으로 통일
                for key in ["지표명", "과제명", "세부추진과제명"]:
                    if key in item and key != name_field:
                        item[name_field] = item.pop(key)
                normalized_items.append(item)
            
            return normalized_items
        except Exception as e:
            logger.warning(f"페이지 {current_page} 상세 추출 실패: {e}")
            return []
    
    # ========================================================================
    # 검색 메서드 (20개 후보 반환)
    # ========================================================================
    
    def search_projects(self, query: str, k: int = 20) -> List[Dict]:
        """국정과제 검색 (20개 후보)"""
        return self.store.search_projects(query, k)
    
    def search_management_evals(self, query: str, k: int = 20) -> List[Dict]:
        """경영평가 검색 (20개 후보)"""
        return self.store.search_management_evals(query, k)
    
    def search_inclusive_growth(self, query: str, k: int = 20) -> List[Dict]:
        """동반성장 검색 (20개 후보)"""
        return self.store.search_inclusive_growth(query, k)
    
    def search_all(self, query: str, k: int = 20) -> Dict[str, List[Dict]]:
        """전체 검색 (각 타입별 20개 후보)"""
        return {
            "projects": self.search_projects(query, k),
            "management_evals": self.search_management_evals(query, k),
            "inclusive_growth": self.search_inclusive_growth(query, k)
        }
    
    def get_stats(self) -> Dict[str, int]:
        """DB 통계 반환"""
        return self.store.get_stats()


# ============================================================================
# 편의 함수
# ============================================================================

def create_pipeline(
    db_host: str = "localhost",
    db_port: int = 3306,
    db_name: str = "b2g_data",
    db_user: str = "root",
    db_password: str = ""
) -> B2GPipeline:
    """파이프라인 인스턴스 생성 헬퍼 함수"""
    return B2GPipeline(
        db_host=db_host,
        db_port=db_port,
        db_name=db_name,
        db_user=db_user,
        db_password=db_password
    )


def create_store(
    db_host: str = "localhost",
    db_port: int = 3306,
    db_name: str = "b2g_data",
    db_user: str = "root",
    db_password: str = ""
) -> MySQLStore:
    """MySQL 저장소만 생성 (검색 전용)"""
    return MySQLStore(
        host=db_host,
        port=db_port,
        database=db_name,
        user=db_user,
        password=db_password
    )


# 기존 코드 호환성을 위한 별칭
B2GDataPipeline = B2GPipeline
B2GVectorStore = MySQLStore
InclusiveGrowthPipeline = B2GPipeline
InclusiveGrowthVectorStore = MySQLStore
StructuredIndicator = StructuredManagementEval


async def example_usage():
    """사용 예시"""
    # 파이프라인 생성
    pipeline = create_pipeline(
        db_host="localhost",
        db_port=3306,
        db_name="b2g_data",
        db_user="root",
        db_password=""
    )
    
    # PDF 처리 예시
    # await pipeline.process_pdf(
    #     pdf_path="/path/to/document.pdf",
    #     data_type="project",  # "project" | "management_eval" | "inclusive_growth"
    #     index_pages=(3, 5),   # 목록 페이지 범위
    #     detail_pages=(6, 50), # 세부내용 페이지 범위
    #     save_intermediate=True,
    #     output_dir="./output"
    # )
    
    # 검색 예시 (20개 후보 반환)
    # results = pipeline.search_projects("인공지능 AI 기술", k=20)
    # for r in results:
    #     print(f"[{r['score']:.2f}] {r['과제명']}")
    
    # DB 통계 확인
    print(pipeline.get_stats())


if __name__ == "__main__":
    asyncio.run(example_usage())
