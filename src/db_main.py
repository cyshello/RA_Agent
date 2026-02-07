"""
B2G 기준데이터 DB 모듈 - Schema-first Dynamic 버전 (MySQL/MariaDB)

- PDF -> OCR -> LLM 구조화 -> MySQL 저장 -> 임베딩 검색
- 모든 타입별 동작은 SCHEMA_REGISTRY에 의해 동적으로 결정됨
"""

import os
import io
import json
import logging
import asyncio
import base64
import struct
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, List, Dict, Any, Tuple

import dotenv
import fitz  # PyMuPDF
import PIL.Image
import numpy as np

from langchain_core.output_parsers import JsonOutputParser
from langchain_core.messages import HumanMessage, SystemMessage

from .utils import CLOVA_ocr, CLOVA_ocr_with_table
from .api import ModelFactory

import pymysql
from pymysql.cursors import DictCursor
from openai import OpenAI

# -----------------------------------------------------------------------------
# 환경/로깅
# -----------------------------------------------------------------------------
env_path = os.path.join(os.path.dirname(__file__), ".env")
OPENAI_KEY = dotenv.get_key(env_path, "OPENAI_KEY")

EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_DIM = 1536  # text-embedding-3-small

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# -----------------------------------------------------------------------------
# Schema Registry (핵심)
# -----------------------------------------------------------------------------
# field spec:
# - type: "string" | "array" | "object"
# - db: "VARCHAR(500)" | "JSON" | ...
# - required: bool
# - searchable: 검색용 텍스트에 포함 여부
# - extract_detail: 상세 추출 단계에서 리스트로 뽑아야 하는 필드 여부
#
# type-level:
# - table / embedding_table
# - name_field / number_field
# - type_display / item_display
# - index_output_key / detail_output_key (LLM 응답 JSON key)
# - aliases: 입력 키를 canonical key로 매핑
SCHEMA_REGISTRY: Dict[str, Dict[str, Any]] = {
    "project": {
        "table": "national_projects",
        "embedding_table": "embedding_chunks_project",
        "json_path": os.path.join(os.path.dirname(os.path.dirname(__file__)), "DB_data", "project.json"),
        "name_field": "과제명",
        "number_field": "과제번호",
        "type_display": "국정과제",
        "item_display": "과제",
        "index_output_key": "indicators",
        "detail_output_key": "page_indicators",
        "fields": {
            "과제번호": {"type": "string", "db": "VARCHAR(50)", "required": False, "searchable": False, "extract_detail": False},
            "과제명": {"type": "string", "db": "VARCHAR(500)", "required": True, "searchable": True, "extract_detail": False},
            "과제_목표": {"type": "array", "db": "JSON", "required": False, "searchable": True, "extract_detail": True},
            "주요내용": {"type": "array", "db": "JSON", "required": False, "searchable": True, "extract_detail": True},
            "기대효과": {"type": "array", "db": "JSON", "required": False, "searchable": True, "extract_detail": True},
        },
        "aliases": {
            "번호": "과제번호",
            "과제 번호": "과제번호",
            "과제 목표": "과제_목표",
        },
    },
    "management_eval": {
        "table": "management_evals",
        "embedding_table": "embedding_chunks_management",
        "json_path": os.path.join(os.path.dirname(os.path.dirname(__file__)), "DB_data", "management.json"),
        "name_field": "지표명",
        "number_field": "번호",
        "type_display": "경영평가 지표",
        "item_display": "지표",
        "index_output_key": "indicators",
        "detail_output_key": "page_indicators",
        "fields": {
            "번호": {"type": "string", "db": "VARCHAR(50)", "required": False, "searchable": False, "extract_detail": False},
            "지표명": {"type": "string", "db": "VARCHAR(500)", "required": True, "searchable": True, "extract_detail": False},
            "평가기준": {"type": "array", "db": "JSON", "required": False, "searchable": True, "extract_detail": True},
            "평가방법": {"type": "array", "db": "JSON", "required": False, "searchable": True, "extract_detail": True},
            "참고사항": {"type": "array", "db": "JSON", "required": False, "searchable": True, "extract_detail": True},
            "증빙자료": {"type": "array", "db": "JSON", "required": False, "searchable": True, "extract_detail": True},
        },
        "aliases": {
            "증빙사료": "증빙자료",
            "과제명": "지표명",
            "세부추진과제명": "지표명",
        },
    },
    "inclusive_growth": {
        "table": "inclusive_growth",
        "embedding_table": "embedding_chunks_inclusive",
        "json_path": os.path.join(os.path.dirname(os.path.dirname(__file__)), "DB_data", "inclusive.json"),
        "name_field": "세부추진과제명",
        "number_field": "번호",
        "type_display": "동반성장 평가지표",
        "item_display": "지표",
        "index_output_key": "indicators",
        "detail_output_key": "page_indicators",
        "fields": {
            "번호": {"type": "string", "db": "VARCHAR(50)", "required": False, "searchable": False, "extract_detail": False},
            "세부추진과제명": {"type": "string", "db": "VARCHAR(500)", "required": True, "searchable": True, "extract_detail": False},
            "세부내용": {"type": "array", "db": "JSON", "required": False, "searchable": True, "extract_detail": True},
        },
        "aliases": {
            "지표명": "세부추진과제명",
            "과제명": "세부추진과제명",
        },
    },
}


# -----------------------------------------------------------------------------
# 공통 유틸
# -----------------------------------------------------------------------------
def get_meta(data_type: str) -> Dict[str, Any]:
    if data_type not in SCHEMA_REGISTRY:
        raise ValueError(f"지원하지 않는 data_type: {data_type}")
    return SCHEMA_REGISTRY[data_type]


def schema_for_prompt(data_type: str, include_meta_fields: bool = False) -> Dict[str, Any]:
    """
    LLM 프롬프트용 스키마 템플릿
    - array는 [설명] 형식
    - string은 "설명" 형식
    """
    meta = get_meta(data_type)
    result: Dict[str, Any] = {}
    for k, spec in meta["fields"].items():
        t = spec["type"]
        if t == "array":
            result[k] = [f"{k} 항목"]
        elif t == "object":
            result[k] = {f"{k}_key": f"{k}_value"}
        else:
            result[k] = f"{k}"
    if include_meta_fields:
        result.update({
            "source_document": "원본 문서명",
            "page_range": "페이지 범위",
            "extraction_date": "ISO datetime",
        })
    return result


def normalize_item_keys(data_type: str, item: Dict[str, Any]) -> Dict[str, Any]:
    """
    입력 dict를 canonical key로 정규화
    - aliases 반영
    - 타 타입 이름필드 혼입 시 현재 타입 이름필드로 통일
    """
    meta = get_meta(data_type)
    aliases = meta.get("aliases", {})
    canonical = set(meta["fields"].keys())
    out: Dict[str, Any] = {}

    # 1) alias 적용
    for k, v in item.items():
        ck = aliases.get(k, k)
        out[ck] = v

    # 2) 이름 필드 통일 (타 스키마 키가 들어온 경우)
    name_field = meta["name_field"]
    alt_name_fields = ["과제명", "지표명", "세부추진과제명"]
    for nk in alt_name_fields:
        if nk in out and nk != name_field and name_field not in out:
            out[name_field] = out[nk]

    # 3) canonical 필드만 남기기(없는 건 초기값 세팅)
    clean: Dict[str, Any] = {}
    for k, spec in meta["fields"].items():
        if k in out:
            clean[k] = out[k]
        else:
            clean[k] = [] if spec["type"] == "array" else ""

    # 4) 타입 정리
    for k, spec in meta["fields"].items():
        if spec["type"] == "array":
            v = clean.get(k, [])
            if isinstance(v, str):
                clean[k] = [v] if v.strip() else []
            elif isinstance(v, list):
                clean[k] = [str(x).strip() for x in v if str(x).strip()]
            else:
                clean[k] = []
        elif spec["type"] == "string":
            v = clean.get(k, "")
            clean[k] = str(v).strip() if v is not None else ""
        # object 필요시 확장

    return clean


def build_searchable_text(data_type: str, values: Dict[str, Any]) -> str:
    meta = get_meta(data_type)
    parts: List[str] = []
    for k, spec in meta["fields"].items():
        if not spec.get("searchable", False):
            continue
        v = values.get(k)
        if isinstance(v, list):
            parts.extend([str(x) for x in v if str(x).strip()])
        elif isinstance(v, str):
            if v.strip():
                parts.append(v.strip())
        elif v is not None:
            parts.append(str(v))
    return " ".join(parts).strip()


def safe_json_parse(data: Any) -> Any:
    if data is None:
        return []
    if isinstance(data, (list, dict)):
        return data
    if isinstance(data, str):
        s = data.strip()
        if not s:
            return []
        try:
            return json.loads(s)
        except json.JSONDecodeError:
            return s
    return data


# -----------------------------------------------------------------------------
# 데이터 모델 (단일)
# -----------------------------------------------------------------------------
@dataclass
class StructuredRecord:
    data_type: str
    values: Dict[str, Any]  # canonical keys only
    source_document: Optional[str] = None
    page_range: Optional[str] = None
    extraction_date: Optional[str] = None

    @classmethod
    def from_dict(
        cls,
        data_type: str,
        data: Dict[str, Any],
        source_document: str = "",
        page_range: str = "",
    ) -> "StructuredRecord":
        normalized = normalize_item_keys(data_type, data)
        return cls(
            data_type=data_type,
            values=normalized,
            source_document=source_document,
            page_range=page_range,
            extraction_date=datetime.now().isoformat(),
        )

    def to_dict(self) -> Dict[str, Any]:
        # canonical key만 반환
        return {
            **self.values,
            "source_document": self.source_document,
            "page_range": self.page_range,
            "extraction_date": self.extraction_date,
        }

    def get_searchable_text(self) -> str:
        return build_searchable_text(self.data_type, self.values)


@dataclass
class PageOCRResult:
    page_num: int
    text: str
    fields: List[str]
    tables: List[Dict]
    raw_response: Dict


# -----------------------------------------------------------------------------
# PDF 처리
# -----------------------------------------------------------------------------
class PDFProcessor:
    def __init__(self, dpi: int = 200, use_table: bool = True):
        self.dpi = dpi
        self.zoom = dpi / 72
        self.use_table = use_table

    def pdf_to_images(self, pdf_path: str) -> List[PIL.Image.Image]:
        images: List[PIL.Image.Image] = []
        doc = fitz.open(pdf_path)
        mat = fitz.Matrix(self.zoom, self.zoom)
        for i in range(len(doc)):
            page = doc[i]
            pix = page.get_pixmap(matrix=mat)
            img = PIL.Image.open(io.BytesIO(pix.tobytes("png")))
            images.append(img)
            logger.info(f"페이지 {i+1}/{len(doc)} 이미지 변환 완료")
        doc.close()
        return images

    def process_page(self, image: PIL.Image.Image, page_num: int) -> PageOCRResult:
        buf = io.BytesIO()
        image.save(buf, format="PNG")
        img_bytes = buf.getvalue()

        if self.use_table:
            ocr = CLOVA_ocr_with_table(img_bytes)
            return PageOCRResult(
                page_num=page_num,
                text=ocr.get("text", ""),
                fields=ocr.get("fields", []),
                tables=ocr.get("tables", []),
                raw_response=ocr.get("raw_response", {}),
            )
        else:
            lines = CLOVA_ocr(img_bytes)
            text = "\n".join(lines)
            return PageOCRResult(
                page_num=page_num,
                text=text,
                fields=lines,
                tables=[],
                raw_response={"text_lines": lines},
            )


# -----------------------------------------------------------------------------
# Dynamic MySQL Store
# -----------------------------------------------------------------------------
class DynamicMySQLStore:
    """
    schema-first 동적 저장소
    - 타입별 CRUD 하드코딩 없음
    """

    def __init__(
        self,
        host: str = "localhost",
        port: int = 3306,
        database: str = "b2g_data",
        user: str = "root",
        password: str = "",
    ):
        self.connection_params = {
            "host": host,
            "port": port,
            "database": database,
            "user": user,
            "password": password,
            "charset": "utf8mb4",
            "cursorclass": DictCursor,
        }
        self._openai_client = None
        self._init_tables()
        logger.info(f"DynamicMySQLStore 초기화 완료: {database}")

    def _get_connection(self):
        return pymysql.connect(**self.connection_params)

    # -------------------------------
    # Table init (dynamic)
    # -------------------------------
    def _build_main_table_ddl(self, data_type: str) -> str:
        meta = get_meta(data_type)
        table = meta["table"]

        col_defs = ["id INT AUTO_INCREMENT PRIMARY KEY"]
        for fname, spec in meta["fields"].items():
            db_type = spec["db"]
            required = "NOT NULL" if spec.get("required", False) else "NULL"
            col_defs.append(f"`{fname}` {db_type} {required}")

        col_defs.extend([
            "source_document VARCHAR(255)",
            "page_range VARCHAR(50)",
            "extraction_date DATETIME",
            "searchable_text TEXT",
            "created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP",
            "FULLTEXT INDEX ft_searchable (searchable_text) WITH PARSER ngram",
        ])

        return f"""
        CREATE TABLE IF NOT EXISTS `{table}` (
            {", ".join(col_defs)}
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
        """

    def _build_embedding_table_ddl(self, data_type: str) -> str:
        meta = get_meta(data_type)
        et = meta["embedding_table"]
        return f"""
        CREATE TABLE IF NOT EXISTS `{et}` (
            id INT AUTO_INCREMENT PRIMARY KEY,
            source_id INT NOT NULL,
            item_name VARCHAR(500) NOT NULL,
            field_type VARCHAR(100) NOT NULL,
            chunk_text TEXT NOT NULL,
            embedding BLOB,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            INDEX idx_source_id (source_id),
            INDEX idx_item_name (item_name(255)),
            INDEX idx_field_type (field_type)
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
        """

    def _init_tables(self):
        conn = self._get_connection()
        try:
            with conn.cursor() as cursor:
                for dt in SCHEMA_REGISTRY.keys():
                    cursor.execute(self._build_main_table_ddl(dt))
                    cursor.execute(self._build_embedding_table_ddl(dt))
            conn.commit()
        finally:
            conn.close()

    # -------------------------------
    # OpenAI embedding utils
    # -------------------------------
    def _init_openai_client(self):
        if self._openai_client is None:
            if not OPENAI_KEY:
                raise ValueError("OPENAI_KEY not found in .env")
            self._openai_client = OpenAI(api_key=OPENAI_KEY)
        return self._openai_client

    def _get_embeddings_batch(self, texts: List[str], batch_size: int = 100) -> List[Optional[bytes]]:
        client = self._init_openai_client()
        results: List[Optional[bytes]] = []

        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            batch = [t[:8000] if t and t.strip() else " " for t in batch]
            try:
                resp = client.embeddings.create(model=EMBEDDING_MODEL, input=batch)
                for item in resp.data:
                    emb = item.embedding
                    results.append(struct.pack(f"{len(emb)}f", *emb))
            except Exception as e:
                logger.warning(f"배치 임베딩 실패: {e}")
                results.extend([None] * len(batch))
        return results

    def _get_query_embedding(self, text: str) -> List[float]:
        client = self._init_openai_client()
        resp = client.embeddings.create(model=EMBEDDING_MODEL, input=text)
        return resp.data[0].embedding

    @staticmethod
    def _blob_to_vector(blob: bytes) -> Optional[np.ndarray]:
        if blob is None:
            return None
        return np.array(struct.unpack(f"{EMBEDDING_DIM}f", blob))

    @staticmethod
    def _cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
        if vec1 is None or vec2 is None:
            return 0.0
        dot = np.dot(vec1, vec2)
        n1 = np.linalg.norm(vec1)
        n2 = np.linalg.norm(vec2)
        if n1 == 0 or n2 == 0:
            return 0.0
        return float(dot / (n1 * n2))

    # -------------------------------
    # CRUD
    # -------------------------------
    def add_record(self, record: StructuredRecord) -> int:
        data_type = record.data_type
        meta = get_meta(data_type)
        table = meta["table"]

        conn = self._get_connection()
        try:
            with conn.cursor() as cursor:
                # main insert columns
                columns = list(meta["fields"].keys()) + ["source_document", "page_range", "extraction_date", "searchable_text"]
                placeholders = ", ".join(["%s"] * len(columns))
                col_sql = ", ".join([f"`{c}`" for c in columns])

                values_for_db: List[Any] = []
                for c in meta["fields"].keys():
                    spec = meta["fields"][c]
                    v = record.values.get(c)
                    if spec["type"] in ("array", "object"):
                        values_for_db.append(json.dumps(v if v is not None else ([] if spec["type"] == "array" else {}), ensure_ascii=False))
                    else:
                        values_for_db.append(v)

                values_for_db.extend([
                    record.source_document,
                    record.page_range,
                    record.extraction_date,
                    record.get_searchable_text(),
                ])

                sql = f"INSERT INTO `{table}` ({col_sql}) VALUES ({placeholders})"
                cursor.execute(sql, tuple(values_for_db))
                source_id = cursor.lastrowid

            # embedding chunks
            self._add_embedding_chunks(conn, record, source_id)

            conn.commit()
            return source_id
        finally:
            conn.close()

    def add_records(self, records: List[StructuredRecord]) -> List[int]:
        ids = [self.add_record(r) for r in records]
        logger.info(f"{len(ids)}개 레코드 저장 완료")
        return ids

    def _add_embedding_chunks(self, conn, record: StructuredRecord, source_id: int):
        meta = get_meta(record.data_type)
        embedding_table = meta["embedding_table"]
        item_name = str(record.values.get(meta["name_field"], "")).strip()
        if not item_name:
            return

        chunks: List[Dict[str, str]] = []
        for fname, spec in meta["fields"].items():
            if not spec.get("extract_detail", False):
                continue
            v = record.values.get(fname)
            if isinstance(v, list):
                for x in v:
                    s = str(x).strip()
                    if s:
                        chunks.append({"field_type": fname, "chunk_text": s})
            elif isinstance(v, str):
                s = v.strip()
                if s:
                    chunks.append({"field_type": fname, "chunk_text": s})

        if not chunks:
            return

        texts = [c["chunk_text"] for c in chunks]
        embs = self._get_embeddings_batch(texts)

        with conn.cursor() as cursor:
            for c, e in zip(chunks, embs):
                cursor.execute(
                    f"""
                    INSERT INTO `{embedding_table}`
                    (source_id, item_name, field_type, chunk_text, embedding)
                    VALUES (%s, %s, %s, %s, %s)
                    """,
                    (source_id, item_name, c["field_type"], c["chunk_text"], e),
                )

    # -------------------------------
    # Stats / delete
    # -------------------------------
    def delete_all_data(self, data_type: Optional[str] = None):
        conn = self._get_connection()
        try:
            with conn.cursor() as cursor:
                targets = [data_type] if data_type else list(SCHEMA_REGISTRY.keys())
                for dt in targets:
                    meta = get_meta(dt)
                    cursor.execute(f"TRUNCATE TABLE `{meta['table']}`")
                    cursor.execute(f"TRUNCATE TABLE `{meta['embedding_table']}`")
            conn.commit()
            logger.info(f"데이터 삭제 완료: {data_type or '전체'}")
        finally:
            conn.close()

    def get_stats(self) -> Dict[str, int]:
        conn = self._get_connection()
        out: Dict[str, int] = {}
        try:
            with conn.cursor() as cursor:
                for dt, meta in SCHEMA_REGISTRY.items():
                    cursor.execute(f"SELECT COUNT(*) as cnt FROM `{meta['table']}`")
                    out[meta["table"]] = cursor.fetchone()["cnt"]
                    cursor.execute(f"SELECT COUNT(*) as cnt FROM `{meta['embedding_table']}`")
                    out[meta["embedding_table"]] = cursor.fetchone()["cnt"]
            return out
        finally:
            conn.close()

    def get_records(self, data_type: str, limit: int = 20) -> List[Dict[str, Any]]:
        """
        임베딩 검색 없이 테이블에서 직접 레코드를 조회 (Fallback용)
        """
        meta = get_meta(data_type)
        table = meta["table"]
        
        conn = self._get_connection()
        try:
            with conn.cursor() as cursor:
                cursor.execute(f"SELECT * FROM `{table}` LIMIT %s", (limit,))
                rows = cursor.fetchall()
                
            results = []
            for row in rows:
                item = {"id": row["id"], "score": 0.0} # 기본 점수 0
                for fname, spec in meta["fields"].items():
                    val = row.get(fname)
                    if spec["type"] in ("array", "object"):
                        item[fname] = safe_json_parse(val)
                    else:
                        item[fname] = val
                item["source_document"] = row.get("source_document")
                item["page_range"] = row.get("page_range")
                results.append(item)
            return results
        finally:
            conn.close()

    # -------------------------------
    # Search (embedding)
    # -------------------------------
    def search_by_embedding(self, data_type: str, query: str, k: int = 20) -> List[Dict[str, Any]]:
        meta = get_meta(data_type)
        table = meta["table"]
        embedding_table = meta["embedding_table"]
        name_field = meta["name_field"]

        qvec = np.array(self._get_query_embedding(query))
        conn = self._get_connection()
        try:
            with conn.cursor() as cursor:
                cursor.execute(
                    f"""
                    SELECT id, source_id, item_name, field_type, chunk_text, embedding
                    FROM `{embedding_table}`
                    WHERE embedding IS NOT NULL
                    """
                )
                chunks = cursor.fetchall()

            scored: List[Dict[str, Any]] = []
            for ch in chunks:
                dbv = self._blob_to_vector(ch["embedding"])
                if dbv is None:
                    continue
                score = self._cosine_similarity(qvec, dbv)
                scored.append({
                    "source_id": ch["source_id"],
                    "item_name": ch["item_name"],
                    "field_type": ch["field_type"],
                    "chunk_text": ch["chunk_text"],
                    "score": score,
                })

            scored.sort(key=lambda x: x["score"], reverse=True)

            seen = set()
            top_sources: List[Tuple[int, float]] = []
            for s in scored:
                nm = s["item_name"]
                if nm in seen:
                    continue
                seen.add(nm)
                top_sources.append((s["source_id"], s["score"]))
                if len(top_sources) >= k:
                    break

            results: List[Dict[str, Any]] = []
            with conn.cursor() as cursor:
                for sid, sc in top_sources:
                    cursor.execute(f"SELECT * FROM `{table}` WHERE id=%s", (sid,))
                    row = cursor.fetchone()
                    if not row:
                        continue

                    item: Dict[str, Any] = {"id": row["id"], "score": sc}
                    for fname, spec in meta["fields"].items():
                        val = row.get(fname)
                        if spec["type"] in ("array", "object"):
                            item[fname] = safe_json_parse(val)
                        else:
                            item[fname] = val
                    item["source_document"] = row.get("source_document")
                    item["page_range"] = row.get("page_range")
                    results.append(item)

            return results
        finally:
            conn.close()

    def search_all_by_embedding(self, query: str, k: int = 20) -> Dict[str, List[Dict[str, Any]]]:
        return {dt: self.search_by_embedding(dt, query, k=k) for dt in SCHEMA_REGISTRY.keys()}

    # -------------------------------
    # Backward Compatibility
    # -------------------------------
    def search_projects_by_embedding(self, query: str, k: int = 20) -> List[Dict[str, Any]]:
        return self.search_by_embedding("project", query, k)

    def search_management_evals_by_embedding(self, query: str, k: int = 20) -> List[Dict[str, Any]]:
        return self.search_by_embedding("management_eval", query, k)

    def search_inclusive_growth_by_embedding(self, query: str, k: int = 20) -> List[Dict[str, Any]]:
        return self.search_by_embedding("inclusive_growth", query, k)


# -----------------------------------------------------------------------------
# B2G Pipeline (dynamic)
# -----------------------------------------------------------------------------
class B2GPipeline:
    """
    PDF -> OCR -> 목록 추출 -> 상세 추출 -> StructuredRecord -> DB 저장
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
        max_rps: float = 2.0,
        use_table: bool = True,
    ):
        self.pdf_processor = PDFProcessor(use_table=use_table)
        self.store = DynamicMySQLStore(
            host=db_host,
            port=db_port,
            database=db_name,
            user=db_user,
            password=db_password,
        )
        self.model_provider = model_provider
        self.extraction_model = extraction_model
        self.structuring_model = structuring_model
        self.max_rps = max_rps
        self._pdf_images: Optional[List[PIL.Image.Image]] = None
        self.use_table = use_table

        logger.info(f"B2GPipeline 초기화 완료 (OCR mode: {'table' if use_table else 'text'})")

    # -------------------------------
    # public API
    # -------------------------------
    async def process_pdf(
        self,
        pdf_path: str,
        data_type: str,
        index_pages: Tuple[int, int],
        detail_pages: Tuple[int, int],
        save_intermediate: bool = False,
        output_dir: Optional[str] = None,
        reuse_ocr_dir: Optional[str] = None,
    ) -> List[StructuredRecord]:
        records = await self.extract_from_pdf(
            pdf_path=pdf_path,
            data_type=data_type,
            index_pages=index_pages,
            detail_pages=detail_pages,
            save_intermediate=save_intermediate,
            output_dir=output_dir,
            reuse_ocr_dir=reuse_ocr_dir,
        )
        if records:
            self.store.add_records(records)
        return records

    async def extract_from_pdf(
        self,
        pdf_path: str,
        data_type: str,
        index_pages: Tuple[int, int],
        detail_pages: Tuple[int, int],
        save_intermediate: bool = False,
        output_dir: Optional[str] = None,
        reuse_ocr_dir: Optional[str] = None,
    ) -> List[StructuredRecord]:
        meta = get_meta(data_type)
        source_document = os.path.basename(pdf_path)

        idx_s, idx_e = index_pages
        det_s, det_e = detail_pages
        all_pages = sorted(set(list(range(idx_s, idx_e + 1)) + list(range(det_s, det_e + 1))))

        logger.info(f"PDF 데이터 추출 시작: {source_document} ({data_type})")

        # OCR cache load
        ocr_data_map: Dict[int, Dict[str, Any]] = {}
        missing: List[int] = []
        effective_ocr_dir = reuse_ocr_dir or output_dir

        if effective_ocr_dir and os.path.exists(effective_ocr_dir):
            loaded = 0
            for p in all_pages:
                x = self._load_ocr_from_json(effective_ocr_dir, p)
                if x:
                    ocr_data_map[p] = x
                    loaded += 1
                else:
                    missing.append(p)
            logger.info(f"OCR cache 로드: {loaded}개, 누락: {len(missing)}개")
        else:
            missing = all_pages

        if missing:
            if self._pdf_images is None:
                self._pdf_images = self.pdf_processor.pdf_to_images(pdf_path)
            for p in missing:
                idx = p - 1
                if idx < 0 or idx >= len(self._pdf_images):
                    logger.warning(f"페이지 범위 초과: {p}")
                    continue
                ocr_res = self.pdf_processor.process_page(self._pdf_images[idx], p)
                ocr_data = {
                    "page_num": ocr_res.page_num,
                    "text": ocr_res.text,
                    "fields": ocr_res.fields,
                    "tables": ocr_res.tables,
                }
                ocr_data_map[p] = ocr_data
                if save_intermediate and output_dir:
                    os.makedirs(output_dir, exist_ok=True)
                    with open(os.path.join(output_dir, f"ocr_page_{p}.json"), "w", encoding="utf-8") as f:
                        json.dump(ocr_data, f, ensure_ascii=False, indent=2)

        # step1: index list
        index_text_parts: List[str] = []
        for p in range(idx_s, idx_e + 1):
            if p not in ocr_data_map:
                continue
            table_text = self._tables_to_text(ocr_data_map[p].get("tables", []))
            page_text = table_text if table_text else ocr_data_map[p].get("text", "")
            if page_text:
                index_text_parts.append(f"=== 페이지 {p} ===\n{page_text}")
        index_text = "\n\n".join(index_text_parts)

        item_list = await self._extract_item_list(index_text, idx_s, idx_e, data_type)
        logger.info(f"목록 추출 완료: {len(item_list)}개")

        if save_intermediate and output_dir:
            with open(os.path.join(output_dir, "item_list.json"), "w", encoding="utf-8") as f:
                json.dump(item_list, f, ensure_ascii=False, indent=2)

        # step2: details merge
        name_field = meta["name_field"]
        item_details: Dict[str, Dict[str, Any]] = {
            it.get(name_field, ""): normalize_item_keys(data_type, it)
            for it in item_list
            if it.get(name_field, "")
        }

        if self._pdf_images is None:
            self._pdf_images = self.pdf_processor.pdf_to_images(pdf_path)

        prev_text = ""
        for p in range(det_s, det_e + 1):
            if p not in ocr_data_map:
                continue
            cur_text = ocr_data_map[p].get("text", "")
            cur_img = self._pdf_images[p - 1] if (self._pdf_images and p - 1 < len(self._pdf_images)) else None

            page_items = await self._extract_page_details(
                item_list=item_list,
                prev_page=p - 1 if p > det_s else 0,
                prev_page_text=prev_text,
                current_page=p,
                current_page_text=cur_text,
                data_type=data_type,
                current_page_image=cur_img,
            )

            for pi in page_items:
                npi = normalize_item_keys(data_type, pi)
                name = npi.get(name_field, "")
                if not name:
                    continue
                
                # 목록에 있는 항목만 처리 (새로운 항목 추가 방지)
                if name in item_details:
                    self._merge_item_details(data_type, item_details[name], npi)

            prev_text = cur_text

        if save_intermediate and output_dir:
            with open(os.path.join(output_dir, "item_details.json"), "w", encoding="utf-8") as f:
                json.dump(item_details, f, ensure_ascii=False, indent=2)

        # step3: records build
        records: List[StructuredRecord] = []
        for _, details in item_details.items():
            rec = StructuredRecord.from_dict(
                data_type=data_type,
                data=details,
                source_document=source_document,
                page_range=f"{det_s}-{det_e}",
            )
            records.append(rec)

        if save_intermediate and output_dir:
            with open(os.path.join(output_dir, "structured_items.json"), "w", encoding="utf-8") as f:
                json.dump([r.to_dict() for r in records], f, ensure_ascii=False, indent=2)

        logger.info(f"추출 완료: {len(records)}개")
        return records

    def get_stats(self) -> Dict[str, int]:
        return self.store.get_stats()

    def search_by_embedding(self, data_type: str, query: str, k: int = 20) -> List[Dict[str, Any]]:
        return self.store.search_by_embedding(data_type=data_type, query=query, k=k)

    def search_all_by_embedding(self, query: str, k: int = 20) -> Dict[str, List[Dict[str, Any]]]:
        return self.store.search_all_by_embedding(query=query, k=k)

    # -------------------------------
    # internal helpers
    # -------------------------------
    @staticmethod
    def _load_ocr_from_json(ocr_dir: str, page_num: int) -> Optional[Dict]:
        p = os.path.join(ocr_dir, f"ocr_page_{page_num}.json")
        if os.path.exists(p):
            with open(p, "r", encoding="utf-8") as f:
                return json.load(f)
        return None

    @staticmethod
    def _tables_to_text(tables: List[Dict]) -> str:
        if not tables:
            return ""
        out: List[str] = []
        for i, tb in enumerate(tables):
            if tb.get("markdown"):
                out.append(f"[표 {i+1}]\n{tb['markdown']}")
            elif tb.get("rows"):
                rows = tb["rows"]
                lines = []
                for ridx in sorted(rows.keys(), key=int):
                    cols = rows[ridx]
                    line = " | ".join([cols.get(str(c), "") for c in sorted(cols.keys(), key=int)])
                    lines.append(line)
                out.append(f"[표 {i+1}]\n" + "\n".join(lines))
        return "\n\n".join(out)

    @staticmethod
    def _image_to_base64(image: PIL.Image.Image) -> str:
        buf = io.BytesIO()
        image.save(buf, format="PNG")
        return base64.b64encode(buf.getvalue()).decode("utf-8")

    def _merge_item_details(self, data_type: str, target: Dict[str, Any], source: Dict[str, Any]):
        meta = get_meta(data_type)
        for k, spec in meta["fields"].items():
            if spec["type"] == "array":
                tlist = target.get(k, [])
                if not isinstance(tlist, list):
                    tlist = [str(tlist)] if tlist else []
                sval = source.get(k, [])
                if isinstance(sval, str):
                    sval = [sval]
                if not isinstance(sval, list):
                    sval = []
                for x in sval:
                    sx = str(x).strip()
                    if sx and sx not in tlist:
                        tlist.append(sx)
                target[k] = tlist
            elif spec["type"] == "string":
                if not target.get(k) and source.get(k):
                    target[k] = str(source.get(k)).strip()

    async def _extract_item_list(self, ocr_text: str, start_page: int, end_page: int, data_type: str) -> List[Dict[str, Any]]:
        meta = get_meta(data_type)
        model = ModelFactory.create_model_chain(
            provider=self.model_provider,
            model_name=self.structuring_model,
            output_format="json",
            max_rps=self.max_rps,
        )
        parser = JsonOutputParser()

        name_field = meta["name_field"]
        number_field = meta["number_field"]
        type_name = meta["type_display"]
        item_name = meta["item_display"]
        output_key = meta["index_output_key"]

        system_prompt = f"""당신은 {type_name} 문서를 분석하는 전문가입니다.
주어진 OCR 텍스트에서 {type_name} 목록을 추출합니다.

중요 원칙:
- 목록/목차 페이지에서 각 {item_name}의 번호와 이름만 추출
- 세부 내용은 이 단계에서 추출하지 않음
- 출력은 JSON만 반환

"""
        extra = """세부평가지표를 추출할 때는 큰 범위의 항목명을 무시하고 최대한 세부적인 항목명을 추출하며, 표의 "평가지표" 라는 단어가 포함된 열을 우선적으로 참고합니다.
가점, 주요사업 등의 추가적인 세부지표도 가능한 한 모두 추출합니다.
마지막으로, 여러 페이지에서 한번이라도 등장한 지표는 반드시 모두 포함시킵니다.

""" if data_type == "management_eval" else ""
        example_item = {number_field: "1", name_field: f"예시 {item_name}명"}
        user_text = f"""OCR 텍스트 ({type_name} 목록 페이지 {start_page} ~ {end_page}):
{ocr_text[:20000]}

위에 주어진 OCR를 이용해 목록/목차 페이지에서 {type_name} 목록을 추출하세요.
이때, OCR 텍스트에서 구조를 파악하지 못할 경우 추가로 주어지는 이미지를 참고하여 추출합니다.
최대한 정확하게 추출하며, 이 문서는 대부분 표로 이루어져 있음을 감안하여 추출에 임하세요.
또한, 중복된 항목이 있을 경우 하나로 합쳐서 추출합니다.
{extra}
아래 형식으로 추출하세요:
{{
  "{output_key}": [
    {json.dumps(example_item, ensure_ascii=False)}
  ]
}}"""

        msgs = [SystemMessage(content=system_prompt), HumanMessage(content=user_text)]
        resp = await model.ainvoke(msgs)

        try:
            result = parser.parse(resp.content)
            items = result.get(output_key, [])
            return [normalize_item_keys(data_type, x) for x in items if isinstance(x, dict)]
        except Exception as e:
            logger.warning(f"목록 추출 실패: {e}")
            return []

    async def _extract_page_details(
        self,
        item_list: List[Dict[str, Any]],
        prev_page: int,
        prev_page_text: str,
        current_page: int,
        current_page_text: str,
        data_type: str,
        current_page_image: Optional[PIL.Image.Image] = None,
    ) -> List[Dict[str, Any]]:
        meta = get_meta(data_type)
        model = ModelFactory.create_model_chain(
            provider=self.model_provider,
            model_name=self.structuring_model,
            output_format="json",
            max_rps=self.max_rps,
        )
        parser = JsonOutputParser()

        name_field = meta["name_field"]
        number_field = meta["number_field"]
        type_name = meta["type_display"]
        item_name = meta["item_display"]
        output_key = meta["detail_output_key"]

        item_list_str = "\n".join([f"- {x.get(number_field, '')} {x.get(name_field, '')}" for x in item_list])
        prompt_schema = schema_for_prompt(data_type, include_meta_fields=False)
        detail_fields = [k for k, s in meta["fields"].items() if s.get("extract_detail", False)]

        system_prompt = f"""당신은 {type_name} 문서 상세정보 추출 전문가입니다.

규칙:
1) 현재 페이지에 등장하거나 이전 페이지에서 이어지는 {item_name}만 추출
2) 이미지(표) 정보를 OCR 텍스트 보완에 활용하되, OCR 텍스트 이외의 텍스트를 생성하지 말 것.
3) 반드시 JSON만 출력하며, 출력 형식은 아래 예시를 엄격히 따를 것.
4) 상세 필드({", ".join(detail_fields)})는 리스트로 추출
5) 누락된 필드는 빈 문자열("") 또는 빈 리스트([])로 채울 것.
6) 최대한 상세한 내용까지 추출할 것.

출력 형식:
{{
  "{output_key}": [
    {json.dumps(prompt_schema, ensure_ascii=False, indent=2)}
  ]
}}"""

        user_text = f"""## 추출 대상 목록
{item_list_str}

## 이전 페이지({prev_page}) 텍스트
{prev_page_text[:5000] if prev_page_text else "(없음)"}

## 현재 페이지({current_page}) OCR 텍스트
{current_page_text[:10000]}
"""

        if current_page_image is not None:
            b64 = self._image_to_base64(current_page_image)
            msgs = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=[
                    {"type": "text", "text": user_text},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}", "detail": "high"}},
                ]),
            ]
        else:
            msgs = [SystemMessage(content=system_prompt), HumanMessage(content=user_text)]

        resp = await model.ainvoke(msgs)

        try:
            result = parser.parse(resp.content)
            rows = result.get(output_key, [])
            return [normalize_item_keys(data_type, x) for x in rows if isinstance(x, dict)]
        except Exception as e:
            logger.warning(f"페이지 {current_page} 상세 추출 실패: {e}")
            return []


# -----------------------------------------------------------------------------
# 편의 함수 / 별칭
# -----------------------------------------------------------------------------
def create_pipeline(
    db_host: str = "localhost",
    db_port: int = 3306,
    db_name: str = "b2g_data",
    db_user: str = "root",
    db_password: str = "",
    use_table: bool = True,
) -> B2GPipeline:
    return B2GPipeline(
        db_host=db_host,
        db_port=db_port,
        db_name=db_name,
        db_user=db_user,
        db_password=db_password,
        use_table=use_table,
    )


def create_store(
    db_host: str = "localhost",
    db_port: int = 3306,
    db_name: str = "b2g_data",
    db_user: str = "root",
    db_password: str = "",
) -> DynamicMySQLStore:
    return DynamicMySQLStore(
        host=db_host,
        port=db_port,
        database=db_name,
        user=db_user,
        password=db_password,
    )


# 기존 호환 별칭
MySQLStore = DynamicMySQLStore
B2GDataPipeline = B2GPipeline
B2GVectorStore = DynamicMySQLStore
InclusiveGrowthPipeline = B2GPipeline
InclusiveGrowthVectorStore = DynamicMySQLStore
StructuredIndicator = StructuredRecord


# -----------------------------------------------------------------------------
# Example
# -----------------------------------------------------------------------------
async def example_usage():
    pipeline = create_pipeline(
        db_host="localhost",
        db_port=3306,
        db_name="b2g_data",
        db_user="root",
        db_password=""
    )

    # 예시: 추출만
    # records = await pipeline.extract_from_pdf(
    #     pdf_path="/path/to/document.pdf",
    #     data_type="project",
    #     index_pages=(3, 5),
    #     detail_pages=(6, 50),
    #     save_intermediate=True,
    #     output_dir="./output"
    # )

    # 예시: 추출 + 저장
    # records = await pipeline.process_pdf(
    #     pdf_path="/path/to/document.pdf",
    #     data_type="management_eval",
    #     index_pages=(2, 4),
    #     detail_pages=(5, 60),
    # )

    # 예시: 검색
    # results = pipeline.search_by_embedding("project", "인공지능 인재 양성", k=20)
    # for r in results:
    #     print(r["score"], r.get("과제명"))

    print(pipeline.get_stats())


if __name__ == "__main__":
    asyncio.run(example_usage())
