#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
JSON 파일을 MariaDB/MySQL에 로드하는 스크립트 (Schema-first, 임베딩 포함)

사용법:
    python load_json_to_db.py [options]

예시:
    python load_json_to_db.py
    python load_json_to_db.py --type project
    python load_json_to_db.py --type management_eval
    python load_json_to_db.py --type inclusive_growth
    python load_json_to_db.py --reset
"""

import argparse
import json
import os
import sys
import struct
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple

import pymysql
import dotenv
from openai import OpenAI


# =========================================================
# 기본 설정
# =========================================================
DEFAULT_DB_HOST = "localhost"
DEFAULT_DB_PORT = 3306
DEFAULT_DB_NAME = "b2g_data"
DEFAULT_DB_USER = "root"
DEFAULT_DB_PASSWORD = ""

EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_DIM = 1536  # 현재 검증/참고용

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# OpenAI API 키 로드
env_path = os.path.join(SCRIPT_DIR, ".env")
if os.path.exists(env_path):
    OPENAI_KEY = dotenv.get_key(env_path, "OPENAI_KEY")
else:
    OPENAI_KEY = os.environ.get("OPENAI_KEY")

openai_client = None


# =========================================================
# Schema Registry (단일 진실원천)
# =========================================================
from src.db_main import SCHEMA_REGISTRY


# =========================================================
# 공통 유틸
# =========================================================
def get_schema(data_type: str) -> Dict[str, Any]:
    if data_type not in SCHEMA_REGISTRY:
        raise ValueError(f"지원하지 않는 타입: {data_type}")
    return SCHEMA_REGISTRY[data_type]


def get_connection(host: str, port: int, database: str, user: str, password: str):
    return pymysql.connect(
        host=host,
        port=port,
        database=database,
        user=user,
        password=password,
        charset="utf8mb4",
        cursorclass=pymysql.cursors.DictCursor,
    )


def init_openai_client():
    global openai_client
    if openai_client is None:
        if not OPENAI_KEY:
            print("⚠️ OPENAI_KEY 없음. 임베딩 없이 진행합니다.")
            return None
        openai_client = OpenAI(api_key=OPENAI_KEY)
    return openai_client


def get_embeddings_batch(texts: List[str], batch_size: int = 100) -> List[Optional[bytes]]:
    client = init_openai_client()
    if client is None:
        return [None] * len(texts)

    result: List[Optional[bytes]] = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        batch = [t[:8000] if isinstance(t, str) and t.strip() else " " for t in batch]
        try:
            resp = client.embeddings.create(
                model=EMBEDDING_MODEL,
                input=batch
            )
            for item in resp.data:
                emb = item.embedding
                result.append(struct.pack(f"{len(emb)}f", *emb))
        except Exception as e:
            print(f"  ⚠️ 배치 임베딩 실패: {e}")
            result.extend([None] * len(batch))
    return result


def normalize_item_keys(data_type: str, item: Dict[str, Any]) -> Dict[str, Any]:
    """
    alias 처리 + canonical key 정규화 + 타입 정리
    """
    schema = get_schema(data_type)
    aliases = schema.get("aliases", {})
    fields = schema["fields"]

    # 1) alias 매핑
    tmp: Dict[str, Any] = {}
    for k, v in item.items():
        ck = aliases.get(k, k)
        tmp[ck] = v

    # 2) name field 보정 (타 타입 이름필드가 들어온 경우)
    name_field = schema["name_field"]
    alt_name_fields = ["과제명", "지표명", "세부추진과제명"]
    if name_field not in tmp:
        for nk in alt_name_fields:
            if nk in tmp and tmp[nk]:
                tmp[name_field] = tmp[nk]
                break

    # 3) canonical 필드만 유지 + 타입 정리
    out: Dict[str, Any] = {}
    for fname, spec in fields.items():
        t = spec["type"]
        val = tmp.get(fname, None)

        if t == "array":
            if val is None:
                out[fname] = []
            elif isinstance(val, list):
                out[fname] = [str(x).strip() for x in val if str(x).strip()]
            elif isinstance(val, str):
                s = val.strip()
                out[fname] = [s] if s else []
            else:
                out[fname] = [str(val).strip()] if str(val).strip() else []
        elif t == "object":
            if isinstance(val, dict):
                out[fname] = val
            else:
                out[fname] = {}
        else:  # string
            out[fname] = "" if val is None else str(val).strip()

    return out


def serialize_for_db(spec_type: str, value: Any) -> Any:
    """
    DB 저장 전 직렬화:
    - array/object -> JSON 문자열
    - string -> 문자열
    """
    if spec_type == "array":
        if value is None:
            return json.dumps([], ensure_ascii=False)
        if isinstance(value, list):
            return json.dumps(value, ensure_ascii=False)
        if isinstance(value, str):
            s = value.strip()
            return json.dumps([s] if s else [], ensure_ascii=False)
        return json.dumps([str(value)], ensure_ascii=False)

    if spec_type == "object":
        if isinstance(value, dict):
            return json.dumps(value, ensure_ascii=False)
        return json.dumps({}, ensure_ascii=False)

    # string
    return "" if value is None else str(value)


def build_embedding_chunks(data_type: str, normalized_item: Dict[str, Any]) -> List[Dict[str, str]]:
    schema = get_schema(data_type)
    chunks: List[Dict[str, str]] = []

    for fname, spec in schema["fields"].items():
        if not spec.get("extract_detail", False):
            continue
        v = normalized_item.get(fname)

        if isinstance(v, list):
            for x in v:
                s = str(x).strip()
                if s:
                    chunks.append({"field_type": fname, "chunk_text": s})
        elif isinstance(v, str):
            s = v.strip()
            if s:
                chunks.append({"field_type": fname, "chunk_text": s})
        elif v is not None:
            s = str(v).strip()
            if s:
                chunks.append({"field_type": fname, "chunk_text": s})

    return chunks


# =========================================================
# 테이블 생성/리셋 (동적)
# =========================================================
def create_tables(conn):
    with conn.cursor() as cursor:
        # 레거시
        cursor.execute("DROP TABLE IF EXISTS embedding_chunks")

        # 타입별 임베딩/메인 테이블 드롭
        for dt, meta in SCHEMA_REGISTRY.items():
            cursor.execute(f"DROP TABLE IF EXISTS `{meta['embedding_table']}`")
        for dt, meta in SCHEMA_REGISTRY.items():
            cursor.execute(f"DROP TABLE IF EXISTS `{meta['table']}`")

        # 메인 테이블 생성
        for dt, meta in SCHEMA_REGISTRY.items():
            col_defs = ["id INT AUTO_INCREMENT PRIMARY KEY"]
            for fname, spec in meta["fields"].items():
                db_t = spec["db"]
                required = "NOT NULL" if spec.get("required", False) else "NULL"
                col_defs.append(f"`{fname}` {db_t} {required}")

            col_defs.extend([
                "source_document VARCHAR(255)",
                "page_range VARCHAR(50)",
                "extraction_date VARCHAR(50)",
                "created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP"
            ])

            ddl = f"""
                CREATE TABLE `{meta['table']}` (
                    {", ".join(col_defs)}
                ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
            """
            cursor.execute(ddl)

        # 임베딩 테이블 생성
        for dt, meta in SCHEMA_REGISTRY.items():
            cursor.execute(f"""
                CREATE TABLE `{meta['embedding_table']}` (
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
            """)

    conn.commit()
    print("✅ 테이블 생성 완료 (schema-first)")


def reset_tables(conn):
    with conn.cursor() as cursor:
        for _, meta in SCHEMA_REGISTRY.items():
            cursor.execute(f"TRUNCATE TABLE `{meta['embedding_table']}`")
        for _, meta in SCHEMA_REGISTRY.items():
            cursor.execute(f"TRUNCATE TABLE `{meta['table']}`")
    conn.commit()
    print("🗑️ 모든 테이블 데이터 삭제 완료")


# =========================================================
# 로딩 (완전 동적)
# =========================================================
def load_json_file(path: str) -> List[Dict[str, Any]]:
    if not os.path.exists(path):
        print(f"⚠️ 파일 없음: {path}")
        return []
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        print(f"⚠️ JSON 루트가 list가 아님: {path}")
        return []
    return data


def add_embedding_chunks(cursor, data_type: str, source_id: int, item_name: str, chunks: List[Dict[str, str]]) -> int:
    if not item_name or not chunks:
        return 0

    schema = get_schema(data_type)
    emb_table = schema["embedding_table"]

    texts = [c["chunk_text"] for c in chunks]
    embeddings = get_embeddings_batch(texts)

    for c, emb in zip(chunks, embeddings):
        cursor.execute(f"""
            INSERT INTO `{emb_table}` (source_id, item_name, field_type, chunk_text, embedding)
            VALUES (%s, %s, %s, %s, %s)
        """, (
            source_id,
            item_name,
            c["field_type"],
            c["chunk_text"],
            emb
        ))

    return len(chunks)


def load_by_type(conn, data_type: str) -> Tuple[int, int]:
    """
    return: (loaded_count, chunk_count)
    """
    schema = get_schema(data_type)
    data = load_json_file(schema["json_path"])
    if not data:
        print(f"⚠️ {data_type} 데이터가 비어있습니다")
        return 0, 0

    table = schema["table"]
    name_field = schema["name_field"]
    fields = schema["fields"]

    loaded = 0
    total_chunks = 0

    with conn.cursor() as cursor:
        for raw_item in data:
            try:
                item = normalize_item_keys(data_type, raw_item)

                # 메인 INSERT 동적 생성
                field_names = list(fields.keys())
                db_columns = ", ".join([f"`{c}`" for c in field_names] + ["source_document", "page_range", "extraction_date"])
                placeholders = ", ".join(["%s"] * (len(field_names) + 3))

                field_values = [
                    serialize_for_db(fields[c]["type"], item.get(c))
                    for c in field_names
                ]

                source_document = raw_item.get("source_document", schema["json_path"])
                page_range = raw_item.get("page_range", "")
                extraction_date = raw_item.get("extraction_date", datetime.now().isoformat())

                cursor.execute(
                    f"INSERT INTO `{table}` ({db_columns}) VALUES ({placeholders})",
                    field_values + [source_document, page_range, extraction_date]
                )
                source_id = cursor.lastrowid

                # 임베딩 청크
                chunks = build_embedding_chunks(data_type, item)
                chunks_added = add_embedding_chunks(
                    cursor=cursor,
                    data_type=data_type,
                    source_id=source_id,
                    item_name=item.get(name_field, ""),
                    chunks=chunks
                )
                total_chunks += chunks_added
                loaded += 1

            except Exception as e:
                title = raw_item.get(name_field) or raw_item.get("과제명") or raw_item.get("지표명") or raw_item.get("세부추진과제명") or "Unknown"
                print(f"  ❌ 오류: {str(title)[:30]} - {e}")

    conn.commit()
    return loaded, total_chunks


# =========================================================
# 통계
# =========================================================
def show_stats(conn):
    print()
    print("=" * 70)
    print("📊 DB 현황")
    print("=" * 70)

    total_items = 0
    total_chunks = 0

    with conn.cursor() as cursor:
        for dt, meta in SCHEMA_REGISTRY.items():
            cursor.execute(f"SELECT COUNT(*) AS cnt FROM `{meta['table']}`")
            item_cnt = cursor.fetchone()["cnt"]

            cursor.execute(f"SELECT COUNT(*) AS cnt FROM `{meta['embedding_table']}`")
            chunk_cnt = cursor.fetchone()["cnt"]

            total_items += item_cnt
            total_chunks += chunk_cnt

            label = meta.get("type_display", dt)
            print(f"  {label:<12}: {item_cnt:>5}개 (임베딩 청크: {chunk_cnt}개)")

    print(f"  {'─' * 55}")
    print(f"  총 항목      : {total_items:>5}개")
    print(f"  총 임베딩청크: {total_chunks:>5}개")
    print("=" * 70)


# =========================================================
# CLI
# =========================================================
def main():
    parser = argparse.ArgumentParser(
        description="JSON 파일을 MariaDB/MySQL에 로드 (schema-first)"
    )

    parser.add_argument(
        "--type", "-t",
        choices=["all"] + list(SCHEMA_REGISTRY.keys()),
        default="all",
        help="로드할 데이터 타입"
    )
    parser.add_argument("--reset", "-r", action="store_true", help="기존 데이터 TRUNCATE 후 로드")
    parser.add_argument("--db-host", default=DEFAULT_DB_HOST, help="DB 호스트")
    parser.add_argument("--db-port", type=int, default=DEFAULT_DB_PORT, help="DB 포트")
    parser.add_argument("--db-name", default=DEFAULT_DB_NAME, help="DB 이름")
    parser.add_argument("--db-user", default=DEFAULT_DB_USER, help="DB 사용자")
    parser.add_argument("--db-password", default=DEFAULT_DB_PASSWORD, help="DB 비밀번호")

    args = parser.parse_args()

    print("=" * 60)
    print("🚀 JSON → MariaDB 로드 시작 (schema-first)")
    print("=" * 60)
    print(f"  DB: {args.db_user}@{args.db_host}:{args.db_port}/{args.db_name}")
    print(f"  타입: {args.type}")
    print(f"  초기화: {'예' if args.reset else '아니오'}")
    print("=" * 60)

    try:
        conn = get_connection(
            host=args.db_host,
            port=args.db_port,
            database=args.db_name,
            user=args.db_user,
            password=args.db_password,
        )
        print("✅ DB 연결 성공")
    except Exception as e:
        print(f"❌ DB 연결 실패: {e}")
        sys.exit(1)

    try:
        # 항상 registry 기준으로 테이블 생성
        create_tables(conn)

        if args.reset:
            reset_tables(conn)

        print()
        total_loaded = 0
        total_chunks = 0

        targets = list(SCHEMA_REGISTRY.keys()) if args.type == "all" else [args.type]

        for dt in targets:
            meta = get_schema(dt)
            print(f"📥 {meta.get('type_display', dt)} 로드 중... ({meta['json_path']})")
            loaded, chunks = load_by_type(conn, dt)
            print(f"   → {loaded}개 로드 완료 / 임베딩 청크 {chunks}개")
            total_loaded += loaded
            total_chunks += chunks

        print()
        print(f"✅ 총 {total_loaded}개 항목 로드 완료")
        print(f"✅ 총 {total_chunks}개 임베딩 청크 생성 완료")

        show_stats(conn)

    finally:
        conn.close()


if __name__ == "__main__":
    main()
