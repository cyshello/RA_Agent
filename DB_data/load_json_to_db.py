#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
JSON 파일을 MariaDB/MySQL에 로드하는 스크립트 (임베딩 포함)

사용법:
    python load_json_to_db.py [options]
    
예시:
    # 전체 로드 (임베딩 포함)
    python load_json_to_db.py
    
    # 특정 타입만 로드
    python load_json_to_db.py --type project
    python load_json_to_db.py --type management
    python load_json_to_db.py --type inclusive
    
    # DB 초기화 후 로드
    python load_json_to_db.py --reset
"""

import argparse
import json
import os
import sys
from datetime import datetime
from typing import List, Dict, Any

import pymysql
import dotenv
from openai import OpenAI

# 기본 설정
DEFAULT_DB_HOST = "localhost"
DEFAULT_DB_PORT = 3306
DEFAULT_DB_NAME = "b2g_data"
DEFAULT_DB_USER = "root"
DEFAULT_DB_PASSWORD = ""

# 임베딩 설정
EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_DIM = 1536

# JSON 파일 경로 (이 스크립트와 같은 디렉토리)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_JSON = os.path.join(SCRIPT_DIR, "project.json")
MANAGEMENT_JSON = os.path.join(SCRIPT_DIR, "management.json")
INCLUSIVE_JSON = os.path.join(SCRIPT_DIR, "inclusive.json")

# OpenAI API 키 로드
env_path = os.path.join(os.path.dirname(SCRIPT_DIR), "src", ".env")
if os.path.exists(env_path):
    OPENAI_KEY = dotenv.get_key(env_path, "OPENAI_KEY")
else:
    OPENAI_KEY = os.environ.get("OPENAI_API_KEY")

# OpenAI 클라이언트
openai_client = None


def get_connection(host: str, port: int, database: str, user: str, password: str):
    """MySQL/MariaDB 연결 생성"""
    return pymysql.connect(
        host=host,
        port=port,
        database=database,
        user=user,
        password=password,
        charset='utf8mb4',
        cursorclass=pymysql.cursors.DictCursor
    )


def create_tables(conn):
    """테이블 생성 (삭제 후 재생성) - 임베딩 컬럼 포함"""
    with conn.cursor() as cursor:
        # 기존 테이블 삭제 후 재생성
        cursor.execute("DROP TABLE IF EXISTS national_projects")
        cursor.execute("DROP TABLE IF EXISTS management_evals")
        cursor.execute("DROP TABLE IF EXISTS inclusive_growth")
        
        # 국정과제 테이블 (임베딩 포함)
        cursor.execute(f"""
            CREATE TABLE national_projects (
                id INT AUTO_INCREMENT PRIMARY KEY,
                과제명 VARCHAR(500) NOT NULL,
                과제번호 VARCHAR(50),
                과제_목표 LONGTEXT,
                주요내용 LONGTEXT,
                기대효과 LONGTEXT,
                source_document VARCHAR(255),
                page_range VARCHAR(50),
                extraction_date VARCHAR(50),
                embedding_text LONGTEXT,
                embedding BLOB,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
        """)
        
        # FULLTEXT 인덱스 별도 생성
        cursor.execute("""
            ALTER TABLE national_projects 
            ADD FULLTEXT INDEX ft_project (과제명, 과제_목표, 주요내용, 기대효과) WITH PARSER ngram
        """)
        
        # 경영평가 테이블 (임베딩 포함)
        cursor.execute(f"""
            CREATE TABLE management_evals (
                id INT AUTO_INCREMENT PRIMARY KEY,
                지표명 VARCHAR(500) NOT NULL,
                평가기준 LONGTEXT,
                평가방법 LONGTEXT,
                참고사항 LONGTEXT,
                증빙자료 LONGTEXT,
                source_document VARCHAR(255),
                page_range VARCHAR(50),
                extraction_date VARCHAR(50),
                embedding_text LONGTEXT,
                embedding BLOB,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
        """)
        
        cursor.execute("""
            ALTER TABLE management_evals 
            ADD FULLTEXT INDEX ft_management (지표명, 평가기준, 평가방법, 참고사항) WITH PARSER ngram
        """)
        
        # 동반성장 테이블 (임베딩 포함)
        cursor.execute(f"""
            CREATE TABLE inclusive_growth (
                id INT AUTO_INCREMENT PRIMARY KEY,
                지표명 VARCHAR(500) NOT NULL,
                평가기준 LONGTEXT,
                평가방법 LONGTEXT,
                참고사항 LONGTEXT,
                증빙자료 LONGTEXT,
                source_document VARCHAR(255),
                page_range VARCHAR(50),
                extraction_date VARCHAR(50),
                embedding_text LONGTEXT,
                embedding BLOB,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
        """)
        
        cursor.execute("""
            ALTER TABLE inclusive_growth 
            ADD FULLTEXT INDEX ft_inclusive (지표명, 평가기준, 평가방법, 참고사항) WITH PARSER ngram
        """)
        
    conn.commit()
    print("✅ 테이블 생성 완료 (임베딩 컬럼 포함)")


def reset_tables(conn):
    """테이블 데이터만 초기화 (테이블 유지)"""
    with conn.cursor() as cursor:
        cursor.execute("TRUNCATE TABLE national_projects")
        cursor.execute("TRUNCATE TABLE management_evals")
        cursor.execute("TRUNCATE TABLE inclusive_growth")
    conn.commit()
    print("🗑️  모든 테이블 데이터 삭제 완료")


def init_openai_client():
    """OpenAI 클라이언트 초기화"""
    global openai_client
    if openai_client is None:
        if not OPENAI_KEY:
            print("⚠️  OpenAI API 키가 없습니다. 임베딩 없이 진행합니다.")
            return None
        openai_client = OpenAI(api_key=OPENAI_KEY)
    return openai_client


def get_embedding(text: str) -> bytes:
    """텍스트의 임베딩 벡터를 생성하여 바이너리로 반환"""
    import struct
    
    client = init_openai_client()
    if client is None or not text.strip():
        return None
    
    try:
        response = client.embeddings.create(
            model=EMBEDDING_MODEL,
            input=text[:8000]  # 토큰 제한
        )
        embedding = response.data[0].embedding
        # float 리스트를 바이너리로 변환
        return struct.pack(f'{len(embedding)}f', *embedding)
    except Exception as e:
        print(f"  ⚠️ 임베딩 생성 실패: {e}")
        return None


def get_embeddings_batch(texts: List[str], batch_size: int = 100) -> List[bytes]:
    """배치로 임베딩 생성"""
    import struct
    
    client = init_openai_client()
    if client is None:
        return [None] * len(texts)
    
    results = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        # 빈 텍스트 처리
        batch = [t[:8000] if t.strip() else " " for t in batch]
        
        try:
            response = client.embeddings.create(
                model=EMBEDDING_MODEL,
                input=batch
            )
            for item in response.data:
                embedding = item.embedding
                results.append(struct.pack(f'{len(embedding)}f', *embedding))
        except Exception as e:
            print(f"  ⚠️ 배치 임베딩 실패: {e}")
            results.extend([None] * len(batch))
    
    return results


def list_to_text(items: List) -> str:
    """리스트를 텍스트로 변환"""
    if not items:
        return ""
    if isinstance(items, list):
        return "\n".join(str(item) for item in items if item)
    return str(items)


def load_projects(conn, json_path: str) -> int:
    """국정과제 데이터 로드 (임베딩 포함)"""
    if not os.path.exists(json_path):
        print(f"⚠️  파일 없음: {json_path}")
        return 0
    
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    if not data:
        print("⚠️  국정과제 데이터가 비어있습니다")
        return 0
    
    # 임베딩용 텍스트 준비
    print("   📊 임베딩 생성 중...")
    embedding_texts = []
    for item in data:
        과제명 = item.get('과제명', '')
        과제목표 = list_to_text(item.get('과제 목표', []))
        주요내용 = list_to_text(item.get('주요내용', []))
        기대효과 = list_to_text(item.get('기대효과', []))
        
        # 임베딩용 통합 텍스트
        embed_text = f"과제명: {과제명}\n목표: {과제목표}\n주요내용: {주요내용}\n기대효과: {기대효과}"
        embedding_texts.append(embed_text)
    
    # 배치 임베딩 생성
    embeddings = get_embeddings_batch(embedding_texts)
    
    count = 0
    with conn.cursor() as cursor:
        for i, item in enumerate(data):
            try:
                과제명 = item.get('과제명', '')
                과제목표 = list_to_text(item.get('과제 목표', []))
                주요내용 = list_to_text(item.get('주요내용', []))
                기대효과 = list_to_text(item.get('기대효과', []))
                
                cursor.execute("""
                    INSERT INTO national_projects 
                    (과제명, 과제번호, 과제_목표, 주요내용, 기대효과, source_document, page_range, extraction_date, embedding_text, embedding)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                """, (
                    과제명,
                    item.get('과제번호', ''),
                    과제목표,
                    주요내용,
                    기대효과,
                    item.get('source_document', 'project.json'),
                    item.get('page_range', ''),
                    item.get('extraction_date', datetime.now().isoformat()),
                    embedding_texts[i],
                    embeddings[i]
                ))
                count += 1
            except Exception as e:
                print(f"  ❌ 오류: {item.get('과제명', 'Unknown')[:30]} - {e}")
    
    conn.commit()
    return count


def load_management(conn, json_path: str) -> int:
    """경영평가 데이터 로드 (임베딩 포함)"""
    if not os.path.exists(json_path):
        print(f"⚠️  파일 없음: {json_path}")
        return 0
    
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    if not data:
        print("⚠️  경영평가 데이터가 비어있습니다")
        return 0
    
    # 임베딩용 텍스트 준비
    print("   📊 임베딩 생성 중...")
    embedding_texts = []
    for item in data:
        지표명 = item.get('지표명', '')
        평가기준 = list_to_text(item.get('평가기준', []))
        평가방법 = list_to_text(item.get('평가방법', []))
        참고사항 = list_to_text(item.get('참고사항', []))
        
        embed_text = f"지표명: {지표명}\n평가기준: {평가기준}\n평가방법: {평가방법}\n참고사항: {참고사항}"
        embedding_texts.append(embed_text)
    
    # 배치 임베딩 생성
    embeddings = get_embeddings_batch(embedding_texts)
    
    count = 0
    with conn.cursor() as cursor:
        for i, item in enumerate(data):
            try:
                지표명 = item.get('지표명', '')
                평가기준 = list_to_text(item.get('평가기준', []))
                평가방법 = list_to_text(item.get('평가방법', []))
                참고사항 = list_to_text(item.get('참고사항', []))
                증빙자료 = list_to_text(item.get('증빙자료', []))
                
                cursor.execute("""
                    INSERT INTO management_evals 
                    (지표명, 평가기준, 평가방법, 참고사항, 증빙자료, source_document, page_range, extraction_date, embedding_text, embedding)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                """, (
                    지표명,
                    평가기준,
                    평가방법,
                    참고사항,
                    증빙자료,
                    item.get('source_document', 'management.json'),
                    item.get('page_range', ''),
                    item.get('extraction_date', datetime.now().isoformat()),
                    embedding_texts[i],
                    embeddings[i]
                ))
                count += 1
            except Exception as e:
                print(f"  ❌ 오류: {item.get('지표명', 'Unknown')[:30]} - {e}")
    
    conn.commit()
    return count


def load_inclusive(conn, json_path: str) -> int:
    """동반성장 데이터 로드 (임베딩 포함)"""
    if not os.path.exists(json_path):
        print(f"⚠️  파일 없음: {json_path}")
        return 0
    
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    if not data:
        print("⚠️  동반성장 데이터가 비어있습니다")
        return 0
    
    # 임베딩용 텍스트 준비
    print("   📊 임베딩 생성 중...")
    embedding_texts = []
    for item in data:
        지표명 = item.get('지표명', '')
        평가기준 = list_to_text(item.get('평가기준', []))
        평가방법 = list_to_text(item.get('평가방법', []))
        참고사항 = list_to_text(item.get('참고사항', []))
        
        embed_text = f"지표명: {지표명}\n평가기준: {평가기준}\n평가방법: {평가방법}\n참고사항: {참고사항}"
        embedding_texts.append(embed_text)
    
    # 배치 임베딩 생성
    embeddings = get_embeddings_batch(embedding_texts)
    
    count = 0
    with conn.cursor() as cursor:
        for i, item in enumerate(data):
            try:
                지표명 = item.get('지표명', '')
                평가기준 = list_to_text(item.get('평가기준', []))
                평가방법 = list_to_text(item.get('평가방법', []))
                참고사항 = list_to_text(item.get('참고사항', []))
                증빙자료 = list_to_text(item.get('증빙자료', []))
                
                cursor.execute("""
                    INSERT INTO inclusive_growth 
                    (지표명, 평가기준, 평가방법, 참고사항, 증빙자료, source_document, page_range, extraction_date, embedding_text, embedding)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                """, (
                    지표명,
                    평가기준,
                    평가방법,
                    참고사항,
                    증빙자료,
                    item.get('source_document', 'inclusive.json'),
                    item.get('page_range', ''),
                    item.get('extraction_date', datetime.now().isoformat()),
                    embedding_texts[i],
                    embeddings[i]
                ))
                count += 1
            except Exception as e:
                print(f"  ❌ 오류: {item.get('지표명', 'Unknown')[:30]} - {e}")
    
    conn.commit()
    return count


def show_stats(conn):
    """DB 통계 출력"""
    with conn.cursor() as cursor:
        cursor.execute("SELECT COUNT(*) as cnt FROM national_projects")
        projects = cursor.fetchone()['cnt']
        
        cursor.execute("SELECT COUNT(*) as cnt FROM management_evals")
        management = cursor.fetchone()['cnt']
        
        cursor.execute("SELECT COUNT(*) as cnt FROM inclusive_growth")
        inclusive = cursor.fetchone()['cnt']
        
        # 임베딩 있는 항목 수 확인
        cursor.execute("SELECT COUNT(*) as cnt FROM national_projects WHERE embedding IS NOT NULL")
        projects_embed = cursor.fetchone()['cnt']
        
        cursor.execute("SELECT COUNT(*) as cnt FROM management_evals WHERE embedding IS NOT NULL")
        management_embed = cursor.fetchone()['cnt']
        
        cursor.execute("SELECT COUNT(*) as cnt FROM inclusive_growth WHERE embedding IS NOT NULL")
        inclusive_embed = cursor.fetchone()['cnt']
    
    print()
    print("=" * 50)
    print("📊 DB 현황")
    print("=" * 50)
    print(f"  국정과제:    {projects:>5}개 (임베딩: {projects_embed}개)")
    print(f"  경영평가:    {management:>5}개 (임베딩: {management_embed}개)")
    print(f"  동반성장:    {inclusive:>5}개 (임베딩: {inclusive_embed}개)")
    print(f"  ─────────────────────")
    print(f"  총계:        {projects + management + inclusive:>5}개")
    print("=" * 50)


def main():
    parser = argparse.ArgumentParser(
        description='JSON 파일을 MariaDB/MySQL에 로드',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
예시:
  python load_json_to_db.py                    # 전체 로드
  python load_json_to_db.py --type project     # 국정과제만 로드
  python load_json_to_db.py --type management  # 경영평가만 로드
  python load_json_to_db.py --type inclusive   # 동반성장만 로드
  python load_json_to_db.py --reset            # DB 초기화 후 로드
        """
    )
    
    parser.add_argument('--type', '-t', choices=['project', 'management', 'inclusive', 'all'],
                        default='all', help='로드할 데이터 타입 (기본: all)')
    parser.add_argument('--reset', '-r', action='store_true', help='기존 데이터 삭제 후 로드')
    parser.add_argument('--db-host', default=DEFAULT_DB_HOST, help='DB 호스트')
    parser.add_argument('--db-port', type=int, default=DEFAULT_DB_PORT, help='DB 포트')
    parser.add_argument('--db-name', default=DEFAULT_DB_NAME, help='DB 이름')
    parser.add_argument('--db-user', default=DEFAULT_DB_USER, help='DB 사용자')
    parser.add_argument('--db-password', default=DEFAULT_DB_PASSWORD, help='DB 비밀번호')
    
    args = parser.parse_args()
    
    print("=" * 50)
    print("🚀 JSON → MariaDB 로드 시작")
    print("=" * 50)
    print(f"  DB: {args.db_user}@{args.db_host}:{args.db_port}/{args.db_name}")
    print(f"  타입: {args.type}")
    print(f"  초기화: {'예' if args.reset else '아니오'}")
    print("=" * 50)
    
    try:
        conn = get_connection(
            host=args.db_host,
            port=args.db_port,
            database=args.db_name,
            user=args.db_user,
            password=args.db_password
        )
        print("✅ DB 연결 성공")
    except Exception as e:
        print(f"❌ DB 연결 실패: {e}")
        sys.exit(1)
    
    try:
        # 테이블 생성
        create_tables(conn)
        
        # 초기화
        if args.reset:
            reset_tables(conn)
        
        # 데이터 로드
        print()
        total = 0
        
        if args.type in ['project', 'all']:
            print(f"📥 국정과제 로드 중... ({PROJECT_JSON})")
            count = load_projects(conn, PROJECT_JSON)
            print(f"   → {count}개 로드 완료")
            total += count
        
        if args.type in ['management', 'all']:
            print(f"📥 경영평가 로드 중... ({MANAGEMENT_JSON})")
            count = load_management(conn, MANAGEMENT_JSON)
            print(f"   → {count}개 로드 완료")
            total += count
        
        if args.type in ['inclusive', 'all']:
            print(f"📥 동반성장 로드 중... ({INCLUSIVE_JSON})")
            count = load_inclusive(conn, INCLUSIVE_JSON)
            print(f"   → {count}개 로드 완료")
            total += count
        
        print()
        print(f"✅ 총 {total}개 항목 로드 완료")
        
        # 통계 출력
        show_stats(conn)
        
    finally:
        conn.close()


if __name__ == '__main__':
    main()
