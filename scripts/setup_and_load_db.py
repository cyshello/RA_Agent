#!/usr/bin/env python3
"""
B2G 기준데이터 DB 구축 스크립트

PostgreSQL DB 생성부터 PDF 처리 및 벡터 저장소 저장까지 수행합니다.

사용법:
    python scripts/setup_and_load_db.py
"""

import os
import sys
import asyncio
import subprocess
import logging
from pathlib import Path

# 프로젝트 루트를 path에 추가
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from dotenv import load_dotenv, find_dotenv

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ============================================================================
# 설정
# ============================================================================

# PDF 파일 경로
PDF_PATH = "/Users/youngseocho/Desktop/AX/RA_Agent/data/presidential_agenda.pdf"

# 중간 결과 저장 디렉토리
OUTPUT_DIR = "/Users/youngseocho/Desktop/AX/RA_Agent/data/output"

# PostgreSQL 설정
DB_CONFIG = {
    "host": "localhost",
    "port": 5432,
    "database": "b2g_data",
    "user": "youngseocho",  # macOS Homebrew PostgreSQL 기본 사용자
    "password": "",  # 비밀번호가 없으면 빈 문자열 사용
}

# 벡터 컬렉션 이름
COLLECTION_NAME = "b2g_projects"


# ============================================================================
# PostgreSQL 설정 함수
# ============================================================================

def check_postgres_connection():
    """PostgreSQL 연결 확인"""
    import psycopg2
    
    try:
        conn = psycopg2.connect(
            host=DB_CONFIG["host"],
            port=DB_CONFIG["port"],
            user=DB_CONFIG["user"],
            password=DB_CONFIG["password"],
            database="postgres"  # 기본 DB로 연결
        )
        conn.close()
        logger.info("✅ PostgreSQL 연결 성공")
        return True
    except Exception as e:
        logger.error(f"❌ PostgreSQL 연결 실패: {e}")
        return False


def create_database():
    """데이터베이스 생성"""
    import psycopg2
    from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT
    
    try:
        # postgres DB에 연결
        conn = psycopg2.connect(
            host=DB_CONFIG["host"],
            port=DB_CONFIG["port"],
            user=DB_CONFIG["user"],
            password=DB_CONFIG["password"],
            database="postgres"
        )
        conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
        cursor = conn.cursor()
        
        # DB 존재 여부 확인
        cursor.execute(
            "SELECT 1 FROM pg_database WHERE datname = %s",
            (DB_CONFIG["database"],)
        )
        
        if cursor.fetchone():
            logger.info(f"✅ 데이터베이스 '{DB_CONFIG['database']}' 이미 존재함")
        else:
            cursor.execute(f"CREATE DATABASE {DB_CONFIG['database']}")
            logger.info(f"✅ 데이터베이스 '{DB_CONFIG['database']}' 생성 완료")
        
        cursor.close()
        conn.close()
        return True
        
    except Exception as e:
        logger.error(f"❌ 데이터베이스 생성 실패: {e}")
        return False


def setup_pgvector():
    """pgvector 확장 활성화"""
    import psycopg2
    
    try:
        conn = psycopg2.connect(
            host=DB_CONFIG["host"],
            port=DB_CONFIG["port"],
            user=DB_CONFIG["user"],
            password=DB_CONFIG["password"],
            database=DB_CONFIG["database"]
        )
        cursor = conn.cursor()
        
        # pgvector 확장 활성화
        cursor.execute("CREATE EXTENSION IF NOT EXISTS vector")
        conn.commit()
        
        logger.info("✅ pgvector 확장 활성화 완료")
        
        cursor.close()
        conn.close()
        return True
        
    except Exception as e:
        logger.error(f"❌ pgvector 확장 활성화 실패: {e}")
        logger.error("pgvector가 설치되어 있는지 확인하세요.")
        return False


# ============================================================================
# PDF 처리 함수
# ============================================================================

async def process_pdf():
    """PDF를 처리하여 벡터 저장소에 저장"""
    from src.db import create_pipeline
    
    # PDF 파일 존재 확인
    if not os.path.exists(PDF_PATH):
        logger.error(f"❌ PDF 파일을 찾을 수 없습니다: {PDF_PATH}")
        return False
    
    logger.info(f"📄 PDF 파일: {PDF_PATH}")
    
    # 출력 디렉토리 생성
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    logger.info(f"📁 출력 디렉토리: {OUTPUT_DIR}")
    
    # Connection string 생성
    connection_string = (
        f"postgresql://{DB_CONFIG['user']}:{DB_CONFIG['password']}"
        f"@{DB_CONFIG['host']}:{DB_CONFIG['port']}/{DB_CONFIG['database']}"
    )
    
    try:
        # 파이프라인 생성
        logger.info("🔧 파이프라인 초기화 중...")
        from src.db import B2GDataPipeline
        
        pipeline = B2GDataPipeline(
            connection_string=connection_string,
            collection_name=COLLECTION_NAME,
            model_provider="openai",
            extraction_model="gpt-4o-mini",
            structuring_model="gpt-4o-mini",  # 비용 절감을 위해 mini 사용
            max_rps=2.0
        )
        
        # PDF 처리
        logger.info("🚀 PDF 처리 시작...")
        projects = await pipeline.process_pdf(
            pdf_path=PDF_PATH,
            save_intermediate=True,
            output_dir=OUTPUT_DIR
        )
        
        logger.info(f"✅ PDF 처리 완료: {len(projects)}개 과제 저장됨")
        
        # 결과 요약 출력
        print("\n" + "="*60)
        print("📊 처리 결과 요약")
        print("="*60)
        for i, p in enumerate(projects, 1):
            print(f"\n{i}. [{p.과제번호}] {p.과제명}")
            print(f"   - 과제 목표: {len(p.과제_목표)}개")
            print(f"   - 주요내용: {len(p.주요내용)}개")
            print(f"   - 기대효과: {len(p.기대효과)}개")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ PDF 처리 실패: {e}")
        import traceback
        traceback.print_exc()
        return False


# ============================================================================
# 테스트 검색 함수
# ============================================================================

def test_search():
    """검색 테스트"""
    from src.db import create_vector_store
    
    connection_string = (
        f"postgresql://{DB_CONFIG['user']}:{DB_CONFIG['password']}"
        f"@{DB_CONFIG['host']}:{DB_CONFIG['port']}/{DB_CONFIG['database']}"
    )
    
    try:
        vector_store = create_vector_store(
            db_host=DB_CONFIG["host"],
            db_port=DB_CONFIG["port"],
            db_name=DB_CONFIG["database"],
            db_user=DB_CONFIG["user"],
            db_password=DB_CONFIG["password"],
            collection_name=COLLECTION_NAME
        )
        
        # 테스트 검색
        test_queries = [
            "인공지능 기술",
            "탄소중립",
            "디지털 전환"
        ]
        
        print("\n" + "="*60)
        print("🔍 검색 테스트")
        print("="*60)
        
        for query in test_queries:
            print(f"\n검색어: '{query}'")
            print("-"*40)
            
            results = vector_store.search_unique_projects(
                query=query,
                k=3
            )
            
            if results:
                for i, r in enumerate(results, 1):
                    print(f"  {i}. [{r['score']:.3f}] {r['과제명']}")
                    print(f"     과제번호: {r['과제번호']}")
                    print(f"     매칭 필드: {r['matched_field']}")
            else:
                print("  결과 없음")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ 검색 테스트 실패: {e}")
        return False


# ============================================================================
# 메인 함수
# ============================================================================

async def main():
    """메인 실행 함수"""
    print("="*60)
    print("🚀 B2G 기준데이터 DB 구축 스크립트")
    print("="*60)
    
    # .env 파일 로드 (src/.env 경로 지정)
    env_path = project_root / "src" / ".env"
    if env_path.exists():
        load_dotenv(env_path)
        logger.info(f"✅ .env 파일 로드: {env_path}")
    else:
        load_dotenv(find_dotenv())
    
    # OpenAI API 키 설정
    openai_key = os.getenv("OPENAI_KEY")
    if openai_key:
        os.environ["OPENAI_API_KEY"] = openai_key
        logger.info("✅ OpenAI API 키 설정 완료")
    
    # CLOVA API 키 설정 (환경변수로도 접근 가능하게)
    clova_api_url = os.getenv("CLOVA_api_url")
    clova_secret_key = os.getenv("CLOVA_secret_key")
    if clova_api_url and clova_secret_key:
        os.environ["CLOVA_api_url"] = clova_api_url
        os.environ["CLOVA_secret_key"] = clova_secret_key
        logger.info("✅ CLOVA API 키 설정 완료")
    else:
        logger.warning("⚠️ CLOVA API 키가 설정되지 않았습니다. OCR이 동작하지 않을 수 있습니다.")
    
    # DB 비밀번호가 비어있으면 환경변수에서 로드
    if not DB_CONFIG["password"]:
        DB_CONFIG["password"] = os.getenv("POSTGRES_PASSWORD", "")
    
    # Step 1: PostgreSQL 연결 확인
    print("\n📌 Step 1: PostgreSQL 연결 확인")
    if not check_postgres_connection():
        print("\nPostgreSQL이 실행 중인지 확인하세요.")
        print("macOS: brew services start postgresql")
        return
    
    # Step 2: 데이터베이스 생성
    print("\n📌 Step 2: 데이터베이스 생성")
    if not create_database():
        return
    
    # Step 3: pgvector 확장 활성화
    print("\n📌 Step 3: pgvector 확장 활성화")
    if not setup_pgvector():
        return
    
    # Step 4: PDF 처리 및 저장
    print("\n📌 Step 4: PDF 처리 및 벡터 저장소 저장")
    if not await process_pdf():
        return
    
    # Step 5: 검색 테스트
    print("\n📌 Step 5: 검색 테스트")
    test_search()
    
    print("\n" + "="*60)
    print("✅ DB 구축 완료!")
    print("="*60)


if __name__ == "__main__":
    asyncio.run(main())
