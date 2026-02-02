#!/usr/bin/env python3
"""
동반성장 평가지표 PDF 처리 스크립트 (2단계 처리)

사용 방법:
    # 2단계 처리: 목록 페이지 + 세부 페이지
    python scripts/process_inclusive_growth.py --index-start 5 --index-end 7 --detail-start 15 --detail-end 72 --clear

    # 레거시: 단일 범위 처리
    python scripts/process_inclusive_growth.py --start 15 --end 72 --reuse-ocr --clear

인자:
    --index-start: 목록(목차) 시작 페이지
    --index-end: 목록(목차) 끝 페이지
    --detail-start: 세부 내용 시작 페이지
    --detail-end: 세부 내용 끝 페이지
    --clear: 기존 컬렉션 삭제 후 시작
    --save-intermediate: 중간 결과 저장
    
    (레거시 옵션)
    --start: 시작 페이지 (단일 범위)
    --end: 끝 페이지 (단일 범위)
    --reuse-ocr: 기존 OCR 결과 재사용
"""

import os
import sys
import argparse
import asyncio
import logging

# 상위 디렉토리를 path에 추가
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.db import (
    create_inclusive_growth_pipeline,
    create_inclusive_growth_vector_store
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 기본 설정
DEFAULT_PDF_PATH = "/Users/youngseocho/Desktop/AX/RA_Agent/data/inclusive_growth.pdf"
OUTPUT_DIR = "/Users/youngseocho/Desktop/AX/RA_Agent/data/output_inclusive_growth"
CONNECTION_STRING = "postgresql://youngseocho:@localhost:5432/b2g_data"


async def process_two_stage(
    index_start: int,
    index_end: int,
    detail_start: int,
    detail_end: int,
    save_intermediate: bool = False,
    clear_collection: bool = False
):
    """
    2단계 처리: 목록 페이지에서 지표명 추출 → 세부 페이지에서 상세 정보 채우기
    """
    logger.info("="*60)
    logger.info("2단계 처리 시작")
    logger.info(f"  - 목록 페이지: {index_start} ~ {index_end}")
    logger.info(f"  - 세부 페이지: {detail_start} ~ {detail_end}")
    logger.info("="*60)
    
    # 기존 컬렉션 삭제 (옵션)
    if clear_collection:
        logger.info("기존 컬렉션 삭제 중...")
        vs = create_inclusive_growth_vector_store(
            db_user='youngseocho',
            db_password=''
        )
        vs.delete_collection()
        logger.info("✅ 기존 컬렉션 삭제 완료")
    
    # 파이프라인 생성
    pipeline = create_inclusive_growth_pipeline(
        db_user='youngseocho',
        db_password=''
    )
    
    # 출력 디렉토리 생성
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # 2단계 처리 실행
    indicators = await pipeline.process_two_stage(
        index_start=index_start,
        index_end=index_end,
        detail_start=detail_start,
        detail_end=detail_end,
        save_intermediate=save_intermediate,
        output_dir=OUTPUT_DIR,
        reuse_ocr_dir=OUTPUT_DIR,  # 기존 OCR 재사용
        source_document="inclusive_growth.pdf"
    )
    
    logger.info(f"✅ 처리 완료: {len(indicators)}개 평가지표 저장됨")
    
    # 결과 요약 출력
    print_results(indicators)
    
    return indicators


async def process_pdf_by_range(
    pdf_path: str,
    start_page: int,
    end_page: int,
    save_intermediate: bool = False,
    clear_collection: bool = False,
    reuse_ocr: bool = False
):
    """
    레거시: 단일 페이지 범위로 PDF 처리
    """
    logger.info(f"PDF 처리 시작: {pdf_path}")
    logger.info(f"페이지 범위: {start_page} ~ {end_page}")
    if reuse_ocr:
        logger.info(f"기존 OCR 결과 재사용: {OUTPUT_DIR}")
    
    # 기존 컬렉션 삭제 (옵션)
    if clear_collection:
        logger.info("기존 컬렉션 삭제 중...")
        vs = create_inclusive_growth_vector_store(
            db_user='youngseocho',
            db_password=''
        )
        vs.delete_collection()
        logger.info("✅ 기존 컬렉션 삭제 완료")
    
    # 파이프라인 생성
    pipeline = create_inclusive_growth_pipeline(
        db_user='youngseocho',
        db_password=''
    )
    
    # 출력 디렉토리 생성
    if save_intermediate:
        os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # 페이지 범위 기반 처리
    indicators = await pipeline.process_pdf_by_page_range(
        pdf_path=pdf_path,
        start_page=start_page,
        end_page=end_page,
        save_intermediate=save_intermediate,
        output_dir=OUTPUT_DIR if save_intermediate else None,
        reuse_ocr_dir=OUTPUT_DIR if reuse_ocr else None
    )
    
    logger.info(f"✅ 처리 완료: {len(indicators)}개 평가지표 저장됨")
    
    # 결과 요약 출력
    print_results(indicators)
    
    return indicators


def print_results(indicators):
    """결과 요약 출력"""
    print("\n" + "="*60)
    print("처리된 평가지표 목록")
    print("="*60)
    for i, indicator in enumerate(indicators, 1):
        print(f"{i}. {indicator.지표명}")
        print(f"   - 평가기준: {len(indicator.평가기준)}개")
        print(f"   - 평가방법: {len(indicator.평가방법)}개")
        print(f"   - 참고사항: {len(indicator.참고사항)}개")
        print(f"   - 증빙자료: {len(indicator.증빙자료)}개")
        print()


async def test_search():
    """검색 테스트"""
    vs = create_inclusive_growth_vector_store(
        db_user='youngseocho',
        db_password=''
    )
    
    print("\n" + "="*60)
    print("벡터 검색 테스트")
    print("="*60)
    
    test_queries = ["중소기업 협력", "동반성장 평가", "공정거래", "기술 지원"]
    
    for query in test_queries:
        print(f"\n🔍 검색어: '{query}'")
        results = vs.search_unique_indicators(query=query, k=3)
        for r in results:
            print(f"   {r['지표명'][:40]}... (score: {r['score']:.3f})")


async def main():
    parser = argparse.ArgumentParser(description="동반성장 평가지표 PDF 처리")
    
    # 2단계 처리 옵션
    parser.add_argument("--index-start", type=int, help="목록(목차) 시작 페이지")
    parser.add_argument("--index-end", type=int, help="목록(목차) 끝 페이지")
    parser.add_argument("--detail-start", type=int, help="세부 내용 시작 페이지")
    parser.add_argument("--detail-end", type=int, help="세부 내용 끝 페이지")
    
    # 레거시: 단일 범위 옵션
    parser.add_argument("--pdf", type=str, default=DEFAULT_PDF_PATH, help="PDF 파일 경로")
    parser.add_argument("--start", type=int, help="시작 페이지 (단일 범위)")
    parser.add_argument("--end", type=int, help="끝 페이지 (단일 범위)")
    parser.add_argument("--reuse-ocr", action="store_true", help="기존 OCR 결과 재사용")
    
    # 공통 옵션
    parser.add_argument("--save-intermediate", action="store_true", help="중간 결과 저장")
    parser.add_argument("--clear", action="store_true", help="기존 컬렉션 삭제 후 시작")
    parser.add_argument("--test-only", action="store_true", help="검색 테스트만 실행")
    
    args = parser.parse_args()
    
    if args.test_only:
        await test_search()
        return
    
    # 2단계 처리 모드
    if args.index_start is not None and args.detail_start is not None:
        if args.index_end is None or args.detail_end is None:
            print("❌ 2단계 처리 모드에서는 모든 범위가 필요합니다.")
            print("사용 예: python scripts/process_inclusive_growth.py --index-start 5 --index-end 7 --detail-start 15 --detail-end 72 --clear")
            sys.exit(1)
        
        indicators = await process_two_stage(
            index_start=args.index_start,
            index_end=args.index_end,
            detail_start=args.detail_start,
            detail_end=args.detail_end,
            save_intermediate=args.save_intermediate,
            clear_collection=args.clear
        )
    
    # 레거시: 단일 범위 처리 모드
    elif args.start is not None and args.end is not None:
        # PDF 파일 존재 확인 (OCR 재사용시 불필요)
        if not args.reuse_ocr and not os.path.exists(args.pdf):
            print(f"❌ PDF 파일을 찾을 수 없습니다: {args.pdf}")
            sys.exit(1)
        
        indicators = await process_pdf_by_range(
            pdf_path=args.pdf,
            start_page=args.start,
            end_page=args.end,
            save_intermediate=args.save_intermediate,
            clear_collection=args.clear,
            reuse_ocr=args.reuse_ocr
        )
    
    else:
        print("❌ 페이지 범위를 지정해주세요.")
        print()
        print("2단계 처리 (권장):")
        print("  python scripts/process_inclusive_growth.py --index-start 5 --index-end 7 --detail-start 15 --detail-end 72 --clear --save-intermediate")
        print()
        print("단일 범위 처리 (레거시):")
        print("  python scripts/process_inclusive_growth.py --start 15 --end 72 --reuse-ocr --clear")
        sys.exit(1)
    
    # 검색 테스트
    if indicators:
        await test_search()
    
    print("\n✅ 동반성장 평가지표 DB 구축 완료!")


if __name__ == "__main__":
    asyncio.run(main())
