#!/usr/bin/env python3
"""
평가지표 PDF 처리 스크립트 (2단계 처리) - 범용

지원 문서 유형:
    - inclusive_growth: 동반성장 평가지표
    - management_eval: 공공기관 경영평가

사용 방법:
    # 동반성장 평가지표
    python scripts/process_evaluation_indicators.py --doc-type inclusive_growth \
        --index-start 12 --index-end 13 --detail-start 15 --detail-end 72 --clear

    # 공공기관 경영평가
    python scripts/process_evaluation_indicators.py --doc-type management_eval \
        --index-start 5 --index-end 10 --detail-start 20 --detail-end 100 --clear

인자:
    --doc-type: 문서 유형 (inclusive_growth, management_eval)
    --index-start: 목록(목차) 시작 페이지
    --index-end: 목록(목차) 끝 페이지
    --detail-start: 세부 내용 시작 페이지
    --detail-end: 세부 내용 끝 페이지
    --pdf: PDF 파일 경로 (지정하지 않으면 기본값 사용)
    --ocr-dir: OCR 결과 저장/재사용 디렉토리 (지정하지 않으면 기본값 사용)
    --clear: 기존 컬렉션 삭제 후 시작
    --save-intermediate: 중간 결과 저장
    --test-only: 검색 테스트만 실행
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

# ============================================================================
# 문서 유형별 설정
# ============================================================================

DOC_TYPE_CONFIGS = {
    "inclusive_growth": {
        "name": "동반성장 평가지표",
        "collection_name": "inclusive_growth_indicators",
        "default_pdf": "/Users/youngseocho/Desktop/AX/RA_Agent/data/inclusive_growth.pdf",
        "default_ocr_dir": "/Users/youngseocho/Desktop/AX/RA_Agent/data/output_inclusive_growth",
        "test_queries": ["중소기업 협력", "동반성장 평가", "공정거래", "기술 지원"]
    },
    "management_eval": {
        "name": "공공기관 경영평가",
        "collection_name": "management_eval_indicators",
        "default_pdf": "/Users/youngseocho/Desktop/AX/RA_Agent/data/management_eval.pdf",
        "default_ocr_dir": "/Users/youngseocho/Desktop/AX/RA_Agent/data/output_management_eval",
        "test_queries": ["경영평가", "성과관리", "조직운영", "재무관리"]
    }
}


def get_config(doc_type: str) -> dict:
    """문서 유형별 설정 반환"""
    if doc_type not in DOC_TYPE_CONFIGS:
        raise ValueError(f"지원하지 않는 문서 유형: {doc_type}. 지원: {list(DOC_TYPE_CONFIGS.keys())}")
    return DOC_TYPE_CONFIGS[doc_type]


async def process_two_stage(
    doc_type: str,
    index_start: int,
    index_end: int,
    detail_start: int,
    detail_end: int,
    pdf_path: str = None,
    ocr_dir: str = None,
    save_intermediate: bool = False,
    clear_collection: bool = False
):
    """
    2단계 처리: 목록 페이지에서 지표명 추출 → 세부 페이지에서 상세 정보 채우기
    """
    config = get_config(doc_type)
    
    # 기본값 설정
    if pdf_path is None:
        pdf_path = config["default_pdf"]
    if ocr_dir is None:
        ocr_dir = config["default_ocr_dir"]
    
    logger.info("="*60)
    logger.info(f"{config['name']} 2단계 처리 시작")
    logger.info(f"  - 컬렉션: {config['collection_name']}")
    logger.info(f"  - PDF: {pdf_path}")
    logger.info(f"  - 목록 페이지: {index_start} ~ {index_end}")
    logger.info(f"  - 세부 페이지: {detail_start} ~ {detail_end}")
    logger.info(f"  - OCR 디렉토리: {ocr_dir}")
    logger.info("="*60)
    
    # 기존 컬렉션 삭제 (옵션)
    if clear_collection:
        logger.info("기존 컬렉션 삭제 중...")
        vs = create_inclusive_growth_vector_store(
            db_user='youngseocho',
            db_password='',
            collection_name=config['collection_name']
        )
        vs.delete_collection()
        logger.info("✅ 기존 컬렉션 삭제 완료")
    
    # 파이프라인 생성
    pipeline = create_inclusive_growth_pipeline(
        db_user='youngseocho',
        db_password='',
        collection_name=config['collection_name']
    )
    
    # 출력 디렉토리 생성
    os.makedirs(ocr_dir, exist_ok=True)
    
    # 2단계 처리 실행 (OCR이 없으면 자동으로 수행)
    indicators = await pipeline.process_two_stage(
        index_start=index_start,
        index_end=index_end,
        detail_start=detail_start,
        detail_end=detail_end,
        save_intermediate=save_intermediate,
        output_dir=ocr_dir,
        reuse_ocr_dir=ocr_dir,
        source_document=os.path.basename(pdf_path),
        pdf_path=pdf_path  # OCR 수행시 필요
    )
    
    logger.info(f"✅ 처리 완료: {len(indicators)}개 평가지표 저장됨")
    
    # 결과 요약 출력
    print_results(indicators, config['name'])
    
    return indicators


def print_results(indicators, doc_name: str):
    """결과 요약 출력"""
    print("\n" + "="*60)
    print(f"{doc_name} - 처리된 평가지표 목록")
    print("="*60)
    for i, indicator in enumerate(indicators, 1):
        print(f"{i}. {indicator.지표명}")
        print(f"   - 평가기준: {len(indicator.평가기준)}개")
        print(f"   - 평가방법: {len(indicator.평가방법)}개")
        print(f"   - 참고사항: {len(indicator.참고사항)}개")
        print(f"   - 증빙자료: {len(indicator.증빙자료)}개")
        print()


async def test_search(doc_type: str):
    """검색 테스트"""
    config = get_config(doc_type)
    
    vs = create_inclusive_growth_vector_store(
        db_user='youngseocho',
        db_password='',
        collection_name=config['collection_name']
    )
    
    print("\n" + "="*60)
    print(f"{config['name']} 벡터 검색 테스트")
    print("="*60)
    
    for query in config['test_queries']:
        print(f"\n🔍 검색어: '{query}'")
        results = vs.search_unique_indicators(query=query, k=3)
        for r in results:
            name = r['지표명'][:40] if len(r['지표명']) > 40 else r['지표명']
            print(f"   {name}... (score: {r['score']:.3f})")


async def main():
    parser = argparse.ArgumentParser(
        description="평가지표 PDF 처리 (동반성장/공공기관 경영평가)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
사용 예시:
  # 동반성장 평가지표
  python scripts/process_evaluation_indicators.py --doc-type inclusive_growth \\
      --index-start 12 --index-end 13 --detail-start 15 --detail-end 72 --clear --save-intermediate

  # 공공기관 경영평가
  python scripts/process_evaluation_indicators.py --doc-type management_eval \\
      --index-start 5 --index-end 10 --detail-start 20 --detail-end 100 --clear --save-intermediate

  # 검색 테스트만
  python scripts/process_evaluation_indicators.py --doc-type inclusive_growth --test-only
        """
    )
    
    # 문서 유형
    parser.add_argument("--doc-type", type=str, required=True,
                        choices=list(DOC_TYPE_CONFIGS.keys()),
                        help="문서 유형 (inclusive_growth: 동반성장, management_eval: 공공기관 경영평가)")
    
    # 2단계 처리 옵션
    parser.add_argument("--index-start", type=int, help="목록(목차) 시작 페이지")
    parser.add_argument("--index-end", type=int, help="목록(목차) 끝 페이지")
    parser.add_argument("--detail-start", type=int, help="세부 내용 시작 페이지")
    parser.add_argument("--detail-end", type=int, help="세부 내용 끝 페이지")
    
    # 경로 옵션
    parser.add_argument("--pdf", type=str, help="PDF 파일 경로 (지정하지 않으면 기본값)")
    parser.add_argument("--ocr-dir", type=str, help="OCR 결과 디렉토리 (지정하지 않으면 기본값)")
    
    # 공통 옵션
    parser.add_argument("--save-intermediate", action="store_true", help="중간 결과 저장")
    parser.add_argument("--clear", action="store_true", help="기존 컬렉션 삭제 후 시작")
    parser.add_argument("--test-only", action="store_true", help="검색 테스트만 실행")
    
    args = parser.parse_args()
    
    # 검색 테스트만
    if args.test_only:
        await test_search(args.doc_type)
        return
    
    # 2단계 처리 모드
    if args.index_start is not None and args.detail_start is not None:
        if args.index_end is None or args.detail_end is None:
            print("❌ 2단계 처리 모드에서는 모든 범위가 필요합니다.")
            config = get_config(args.doc_type)
            print(f"\n사용 예:")
            print(f"  python scripts/process_evaluation_indicators.py --doc-type {args.doc_type} \\")
            print(f"      --index-start 5 --index-end 10 --detail-start 15 --detail-end 100 --clear")
            sys.exit(1)
        
        indicators = await process_two_stage(
            doc_type=args.doc_type,
            index_start=args.index_start,
            index_end=args.index_end,
            detail_start=args.detail_start,
            detail_end=args.detail_end,
            pdf_path=args.pdf,
            ocr_dir=args.ocr_dir,
            save_intermediate=args.save_intermediate,
            clear_collection=args.clear
        )
        
        # 검색 테스트
        if indicators:
            await test_search(args.doc_type)
        
        config = get_config(args.doc_type)
        print(f"\n✅ {config['name']} DB 구축 완료!")
    
    else:
        config = get_config(args.doc_type)
        print("❌ 페이지 범위를 지정해주세요.")
        print(f"\n{config['name']} 처리 예시:")
        print(f"  python scripts/process_evaluation_indicators.py --doc-type {args.doc_type} \\")
        print(f"      --index-start 5 --index-end 10 --detail-start 15 --detail-end 100 \\")
        print(f"      --clear --save-intermediate")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
