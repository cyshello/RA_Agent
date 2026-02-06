#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
PDF를 처리하여 DB에 저장하는 스크립트

사용법:
    python scripts/load_pdf_to_db.py <pdf_path> <data_type> [options]

예시:
    # 새로운 PDF 처리 (OCR 포함)
    python scripts/load_pdf_to_db.py ./data/국정과제.pdf project --index-pages 3-5 --detail-pages 6-100
    
    # 기존 OCR 결과 재사용 (빠름)
    python scripts/load_pdf_to_db.py ./data/국정과제.pdf project --index-pages 3-5 --detail-pages 6-100 --reuse-ocr ./ocr_cache/project
    
    # 중간 결과 저장 (디버깅용)
    python scripts/load_pdf_to_db.py ./data/경영평가.pdf management_eval --index-pages 2-3 --detail-pages 4-50 --output-dir ./output/management
"""

import argparse
import asyncio
import os
import sys

# 상위 디렉토리를 path에 추가
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.db_main import create_pipeline


def parse_page_range(page_range_str: str) -> tuple:
    """페이지 범위 문자열을 튜플로 변환 (예: '3-5' -> (3, 5))"""
    parts = page_range_str.split('-')
    if len(parts) != 2:
        raise ValueError(f"잘못된 페이지 범위 형식: {page_range_str} (예: '3-5')")
    return int(parts[0]), int(parts[1])


async def main():
    parser = argparse.ArgumentParser(
        description='PDF를 처리하여 MySQL DB에 저장',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
데이터 타입:
  project          국정과제 (과제명, 과제_목표, 주요내용, 기대효과)
  management_eval  경영평가 (지표명, 평가기준, 평가방법, 참고사항, 증빙자료)
  inclusive_growth 동반성장 (세부추진과제명, 세부내용)

예시:
  # 새로운 PDF 처리
  python scripts/load_pdf_to_db.py ./국정과제.pdf project --index-pages 13-17 --detail-pages 21-195
  
  # 기존 OCR 결과 재사용
  python scripts/load_pdf_to_db.py ./국정과제.pdf project --index-pages 13-17 --detail-pages 21-195 --reuse-ocr ./ocr_cache/project
  
  # OCR만 먼저 수행하려면:
  python scripts/run_ocr_only.py ./국정과제.pdf --pages 13-195 --output-dir ./ocr_cache/project
        """
    )
    
    parser.add_argument('pdf_path', help='PDF 파일 경로')
    parser.add_argument('data_type', choices=['project', 'management_eval', 'inclusive_growth'],
                        help='데이터 타입 (project/management_eval/inclusive_growth)')
    parser.add_argument('--index-pages', required=True, help='목록 페이지 범위 (예: 3-5)')
    parser.add_argument('--detail-pages', required=True, help='세부내용 페이지 범위 (예: 6-100)')
    parser.add_argument('--output-dir', default=None, help='중간 결과 저장 디렉토리 (OCR, item_list, item_details 등)')
    parser.add_argument('--reuse-ocr', default=None, help='기존 OCR 결과 디렉토리 (재사용하여 OCR 단계 스킵)')
    parser.add_argument('--db-host', default='localhost', help='MySQL 호스트')
    parser.add_argument('--db-port', type=int, default=3306, help='MySQL 포트')
    parser.add_argument('--db-name', default='b2g_data', help='데이터베이스 이름')
    parser.add_argument('--db-user', default='root', help='MySQL 사용자')
    parser.add_argument('--db-password', default='', help='MySQL 비밀번호')
    
    args = parser.parse_args()
    
    # PDF 파일 확인
    if not os.path.exists(args.pdf_path):
        print(f"오류: PDF 파일을 찾을 수 없습니다: {args.pdf_path}")
        sys.exit(1)
    
    # 페이지 범위 파싱
    try:
        index_pages = parse_page_range(args.index_pages)
        detail_pages = parse_page_range(args.detail_pages)
    except ValueError as e:
        print(f"오류: {e}")
        sys.exit(1)
    
    # 데이터 타입 한글 변환
    type_names = {
        'project': '국정과제',
        'management_eval': '경영평가',
        'inclusive_growth': '동반성장'
    }
    
    type_fields = {
        'project': '과제명, 과제_목표, 주요내용, 기대효과',
        'management_eval': '지표명, 평가기준, 평가방법, 참고사항, 증빙자료',
        'inclusive_growth': '세부추진과제명, 세부내용'
    }
    
    print("=" * 60)
    print(f"PDF 처리 시작")
    print("=" * 60)
    print(f"  PDF 파일: {args.pdf_path}")
    print(f"  데이터 타입: {type_names[args.data_type]}")
    print(f"  추출 필드: {type_fields[args.data_type]}")
    print(f"  목록 페이지: {index_pages[0]} ~ {index_pages[1]}")
    print(f"  세부 페이지: {detail_pages[0]} ~ {detail_pages[1]}")
    if args.reuse_ocr:
        print(f"  OCR 재사용: {args.reuse_ocr}")
    if args.output_dir:
        print(f"  중간결과 저장: {args.output_dir}")
    print(f"  DB: {args.db_user}@{args.db_host}:{args.db_port}/{args.db_name}")
    print("=" * 60)
    
    # 파이프라인 생성
    pipeline = create_pipeline(
        db_host=args.db_host,
        db_port=args.db_port,
        db_name=args.db_name,
        db_user=args.db_user,
        db_password=args.db_password
    )
    
    # PDF 처리
    save_intermediate = args.output_dir is not None
    
    results = await pipeline.process_pdf(
        pdf_path=args.pdf_path,
        data_type=args.data_type,
        index_pages=index_pages,
        detail_pages=detail_pages,
        save_intermediate=save_intermediate,
        output_dir=args.output_dir,
        reuse_ocr_dir=args.reuse_ocr
    )
    
    print()
    print("=" * 60)
    print(f"처리 완료: {len(results)}개 항목 저장")
    print("=" * 60)
    
    # 저장된 항목 요약 출력
    for i, item in enumerate(results[:5], 1):
        data = item.to_dict()
        if args.data_type == 'project':
            print(f"  {i}. [{data.get('과제번호', '')}] {data.get('과제명', '')[:40]}")
        elif args.data_type == 'management_eval':
            print(f"  {i}. {data.get('지표명', '')[:50]}")
        else:
            print(f"  {i}. {data.get('세부추진과제명', '')[:50]}")
    
    if len(results) > 5:
        print(f"  ... 외 {len(results) - 5}개")
    
    # DB 통계
    print()
    stats = pipeline.get_stats()
    print(f"DB 현황: 국정과제 {stats['national_projects']}개 | "
          f"경영평가 {stats['management_evals']}개 | "
          f"동반성장 {stats['inclusive_growth']}개")


if __name__ == '__main__':
    asyncio.run(main())
