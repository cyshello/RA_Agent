#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
PDF에서 OCR만 수행하여 저장하는 스크립트 (LLM 처리 없이)

사용법:
    python scripts/run_ocr_only.py <pdf_path> --pages <start>-<end> --output-dir <dir>

예시:
    # 전체 페이지 OCR
    python scripts/run_ocr_only.py ./data/criteria/management_eval.pdf --output-dir ./ocr_cache/management_eval
    
    # 특정 페이지 범위만 OCR  
    python scripts/run_ocr_only.py ./data/criteria/presidential_agenda.pdf --pages 13-195 --output-dir ./ocr_cache/project
"""

import argparse
import os
import sys
import json

# 상위 디렉토리를 path에 추가
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.pdf_extractor import PDFProcessor


def parse_page_range(page_range_str: str) -> tuple:
    """페이지 범위 문자열을 튜플로 변환 (예: '3-5' -> (3, 5))"""
    parts = page_range_str.split('-')
    if len(parts) != 2:
        raise ValueError(f"잘못된 페이지 범위 형식: {page_range_str} (예: '3-5')")
    return int(parts[0]), int(parts[1])


def main():
    parser = argparse.ArgumentParser(
        description='PDF에서 OCR만 수행하여 저장 (LLM 처리 없이)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
예시:
  # 전체 페이지 OCR
  python scripts/run_ocr_only.py ./data/criteria/management_eval.pdf --output-dir ./ocr_cache/management_eval
  
  # 특정 페이지 범위만 OCR
  python scripts/run_ocr_only.py ./data/criteria/presidential_agenda.pdf --pages 13-195 --output-dir ./ocr_cache/project
  
  # 기존 OCR이 있는 페이지는 건너뛰기
  python scripts/run_ocr_only.py ./data/criteria/inclusive_growth.pdf --pages 2-17 --output-dir ./ocr_cache/inclusive --skip-existing
        """
    )
    
    parser.add_argument('pdf_path', help='PDF 파일 경로')
    parser.add_argument('--pages', default=None, help='OCR할 페이지 범위 (예: 1-100). 생략시 전체 페이지')
    parser.add_argument('--output-dir', required=True, help='OCR 결과 저장 디렉토리')
    parser.add_argument('--skip-existing', action='store_true', help='이미 OCR 결과가 있는 페이지는 건너뛰기')
    parser.add_argument('--dpi', type=int, default=200, help='PDF 변환 DPI (기본: 200)')
    
    args = parser.parse_args()
    
    # PDF 파일 확인
    if not os.path.exists(args.pdf_path):
        print(f"오류: PDF 파일을 찾을 수 없습니다: {args.pdf_path}")
        sys.exit(1)
    
    # 출력 디렉토리 생성
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("=" * 60)
    print(f"OCR 전용 처리 시작")
    print("=" * 60)
    print(f"  PDF 파일: {args.pdf_path}")
    print(f"  출력 디렉토리: {args.output_dir}")
    print(f"  DPI: {args.dpi}")
    print("=" * 60)
    
    # PDF 프로세서 생성 및 이미지 변환
    processor = PDFProcessor()
    
    print(f"\n📄 PDF를 이미지로 변환 중...")
    pdf_images = processor.pdf_to_images(args.pdf_path, dpi=args.dpi)
    total_pages = len(pdf_images)
    print(f"   총 {total_pages} 페이지")
    
    # 페이지 범위 결정
    if args.pages:
        try:
            start_page, end_page = parse_page_range(args.pages)
        except ValueError as e:
            print(f"오류: {e}")
            sys.exit(1)
    else:
        start_page, end_page = 1, total_pages
    
    # 범위 검증
    if start_page < 1:
        start_page = 1
    if end_page > total_pages:
        end_page = total_pages
    
    print(f"\n🔍 OCR 수행: 페이지 {start_page} ~ {end_page}")
    
    processed = 0
    skipped = 0
    failed = 0
    
    for page_num in range(start_page, end_page + 1):
        output_file = os.path.join(args.output_dir, f"ocr_page_{page_num}.json")
        
        # 기존 파일 건너뛰기
        if args.skip_existing and os.path.exists(output_file):
            print(f"   페이지 {page_num}: 이미 존재 (건너뜀)")
            skipped += 1
            continue
        
        try:
            idx = page_num - 1
            result = processor.process_page(pdf_images[idx], page_num)
            
            ocr_data = {
                'page_num': result.page_num,
                'text': result.text,
                'fields': result.fields,
                'tables': result.tables
            }
            
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(ocr_data, f, ensure_ascii=False, indent=2)
            
            text_preview = result.text[:50].replace('\n', ' ') if result.text else "(텍스트 없음)"
            table_count = len(result.tables) if result.tables else 0
            print(f"   페이지 {page_num}: ✅ (표 {table_count}개) {text_preview}...")
            processed += 1
            
        except Exception as e:
            print(f"   페이지 {page_num}: ❌ 오류 - {e}")
            failed += 1
    
    print()
    print("=" * 60)
    print(f"OCR 완료")
    print("=" * 60)
    print(f"  처리됨: {processed}개")
    print(f"  건너뜀: {skipped}개")
    print(f"  실패: {failed}개")
    print(f"  저장 위치: {args.output_dir}")
    print()
    print("이 OCR 결과를 재사용하려면:")
    print(f"  python scripts/load_pdf_to_db.py <pdf> <type> --reuse-ocr {args.output_dir} ...")


if __name__ == '__main__':
    main()
