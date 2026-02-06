#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
DB 내용을 확인하는 스크립트

사용법:
    python scripts/check_db.py [table] [options]

예시:
    python scripts/check_db.py                    # 전체 통계
    python scripts/check_db.py project            # 국정과제 테이블 조회
    python scripts/check_db.py management_eval    # 경영평가 테이블 조회
    python scripts/check_db.py inclusive_growth   # 동반성장 테이블 조회
    python scripts/check_db.py project --limit 5  # 상위 5개만 조회
    python scripts/check_db.py --delete project   # 국정과제 테이블 삭제
    python scripts/check_db.py --delete-all       # 전체 삭제
"""

import argparse
import json
import os
import sys

# 상위 디렉토리를 path에 추가
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.db_main import create_store


def safe_join(data, max_items=2):
    """리스트를 안전하게 문자열로 변환"""
    if not data:
        return ''
    if isinstance(data, str):
        return data[:100]
    if isinstance(data, list):
        items = [str(x)[:50] for x in data[:max_items]]
        suffix = '...' if len(data) > max_items else ''
        return ', '.join(items) + suffix
    return str(data)[:100]


def print_project(item: dict, idx: int):
    """국정과제 출력"""
    print(f"\n[{idx}] {item.get('과제명', '')}")
    print(f"    과제번호: {item.get('과제번호', '')}")
    if item.get('과제 목표'):
        print(f"    과제 목표: {safe_join(item['과제 목표'])}")
    if item.get('주요내용'):
        print(f"    주요내용: {safe_join(item['주요내용'])}")
    if item.get('기대효과'):
        print(f"    기대효과: {safe_join(item['기대효과'])}")
    print(f"    출처: {item.get('source_document', '')} (p.{item.get('page_range', '')})")


def print_management_eval(item: dict, idx: int):
    """경영평가 출력"""
    print(f"\n[{idx}] {item.get('지표명', '')}")
    if item.get('평가기준'):
        print(f"    평가기준: {safe_join(item['평가기준'])}")
    if item.get('평가방법'):
        print(f"    평가방법: {safe_join(item['평가방법'])}")
    if item.get('참고사항'):
        print(f"    참고사항: {safe_join(item['참고사항'])}")
    if item.get('증빙자료'):
        print(f"    증빙자료: {safe_join(item['증빙자료'])}")
    print(f"    출처: {item.get('source_document', '')} (p.{item.get('page_range', '')})")


def print_inclusive_growth(item: dict, idx: int):
    """동반성장 출력 (DB 스키마: 지표명, 평가기준, 평가방법)"""
    print(f"\n[{idx}] {item.get('지표명', '')}")
    if item.get('평가기준'):
        criteria = item['평가기준']
        if isinstance(criteria, list):
            for content in criteria[:3]:
                text = str(content)
                print(f"    - {text[:60]}{'...' if len(text) > 60 else ''}")
            if len(criteria) > 3:
                print(f"    ... 외 {len(criteria) - 3}개")
        else:
            print(f"    - {str(criteria)[:60]}...")
    if item.get('평가방법'):
        method = item['평가방법']
        if isinstance(method, list) and len(method) > 0:
            print(f"    평가방법: {str(method[0])[:50]}...")
    print(f"    출처: {item.get('source_document', '')} (p.{item.get('page_range', '')})")


def main():
    parser = argparse.ArgumentParser(
        description='MySQL DB 내용 확인',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
테이블 이름:
  project          국정과제 (national_projects)
  management_eval  경영평가 (management_evals)
  inclusive_growth 동반성장 (inclusive_growth)

예시:
  python scripts/check_db.py                    # 전체 통계만 보기
  python scripts/check_db.py project            # 국정과제 전체 조회
  python scripts/check_db.py project --limit 5  # 국정과제 5개만 조회
  python scripts/check_db.py --delete project   # 국정과제 테이블 삭제
  python scripts/check_db.py --delete-all       # 전체 삭제
        """
    )
    
    parser.add_argument('table', nargs='?', choices=['project', 'management_eval', 'inclusive_growth'],
                        help='조회할 테이블 (생략시 통계만 출력)')
    parser.add_argument('--limit', type=int, default=20, help='조회할 최대 개수 (기본: 20)')
    parser.add_argument('--json', action='store_true', help='JSON 형식으로 출력')
    parser.add_argument('--delete', metavar='TABLE', help='특정 테이블 데이터 삭제')
    parser.add_argument('--delete-all', action='store_true', help='전체 데이터 삭제')
    parser.add_argument('--db-host', default='localhost', help='MySQL 호스트')
    parser.add_argument('--db-port', type=int, default=3306, help='MySQL 포트')
    parser.add_argument('--db-name', default='b2g_data', help='데이터베이스 이름')
    parser.add_argument('--db-user', default='root', help='MySQL 사용자')
    parser.add_argument('--db-password', default='', help='MySQL 비밀번호')
    
    args = parser.parse_args()
    
    # 스토어 생성
    store = create_store(
        db_host=args.db_host,
        db_port=args.db_port,
        db_name=args.db_name,
        db_user=args.db_user,
        db_password=args.db_password
    )
    
    # 삭제 처리
    if args.delete_all:
        confirm = input("정말 전체 데이터를 삭제하시겠습니까? (yes 입력): ")
        if confirm.lower() == 'yes':
            store.delete_all_data()
            print("전체 데이터 삭제 완료")
        else:
            print("취소됨")
        return
    
    if args.delete:
        table_map = {
            'project': 'national_projects',
            'management_eval': 'management_evals',
            'inclusive_growth': 'inclusive_growth'
        }
        if args.delete not in table_map:
            print(f"오류: 알 수 없는 테이블: {args.delete}")
            sys.exit(1)
        confirm = input(f"{args.delete} 테이블 데이터를 삭제하시겠습니까? (yes 입력): ")
        if confirm.lower() == 'yes':
            store.delete_all_data(table_map[args.delete])
            print(f"{args.delete} 테이블 데이터 삭제 완료")
        else:
            print("취소됨")
        return
    
    # 통계 출력
    stats = store.get_stats()
    print("=" * 60)
    print("DB 현황")
    print("=" * 60)
    print(f"  국정과제 (national_projects):    {stats['national_projects']:>5}개")
    print(f"  경영평가 (management_evals):     {stats['management_evals']:>5}개")
    print(f"  동반성장 (inclusive_growth):     {stats['inclusive_growth']:>5}개")
    print(f"  {'─' * 40}")
    print(f"  총계:                            {sum(stats.values()):>5}개")
    print("=" * 60)
    
    # 테이블 조회
    if args.table:
        type_names = {
            'project': '국정과제',
            'management_eval': '경영평가',
            'inclusive_growth': '동반성장'
        }
        print(f"\n{type_names[args.table]} 목록 (최대 {args.limit}개)")
        print("-" * 60)
        
        # 빈 쿼리로 전체 검색 (LIKE 검색 fallback 이용)
        if args.table == 'project':
            results = store.search_projects('과제', k=args.limit)
            if args.json:
                print(json.dumps(results, ensure_ascii=False, indent=2))
            else:
                for i, item in enumerate(results, 1):
                    print_project(item, i)
        
        elif args.table == 'management_eval':
            results = store.search_management_evals('평가', k=args.limit)
            if args.json:
                print(json.dumps(results, ensure_ascii=False, indent=2))
            else:
                for i, item in enumerate(results, 1):
                    print_management_eval(item, i)
        
        else:  # inclusive_growth
            results = store.search_inclusive_growth('추진', k=args.limit)
            if args.json:
                print(json.dumps(results, ensure_ascii=False, indent=2))
            else:
                for i, item in enumerate(results, 1):
                    print_inclusive_growth(item, i)
        
        if not results:
            print("  (데이터 없음)")


if __name__ == '__main__':
    main()
