#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
DB 내용을 확인하는 스크립트

사용법:
    python db_scripts/check_db.py [table] [options]

예시:
    python db_scripts/check_db.py                    # 전체 통계
    python db_scripts/check_db.py project            # 국정과제 테이블 조회
    python db_scripts/check_db.py management_eval    # 경영평가 테이블 조회
    python db_scripts/check_db.py inclusive_growth   # 동반성장 테이블 조회
    python db_scripts/check_db.py project --limit 5  # 상위 5개만 조회
    python db_scripts/check_db.py --delete project   # 국정과제 테이블 삭제
    python db_scripts/check_db.py --delete-all       # 전체 삭제
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
    """동반성장 출력 (DB 스키마: 세부추진과제명, 세부내용)"""
    print(f"\n[{idx}] {item.get('세부추진과제명', '')}")
    if item.get('세부내용'):
        details = item['세부내용']
        if isinstance(details, list):
            for content in details[:3]:
                text = str(content)
                print(f"    - {text[:60]}{'...' if len(text) > 60 else ''}")
            if len(details) > 3:
                print(f"    ... 외 {len(details) - 3}개")
        else:
            print(f"    - {str(details)[:60]}...")
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
  python db_scripts/check_db.py                    # 전체 통계만 보기
  python db_scripts/check_db.py project            # 국정과제 전체 조회
  python db_scripts/check_db.py project --limit 5  # 국정과제 5개만 조회
  python db_scripts/check_db.py --delete project   # 국정과제 테이블 삭제
  python db_scripts/check_db.py --delete-all       # 전체 삭제
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
    
    # 메인 테이블 개수
    main_tables = ['national_projects', 'management_evals', 'inclusive_growth']
    main_count = sum(stats.get(t, 0) for t in main_tables)
    
    # 임베딩 청크 테이블별 개수
    chunks_project = stats.get('embedding_chunks_project', 0)
    chunks_management = stats.get('embedding_chunks_management', 0)
    chunks_inclusive = stats.get('embedding_chunks_inclusive', 0)
    total_chunks = chunks_project + chunks_management + chunks_inclusive
    
    print("=" * 60)
    print("DB 현황")
    print("=" * 60)
    print(f"  국정과제 (national_projects):    {stats.get('national_projects', 0):>5}개")
    print(f"  경영평가 (management_evals):     {stats.get('management_evals', 0):>5}개")
    print(f"  동반성장 (inclusive_growth):     {stats.get('inclusive_growth', 0):>5}개")
    print(f"  {'─' * 40}")
    print(f"  지표/과제 총계:                  {main_count:>5}개")
    print()
    print(f"  [임베딩 청크 테이블]")
    print(f"    embedding_chunks_project:      {chunks_project:>5}개")
    print(f"    embedding_chunks_management:   {chunks_management:>5}개")
    print(f"    embedding_chunks_inclusive:    {chunks_inclusive:>5}개")
    print(f"    ────────────────────────────────────")
    print(f"    임베딩 총계:                   {total_chunks:>5}개")
    print("=" * 60)
    
    # 테이블 조회
    if args.table:
        type_names = {
            'project': '국정과제',
            'management_eval': '경영평가',
            'inclusive_growth': '동반성장'
        }
        table_map = {
            'project': 'national_projects',
            'management_eval': 'management_evals',
            'inclusive_growth': 'inclusive_growth'
        }
        print(f"\n{type_names[args.table]} 목록 (최대 {args.limit}개)")
        print("-" * 60)
        
        # 직접 SQL로 데이터 조회 (pymysql 사용)
        import pymysql
        from pymysql.cursors import DictCursor
        conn = pymysql.connect(
            host=args.db_host,
            port=args.db_port,
            database=args.db_name,
            user=args.db_user,
            password=args.db_password,
            charset='utf8mb4',
            cursorclass=DictCursor
        )
        table_name = table_map[args.table]
        with conn.cursor() as cursor:
            cursor.execute(f"SELECT * FROM {table_name} ORDER BY id LIMIT %s", (args.limit,))
            results = cursor.fetchall()
        conn.close()
        
        if args.json:
            # JSON 필드 파싱
            for r in results:
                for key, val in r.items():
                    if isinstance(val, str) and val.startswith('['):
                        try:
                            r[key] = json.loads(val)
                        except:
                            pass
            print(json.dumps(results, ensure_ascii=False, indent=2, default=str))
        else:
            for i, item in enumerate(results, 1):
                # JSON 필드 파싱
                for key, val in item.items():
                    if isinstance(val, str) and val.startswith('['):
                        try:
                            item[key] = json.loads(val)
                        except:
                            pass
                
                if args.table == 'project':
                    print_project(item, i)
                elif args.table == 'management_eval':
                    print_management_eval(item, i)
                else:
                    print_inclusive_growth(item, i)
        
        if not results:
            print("  (데이터 없음)")


if __name__ == '__main__':
    main()
