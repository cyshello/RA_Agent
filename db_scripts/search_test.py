#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
검색 테스트 스크립트 (임베딩 기반 벡터 검색)

사용법:
    python db_scripts/search_test.py <query> [options]

예시:
    python db_scripts/search_test.py "인공지능 AI"
    python db_scripts/search_test.py "리더십 전략" --type management_eval
    python db_scripts/search_test.py "상생 협력" --type inclusive_growth
    python db_scripts/search_test.py "디지털 전환" --all
"""

import argparse
import json
import os
import sys

# 상위 디렉토리를 path에 추가
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.db_main import create_pipeline


def print_project_result(item: dict, idx: int):
    """국정과제 검색 결과 출력"""
    print(f"\n  [{idx}] (score: {item.get('score', 0):.4f}) {item.get('과제명', '')}")
    print(f"      과제번호: {item.get('과제번호', '')}")
    if item.get('과제 목표'):
        goals = item['과제 목표']
        if isinstance(goals, list) and len(goals) > 0:
            text = str(goals[0])[:50]
            print(f"      과제 목표: {text}..." if len(str(goals[0])) > 50 else f"      과제 목표: {text}")


def print_management_eval_result(item: dict, idx: int):
    """경영평가 검색 결과 출력"""
    print(f"\n  [{idx}] (score: {item.get('score', 0):.4f}) {item.get('지표명', '')}")
    if item.get('평가기준'):
        criteria = item['평가기준']
        if isinstance(criteria, list) and len(criteria) > 0:
            text = str(criteria[0])[:50]
            print(f"      평가기준: {text}..." if len(str(criteria[0])) > 50 else f"      평가기준: {text}")


def print_inclusive_growth_result(item: dict, idx: int):
    """동반성장 검색 결과 출력 (DB 스키마: 지표명, 평가기준)"""
    print(f"\n  [{idx}] (score: {item.get('score', 0):.4f}) {item.get('지표명', '')}")
    if item.get('평가기준'):
        criteria = item['평가기준']
        if isinstance(criteria, list) and len(criteria) > 0:
            text = str(criteria[0])[:50]
            print(f"      평가기준: {text}..." if len(str(criteria[0])) > 50 else f"      평가기준: {text}")


def main():
    parser = argparse.ArgumentParser(
        description='B2G 데이터 검색 테스트',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
검색 타입:
  project          국정과제만 검색
  management_eval  경영평가만 검색
  inclusive_growth 동반성장만 검색
  (기본: project)

예시:
  python db_scripts/search_test.py "인공지능 AI"
  python db_scripts/search_test.py "리더십 전략" --type management_eval
  python db_scripts/search_test.py "디지털 전환" --all --limit 5
  python db_scripts/search_test.py "탄소중립" --json
        """
    )
    
    parser.add_argument('query', help='검색어')
    parser.add_argument('--type', '-t', choices=['project', 'management_eval', 'inclusive_growth'],
                        default='project', help='검색 타입 (기본: project)')
    parser.add_argument('--all', '-a', action='store_true', help='모든 타입에서 검색')
    parser.add_argument('--limit', '-k', type=int, default=20, help='검색 결과 개수 (기본: 20)')
    parser.add_argument('--json', action='store_true', help='JSON 형식으로 출력')
    parser.add_argument('--db-host', default='localhost', help='MySQL 호스트')
    parser.add_argument('--db-port', type=int, default=3306, help='MySQL 포트')
    parser.add_argument('--db-name', default='b2g_data', help='데이터베이스 이름')
    parser.add_argument('--db-user', default='root', help='MySQL 사용자')
    parser.add_argument('--db-password', default='', help='MySQL 비밀번호')
    
    args = parser.parse_args()
    
    # 파이프라인 생성
    pipeline = create_pipeline(
        db_host=args.db_host,
        db_port=args.db_port,
        db_name=args.db_name,
        db_user=args.db_user,
        db_password=args.db_password
    )
    
    print("=" * 60)
    print(f"검색어: \"{args.query}\"")
    print(f"검색 개수: {args.limit}개")
    print(f"검색 방식: 임베딩 기반 벡터 검색")
    print("=" * 60)
    
    if args.all:
        # 전체 검색 (임베딩 기반)
        all_results = pipeline.store.search_all_by_embedding(args.query, k=args.limit)
        
        if args.json:
            print(json.dumps(all_results, ensure_ascii=False, indent=2))
        else:
            # 국정과제
            print(f"\n▶ 국정과제 ({len(all_results['projects'])}개)")
            print("-" * 50)
            if all_results['projects']:
                for i, item in enumerate(all_results['projects'][:10], 1):
                    print_project_result(item, i)
            else:
                print("  (검색 결과 없음)")
            
            # 경영평가
            print(f"\n▶ 경영평가 ({len(all_results['management_evals'])}개)")
            print("-" * 50)
            if all_results['management_evals']:
                for i, item in enumerate(all_results['management_evals'][:10], 1):
                    print_management_eval_result(item, i)
            else:
                print("  (검색 결과 없음)")
            
            # 동반성장
            print(f"\n▶ 동반성장 ({len(all_results['inclusive_growth'])}개)")
            print("-" * 50)
            if all_results['inclusive_growth']:
                for i, item in enumerate(all_results['inclusive_growth'][:10], 1):
                    print_inclusive_growth_result(item, i)
            else:
                print("  (검색 결과 없음)")
    
    else:
        # 특정 타입 검색
        type_names = {
            'project': '국정과제',
            'management_eval': '경영평가',
            'inclusive_growth': '동반성장'
        }
        
        if args.type == 'project':
            results = pipeline.store.search_projects_by_embedding(args.query, k=args.limit)
        elif args.type == 'management_eval':
            results = pipeline.store.search_management_evals_by_embedding(args.query, k=args.limit)
        else:
            results = pipeline.store.search_inclusive_growth_by_embedding(args.query, k=args.limit)
        
        print(f"\n▶ {type_names[args.type]} 검색 결과 ({len(results)}개)")
        print("-" * 50)
        
        if args.json:
            print(json.dumps(results, ensure_ascii=False, indent=2))
        else:
            if results:
                for i, item in enumerate(results, 1):
                    if args.type == 'project':
                        print_project_result(item, i)
                    elif args.type == 'management_eval':
                        print_management_eval_result(item, i)
                    else:
                        print_inclusive_growth_result(item, i)
            else:
                print("  (검색 결과 없음)")
    
    print()
    print("=" * 60)
    print("검색 완료")
    print("=" * 60)


if __name__ == '__main__':
    main()
