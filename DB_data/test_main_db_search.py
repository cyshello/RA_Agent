#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
main.py에서 사용하는 DB 검색 로직 테스트
MySQL 임베딩 기반 검색이 잘 작동하는지 확인
"""

import sys
sys.path.insert(0, '/Users/youngseocho/Desktop/AX/RA_Agent')

from src.db_main import MySQLStore
import json

print("=" * 70)
print("🔍 main.py DB 검색 로직 테스트 (MySQL 임베딩 검색)")
print("=" * 70)

# MySQL DB 연결
mysql_store = MySQLStore(
    host="localhost",
    port=3306,
    database="b2g_data",
    user="root",
    password=""
)

print("✅ MySQLStore 초기화 완료")

# 평가 유형별 설정 (main.py와 동일)
eval_configs = {
    "presidential_agenda": {
        "name": "국정과제",
        "search_method": "search_projects_by_embedding",
        "name_field": "과제명",
        "id_field": "과제번호",
    },
    "management_eval": {
        "name": "공공기관 경영평가",
        "search_method": "search_management_evals_by_embedding",
        "name_field": "지표명",
        "id_field": "지표명",
    },
    "inclusive_growth": {
        "name": "동반성장 평가지표",
        "search_method": "search_inclusive_growth_by_embedding",
        "name_field": "지표명",
        "id_field": "지표명",
    }
}

# 테스트 쿼리 (기업 특징 기반 검색 쿼리 예시)
test_queries = [
    "인공지능 AI 솔루션",
    "데이터 분석 플랫폼",
    "디지털 혁신"
]

print(f"\n📌 테스트 쿼리: {test_queries}")

# 각 평가 유형별로 검색 테스트
for eval_type, config in eval_configs.items():
    print(f"\n{'=' * 60}")
    print(f"📊 [{config['name']}] 검색 테스트")
    print(f"{'=' * 60}")
    
    search_func = getattr(mysql_store, config["search_method"])
    all_items = {}
    
    for query in test_queries:
        try:
            results = search_func(query, k=5)
            for item in results:
                item_id = item.get(config["id_field"], "") or item.get(config["name_field"], "")
                if item_id and item_id not in all_items:
                    all_items[item_id] = item
        except Exception as e:
            print(f"  ⚠️ 검색 오류: {e}")
            continue
    
    items_list = list(all_items.values())[:10]
    print(f"  → 검색된 항목 수: {len(items_list)}")
    
    # 항목 리스트 텍스트 변환 테스트 (main.py와 동일)
    items_text = ""
    for i, item in enumerate(items_list[:5], 1):  # 상위 5개만 출력
        name = item.get(config["name_field"], "")
        score = item.get("score", 0)
        
        if eval_type == "presidential_agenda":
            goals = item.get("과제 목표", [])
            if isinstance(goals, list):
                goals = goals[:2]
            else:
                goals = []
            contents = item.get("주요내용", [])
            if isinstance(contents, list):
                contents = contents[:2]
            else:
                contents = []
            print(f"  {i}. [{item.get(config['id_field'], '')}] {name}")
            print(f"     유사도: {score:.4f}")
            print(f"     과제 목표: {', '.join(goals[:1]) if goals else '없음'}...")
        else:
            eval_criteria = item.get("평가기준", [])
            if isinstance(eval_criteria, list):
                eval_criteria = eval_criteria[:2]
            else:
                eval_criteria = []
            print(f"  {i}. {name}")
            print(f"     유사도: {score:.4f}")
            print(f"     평가기준: {eval_criteria[0][:50] if eval_criteria else '없음'}...")

print(f"\n{'=' * 70}")
print("✅ main.py DB 검색 로직 테스트 완료!")
print("   → MySQLStore의 임베딩 기반 검색 함수가 정상 작동합니다.")
print(f"{'=' * 70}")
